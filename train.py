# train.py
import argparse
import os
import csv
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from model import VAEUNet
from data_loader import get_dataloader
from losses import (
    reconstruction_loss,
    kl_divergence,
    segmentation_loss,
    dice_loss,
)
# -----------------------
# EMA class
# -----------------------

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p, alpha=1 - self.decay)

    def apply_to(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[name])

def save_checkpoint(path, model, ema, optimizer, epoch, history, best_val):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.shadow if ema else None,
            "optimizer": optimizer.state_dict(),
            "history": history,
            "best_val": best_val,
        },
        path,
    )
    print(f"[Checkpoint saved] {path}")


def save_history_csv(history, out_dir):
    path = os.path.join(out_dir, "loss_history.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        for i in range(len(history["train_total"])):
            writer.writerow([history[k][i] for k in history])

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = get_dataloader(
                                    args.train_dir,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    downsample_factor=args.downsample_factor,
                                )


    val_loader = get_dataloader(
                                    args.val_dir,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    downsample_factor=1.0,
                                )

    model = VAEUNet(
        in_ch=args.in_ch,
        seg_out_ch=args.seg_out_ch,
        base=args.base,
        depth=args.depth,
        block_type=args.block_type,
        dim=args.dim,
        enable_reconstruction=True,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=args.ema_decay)

    history = {
        "train_total": [],
        "train_seg": [],
        "train_recon": [],
        "train_kl": [],
        "val_dice": [],
    }

    start_epoch = 0
    best_val = float("inf")

    # -------- Resume --------
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.shadow = ckpt["ema"]
        history = ckpt["history"]
        best_val = ckpt["best_val"]
        start_epoch = ckpt["epoch"]
        print(f"[Resumed] from epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_loss = 0.0
            seg_l, rec_l, kl_l = 0.0, 0.0, 0.0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                leave=False,
            )

            for x, y in pbar:
                x, y = x.to(device), y.to(device)

                out = model(x)

                l_recon = reconstruction_loss(out["recon"], x)
                l_kl = kl_divergence(out["mu"], out["logvar"]) / out["mu"].numel() # normalize by number of elements
                l_seg = segmentation_loss(
                    out["seg"],
                    y,
                    use_dice=args.use_dice,
                    use_focal=args.use_focal,
                    use_cldice=args.use_cldice,
                    dice_w=args.dice_w,
                    focal_w=args.focal_w,
                    cldice_w=args.cldice_w,
                )
                kl_weight = args.lambda_kl * min(1.0, epoch / args.kl_warmup_epochs)


                loss = (
                    args.lambda_recon * l_recon
                    + kl_weight * l_kl
                    + args.lambda_seg * l_seg
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                ema.update(model)

                # --- accumulate ---
                epoch_loss += loss.item()
                seg_l += l_seg.item()
                rec_l += l_recon.item()
                kl_l += l_kl.item()

                # --- live logging ---
                pbar.set_postfix({
                    "L": f"{loss.item():.3f}",
                    "Seg": f"{l_seg.item():.3f}",
                    "Rec": f"{l_recon.item():.3f}",
                    "KL": f"{l_kl.item():.3f}",
                })


            history["train_total"].append(epoch_loss / len(train_loader))
            history["train_seg"].append(seg_l / len(train_loader))
            history["train_recon"].append(rec_l / len(train_loader))
            history["train_kl"].append(kl_l / len(train_loader))

            # -------- Validation with EMA --------
            model.eval()
            ema.apply_to(model)

            val_loss = 0.0
            vbar = tqdm(val_loader, desc="Validation", leave=False)

            with torch.no_grad():
                for x, y in vbar:
                    x, y = x.to(device), y.to(device)

                    out = model(x, reconstruct=False)
                    l_val = dice_loss(out["seg"], y)

                    val_loss += l_val.item()
                    vbar.set_postfix({"Dice": f"{l_val.item():.3f}"})

            ema.restore(model)
            val_loss /= len(val_loader)
            history["val_dice"].append(val_loss)


            # -------- Best model --------
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    os.path.join(args.out_dir, "best_model.pt"),
                    model, ema, optimizer, epoch+1, history, best_val
                )

            # -------- Periodic checkpoint --------
            if (epoch + 1) % 20 == 0:
                save_checkpoint(
                    os.path.join(args.out_dir, f"checkpoint_epoch_{epoch+1}.pt"),
                    model, ema, optimizer, epoch+1, history, best_val
                )

    except KeyboardInterrupt:
        print("\n[Interrupted] Saving checkpoint...")
        save_checkpoint(
            os.path.join(args.out_dir, "checkpoint_interrupt.pt"),
            model, ema, optimizer, epoch+1, history, best_val
        )

    save_history_csv(history, args.out_dir)

def main():
    parser = argparse.ArgumentParser("Train VAE-UNet with EMA")

    # data
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument(
    "--downsample_factor",
    type=float,
    default=1.0,
    help="Downsample factor for training data (e.g. 0.5, 0.25)"
)


    # model
    parser.add_argument("--in_ch", type=int, default=1)
    parser.add_argument("--seg_out_ch", type=int, default=1)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--block_type", type=str, default="dense_residual")
    parser.add_argument("--dim", type=int, default=3)

    # training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    # loss weights
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_kl", type=float, default=0.1)
    parser.add_argument("--lambda_seg", type=float, default=1.0)
    parser.add_argument("--kl_warmup_epochs", type=int, default=20)


    # EMA + resume
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--resume", type=str, default=None)

    # loss ablation flags
    parser.add_argument("--use_dice", action="store_true")
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--use_cldice", action="store_true")

    parser.add_argument("--dice_w", type=float, default=1.0)
    parser.add_argument("--focal_w", type=float, default=1.0)
    parser.add_argument("--cldice_w", type=float, default=1.0)




    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
