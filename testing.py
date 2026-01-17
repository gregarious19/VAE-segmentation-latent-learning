# test_model.py
import torch
import torch.nn.functional as F

from model import VAEUNet


# -----------------------
# dummy loss functions
# -----------------------
def reconstruction_loss(recon, x):
    return F.mse_loss(recon, x)


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return 1 - num / den


# -----------------------
# test + backprop
# -----------------------
def test_forward_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    model = VAEUNet(
        in_ch=1,
        seg_out_ch=1,
        base=8,
        depth=4,
        block_type="dense_residual",
        dim=3,
        enable_reconstruction=True
    ).to(device)

    model.train()

    # dummy inputs
    x = torch.randn(1, 1, 200, 200, 128, device=device)
    y = torch.randint(0, 2, (1, 1, 200, 200, 128), device=device).float()

    # forward
    out = model(x)

    # losses
    recon_loss = reconstruction_loss(out["recon"], x)
    kl_loss = kl_divergence(out["mu"], out["logvar"])
    seg_loss = dice_loss(out["seg"], y)

    loss = (
        1.0 * recon_loss +
        0.1 * kl_loss +
        1.0 * seg_loss
    )

    # backward
    loss.backward()

    print("Forward + backward successful")
    print("Total loss:", loss.item())
    print("Recon loss:", recon_loss.item())
    print("KL loss:", kl_loss.item())
    print("Seg loss:", seg_loss.item())


if __name__ == "__main__":
    test_forward_backward()
