# model.py
import torch
import torch.nn as nn

from components import Encoder, Decoder


class VAEUNet(nn.Module):
    def __init__(
        self,
        in_ch=1,
        seg_out_ch=1,
        base=32,
        depth=4,
        block_type="dense_residual",
        dim=2,
        enable_reconstruction=True,   # default behavior
    ):
        super().__init__()

        self.enable_reconstruction = enable_reconstruction

        # ---------- Encoder ----------
        self.encoder = Encoder(
            in_ch=in_ch,
            base=base,
            depth=depth,
            block_type=block_type,
            dim=dim
        )

        # Latent sample z has 1 channel
        latent_ch = 1

        # ---------- Decoders ----------
        self.seg_decoder = Decoder(
            in_ch=latent_ch,
            out_ch=seg_out_ch,
            depth=depth,
            block_type=block_type,
            dim=dim
        )

        if self.enable_reconstruction:
            self.recon_decoder = Decoder(
                in_ch=latent_ch,
                out_ch=in_ch,
                depth=depth,
                block_type=block_type,
                dim=dim
            )
        else:
            self.recon_decoder = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, reconstruct=None):
        """
        reconstruct:
            None  -> use model default
            True  -> force reconstruction
            False -> disable reconstruction
        """
        if reconstruct is None:
            reconstruct = self.enable_reconstruction

        enc_out = self.encoder(x)   # [B, 2, ...]

        mu = enc_out[:, 0:1]
        logvar = torch.clamp(enc_out[:, 1:2], min=-10.0, max=10.0)


        z = self.reparameterize(mu, logvar)

        seg = self.seg_decoder(z)

        out = {
            "seg": seg,
            "mu": mu,
            "logvar": logvar
        }

        if reconstruct:
            recon = self.recon_decoder(z)
            out["recon"] = recon

        return out
