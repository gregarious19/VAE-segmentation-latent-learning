import torch
import torch.nn as nn


def conv_nd(dim, in_ch, out_ch, k=3, p=1):
    if dim == 2:
        return nn.Conv2d(in_ch, out_ch, k, padding=p)
    if dim == 3:
        return nn.Conv3d(in_ch, out_ch, k, padding=p)
    raise ValueError("dim must be 2 or 3")


class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dim=2):
        super().__init__()
        self.block = nn.Sequential(
            conv_nd(dim, in_ch, out_ch),
            nn.ReLU(inplace=True),
            conv_nd(dim, out_ch, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DenseResidualBlock(nn.Module):
    def __init__(self, ch, growth=16, layers=4, dim=2):
        super().__init__()
        self.layers = nn.ModuleList()
        cur = ch
        for _ in range(layers):
            self.layers.append(
                nn.Sequential(
                    conv_nd(dim, cur, growth),
                    nn.ReLU(inplace=True),
                )
            )
            cur += growth
        self.proj = conv_nd(dim, cur, ch, k=1, p=0)

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            feats.append(layer(torch.cat(feats, dim=1)))
        return x + self.proj(torch.cat(feats, dim=1))


class Encoder(nn.Module):
    def __init__(
        self,
        in_ch=1,
        base=32,
        depth=4,
        block_type="dense_residual",  # "cnn" or "dense_residual"
        dim=2
    ):
        super().__init__()

        self.entry = conv_nd(dim, in_ch, base)

        self.blocks = nn.ModuleList()
        self.down = nn.ModuleList()

        ch = base
        for _ in range(depth):
            if block_type == "cnn":
                block = CNNBlock(ch, ch, dim)
            elif block_type == "dense_residual":
                block = DenseResidualBlock(ch, dim=dim)
            else:
                raise ValueError("block_type must be 'cnn' or 'dense_residual'")

            self.blocks.append(block)
            self.down.append(conv_nd(dim, ch, ch * 2, k=2, p=0))
            ch *= 2

        # encoder output channels fixed to 2
        self.out_conv = conv_nd(dim, ch, 2, k=1, p=0)

    def forward(self, x):
        x = self.entry(x)
        for block, down in zip(self.blocks, self.down):
            x = block(x)
            x = down(x)
        x = self.out_conv(x)
        return x


def upconv_nd(dim, in_ch, out_ch):
    if dim == 2:
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=1)
    if dim == 3:
        return nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=1)
    raise ValueError("dim must be 2 or 3")

# ADD this Decoder class (encoder code remains unchanged)

class Decoder(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        depth=4,
        block_type="dense_residual",  # "cnn" or "dense_residual"
        dim=2
    ):
        super().__init__()

        self.up = nn.ModuleList()
        self.blocks = nn.ModuleList()

        ch = in_ch
        for _ in range(depth):
            next_ch = max(ch // 2, 1)
            self.up.append(upconv_nd(dim, ch, next_ch))
            ch = next_ch

            if block_type == "cnn":
                block = CNNBlock(ch, ch, dim)
            elif block_type == "dense_residual":
                block = DenseResidualBlock(ch, dim=dim)
            else:
                raise ValueError("block_type must be 'cnn' or 'dense_residual'")

            self.blocks.append(block)

        self.out_conv = conv_nd(dim, ch, out_ch, k=1, p=0)

    def forward(self, x):
        for up, block in zip(self.up, self.blocks):
            x = up(x)
            x = block(x)
        return self.out_conv(x)
