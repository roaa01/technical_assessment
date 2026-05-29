"""
model.py — Task 2
==================
Lightweight U-Net for remote sensing image segmentation.

Architecture
------------
Encoder: 4 down-sampling blocks (Conv-BN-ReLU ×2 + MaxPool).
Bottleneck: Conv-BN-ReLU ×2.
Decoder: 4 up-sampling blocks (Bilinear upsample + skip concat + Conv ×2).
Head: 1×1 Conv → num_classes logits.

The model is intentionally small so experiments complete quickly on CPU
or a single GPU without a large dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv 3×3 → BN → ReLU (standard U-Net block unit)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """Two consecutive ConvBnRelu blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvBnRelu(in_ch,  out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool 2×2 → DoubleConv (encoder step)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Bilinear upsample → cat skip connection → DoubleConv (decoder step).
    Bilinear upsampling avoids checkerboard artefacts common with
    transposed convolutions.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if skip and x sizes mismatch (when input is not a power of 2)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────
# U-Net
# ──────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net for multi-class semantic segmentation.

    Parameters
    ----------
    in_channels  : number of input image channels (3 for RGB)
    num_classes  : number of segmentation classes
    base_ch      : width multiplier — feature channels at the first
                   encoder level (default 32 keeps the model lightweight)
    dropout_p    : spatial dropout probability applied before the head
                   (set to 0.0 to disable)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        base_ch: int = 32,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        b = base_ch

        # ── Encoder ────────────────────────────────────────────────
        self.enc1 = DoubleConv(in_channels, b)       #  b  maps, H×W
        self.enc2 = Down(b,      b * 2)              # 2b  maps, H/2
        self.enc3 = Down(b * 2,  b * 4)              # 4b  maps, H/4
        self.enc4 = Down(b * 4,  b * 8)              # 8b  maps, H/8

        # ── Bottleneck ─────────────────────────────────────────────
        self.bottleneck = Down(b * 8, b * 16)        # 16b maps, H/16

        # ── Decoder ────────────────────────────────────────────────
        # in_ch = skip_ch + up_ch
        self.dec4 = Up(b * 16 + b * 8,  b * 8)
        self.dec3 = Up(b * 8  + b * 4,  b * 4)
        self.dec2 = Up(b * 4  + b * 2,  b * 2)
        self.dec1 = Up(b * 2  + b,       b)

        # ── Head ───────────────────────────────────────────────────
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.head    = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        logits : (B, num_classes, H, W)
        """
        # Encoder
        s1 = self.enc1(x)         # skip 1
        s2 = self.enc2(s1)        # skip 2
        s3 = self.enc3(s2)        # skip 3
        s4 = self.enc4(s3)        # skip 4

        # Bottleneck
        x = self.bottleneck(s4)

        # Decoder
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        # Head
        x = self.dropout(x)
        return self.head(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────
# Sanity-check
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for base_ch in [16, 32, 64]:
        net    = UNet(in_channels=3, num_classes=6, base_ch=base_ch)
        dummy  = torch.randn(2, 3, 256, 256)
        logits = net(dummy)
        print(
            f"base_ch={base_ch:3d} | "
            f"params={net.n_params():,} | "
            f"output={tuple(logits.shape)}"
        )
