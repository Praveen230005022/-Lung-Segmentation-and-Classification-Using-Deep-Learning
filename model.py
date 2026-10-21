"""U-Net for lung segmentation and a lightweight CNN classifier (both built from scratch)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv 3x3 -> BatchNorm -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Max-pool downsampling followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class DecoderBlock(nn.Module):
    """Transposed-conv upsampling, skip-connection concatenation, DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad when the input size is not divisible by 16 so shapes match the skip tensor.
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Standard 4-level U-Net returning pixel-wise logits of shape (N, num_classes, H, W)."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.in_channels, self.num_classes, self.base_channels = in_channels, num_classes, base_channels
        self.stem = DoubleConv(in_channels, c)
        self.down1, self.down2 = EncoderBlock(c, c * 2), EncoderBlock(c * 2, c * 4)
        self.down3, self.down4 = EncoderBlock(c * 4, c * 8), EncoderBlock(c * 8, c * 16)
        self.up1, self.up2 = DecoderBlock(c * 16, c * 8), DecoderBlock(c * 8, c * 4)
        self.up3, self.up4 = DecoderBlock(c * 4, c * 2), DecoderBlock(c * 2, c)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.head(x)


class LungClassifier(nn.Module):
    """Small CNN (4 conv stages + global average pooling) for binary or multiclass labels."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_channels: int = 16, dropout: float = 0.3):
        super().__init__()
        self.in_channels, self.num_classes, self.base_channels = in_channels, num_classes, base_channels
        layers, ch = [], in_channels
        for i in range(4):
            out = base_channels * 2 ** i
            layers += [
                nn.Conv2d(ch, out, 3, padding=1, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            ch = out
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(ch, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(self.features(x), 1)
        return self.classifier(x)


def build_unet(cfg: dict) -> UNet:
    return UNet(cfg["image"]["in_channels"], cfg["segmentation"]["num_classes"], cfg["segmentation"]["base_channels"])


def build_classifier(cfg: dict) -> LungClassifier:
    return LungClassifier(cfg["image"]["in_channels"], len(cfg["classification"]["class_names"]))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
