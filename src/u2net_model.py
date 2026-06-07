"""U2-Net and U2-Net-P (lite) model definitions.

Based on the official implementation: https://github.com/xuebinqin/U-2-Net
Reference: Qin et al., "U^2-Net: Going Deeper with Nested U-Structure
for Salient Object Detection", Pattern Recognition 2020.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class REU(nn.Module):
    """Residual U-block: the core building block of U2-Net."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)

        self.bn3 = nn.BatchNorm2d(mid_ch)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)

        self.bn4 = nn.BatchNorm2d(mid_ch)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)

        self.bn5 = nn.BatchNorm2d(mid_ch)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(mid_ch, out_ch, 3, padding=1)

        # Skip connection
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.conv1(self.relu1(self.bn1(x)))
        h = self.conv2(self.relu2(self.bn2(h)))
        h = self.conv3(self.relu3(self.bn3(h)))
        h = self.conv4(self.relu4(self.bn4(h)))
        h = self.conv5(self.relu5(self.bn5(h)))
        return self.skip(x) + h


class REU_Lite(nn.Module):
    """Lightweight REU for U2-Net-P (uses fewer conv layers)."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)

        self.bn3 = nn.BatchNorm2d(mid_ch)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.conv1(self.relu1(self.bn1(x)))
        h = self.conv2(self.relu2(self.bn2(h)))
        h = self.conv3(self.relu3(self.bn3(h)))
        return self.skip(x) + h


def _make_stage(in_ch, mid_ch, out_ch, lite=False):
    """Create one encoder or decoder stage."""
    Block = REU_Lite if lite else REU
    return Block(in_ch, mid_ch, out_ch)


class U2Net(nn.Module):
    """Full U2-Net for salient object detection."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        lite = False

        # Encoder
        self.stage1 = _make_stage(in_ch, 32, 64, lite)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = _make_stage(64, 32, 128, lite)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = _make_stage(128, 64, 256, lite)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = _make_stage(256, 128, 512, lite)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Bridge
        self.stage5 = _make_stage(512, 256, 512, lite)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = _make_stage(512, 256, 512, lite)

        # Decoder
        self.stage5d = _make_stage(1024, 256, 512, lite)
        self.stage4d = _make_stage(1024, 128, 256, lite)
        self.stage3d = _make_stage(512, 64, 128, lite)
        self.stage2d = _make_stage(256, 32, 64, lite)
        self.stage1d = _make_stage(128, 16, 64, lite)

        # Side outputs
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)

    def forward(self, x):
        # Encoder
        h1 = self.stage1(x)
        h2 = self.stage2(self.pool1(h1))
        h3 = self.stage3(self.pool2(h2))
        h4 = self.stage4(self.pool3(h3))
        h5 = self.stage5(self.pool4(h4))
        h6 = self.stage6(self.pool5(h5))

        # Decoder with skip connections
        h5d = self.stage5d(torch.cat([h5, F.interpolate(h6, size=h5.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        h4d = self.stage4d(torch.cat([h4, F.interpolate(h5d, size=h4.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        h3d = self.stage3d(torch.cat([h3, F.interpolate(h4d, size=h3.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        h2d = self.stage2d(torch.cat([h2, F.interpolate(h3d, size=h2.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        h1d = self.stage1d(torch.cat([h1, F.interpolate(h2d, size=h1.shape[2:], mode='bilinear', align_corners=False)], dim=1))

        # Side outputs
        d6 = self.side6(h6)
        d5 = self.side5(h5d)
        d4 = self.side4(h4d)
        d3 = self.side3(h3d)
        d2 = self.side2(h2d)
        d1 = self.side1(h1d)

        # Upsample all to input size
        d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        d2 = F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=False)
        d3 = F.interpolate(d3, size=x.shape[2:], mode='bilinear', align_corners=False)
        d4 = F.interpolate(d4, size=x.shape[2:], mode='bilinear', align_corners=False)
        d5 = F.interpolate(d5, size=x.shape[2:], mode='bilinear', align_corners=False)
        d6 = F.interpolate(d6, size=x.shape[2:], mode='bilinear', align_corners=False)

        return d1, d2, d3, d4, d5, d6


class U2NetP(nn.Module):
    """U2-Net-P (lite / small) for salient object detection."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        lite = True

        # Encoder
        self.stage1 = _make_stage(in_ch, 16, 64, lite)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = _make_stage(64, 16, 64, lite)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = _make_stage(64, 32, 128, lite)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = _make_stage(128, 32, 128, lite)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Bridge
        self.stage5 = _make_stage(128, 64, 256, lite)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = _make_stage(256, 64, 256, lite)

        # Decoder
        self.stage5d = _make_stage(512, 64, 128, lite)
        self.stage4d = _make_stage(256, 32, 64, lite)
        self.stage3d = _make_stage(192, 16, 64, lite)
        self.stage2d = _make_stage(128, 16, 64, lite)
        self.stage1d = _make_stage(128, 8, 64, lite)

        # Side outputs
        self.side6 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)

    def forward(self, x):
        # Encoder
        h1 = self.stage1(x)
        h2 = self.stage2(self.pool1(h1))
        h3 = self.stage3(self.pool2(h2))
        h4 = self.stage4(self.pool3(h3))
        h5 = self.stage5(self.pool4(h4))
        h6 = self.stage6(self.pool5(h5))

        # Decoder
        h5d = self.stage5d(torch.cat([h5, F.interpolate(h6, size=h5.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        h4d = self.stage4d(torch.cat([h4, F.interpolate(h5d, size=h4.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        h3d = self.stage3d(torch.cat([h3, F.interpolate(h4d, size=h3.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        h2d = self.stage2d(torch.cat([h2, F.interpolate(h3d, size=h2.shape[2:], mode='bilinear', align_corners=False)], dim=1))
        h1d = self.stage1d(torch.cat([h1, F.interpolate(h2d, size=h1.shape[2:], mode='bilinear', align_corners=False)], dim=1))

        # Side outputs
        d6 = self.side6(h6)
        d5 = self.side5(h5d)
        d4 = self.side4(h4d)
        d3 = self.side3(h3d)
        d2 = self.side2(h2d)
        d1 = self.side1(h1d)

        d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        d2 = F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=False)
        d3 = F.interpolate(d3, size=x.shape[2:], mode='bilinear', align_corners=False)
        d4 = F.interpolate(d4, size=x.shape[2:], mode='bilinear', align_corners=False)
        d5 = F.interpolate(d5, size=x.shape[2:], mode='bilinear', align_corners=False)
        d6 = F.interpolate(d6, size=x.shape[2:], mode='bilinear', align_corners=False)

        return d1, d2, d3, d4, d5, d6