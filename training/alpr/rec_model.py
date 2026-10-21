"""PlateNet: a small, fast CTC recogniser (~1.8 M params, ~0.36 GMAC, a few ms on one CPU core).

input  : (B, 3, 48, 168) BGR normalised to [-1, 1]
outputs: char logits (B, 21, NUM_CLASSES)  -- 21 time steps, CTC blank = 0
         colour logits (B, 5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .charset import COLORS, NUM_CLASSES


def cbr(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(cbr(c, c), nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.relu(x + self.body(x), inplace=True)


class PlateNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, num_colors: int = len(COLORS)):
        super().__init__()
        self.features = nn.Sequential(
            cbr(3, 32),                       # 48x168
            nn.MaxPool2d(2),                  # 24x84
            cbr(32, 64),
            nn.MaxPool2d(2),                  # 12x42
            cbr(64, 128),
            ResBlock(128),
            nn.MaxPool2d(2),                  # 6x21
            cbr(128, 192),
            ResBlock(192),
            nn.MaxPool2d((2, 1)),             # 3x21
        )
        self.collapse = cbr(192, 256, k=(3, 1), p=0)   # 1x21
        self.context = nn.Sequential(                  # widen receptive field along the sequence
            nn.Conv1d(256, 256, 3, 1, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 3, 1, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Conv1d(256, num_classes, 1)
        self.color_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(192, num_colors))

    def forward(self, x):
        f = self.features(x)                  # B,192,3,21
        color = self.color_head(f)
        s = self.collapse(f).squeeze(2)       # B,256,21
        s = s + self.context(s)
        logits = self.classifier(self.dropout(s))   # B,C,21
        return logits.permute(0, 2, 1), color       # B,21,C


class PlateNetExport(nn.Module):
    """Wraps PlateNet with softmax so the C++ side reads probabilities directly."""

    def __init__(self, net: PlateNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        logits, color = self.net(x)
        return logits.softmax(-1), color.softmax(-1)
