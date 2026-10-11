import math

import torch.nn as nn


class Head(nn.Module):
    """SSD-style 1x1 conv head: class logits, 7 box residuals and 2 direction logits per anchor."""

    def __init__(self, in_channels=384, num_anchors=2, num_classes=1, prior=0.01):
        super().__init__()
        self.na, self.nc = num_anchors, num_classes
        self.cls = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.box = nn.Conv2d(in_channels, num_anchors * 7, 1)
        self.dir = nn.Conv2d(in_channels, num_anchors * 2, 1)
        nn.init.constant_(self.cls.bias, -math.log((1 - prior) / prior))   # focal-loss prior

    def forward(self, x):
        B = x.shape[0]
        flat = lambda t, d: t.permute(0, 2, 3, 1).reshape(B, -1, d)       # -> (B, fy * fx * na, d)
        return dict(cls=flat(self.cls(x), self.nc), box=flat(self.box(x), 7), dir=flat(self.dir(x), 2))
