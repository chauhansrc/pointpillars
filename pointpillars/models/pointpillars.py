import torch.nn as nn

from .backbone import Backbone
from .head import Head
from .pfn import PillarFeatureNet, scatter_to_bev


class PointPillars(nn.Module):
    """Pillar Feature Net -> pseudo-image -> 2D backbone -> SSD head."""

    def __init__(self, cfg):
        super().__init__()
        self.nx, self.ny = cfg["nx"], cfg["ny"]
        c = cfg.get("pfn_channels", 64)
        self.pfn = PillarFeatureNet(cfg, c)
        self.backbone = Backbone(c)
        self.head = Head(self.backbone.out_channels, num_anchors=len(cfg["anchor_rotations"]))

    def forward(self, batch):
        f = self.pfn(batch["pillars"], batch["coords"], batch["num"])
        x = scatter_to_bev(f, batch["coords"], batch["bs"], self.nx, self.ny)
        return self.head(self.backbone(x))
