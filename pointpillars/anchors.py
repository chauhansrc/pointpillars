"""Anchor grid and per-anchor training targets (paper Sec. 2.3)."""
import torch

from .ops.box_utils import bev_aabb, direction_target, encode, iou_aabb


def make_anchors(cfg):
    """(fy * fx * R, 7) anchors, ordered (y, x, rotation) to match the head's output layout."""
    pr, vs = cfg["pc_range"], cfg["voxel_size"]
    sx, sy = vs[0] * 2, vs[1] * 2                       # head stride 2
    x = pr[0] + (torch.arange(cfg["fx"]) + 0.5) * sx
    y = pr[1] + (torch.arange(cfg["fy"]) + 0.5) * sy
    dx, dy, dz = cfg["anchor_size"]
    z = cfg["anchor_bottom"] + dz / 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    full = lambda v: torch.full_like(xx, v)
    per_rot = [torch.stack([xx, yy, full(z), full(dx), full(dy), full(dz), full(r)], -1)
               for r in cfg["anchor_rotations"]]
    return torch.stack(per_rot, 2).reshape(-1, 7)


def assign_targets(anchors, gt, cfg):
    """Label anchors by BEV IoU with the ground truth.

    Returns:
        labels (K,) 1 = positive, 0 = negative, -1 = ignored,
        reg    (K, 7) box regression targets (valid where labels == 1),
        dirc   (K,) heading direction bin (valid where labels == 1).

    Matching uses the IoU of axis-aligned hulls of the rotated boxes - cheap on the GPU for ~100k
    anchors, slightly looser than true rotated IoU.
    """
    K = anchors.shape[0]
    dev = anchors.device
    labels = torch.full((K,), -1, dtype=torch.long, device=dev)
    reg = torch.zeros(K, 7, device=dev)
    dirc = torch.zeros(K, dtype=torch.long, device=dev)
    if len(gt) == 0:
        labels[:] = 0
        return labels, reg, dirc

    iou = iou_aabb(bev_aabb(anchors), bev_aabb(gt))       # (K, G)
    best_iou, best_gt = iou.max(1)
    labels[best_iou < cfg["unmatch_iou"]] = 0
    pos = best_iou >= cfg["match_iou"]

    # Every GT box also claims its single best anchor, so small / odd-angle cars still get a positive.
    gt_best_iou, gt_best_anchor = iou.max(0)
    ok = gt_best_iou > 0.1
    pos[gt_best_anchor[ok]] = True
    best_gt[gt_best_anchor[ok]] = torch.arange(len(gt), device=dev)[ok]

    labels[pos] = 1
    if pos.any():
        g = gt[best_gt[pos]]
        reg[pos] = encode(g, anchors[pos])
        dirc[pos] = direction_target(g[:, 6], cfg["dir_offset"])
    return labels, reg, dirc
