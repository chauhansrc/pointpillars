"""Box helpers. A box is (x, y, z, dx, dy, dz, heading) in the LiDAR frame, z at the box center."""
import numpy as np
import torch
from shapely.geometry import Polygon


def limit_period(v, offset=0.5, period=np.pi):
    """Wrap ``v`` into ``[-offset * period, (1 - offset) * period)``. Works on tensors and arrays."""
    floor = torch.floor if isinstance(v, torch.Tensor) else np.floor
    return v - floor(v / period + offset) * period


# ----------------------------------------------------------------------------- box coder (Sec. 2.3)

def encode(gt, anchors):
    """Regression targets from Eq. (1): dx = (xg - xa) / da, dw = log(wg / wa), dtheta = tg - ta."""
    a = anchors
    d = torch.sqrt(a[:, 3] ** 2 + a[:, 4] ** 2)
    return torch.stack([(gt[:, 0] - a[:, 0]) / d,
                        (gt[:, 1] - a[:, 1]) / d,
                        (gt[:, 2] - a[:, 2]) / a[:, 5],
                        torch.log(gt[:, 3] / a[:, 3]),
                        torch.log(gt[:, 4] / a[:, 4]),
                        torch.log(gt[:, 5] / a[:, 5]),
                        gt[:, 6] - a[:, 6]], -1)


def decode(pred, anchors):
    a = anchors
    d = torch.sqrt(a[:, 3] ** 2 + a[:, 4] ** 2)
    return torch.stack([pred[:, 0] * d + a[:, 0],
                        pred[:, 1] * d + a[:, 1],
                        pred[:, 2] * a[:, 5] + a[:, 2],
                        torch.exp(pred[:, 3]) * a[:, 3],
                        torch.exp(pred[:, 4]) * a[:, 4],
                        torch.exp(pred[:, 5]) * a[:, 5],
                        pred[:, 6] + a[:, 6]], -1)


# ----------------------------------------------------------------------------- heading direction
# The sine-difference loss cannot tell a heading from its 180-degree flip, so a separate
# 2-way classifier predicts which half-turn the heading is in. Both functions below must use
# the same convention: bin 0 <=> (heading - dir_offset) mod 2pi in [0, pi).

def direction_target(heading, dir_offset):
    return (limit_period(heading - dir_offset, 0.0, 2 * np.pi) >= np.pi).long()


def apply_direction(heading, dir_cls, dir_offset):
    """Snap a regressed heading (ambiguous up to pi) onto the half-turn picked by ``dir_cls``.

    Returns the heading wrapped to [-pi, pi).
    """
    rot = limit_period(heading - dir_offset, 0.0, np.pi)
    out = rot + dir_offset + np.pi * dir_cls.to(heading.dtype)
    return limit_period(out, 0.5, 2 * np.pi)


# ----------------------------------------------------------------------------- axis-aligned BEV IoU (matching)

def bev_aabb(b):
    """Axis-aligned hull (x1, y1, x2, y2) of the rotated BEV footprint."""
    c, s = torch.cos(b[:, 6]).abs(), torch.sin(b[:, 6]).abs()
    ex = (b[:, 3] * c + b[:, 4] * s) / 2
    ey = (b[:, 3] * s + b[:, 4] * c) / 2
    return torch.stack([b[:, 0] - ex, b[:, 1] - ey, b[:, 0] + ex, b[:, 1] + ey], -1)


def iou_aabb(a, b):
    """Pairwise IoU between two sets of (x1, y1, x2, y2) boxes -> (len(a), len(b))."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


# ----------------------------------------------------------------------------- rotated IoU (NMS / eval)

def corners_bev(b):
    """(4, 2) BEV corners of a single numpy box, counter-clockwise."""
    x, y, dx, dy, r = b[0], b[1], b[3], b[4], b[6]
    c, s = np.cos(r), np.sin(r)
    p = np.array([[dx / 2, dy / 2], [-dx / 2, dy / 2], [-dx / 2, -dy / 2], [dx / 2, -dy / 2]])
    return p @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


def corners_3d(b):
    """(8, 3) corners of a single numpy box: 4 bottom then 4 top."""
    bev = corners_bev(b)
    z0, z1 = b[2] - b[5] / 2, b[2] + b[5] / 2
    return np.vstack([np.hstack([bev, np.full((4, 1), z0)]), np.hstack([bev, np.full((4, 1), z1)])])


def _poly(b):
    return Polygon(corners_bev(b))


def iou_bev_rot(a, b):
    pa, pb = _poly(a), _poly(b)
    if not (pa.is_valid and pb.is_valid):
        return 0.0
    inter = pa.intersection(pb).area
    return inter / (pa.area + pb.area - inter + 1e-9)


def iou_3d_rot(a, b):
    pa, pb = _poly(a), _poly(b)
    if not (pa.is_valid and pb.is_valid):
        return 0.0
    zo = min(a[2] + a[5] / 2, b[2] + b[5] / 2) - max(a[2] - a[5] / 2, b[2] - b[5] / 2)
    if zo <= 0:
        return 0.0
    inter = pa.intersection(pb).area * zo
    return inter / (a[3] * a[4] * a[5] + b[3] * b[4] * b[5] - inter + 1e-9)


def nms_bev(boxes, scores, thr=0.1, max_dets=100, pre=500):
    """Greedy NMS on rotated BEV IoU. Returns the kept indices into ``boxes``."""
    order = np.argsort(-scores)[:pre]
    polys = {i: _poly(boxes[i]) for i in order}
    keep = []
    for i in order:
        pi = polys[i]
        ok = True
        for j in keep:
            pj = polys[j]
            inter = pi.intersection(pj).area
            if inter / (pi.area + pj.area - inter + 1e-9) > thr:
                ok = False
                break
        if ok:
            keep.append(i)
            if len(keep) >= max_dets:
                break
    return keep
