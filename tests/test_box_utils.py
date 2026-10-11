import numpy as np
import torch

from pointpillars.ops.box_utils import (apply_direction, decode, direction_target, encode, iou_3d_rot,
                                        iou_bev_rot, limit_period, nms_bev)


def random_boxes(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    xyz = torch.rand(n, 3, generator=g) * torch.tensor([70.0, 80.0, 2.0]) - torch.tensor([0.0, 40.0, 2.0])
    dims = torch.rand(n, 3, generator=g) * 2 + 1
    yaw = (torch.rand(n, 1, generator=g) * 2 - 1) * np.pi
    return torch.cat([xyz, dims, yaw], 1)


def test_limit_period_range():
    v = torch.linspace(-20, 20, 1001)
    w = limit_period(v, 0.5, 2 * np.pi)
    assert (w >= -np.pi).all() and (w < np.pi).all()
    assert torch.allclose(torch.sin(w), torch.sin(v), atol=1e-5)


def test_encode_decode_roundtrip():
    gt, anchors = random_boxes(256, 0), random_boxes(256, 1)
    assert torch.allclose(decode(encode(gt, anchors), anchors), gt, atol=1e-4)


def test_direction_roundtrip_resolves_flip():
    """The regressed heading is only known up to pi; the direction bin must recover the true one.
    (The original notebook decoded with a different offset than training and failed this.)"""
    offset = 0.78539
    heading = torch.linspace(-np.pi + 1e-3, np.pi - 1e-3, 721)
    target = direction_target(heading, offset)
    for ambiguous in (heading, heading + np.pi, heading - np.pi):
        out = apply_direction(ambiguous, target, offset)
        diff = limit_period(out - heading, 0.5, 2 * np.pi)
        assert diff.abs().max() < 1e-4


def test_rotated_iou():
    a = np.array([0, 0, 0, 4, 2, 2, 0.0])
    assert abs(iou_bev_rot(a, a) - 1) < 1e-6
    assert abs(iou_3d_rot(a, a) - 1) < 1e-6
    b = a.copy()
    b[6] = np.pi / 2                       # 4x2 vs 2x4 cross: overlap 2x2 = 4, union 12
    assert abs(iou_bev_rot(a, b) - 4 / 12) < 1e-6
    c = a.copy()
    c[2] = 1.0                             # half the height overlaps
    assert abs(iou_3d_rot(a, c) - 8 / 24) < 1e-6


def test_nms_suppresses_duplicates():
    boxes = np.array([[0, 0, 0, 4, 2, 2, 0], [0.1, 0, 0, 4, 2, 2, 0.05], [10, 0, 0, 4, 2, 2, 0]], np.float32)
    keep = nms_bev(boxes, np.array([0.9, 0.8, 0.7]), thr=0.1)
    assert keep == [0, 2]
