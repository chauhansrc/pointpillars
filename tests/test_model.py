from pathlib import Path

import numpy as np
import pytest
import torch

from pointpillars.anchors import assign_targets, make_anchors
from pointpillars.data import KittiDataset, collate
from pointpillars.eval import compute_metrics, run_inference
from pointpillars.loss import PointPillarsLoss
from pointpillars.models import PointPillars
from pointpillars.postprocess import decode_predictions
from pointpillars.utils import load_weights

ROOT = Path(__file__).resolve().parents[1]


def test_anchor_assignment_finds_positives(cfg):
    anchors = make_anchors(cfg)
    assert anchors.shape == (cfg["fx"] * cfg["fy"] * 2, 7)
    gt = torch.tensor([[20.0, 3.0, -1.0, 3.9, 1.6, 1.56, 0.3]])
    labels, reg, dirc = assign_targets(anchors, gt, cfg)
    pos = labels == 1
    assert pos.sum() >= 1 and (labels == -1).any() and (labels == 0).any()
    assert reg[pos].abs().max() < 1.0


def test_forward_loss_backward_decode(fake_kitti):
    torch.manual_seed(0)
    cfg = fake_kitti
    ds = KittiDataset(cfg, "train", train=True)
    batch = collate([ds[0], ds[1]])
    model = PointPillars(cfg)
    anchors = make_anchors(cfg)

    preds = model(batch)
    n_anchors = len(anchors)
    assert preds["cls"].shape == (2, n_anchors, 1)
    assert preds["box"].shape == (2, n_anchors, 7)
    assert preds["dir"].shape == (2, n_anchors, 2)

    loss, parts = PointPillarsLoss(cfg)(preds, batch["gt"], anchors)
    assert torch.isfinite(loss) and parts["npos"] > 0
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    model.eval()
    dets = decode_predictions(model(batch), anchors, cfg, score_thr=0.0)
    assert len(dets) == 2 and dets[0].shape[1] == 8


def test_eval_loop(fake_kitti):
    cfg = fake_kitti
    ds = KittiDataset(cfg, "val", train=False)
    dl = torch.utils.data.DataLoader(ds, 2, collate_fn=collate)
    model = PointPillars(cfg)
    dets, gts, fids, ms = run_inference(model, dl, make_anchors(cfg), cfg, torch.device("cpu"))
    assert len(dets) == len(gts) == len(fids) == 3
    m = compute_metrics(dets, gts)
    assert all(0 <= v <= 100 for v in m.values())


@pytest.mark.skipif(not (ROOT / "pointpillars.pth").exists(), reason="notebook checkpoint not present")
def test_notebook_checkpoint_loads(cfg):
    """The refactor keeps the notebook's module names, so its weights still load."""
    model = PointPillars(cfg)
    load_weights(model, ROOT / "pointpillars.pth")
    assert np.isfinite(sum(p.sum().item() for p in model.parameters()))
