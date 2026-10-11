"""Simplified KITTI-style evaluation: single class, whole split, 40-recall-point AP (AP_R40).

Unlike the official devkit this ignores the easy / moderate / hard buckets and DontCare regions,
so numbers are close to, but not the same as, leaderboard numbers. Use ``tools/evaluate.py --export``
plus the official evaluator for comparable results.
"""
import time

import numpy as np
import torch

from .data.kitti import to_device
from .ops.box_utils import iou_3d_rot, iou_bev_rot
from .postprocess import decode_predictions


def ap_r40(dets, gts, iou_thr=0.7, iou_fn=iou_bev_rot):
    """dets: per-frame (K, 8) [box, score]; gts: per-frame (G, 7).

    Returns (AP_R40, mean orientation similarity of true positives). Orientation similarity is
    (1 + cos(dtheta)) / 2: 1.0 for a correct heading, 0.0 for a box pointing backwards.
    """
    npos = sum(len(g) for g in gts)
    flat = [(d[7], fi, d[:7]) for fi, D in enumerate(dets) for d in D]
    flat.sort(key=lambda x: -x[0])
    matched = [np.zeros(len(g), bool) for g in gts]
    tp, fp = np.zeros(len(flat)), np.zeros(len(flat))
    osim = []
    for k, (_, fi, b) in enumerate(flat):
        g = gts[fi]
        if len(g) == 0:
            fp[k] = 1
            continue
        ious = np.array([iou_fn(b, gg) for gg in g])
        j = int(ious.argmax())
        if ious[j] >= iou_thr and not matched[fi][j]:
            tp[k] = 1
            matched[fi][j] = True
            osim.append((1 + np.cos(b[6] - g[j][6])) / 2)
        else:
            fp[k] = 1
    if npos == 0 or len(flat) == 0:
        return 0.0, 0.0
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / npos
    prec = tp / np.maximum(tp + fp, 1e-9)
    ap = sum((prec[rec >= r].max() if (rec >= r).any() else 0.0) for r in np.linspace(1 / 40, 1, 40)) / 40
    return float(ap), float(np.mean(osim)) if osim else 0.0


@torch.no_grad()
def run_inference(model, loader, anchors, cfg, device, score_thr=None):
    """Returns (per-frame detections, per-frame GT boxes, frame ids, mean forward ms per frame)."""
    model.eval()
    dets, gts, fids = [], [], []
    fwd_ms, frames = 0.0, 0
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for batch in loader:
        gts += [g.numpy() for g in batch["gt"]]
        fids += batch["fid"]
        batch = to_device(batch, device)
        sync()
        t0 = time.perf_counter()
        preds = model(batch)
        sync()
        fwd_ms += (time.perf_counter() - t0) * 1000
        frames += batch["bs"]
        dets += decode_predictions(preds, anchors, cfg, score_thr)
    return dets, gts, fids, fwd_ms / max(frames, 1)


def compute_metrics(dets, gts):
    m = {}
    for thr in (0.7, 0.5):
        m[f"bev_ap@{thr}"], osim = ap_r40(dets, gts, thr, iou_bev_rot)
        m[f"3d_ap@{thr}"], _ = ap_r40(dets, gts, thr, iou_3d_rot)
        if thr == 0.7:
            m["orientation_sim@0.7"] = osim
    return {k: round(v * 100, 2) for k, v in m.items()}


def evaluate(model, loader, anchors, cfg, device):
    dets, gts, _, ms = run_inference(model, loader, anchors, cfg, device)
    metrics = compute_metrics(dets, gts)
    metrics["forward_ms_per_frame"] = round(ms, 2)
    return metrics
