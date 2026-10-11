import numpy as np
import torch

from .ops.box_utils import apply_direction, decode, nms_bev


@torch.no_grad()
def decode_predictions(preds, anchors, cfg, score_thr=None):
    """Raw head outputs -> per-sample (K, 8) numpy arrays of [x, y, z, dx, dy, dz, heading, score]."""
    tc = cfg["test"]
    score_thr = tc["score_thr"] if score_thr is None else score_thr
    scores = torch.sigmoid(preds["cls"].float()).squeeze(-1)          # (B, A)
    out = []
    for b in range(scores.shape[0]):
        m = scores[b] > score_thr
        if not m.any():
            out.append(np.zeros((0, 8), np.float32))
            continue
        boxes = decode(preds["box"][b][m].float(), anchors[m])
        dir_cls = preds["dir"][b][m].argmax(-1)
        boxes[:, 6] = apply_direction(boxes[:, 6], dir_cls, cfg["dir_offset"])
        bx, sc = boxes.cpu().numpy(), scores[b][m].cpu().numpy()
        keep = nms_bev(bx, sc, tc["nms_thr"], tc["max_dets"], tc["nms_pre"])
        out.append(np.concatenate([bx[keep], sc[keep][:, None]], 1).astype(np.float32))
    return out
