"""Training loss (paper Sec. 2.3): focal classification + smooth-L1 box with sine-difference heading
+ softmax direction classification."""
import torch
import torch.nn.functional as F

from .anchors import assign_targets


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    pt = target * p + (1 - target) * (1 - p)
    w = alpha * target + (1 - alpha) * (1 - target)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return w * (1 - pt).pow(gamma) * ce


def smooth_l1(diff, beta=1 / 9):
    a = diff.abs()
    return torch.where(a < beta, 0.5 * a * a / beta, a - 0.5 * beta)


def sin_difference(pred, target):
    """Replace the heading residual with sin(p - t) = sin p cos t - cos p sin t, so a box and its
    180-degree flip cost the same; the direction classifier resolves the flip."""
    rp = torch.sin(pred[..., 6:7]) * torch.cos(target[..., 6:7])
    rt = torch.cos(pred[..., 6:7]) * torch.sin(target[..., 6:7])
    return torch.cat([pred[..., :6], rp], -1), torch.cat([target[..., :6], rt], -1)


class PointPillarsLoss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lc = cfg["loss"]

    def __call__(self, preds, gt_boxes, anchors):
        lc = self.lc
        cls_p, box_p, dir_p = preds["cls"].float(), preds["box"].float(), preds["dir"].float()

        targets = [assign_targets(anchors, g.to(anchors.device), self.cfg) for g in gt_boxes]
        labels = torch.stack([t[0] for t in targets])
        reg_t = torch.stack([t[1] for t in targets])
        dir_t = torch.stack([t[2] for t in targets])

        pos = labels == 1
        cared = labels >= 0
        npos = pos.sum().clamp(min=1).float()

        cls_t = pos.unsqueeze(-1).to(cls_p.dtype)
        l_cls = (focal_loss(cls_p, cls_t, lc["focal_alpha"], lc["focal_gamma"]).squeeze(-1) * cared).sum() / npos

        pb, tb = sin_difference(box_p[pos], reg_t[pos])
        l_box = smooth_l1(pb - tb, lc["smooth_l1_beta"]).sum() / npos

        if pos.any():
            l_dir = F.cross_entropy(dir_p[pos], dir_t[pos], reduction="sum") / npos
        else:
            l_dir = dir_p.sum() * 0

        loss = lc["w_cls"] * l_cls + lc["w_box"] * l_box + lc["w_dir"] * l_dir
        parts = dict(loss=loss.item(), cls=l_cls.item(), box=l_box.item(), dir=l_dir.item(),
                     npos=int(pos.sum()))
        return loss, parts
