"""Save bird's-eye-view plots of predictions vs. ground truth.

    python tools/visualize.py --ckpt pointpillars.pth --num 6 --out assets
    python tools/visualize.py --ckpt pointpillars.pth --frames 000010 000042
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pointpillars.anchors import make_anchors
from pointpillars.config import load_config, parse_overrides
from pointpillars.data import KittiDataset, collate, to_device
from pointpillars.models import PointPillars
from pointpillars.ops.box_utils import corners_bev
from pointpillars.postprocess import decode_predictions
from pointpillars.utils import get_device, load_weights


def draw_box(ax, box, color, label=None):
    c = corners_bev(box)
    ax.plot(*np.vstack([c, c[:1]]).T, color=color, lw=1.5)
    # Heading arrow from the center to the front edge.
    x, y, r = box[0], box[1], box[6]
    ax.plot([x, x + box[3] / 2 * np.cos(r)], [y, y + box[3] / 2 * np.sin(r)], color=color, lw=1.5)
    if label:
        ax.text(x, y, label, color=color, fontsize=7)


def plot_frame(points, gt, dets, pc_range, title, path):
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.scatter(points[:, 0], points[:, 1], s=0.05, c=points[:, 2], cmap="Greys_r", vmin=-2.5, vmax=1)
    for g in gt:
        draw_box(ax, g, "tab:green")
    for d in dets:
        draw_box(ax, d, "tab:red", f"{d[7]:.2f}")
    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.set_title(f"{title}   green = GT, red = prediction")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/car.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--frames", nargs="*", default=None, help="frame ids, e.g. 000010")
    ap.add_argument("--num", type=int, default=4, help="number of frames when --frames is not given")
    ap.add_argument("--score-thr", type=float, default=0.4)
    ap.add_argument("--out", default="assets")
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_config(args.cfg, parse_overrides(args.set))
    device = get_device(args.device)
    ds = KittiDataset(cfg, args.split, train=False)
    model = PointPillars(cfg).to(device).eval()
    load_weights(model, args.ckpt, device)
    anchors = make_anchors(cfg).to(device)

    if args.frames:
        index = {fid: i for i, fid in enumerate(ds.ids)}
        idxs = [index[f] for f in args.frames]
    else:
        idxs = np.linspace(0, len(ds) - 1, args.num).astype(int).tolist()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in idxs:
        sample = ds[i]
        points, _, fid = ds.load(i)
        preds = model(to_device(collate([sample]), device))
        dets = decode_predictions(preds, anchors, cfg, args.score_thr)[0]
        path = out / f"bev_{fid}.png"
        plot_frame(points, sample["gt"], dets, cfg["pc_range"], fid, path)
        print(f"{fid}: {len(sample['gt'])} GT, {len(dets)} detections -> {path}")


if __name__ == "__main__":
    main()
