"""Evaluate a checkpoint on the KITTI val split.

    python tools/evaluate.py --ckpt pointpillars.pth
    python tools/evaluate.py --ckpt outputs/car/best_model.pth --export outputs/car/val_preds

``--export`` writes one KITTI label file per frame, ready for the official devkit or OpenPCDet's
``kitti_object_eval_python`` (easy / moderate / hard numbers).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from pointpillars.anchors import make_anchors
from pointpillars.config import load_config, parse_overrides
from pointpillars.data import KittiDataset, collate, read_calib, to_kitti_lines
from pointpillars.eval import compute_metrics, run_inference
from pointpillars.models import PointPillars
from pointpillars.utils import get_device, load_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/car.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score-thr", type=float, default=None)
    ap.add_argument("--export", default=None, help="directory for KITTI-format predictions")
    ap.add_argument("--out-json", default=None, help="write metrics here")
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_config(args.cfg, parse_overrides(args.set))
    device = get_device(args.device)
    ds = KittiDataset(cfg, args.split, train=False, limit=args.limit)
    dl = DataLoader(ds, cfg["test"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"],
                    collate_fn=collate)
    model = PointPillars(cfg).to(device)
    load_weights(model, args.ckpt, device)
    anchors = make_anchors(cfg).to(device)

    dets, gts, fids, ms = run_inference(model, dl, anchors, cfg, device, args.score_thr)
    metrics = compute_metrics(dets, gts)
    metrics["forward_ms_per_frame"] = round(ms, 2)
    metrics["frames"] = len(ds)
    print(json.dumps(metrics, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(metrics, indent=2))

    if args.export:
        exp = Path(args.export)
        exp.mkdir(parents=True, exist_ok=True)
        calib_dir = Path(cfg["root"]) / "training" / "calib"
        for fid, d in zip(fids, dets):
            lines = to_kitti_lines(d, read_calib(calib_dir / f"{fid}.txt"), cfg["cls_name"])
            (exp / f"{fid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"wrote {len(fids)} prediction files to {exp}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
