"""Train PointPillars on KITTI.

    python tools/train.py --cfg configs/car.yaml --out outputs/car
    python tools/train.py --cfg configs/car.yaml --out outputs/car --resume outputs/car/last.pth
    python tools/train.py --cfg configs/car.yaml --limit 64 --set train.epochs=1   # smoke test
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch.utils.data import DataLoader

from pointpillars.anchors import make_anchors
from pointpillars.config import load_config, parse_overrides
from pointpillars.data import KittiDataset, collate, to_device
from pointpillars.eval import evaluate
from pointpillars.loss import PointPillarsLoss
from pointpillars.models import PointPillars
from pointpillars.utils import get_device, seed_everything


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/car.yaml")
    ap.add_argument("--out", default="outputs/car")
    ap.add_argument("--resume", default=None, help="full checkpoint (last.pth) to continue from")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N frames of each split")
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides, e.g. train.epochs=1 root=/data/kitti")
    return ap.parse_args()


def make_loader(ds, batch_size, num_workers, train):
    return DataLoader(ds, batch_size, shuffle=train, num_workers=num_workers, collate_fn=collate,
                      pin_memory=torch.cuda.is_available(), drop_last=train,
                      persistent_workers=train and num_workers > 0)


def main():
    args = parse_args()
    cfg = load_config(args.cfg, parse_overrides(args.set))
    tc = cfg["train"]
    seed_everything(tc["seed"])
    device = get_device(args.device)
    use_amp = tc["amp"] and device.type == "cuda"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    ds_tr = KittiDataset(cfg, "train", train=True, limit=args.limit)
    ds_va = KittiDataset(cfg, "val", train=False, limit=args.limit)
    dl_tr = make_loader(ds_tr, tc["batch_size"], tc["num_workers"], train=True)
    dl_va = make_loader(ds_va, cfg["test"]["batch_size"], tc["num_workers"], train=False)
    print(f"{len(ds_tr)} train / {len(ds_va)} val frames | device {device} | amp {use_amp}")

    model = PointPillars(cfg).to(device)
    anchors = make_anchors(cfg).to(device)
    criterion = PointPillarsLoss(cfg)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, {len(anchors)} anchors")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"],
                            betas=tuple(tc["betas"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=tc["lr"], total_steps=tc["epochs"] * len(dl_tr), pct_start=tc["pct_start"],
        div_factor=tc["div_factor"], final_div_factor=tc["final_div_factor"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)   # own file
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", -1.0)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    log = open(out / "log.jsonl", "a")

    def write_log(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    for ep in range(start_epoch, tc["epochs"]):
        model.train()
        t0, run, n = time.time(), {}, 0
        for it, batch in enumerate(dl_tr):
            batch = to_device(batch, device)
            with torch.autocast(device.type, enabled=use_amp):
                preds = model(batch)
            loss, parts = criterion(preds, batch["gt"], anchors)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()

            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
            n += 1
            if it % tc["log_every"] == 0:
                lr = sched.get_last_lr()[0]
                print(f"ep {ep} it {it}/{len(dl_tr)} loss {parts['loss']:.3f} cls {parts['cls']:.3f} "
                      f"box {parts['box']:.3f} dir {parts['dir']:.3f} pos {parts['npos']} lr {lr:.2e}")
                write_log(dict(type="iter", epoch=ep, iter=it, lr=lr, **parts))

        mean = {k: round(v / max(n, 1), 4) for k, v in run.items()}
        secs = round(time.time() - t0)
        print(f"== epoch {ep} mean loss {mean.get('loss', 0):.3f} ({secs}s)")
        write_log(dict(type="epoch", epoch=ep, seconds=secs, **mean))

        last = ep == tc["epochs"] - 1
        if tc["eval_every"] and ((ep + 1) % tc["eval_every"] == 0 or last):
            metrics = evaluate(model, dl_va, anchors, cfg, device)
            print("   val:", metrics)
            write_log(dict(type="val", epoch=ep, **metrics))
            score = metrics["bev_ap@0.7"]
            if score > best:
                best = score
                torch.save(model.state_dict(), out / "best_model.pth")   # weights only, for release
                print(f"   new best BEV AP@0.7 = {best}")

        ck = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                  scaler=scaler.state_dict(), epoch=ep, best=best, cfg=cfg)
        torch.save(ck, out / "last.pth")
        if last:
            torch.save(model.state_dict(), out / "final_model.pth")
    log.close()


if __name__ == "__main__":
    main()
