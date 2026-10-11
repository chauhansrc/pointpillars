import yaml


def load_config(path, overrides=None):
    """Load a YAML config, apply ``{"dotted.key": value}`` overrides and add derived grid sizes."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        _set(cfg, key, value)
    return finalize(cfg)


def parse_overrides(items):
    """``["train.epochs=1", "root=/data/kitti"]`` -> ``{"train.epochs": 1, "root": "/data/kitti"}``."""
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        out[key.strip()] = yaml.safe_load(value)
    return out


def finalize(cfg):
    pr, vs = cfg["pc_range"], cfg["voxel_size"]
    cfg["nx"] = int(round((pr[3] - pr[0]) / vs[0]))   # 432
    cfg["ny"] = int(round((pr[4] - pr[1]) / vs[1]))   # 496
    # The head runs at backbone stride 2, so the anchor grid is half the pillar grid.
    cfg["fx"], cfg["fy"] = cfg["nx"] // 2, cfg["ny"] // 2
    return cfg


def _set(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
