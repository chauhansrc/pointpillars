import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pointpillars.config import load_config  # noqa: E402

# Typical KITTI calibration: velodyne (x fwd, y left, z up) -> camera (x right, y down, z fwd).
CALIB = """P0: 721.5 0 609.6 0 0 721.5 172.9 0 0 0 1 0
P1: 721.5 0 609.6 -387.6 0 721.5 172.9 0 0 0 1 0
P2: 721.5377 0 609.5593 44.85728 0 721.5377 172.854 0.2163791 0 0 1 0.002745884
P3: 721.5 0 609.6 -339.5 0 721.5 172.9 2.2 0 0 1 0.0027
R0_rect: 0.9999239 0.00983776 -0.007445048 -0.009869795 0.9999421 -0.004278459 0.007402527 0.004351614 0.9999631
Tr_velo_to_cam: 0.007533745 -0.9999714 -0.000616602 -0.004069766 0.01480249 0.0007280733 -0.9998902 -0.07631618 0.9998621 0.007523790 0.01480755 -0.2717806
Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0
"""

# Two cars and a pedestrian, KITTI label_2 format (camera frame, bottom-center location).
LABEL = """Car 0.00 0 -1.58 587.0 173.3 614.1 200.1 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59
Car 0.00 0 1.85 387.6 181.5 423.8 203.1 1.67 1.87 3.69 -16.53 2.39 58.49 1.57
Pedestrian 0.00 0 -0.20 712.4 143.0 810.7 307.9 1.89 0.48 1.20 1.84 1.47 8.41 0.01
DontCare -1 -1 -10 503.9 169.7 590.6 190.1 -1 -1 -1 -1000 -1000 -1000 -10
"""


@pytest.fixture
def cfg():
    return load_config(ROOT / "configs" / "car.yaml")


@pytest.fixture
def fake_kitti(tmp_path, cfg):
    """A 3-frame KITTI-shaped directory with random points plus points sampled inside the cars."""
    rng = np.random.default_rng(0)
    tr = tmp_path / "training"
    for d in ("velodyne", "calib", "label_2"):
        (tr / d).mkdir(parents=True)
    (tmp_path / "ImageSets").mkdir()
    ids = ["000000", "000001", "000002"]
    for fid in ids:
        n = 20000
        pts = np.stack([rng.uniform(0, 70, n), rng.uniform(-40, 40, n), rng.uniform(-2.5, 0.5, n),
                        rng.uniform(0, 1, n)], 1).astype(np.float32)
        pts.tofile(tr / "velodyne" / f"{fid}.bin")
        (tr / "calib" / f"{fid}.txt").write_text(CALIB)
        (tr / "label_2" / f"{fid}.txt").write_text(LABEL)
    for split in ("train", "val"):
        (tmp_path / "ImageSets" / f"{split}.txt").write_text("\n".join(ids) + "\n")
    cfg["root"] = str(tmp_path)
    cfg["train"]["num_workers"] = 0
    return cfg
