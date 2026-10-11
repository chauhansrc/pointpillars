import numpy as np

from pointpillars.data.kitti import KittiDataset, read_calib, read_label, to_kitti_lines


def test_label_roundtrip_through_export(fake_kitti):
    """camera label -> LiDAR box -> exported KITTI line should give back the original label."""
    tr = f"{fake_kitti['root']}/training"
    calib = read_calib(f"{tr}/calib/000000.txt")
    boxes = read_label(f"{tr}/label_2/000000.txt", "Car", calib)
    assert boxes.shape == (2, 7)

    dets = np.hstack([boxes, np.ones((2, 1), np.float32)])
    lines = to_kitti_lines(dets, calib, "Car")
    orig = [l.split() for l in open(f"{tr}/label_2/000000.txt") if l.startswith("Car")]
    for line, ref in zip(lines, orig):
        f = line.split()
        assert np.allclose([float(x) for x in f[8:14]], [float(x) for x in ref[8:14]], atol=1e-3)
        dry = (float(f[14]) - float(ref[14]) + np.pi) % (2 * np.pi) - np.pi
        assert abs(dry) < 1e-3


def test_dataset_sample(fake_kitti):
    ds = KittiDataset(fake_kitti, "val", train=False)
    s1, s2 = ds[0], ds[0]
    assert s1["pillars"].shape[1:] == (fake_kitti["max_pts"], 4)
    assert np.array_equal(s1["coords"], s2["coords"])           # eval sampling is deterministic
    assert s1["gt"].shape == (2, 7)

    ds_tr = KittiDataset(fake_kitti, "train", train=True)
    assert ds_tr[1]["gt"].shape[1] == 7
