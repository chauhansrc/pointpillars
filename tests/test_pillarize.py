import numpy as np

from pointpillars.ops.pillarize import pillarize


def test_points_grouped_into_pillars(cfg):
    pr, vs = cfg["pc_range"], cfg["voxel_size"]
    # 40 points in one pillar, 3 in another, 1 outside the range.
    a = np.tile([[10.05, 0.05, -1.0, 0.5]], (40, 1))
    b = np.tile([[20.05, -5.05, -1.0, 0.2]], (3, 1))
    out = np.array([[-5.0, 0.0, 0.0, 0.0]])
    feats, coords, num = pillarize(np.vstack([a, b, out]).astype(np.float32), cfg)

    assert feats.shape == (2, cfg["max_pts"], 4)
    assert sorted(num.tolist()) == [3, cfg["max_pts"]]          # capped at N
    expect = np.floor((np.array([[10.05, 0.05], [20.05, -5.05]]) - pr[:2]) / vs[:2]).astype(int)
    assert {tuple(c) for c in coords} == {tuple(c) for c in expect}
    i = int(np.argmin(num))
    assert np.allclose(feats[i, :3], b[0]) and np.all(feats[i, 3:] == 0)  # zero padded


def test_max_pillars_and_determinism(cfg):
    cfg = dict(cfg, max_pillars=100)
    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(0, 69, 5000), rng.uniform(-39, 39, 5000),
                    rng.uniform(-2, 0, 5000), rng.uniform(0, 1, 5000)], 1).astype(np.float32)
    f1, c1, n1 = pillarize(pts, cfg, np.random.default_rng(7))
    f2, c2, n2 = pillarize(pts, cfg, np.random.default_rng(7))
    assert len(c1) == 100
    assert np.array_equal(c1, c2) and np.array_equal(f1, f2)
    assert (c1[:, 0] < cfg["nx"]).all() and (c1[:, 1] < cfg["ny"]).all() and (c1 >= 0).all()


def test_edge_points_stay_inside_grid(cfg):
    """Points just inside the range can round to index nx / ny in float32; they must be clipped."""
    pr = np.array(cfg["pc_range"], np.float32)
    xmax = np.nextafter(pr[3], np.float32(0))
    ymin, ymax = pr[1], np.nextafter(pr[4], np.float32(0))
    pts = np.array([[xmax, ymax, -1, 0], [xmax, ymin, -1, 0], [pr[0], ymin, -1, 0]], np.float32)
    _, coords, _ = pillarize(pts, cfg)
    assert (coords >= 0).all()
    assert (coords[:, 0] < cfg["nx"]).all() and (coords[:, 1] < cfg["ny"]).all()


def test_empty_cloud(cfg):
    feats, coords, num = pillarize(np.zeros((0, 4), np.float32), cfg)
    assert feats.shape[0] == coords.shape[0] == num.shape[0] == 1
