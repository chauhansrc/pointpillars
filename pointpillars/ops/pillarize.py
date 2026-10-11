import numpy as np


def pillarize(points, cfg, rng=None):
    """Group a point cloud into vertical pillars (paper Sec. 2.1).

    Args:
        points: (n, 4) float32 array of x, y, z, reflectance in the LiDAR frame.
        cfg: config with ``pc_range``, ``voxel_size``, ``nx``, ``max_pts`` (N), ``max_pillars`` (P).
        rng: ``np.random.Generator`` used when more than P pillars are occupied. Pass a seeded
            generator for deterministic evaluation.

    Returns:
        feats  (M, N, 4) raw points per pillar, zero padded,
        coords (M, 2)    integer (x, y) grid index of each pillar,
        num    (M,)      number of valid points in each pillar,
    with M <= P.
    """
    pr, vs = cfg["pc_range"], cfg["voxel_size"]
    N, P = cfg["max_pts"], cfg["max_pillars"]
    m = ((points[:, 0] >= pr[0]) & (points[:, 0] < pr[3]) &
         (points[:, 1] >= pr[1]) & (points[:, 1] < pr[4]) &
         (points[:, 2] >= pr[2]) & (points[:, 2] < pr[5]))
    points = points[m]
    if len(points) == 0:
        return np.zeros((1, N, 4), np.float32), np.zeros((1, 2), np.int32), np.ones(1, np.int32)

    coords = np.floor((points[:, :2] - np.array(pr[:2], np.float32)) /
                      np.array(vs[:2], np.float32)).astype(np.int32)
    # Guard against float rounding at the edges of the range (an out-of-grid index crashes the scatter on GPU).
    coords = np.clip(coords, 0, [cfg["nx"] - 1, cfg["ny"] - 1]).astype(np.int32)

    key = coords[:, 1] * cfg["nx"] + coords[:, 0]
    order = np.argsort(key, kind="stable")
    points, coords, key = points[order], coords[order], key[order]
    _, start, cnt = np.unique(key, return_index=True, return_counts=True)
    n_pillars = len(start)

    slot = np.arange(len(points)) - np.repeat(start, cnt)     # index of each point inside its pillar
    pillar_id = np.repeat(np.arange(n_pillars), cnt)
    keep = slot < N                                           # drop points beyond N per pillar
    feats = np.zeros((n_pillars, N, 4), np.float32)
    feats[pillar_id[keep], slot[keep]] = points[keep, :4]
    coo = coords[start]
    num = np.minimum(cnt, N).astype(np.int32)

    if n_pillars > P:
        rng = rng if rng is not None else np.random.default_rng()
        sel = rng.choice(n_pillars, P, replace=False)
        feats, coo, num = feats[sel], coo[sel], num[sel]
    return feats, coo, num
