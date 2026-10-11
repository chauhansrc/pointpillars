import torch
import torch.nn as nn
import torch.nn.functional as F


class PillarFeatureNet(nn.Module):
    """Simplified PointNet over each pillar (paper Sec. 2.1).

    Every point is decorated to D = 9 features (x, y, z, r, xc, yc, zc, xp, yp), where c is the
    offset from the pillar's point centroid and p the offset from the pillar's grid center. A shared
    Linear + BN + ReLU lifts them to C channels and a max over the N points gives one vector per pillar.
    """

    def __init__(self, cfg, out_channels=64):
        super().__init__()
        self.pc_range = cfg["pc_range"]
        self.voxel_size = cfg["voxel_size"]
        self.lin = nn.Linear(9, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coords, num):
        """pillars (M, N, 4), coords (M, 3) = [b, x, y], num (M,) -> (M, C)."""
        pr, vs = self.pc_range, self.voxel_size
        N = pillars.shape[1]
        mask = (torch.arange(N, device=pillars.device)[None] < num[:, None]).unsqueeze(-1).to(pillars.dtype)

        mean = (pillars[..., :3] * mask).sum(1, keepdim=True) / num.clamp(min=1)[:, None, None].to(pillars.dtype)
        f_cluster = pillars[..., :3] - mean
        cx = (coords[:, 1].to(pillars.dtype) + 0.5) * vs[0] + pr[0]
        cy = (coords[:, 2].to(pillars.dtype) + 0.5) * vs[1] + pr[1]
        f_center = torch.stack([pillars[..., 0] - cx[:, None], pillars[..., 1] - cy[:, None]], -1)

        f = torch.cat([pillars, f_cluster, f_center], -1) * mask            # (M, N, 9)
        f = self.lin(f)
        f = F.relu(self.bn(f.transpose(1, 2)).transpose(1, 2)) * mask      # padded points -> 0
        return f.max(1)[0]


def scatter_to_bev(feats, coords, batch_size, nx, ny):
    """Scatter pillar features back to their grid cells -> pseudo-image (B, C, ny, nx)."""
    C = feats.shape[1]
    canvas = feats.new_zeros(batch_size, C, ny * nx)
    idx = (coords[:, 2] * nx + coords[:, 1]).clamp_(0, ny * nx - 1)
    canvas[coords[:, 0], :, idx] = feats
    return canvas.view(batch_size, C, ny, nx)
