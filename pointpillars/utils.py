import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_weights(model, path, device="cpu"):
    """Load either a bare ``state_dict`` (e.g. the original notebook's ``pointpillars.pth``) or a full
    training checkpoint saved by ``tools/train.py``."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # Full checkpoints also hold optimizer / scheduler state that the safe loader may reject.
        # Only do this for files you produced yourself.
        ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return ckpt
