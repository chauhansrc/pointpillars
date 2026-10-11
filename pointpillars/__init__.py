"""PointPillars (Lang et al., CVPR 2019) in pure PyTorch."""

from .config import load_config
from .models import PointPillars

__all__ = ["load_config", "PointPillars"]
__version__ = "0.1.0"
