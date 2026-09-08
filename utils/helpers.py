"""
PhysioRANO Helper Utilities.

Provides reproducibility and filesystem helper functions.
"""

import os
import random
from pathlib import Path
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, PyTorch, and MONAI for reproducibility.

    Args:
        seed (int): Integer random seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        from monai.utils import set_determinism
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        set_determinism(seed=seed)
    except Exception:
        pass


def ensure_dir(path: Path) -> Path:
    """Ensures a directory path exists, creating parents if missing.

    Args:
        path (Path): Path object or directory path.

    Returns:
        Path: Resolved existing directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_exists(path: Path) -> bool:
    """Checks if a specified file path exists and is a valid file.

    Args:
        path (Path): Path to check.

    Returns:
        bool: True if path exists and is a file, False otherwise.
    """
    return Path(path).is_file()
