from __future__ import annotations

import torch
from monai.config import print_config
from monai.utils import set_determinism

DEFAULT_NUM_WORKERS = 4


def init_device_and_seed(seed: int | None, *, print_monai_config: bool = True) -> torch.device:
    """Select device, optionally print MONAI config, and set determinism.

    Args:
        seed (int | None): Seed used for deterministic behavior. If ``None``, determinism is not enforced.
        print_monai_config (bool): Whether to print MONAI config details.

    Returns:
        torch.device: Selected compute device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if print_monai_config:
        print_config()
    if seed is not None:
        set_determinism(seed)
    return device
