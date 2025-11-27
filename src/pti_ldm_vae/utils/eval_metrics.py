from typing import Any

import torch


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute PSNR for a batch.

    Args:
        pred (torch.Tensor): Predicted images [B, C, H, W].
        target (torch.Tensor): Reference images [B, C, H, W].
        data_range (float): Value range for pixels.

    Returns:
        torch.Tensor: PSNR per sample.
    """
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-12)
    return 10 * torch.log10(torch.tensor(data_range, device=pred.device) ** 2 / mse)


def compute_ssim(
    pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, k1: float = 0.01, k2: float = 0.03
) -> torch.Tensor:
    """Compute SSIM for single-channel images using a Gaussian window.

    Args:
        pred (torch.Tensor): Predicted images [B, C, H, W].
        target (torch.Tensor): Reference images [B, C, H, W].
        data_range (float): Value range for pixels.
        k1 (float): Stabilization constant K1.
        k2 (float): Stabilization constant K2.

    Returns:
        torch.Tensor: SSIM per sample.
    """
    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, device=pred.device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(0)
    kernel_1d = g
    kernel_2d = (kernel_1d.t() @ kernel_1d).unsqueeze(0).unsqueeze(0)
    pad = window_size // 2

    def _filter(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(x, kernel_2d, padding=pad, groups=x.shape[1])

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_x = _filter(pred)
    mu_y = _filter(target)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _filter(pred * pred) - mu_x2
    sigma_y2 = _filter(target * target) - mu_y2
    sigma_xy = _filter(pred * target) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return ssim_map.mean(dim=(1, 2, 3))


def serialize_args(args: Any) -> dict[str, Any]:
    """Convert CLI arguments to JSON-serializable primitives.

    Args:
        args (Any): Parsed CLI arguments (Namespace-like).

    Returns:
        Dict[str, Any]: Mapping of argument names to primitive values.
    """
    serialized: dict[str, Any] = {}
    for key, value in vars(args).items():
        if hasattr(value, "__fspath__"):
            serialized[key] = str(value)
        elif isinstance(value, (list, tuple)):
            serialized[key] = [str(item) for item in value]
        else:
            serialized[key] = value
    return serialized
