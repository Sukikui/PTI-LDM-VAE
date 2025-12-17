import argparse
import json
from pathlib import Path
from typing import Any

import torch
from monai.config import print_config
from monai.utils import set_determinism
from torch.utils.data import DataLoader

from pti_ldm_vae.data import create_vae_inference_dataloader
from pti_ldm_vae.models import VAEModel
from pti_ldm_vae.utils.vae_loader import default_eval_output_dir, load_vae_config, load_vae_model


def add_shared_io_args(parser: argparse.ArgumentParser, output_help: str) -> None:
    """Add common CLI arguments for inference and evaluation vae_scripts.

    Args:
        parser (argparse.ArgumentParser): Parser instance to enrich.
        output_help (str): Help string for the ``--output-dir`` argument.
    """
    parser.add_argument("-c", "--config-file", required=True, help="Config json file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint (e.g., checkpoint_epoch73.pth)"
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input TIF images")
    parser.add_argument("--output-dir", type=str, default=None, help=output_help)
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism (default: 42)")


def init_device_and_seed(seed: int | None) -> torch.device:
    """Select device, print MONAI config, and set determinism.

    Args:
        seed (int | None): Seed used for deterministic behavior. If ``None``, determinism is not enforced.

    Returns:
        torch.device: Selected compute device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print_config()
    if seed is not None:
        set_determinism(seed)
    return device


def load_config_and_model(config_file: str, checkpoint_path: str, device: torch.device) -> tuple[Any, VAEModel]:
    """Load VAE configuration and model weights.

    Args:
        config_file (str): Path to the configuration JSON file.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Target device.

    Returns:
        tuple[Any, VAEModel]: Parsed configuration object and VAE model in eval mode.
    """
    config = load_vae_config(config_file)
    autoencoder = load_vae_model(config, checkpoint_path, device)
    return config, autoencoder


def build_inference_dataloader(
    input_dir: str,
    config: Any,
    batch_size: int,
    num_samples: int | None,
    num_workers: int,
) -> tuple[DataLoader, list[str]]:
    """Construct dataloader for inference/evaluation with consistent preprocessing.

    Args:
        input_dir (str): Folder containing input TIF images.
        config (Any): Parsed configuration with ``autoencoder_train.patch_size``.
        batch_size (int): Loader batch size.
        num_samples (int | None): Optional cap on number of images.
        num_workers (int): Number of worker processes.

    Returns:
        tuple[DataLoader, list[str]]: Dataloader instance and the list of discovered image paths.
    """
    patch_size = tuple(config.autoencoder_train["patch_size"])
    return create_vae_inference_dataloader(
        input_dir=input_dir,
        patch_size=patch_size,
        batch_size=batch_size,
        num_samples=num_samples,
        num_workers=num_workers,
    )


def resolve_inference_output_dirs(checkpoint_path: str, output_dir: str | None) -> tuple[Path, Path, Path]:
    """Compute output directories for inference results (TIF + PNG).

    Args:
        checkpoint_path (str): Path to the checkpoint, used to derive default name.
        output_dir (str | None): Optional user-provided output directory.

    Returns:
        tuple[Path, Path, Path]: Root output directory, TIF subfolder, PNG subfolder.
    """
    checkpoint_name = Path(checkpoint_path).stem
    base_output = Path(f"inference_vae_{checkpoint_name}") if output_dir is None else Path(output_dir)

    out_tif = base_output / "results_tif"
    out_png = base_output / "results_png"
    out_tif.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)
    return base_output, out_tif, out_png


def resolve_eval_output_dir(config_file: str, output_dir: str | None) -> Path:
    """Resolve and create evaluation output directory.

    Args:
        config_file (str): Path to the configuration file (used for default path).
        output_dir (str | None): Optional override path for metrics.

    Returns:
        Path: Existing or newly created output directory.
    """
    concrete_dir = Path(output_dir) if output_dir is not None else default_eval_output_dir(config_file)
    concrete_dir.mkdir(parents=True, exist_ok=True)
    return concrete_dir


def load_json_config(config_file: str) -> dict[str, Any]:
    """Load a JSON configuration file.

    Args:
        config_file (str): Path to the JSON file.

    Returns:
        dict[str, Any]: Parsed configuration dictionary.
    """
    with open(config_file, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_run_dir(config: dict[str, Any], config_file: str) -> Path:
    """Resolve run directory, defaulting to ``runs/<config_stem>`` when missing.

    Args:
        config (dict[str, Any]): Configuration dictionary to enrich.
        config_file (str): Path to the configuration file.

    Returns:
        Path: Absolute or relative run directory path stored back into config.
    """
    if config.get("run_dir"):
        run_dir = Path(config["run_dir"])
    else:
        run_dir = Path("runs") / Path(config_file).stem
        config["run_dir"] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
