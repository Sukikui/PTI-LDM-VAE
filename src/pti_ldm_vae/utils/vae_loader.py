from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from monai.bundle import ConfigParser

from pti_ldm_vae.models import VAEModel


def load_vae_config(config_file: str) -> SimpleNamespace:
    """Load a VAE configuration file with MONAI ConfigParser.

    Args:
        config_file (str): Path to the configuration JSON file.

    Returns:
        SimpleNamespace: Configuration object with resolved references.
    """
    parser = ConfigParser()
    parser.read_config(config_file)
    parser.parse(True)
    cfg_dict = parser.get_parsed_content()
    return SimpleNamespace(**cfg_dict)


def load_vae_model(config: Any, checkpoint_path: str, device: torch.device) -> VAEModel:
    """Load a VAE model from a checkpoint.

    Args:
        config (Any): Parsed configuration containing ``autoencoder_def``.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Torch device to load the model onto.

    Returns:
        VAEModel: Model in evaluation mode with loaded weights.
    """
    autoencoder = VAEModel.from_config(config.autoencoder_def).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("autoencoder_state_dict", checkpoint)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    return autoencoder


def default_eval_output_dir(config_file: str, root_dir: str = "evals") -> Path:
    """Compute the default output directory for evaluation artifacts.

    Args:
        config_file (str): Path to the configuration file.
        root_dir (str): Root folder used to store evaluations.

    Returns:
        Path: Folder path ``<root_dir>/<config_name>/``.
    """
    config_name = Path(config_file).stem
    return Path(root_dir) / config_name
