"""
Models module for PTI-LDM-VAE project.

Simple wrappers around MONAI models with configuration-based instantiation.
"""

from .autoencoder import VAEModel
from .unet import DiffusionUNet, create_condition_projector
from .losses import compute_kl_loss

__all__ = ["VAEModel", "DiffusionUNet", "create_condition_projector", "compute_kl_loss"]