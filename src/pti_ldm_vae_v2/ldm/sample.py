from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import tifffile
import torch
from PIL import Image
from tqdm import tqdm

from pti_ldm_vae_v2.vae.visualization import normalize_batch_for_display
from pti_ldm_vae_v2.common import DEFAULT_NUM_WORKERS, init_device_and_seed, resolve_run_output_dir
from pti_ldm_vae_v2.tools.mask_metrics_utils import (
    binary_mask_from_prediction,
    compute_bbox,
    compute_edente_widths,
)

from .build import build_frozen_regressor, build_frozen_vae, build_unet
from pti_ldm_vae_v2.models.conditioning import CondEnc, ContextBuilder
from .config import load_config, resolve_run_dir
from .data import create_ldm_inference_dataloader
from .noise import read_noise_init_config
from .sampler import LatentDiffusionSampler
from .scheduler import build_ddim_scheduler


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for LDM sampling.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Generate edentulous images with a trained LDM.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to LDM JSON config.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained LDM checkpoint.")
    parser.add_argument("--input-dir", required=True, help="Directory of dentate images to condition on.")
    parser.add_argument(
        "--edente-dir",
        default=None,
        help="Directory of edentulous ground-truth images (defaults to sibling 'edente').",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for generated samples.")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of diffusion steps.")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta parameter.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Classifier-free guidance scale.")
    parser.add_argument("--drop-z", type=float, default=0.0, help="Drop probability for dentate latent conditioning.")
    parser.add_argument("--drop-metrics", type=float, default=0.0, help="Drop probability for metric conditioning.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--num-samples", type=int, default=None, help="Optional limit on processed images.")
    return parser.parse_args()


def load_ldm_checkpoint(
    unet: torch.nn.Module,
    metric_embed: torch.nn.Module,
    condition_builder: torch.nn.Module,
    checkpoint: str,
) -> float:
    """Load LDM checkpoint weights for the UNet and conditioning modules.

    Args:
        unet (torch.nn.Module): UNet instance to populate.
        metric_embed (torch.nn.Module): Metric conditioning embed module.
        condition_builder (torch.nn.Module): Spatial conditioning context builder.
        checkpoint (str): Checkpoint path.

    Returns:
        float: Latent scale factor (defaults to 1.0 when missing).
    """
    payload = torch.load(checkpoint, map_location="cpu")
    unet_state = payload.get("ema_unet_state_dict") or payload.get("unet_state_dict") or payload
    unet.load_state_dict(unet_state)

    metric_state = payload.get("metric_embed_state_dict")
    if metric_state is not None:
        metric_embed.load_state_dict(metric_state)
    else:
        print("[WARN] metric_embed_state_dict not found in checkpoint; using default initialization.")

    condition_state = payload.get("condition_builder_state_dict")
    if condition_state is not None:
        condition_builder.load_state_dict(condition_state)
    else:
        print("[WARN] condition_builder_state_dict not found in checkpoint; using default initialization.")

    return float(payload.get("scale_factor", 1.0))


def _resolve_edente_dir(input_dir: str, edente_dir: str | None) -> Path:
    """Resolve the ground-truth edentulous directory for sampling.

    Args:
        input_dir (str): Dentate input directory.
        edente_dir (str | None): Optional explicit edentulous directory.

    Returns:
        Path: Resolved edentulous directory.

    Raises:
        FileNotFoundError: If the resolved directory does not exist.
        ValueError: If edente directory cannot be inferred.
    """
    if edente_dir is not None:
        resolved = Path(edente_dir).expanduser().resolve()
    else:
        input_path = Path(input_dir).expanduser().resolve()
        if input_path.name == "dente":
            resolved = input_path.parent / "edente"
        elif "dente" in input_path.parts:
            resolved = Path(*["edente" if part == "dente" else part for part in input_path.parts])
        else:
            raise ValueError("Cannot infer edente directory from input_dir; use --edente-dir.")
    if not resolved.exists():
        raise FileNotFoundError(f"Edente directory not found: {resolved}")
    return resolved


def _load_edente_batch(
    edente_dir: Path,
    batch_paths: list[str],
    transform: Callable[[str], torch.Tensor],
) -> torch.Tensor:
    """Load a batch of ground-truth edentulous images matching dentate paths.

    Args:
        edente_dir (Path): Directory containing edentulous images.
        batch_paths (list[str]): Dentate image paths for the current batch.
        transform (Callable[[str], torch.Tensor]): Preprocessing transform.

    Returns:
        torch.Tensor: Batch of edentulous images [B, C, H, W].
    """
    tensors: list[torch.Tensor] = []
    for path in batch_paths:
        filename = Path(path).name
        edente_path = edente_dir / filename
        if not edente_path.exists():
            raise FileNotFoundError(f"Missing edente ground truth: {edente_path}")
        tensors.append(transform(str(edente_path)))
    return torch.stack(tensors, dim=0)


def save_results(
    outputs: torch.Tensor,
    dentate: torch.Tensor,
    edente_gt: torch.Tensor,
    input_paths: list[str],
    out_tif: Path,
    out_png: Path,
) -> None:
    """Save a batch of generated images to disk.

    Args:
        outputs (torch.Tensor): Generated edentulous images.
        dentate (torch.Tensor): Conditioning dentate images.
        edente_gt (torch.Tensor): Ground-truth edentulous images.
        input_paths (list[str]): Input filenames used to name outputs.
        out_tif (Path): Directory for TIF outputs.
        out_png (Path): Directory for PNG outputs.
    """
    outputs_cpu = outputs.cpu()
    dentate_cpu = dentate.cpu()
    edente_cpu = edente_gt.cpu()
    disp_generated = normalize_batch_for_display(outputs_cpu)
    disp_dentate = normalize_batch_for_display(dentate_cpu)
    disp_edente = normalize_batch_for_display(edente_cpu)

    for i, src_path in enumerate(input_paths):
        filename = Path(src_path).name
        stem = Path(src_path).stem
        rotated_pred = torch.rot90(outputs_cpu[i], k=3, dims=[1, 2])
        tifffile.imwrite(out_tif / filename, rotated_pred[0].numpy())

        png_triplet = torch.cat([disp_dentate[i], disp_edente[i], disp_generated[i]], dim=2)[0].numpy()
        png_uint8 = (png_triplet * 255).astype("uint8")
        Image.fromarray(png_uint8).save(out_png / f"{stem}.png")


def _infer_width_samples(targets: list[str]) -> int:
    """Infer the number of width samples from regression target names.

    Args:
        targets (list[str]): Regression target names.

    Returns:
        int: Number of ``width_*`` targets.
    """
    return sum(1 for target in targets if target.startswith("width_"))


def _compute_pred_metrics(
    prediction: torch.Tensor,
    *,
    width_samples: int,
) -> tuple[dict[str, int], bool]:
    """Compute edente metrics from a predicted image.

    Args:
        prediction (torch.Tensor): Predicted edentulous image [C, H, W].
        width_samples (int): Number of width samples to compute.

    Returns:
        tuple[dict[str, int], bool]: Metrics dict and whether the mask was empty.
    """
    image = prediction.detach().cpu().numpy()
    mask = binary_mask_from_prediction(image)
    if mask.sum() == 0:
        metrics = {"height_0": 0}
        for idx in range(width_samples):
            metrics[f"width_{idx}"] = 0
        return metrics, True

    try:
        x_min, y_min, bbox_w, bbox_h = compute_bbox(mask)
    except ValueError:
        metrics = {"height_0": 0}
        for idx in range(width_samples):
            metrics[f"width_{idx}"] = 0
        return metrics, True

    bbox_height_px, widths = compute_edente_widths(
        mask,
        x=x_min,
        y=y_min,
        width=bbox_w,
        height=bbox_h,
        samples=width_samples,
    )
    metrics = {"height_0": int(bbox_height_px)}
    for idx, value in enumerate(widths):
        metrics[f"width_{idx}"] = int(value)
    return metrics, False


def main() -> None:
    """Entry point for LDM sampling."""
    args = parse_args()
    config = load_config(args.config_file)
    data_cfg = dict(config.get("data", {}))
    conditioning_cfg = dict(config.get("conditioning", {}))
    diffusion_cfg = dict(config.get("diffusion", {}))
    unet_cfg = dict(config.get("unet", {}))
    noise_init = read_noise_init_config(config)

    device = init_device_and_seed(config.get("seed"), print_monai_config=False)

    vae, latent_channels = build_frozen_vae(
        config_file=config["vae"]["config_file"],
        checkpoint=config["vae"]["checkpoint"],
        device=device,
    )
    regressor = build_frozen_regressor(
        config_file=config["regressor"]["config_file"],
        checkpoint=config["regressor"]["checkpoint"],
        device=device,
        patch_size=tuple(data_cfg["patch_size"]),
        targets=config["regressor"]["targets"],
    )
    width_samples = _infer_width_samples(config["regressor"]["targets"])

    use_dentate_latent = bool(conditioning_cfg.get("use_dentate_latent", True))
    concat_dentate = bool(conditioning_cfg.get("concat_dentate", True))
    if not use_dentate_latent and concat_dentate:
        print("[INFO] use_dentate_latent is False; forcing concat_dentate to False.")
        concat_dentate = False
    unet_cfg = dict(unet_cfg)
    unet_cfg["in_channels"] = latent_channels * (2 if concat_dentate else 1)
    unet_cfg["out_channels"] = latent_channels
    unet = build_unet(unet_cfg).to(device)

    cross_attention_dim = unet_cfg.get("cross_attention_dim", 256)
    metric_embed = CondEnc(
        input_dim=len(config["regressor"]["targets"]),
        embed_dim=cross_attention_dim,
        dropout=conditioning_cfg.get("metric_dropout", 0.0),
    ).to(device)
    condition_builder = ContextBuilder(latent_channels, cross_attention_dim).to(device)
    scale_factor = load_ldm_checkpoint(unet, metric_embed, condition_builder, args.checkpoint)
    unet.eval()

    ddim_scheduler = build_ddim_scheduler(diffusion_cfg, args.num_steps, device)

    sampler = LatentDiffusionSampler(
        unet=unet,
        vae=vae,
        condition_builder=condition_builder,
        metric_embed=metric_embed,
        ddim_scheduler=ddim_scheduler,
        concat_dentate=concat_dentate,
        use_dentate_latent=use_dentate_latent,
        scale_factor=scale_factor,
    )

    dataloader, image_paths = create_ldm_inference_dataloader(
        input_dir=args.input_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=int(data_cfg.get("num_workers", DEFAULT_NUM_WORKERS)),
    )
    edente_dir = _resolve_edente_dir(args.input_dir, args.edente_dir)
    transform = dataloader.dataset.transform

    run_dir = resolve_run_dir(config, args.config_file)
    output_dir = resolve_run_output_dir(run_dir, args.input_dir, args.output_dir, "results")
    out_tif = output_dir / "results_tif"
    out_png = output_dir / "results_png"
    out_tif.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)
    metrics_dir = resolve_run_output_dir(run_dir, args.input_dir, None, "metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "attributes_edente_pred.json"
    pred_metrics: dict[str, dict[str, int]] = {}
    empty_masks = 0

    idx = 0
    for batch in tqdm(dataloader, desc="Sampling", unit="batch"):
        batch_size = batch.shape[0]
        batch_paths = image_paths[idx : idx + batch_size]
        edente_batch = _load_edente_batch(edente_dir, batch_paths, transform)
        batch = batch.to(device)
        generated = sampler(
            dentate_images=batch,
            regressor=regressor,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            drop_z_prob=args.drop_z,
            drop_metrics_prob=args.drop_metrics,
            eta=args.eta,
            init_mode=str(noise_init["init_mode"]),
            noise_top=float(noise_init["noise_top"]),
            noise_bottom=float(noise_init["noise_bottom"]),
            noise_exponent=float(noise_init["noise_exponent"]),
            noise_direction=str(noise_init["noise_direction"]),
            noise_weight=float(noise_init["noise_weight"]),
        )
        save_results(generated, batch, edente_batch, batch_paths, out_tif, out_png)
        for i, src_path in enumerate(batch_paths):
            rotated_pred = torch.rot90(generated[i], k=3, dims=[1, 2])
            metrics, is_empty = _compute_pred_metrics(rotated_pred, width_samples=width_samples)
            pred_metrics[Path(src_path).name] = metrics
            if is_empty:
                empty_masks += 1
        idx += batch_size

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(pred_metrics, file, indent=4)

    if empty_masks > 0:
        print(f"[WARN] {empty_masks} predictions produced empty masks for metrics.")

    print(f"[INFO] Generated {idx} samples. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
