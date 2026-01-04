import argparse
from pathlib import Path

import tifffile
import torch
from PIL import Image
from tqdm import tqdm

from pti_ldm_vae.ldm import (
    ConditionContextBuilder,
    DiffusionSchedule,
    LatentDiffusionSampler,
    MetricConditioning,
    build_frozen_regressor,
    build_frozen_vae,
    build_unet,
)
from pti_ldm_vae.utils.cli_common import (
    build_inference_dataloader,
    init_device_and_seed,
    load_json_config,
)
from pti_ldm_vae.utils.vae_loader import load_vae_config
from pti_ldm_vae.utils.visualization import normalize_batch_for_display


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for LDM sampling.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Generate edentulous images with a trained LDM.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to LDM JSON config.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained LDM checkpoint.")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of diffusion steps.")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta parameter.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Classifier-free guidance scale.")
    parser.add_argument("--drop-z", type=float, default=0.0, help="Drop probability for dentate latent conditioning.")
    parser.add_argument("--drop-metrics", type=float, default=0.0, help="Drop probability for metric conditioning.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--num-samples", type=int, default=None, help="Optional limit on processed images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--input-dir", required=True, help="Directory of dentate images to condition on.")
    parser.add_argument("--output-dir", default=None, help="Output directory for generated samples.")
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


def save_results(
    outputs: torch.Tensor,
    dentate: torch.Tensor,
    start_idx: int,
    out_tif: Path,
    out_png: Path,
) -> int:
    """Save a batch of generated images to disk.

    Args:
        outputs: Generated edentulous images.
        dentate: Conditioning dentate images.
        start_idx: Starting index for filenames.
        out_tif: Directory for TIF outputs.
        out_png: Directory for PNG outputs.

    Returns:
        Next global index after saving.
    """
    outputs_cpu = outputs.cpu()
    dentate_cpu = dentate.cpu()
    batch_size = outputs_cpu.shape[0]
    disp_generated = normalize_batch_for_display(outputs_cpu)
    disp_dentate = normalize_batch_for_display(dentate_cpu)

    for i in range(batch_size):
        concat = torch.cat([dentate_cpu[i], outputs_cpu[i]], dim=2).numpy()
        tifffile.imwrite(out_tif / f"sample_{start_idx + i:04d}.tif", concat)

        png_pair = torch.cat([disp_dentate[i], disp_generated[i]], dim=2)[0].numpy()
        png_uint8 = (png_pair * 255).astype("uint8")
        Image.fromarray(png_uint8).save(out_png / f"sample_{start_idx + i:04d}.png")
    return start_idx + batch_size


def _default_output_dirs(run_dir: Path, input_dir: str) -> tuple[Path, Path, Path]:
    """Resolve default output directories for LDM sampling.

    Args:
        run_dir (Path): Root directory for the LDM run.
        input_dir (str): Input directory passed to the sampler.

    Returns:
        tuple[Path, Path, Path]: Root output directory, TIF subfolder, PNG subfolder.
    """
    input_path = Path(input_dir).resolve()
    try:
        relative_input = input_path.relative_to(Path.cwd())
    except ValueError:
        relative_input = Path(str(input_path).lstrip("/"))

    base_output = run_dir / "results" / relative_input
    out_tif = base_output / "results_tif"
    out_png = base_output / "results_png"
    out_tif.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)
    return base_output, out_tif, out_png


def main() -> None:
    args = parse_args()
    device = init_device_and_seed(args.seed)
    config = load_json_config(args.config_file)

    data_cfg = config.get("data", {})
    conditioning_cfg = config.get("conditioning", {})
    diffusion_cfg = config.get("diffusion", {})
    unet_cfg = config.get("unet", {})

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

    concat_dentate = bool(conditioning_cfg.get("concat_dentate", True))
    unet_cfg = dict(unet_cfg)
    unet_cfg["in_channels"] = latent_channels * (2 if concat_dentate else 1)
    unet_cfg["out_channels"] = latent_channels
    unet = build_unet(unet_cfg).to(device)

    cross_attention_dim = unet_cfg.get("cross_attention_dim", 256)
    metric_embed = MetricConditioning(
        input_dim=len(config["regressor"]["targets"]),
        embed_dim=cross_attention_dim,
        dropout=0.0,
    ).to(device)
    condition_builder = ConditionContextBuilder(latent_channels, cross_attention_dim).to(device)
    scale_factor = load_ldm_checkpoint(unet, metric_embed, condition_builder, args.checkpoint)
    unet.eval()

    schedule = DiffusionSchedule.linear(
        timesteps=diffusion_cfg.get("num_train_timesteps", 1000),
        beta_start=diffusion_cfg.get("beta_start", 0.00085),
        beta_end=diffusion_cfg.get("beta_end", 0.012),
        device=device,
    )

    sampler = LatentDiffusionSampler(
        unet=unet,
        vae=vae,
        condition_builder=condition_builder,
        metric_embed=metric_embed,
        schedule=schedule,
        concat_dentate=concat_dentate,
        scale_factor=scale_factor,
    )

    # Reuse VAE dataloader helper for dentate images only
    vae_config = load_vae_config(config["vae"]["config_file"])
    dataloader, image_paths = build_inference_dataloader(
        input_dir=args.input_dir,
        config=vae_config,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
    )

    run_dir = Path(config.get("run_dir", "runs/ldm_run"))
    if args.output_dir is None:
        output_dir, out_tif, out_png = _default_output_dirs(run_dir, args.input_dir)
    else:
        output_dir = Path(args.output_dir)
        out_tif = output_dir / "results_tif"
        out_png = output_dir / "results_png"
        out_tif.mkdir(parents=True, exist_ok=True)
        out_png.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_tif.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)

    idx = 0
    for batch in tqdm(dataloader, desc="Sampling", unit="batch"):
        batch = batch.to(device)
        generated = sampler(
            dentate_images=batch,
            regressor=regressor,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            drop_z_prob=args.drop_z,
            drop_metrics_prob=args.drop_metrics,
            eta=args.eta,
        )
        idx = save_results(generated, batch, idx, out_tif, out_png)

    print(f"✅ Generated {idx} samples. Outputs: {output_dir}")


if __name__ == "__main__":
    main()
