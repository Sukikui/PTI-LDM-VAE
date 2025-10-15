import argparse
from pathlib import Path

import numpy as np
from analysis_common import (
    compute_and_save_statistics,
    create_transforms,
    load_and_encode_group,
    load_vae_model,
    save_visualization_and_legend,
    set_seed,
    setup_device_and_output,
)

from pti_ldm_vae.analysis import LatentSpaceAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="t-SNE Analysis of VAE Latent Space")
    parser.add_argument("--vae-weights", type=str, required=True, help="Path to VAE weights file")
    parser.add_argument("--config-file", type=str, required=True, help="Path to model config file")
    parser.add_argument(
        "--folder-group1", type=str, required=True, help="Path to first image group folder (e.g., edentulous)"
    )
    parser.add_argument(
        "--folder-group2", type=str, default=None, help="Path to second image group folder (e.g., dental)"
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--max-images", type=int, default=1000, help="Maximum number of images per group")
    parser.add_argument("--patch-size", type=int, nargs=2, default=[256, 256], help="Image patch size (H W)")
    parser.add_argument("--color-by-patient", action="store_true", help="Color points by patient ID instead of group")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Setup
    device, output_dir = setup_device_and_output(args.output_dir)
    vae = load_vae_model(args.config_file, args.vae_weights, device)
    transforms = create_transforms(tuple(args.patch_size))
    analyzer = LatentSpaceAnalyzer(vae, device, transforms)

    # Load and encode group 1
    latent_group1, ids_group1, paths_group1 = load_and_encode_group(
        analyzer, args.folder_group1, args.max_images, "group 1"
    )

    # Load and encode group 2 if provided
    latent_group2, ids_group2, paths_group2 = None, None, None
    if args.folder_group2:
        latent_group2, ids_group2, paths_group2 = load_and_encode_group(
            analyzer, args.folder_group2, args.max_images, "group 2"
        )

    # t-SNE reduction on combined data
    print("\n⏳ Computing t-SNE projection...")
    print("(This may take a few minutes...)")

    # Combine all data for joint t-SNE
    combined_latent = np.concatenate([latent_group1, latent_group2]) if args.folder_group2 else latent_group1

    tsne_combined = analyzer.reduce_dimensionality_tsne(
        combined_latent, perplexity=args.perplexity, random_state=args.seed
    )
    print(f"t-SNE projection computed: {tsne_combined.shape}")

    # Split back into groups
    split_idx = len(latent_group1)
    tsne_group1 = tsne_combined[:split_idx]
    projections = [(tsne_group1, ids_group1, "o", "group1")]

    tsne_group2 = None
    if args.folder_group2:
        tsne_group2 = tsne_combined[split_idx:]
        projections.append((tsne_group2, ids_group2, "^", "group2"))

    # Visualization
    title = "t-SNE"
    # Prepare image paths list
    image_paths_list = [paths_group1]
    if paths_group2:
        image_paths_list.append(paths_group2)

    save_visualization_and_legend(
        analyzer,
        projections,
        output_dir,
        title,
        args.color_by_patient,
        ids_group1,
        ids_group2,
        "tsne_projection.html",
        image_paths_list=image_paths_list,
    )

    # Compute statistics if two groups
    if args.folder_group2:
        group1_name = Path(args.folder_group1).name
        group2_name = Path(args.folder_group2).name
        compute_and_save_statistics(
            analyzer,
            tsne_group1,
            tsne_group2,
            latent_group1,
            latent_group2,
            ids_group1,
            ids_group2,
            group1_name,
            group2_name,
            output_dir,
        )

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
