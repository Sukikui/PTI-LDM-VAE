#!/usr/bin/env python
"""
t-SNE Analysis of VAE Latent Space

This script performs t-SNE dimensionality reduction on the latent space
of a trained VAE model and visualizes the results for comparative analysis
between different image groups.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from monai.config import print_config
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Resize, EnsureType

from pti_ldm_vae.analysis import LatentSpaceAnalyzer, load_image_paths
from pti_ldm_vae.data.transforms import LocalNormalizeByMask
from pti_ldm_vae.models import VAEModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="t-SNE Analysis of VAE Latent Space")
    parser.add_argument("--vae-weights", type=str, required=True,
                        help="Path to VAE weights file")
    parser.add_argument("--config-file", type=str, required=True,
                        help="Path to model config file")
    parser.add_argument("--folder-group1", type=str, required=True,
                        help="Path to first image group folder (e.g., edentulous)")
    parser.add_argument("--folder-group2", type=str, default=None,
                        help="Path to second image group folder (e.g., dental)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--max-images", type=int, default=1000,
                        help="Maximum number of images per group")
    parser.add_argument("--patch-size", type=int, nargs=2, default=[256, 256],
                        help="Image patch size (H W)")
    parser.add_argument("--color-by-exam", action="store_true",
                        help="Color points by exam ID instead of group")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="t-SNE perplexity parameter")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_config()
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")

    # Load config and model
    with open(args.config_file) as f:
        config_dict = json.load(f)

    vae = VAEModel.from_config(config_dict["autoencoder_def"]).to(device)
    vae.load_state_dict(torch.load(args.vae_weights, map_location=device))
    vae.eval()
    print(f"Loaded VAE from {args.vae_weights}")

    # Setup transforms
    patch_size = tuple(args.patch_size)
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(patch_size),
        LocalNormalizeByMask(),
        EnsureType(dtype=torch.float32)
    ])

    # Create analyzer
    analyzer = LatentSpaceAnalyzer(vae, device, transforms)

    # Load images for group 1
    print(f"\nLoading group 1 images from {args.folder_group1}...")
    paths_group1 = load_image_paths(args.folder_group1, args.max_images)
    print(f"Found {len(paths_group1)} images")

    # Encode group 1
    print("Encoding group 1...")
    latent_group1, ids_group1 = analyzer.encode_images(paths_group1)
    print(f"Encoded {len(latent_group1)} images to latent space")

    # Prepare combined data for t-SNE
    all_latent = [latent_group1]
    all_ids = [ids_group1]
    all_markers = [['o'] * len(latent_group1)]
    all_names = [["group1"] * len(latent_group1)]

    # Load and encode group 2 if provided
    if args.folder_group2:
        print(f"\nLoading group 2 images from {args.folder_group2}...")
        paths_group2 = load_image_paths(args.folder_group2, args.max_images)
        print(f"Found {len(paths_group2)} images")

        print("Encoding group 2...")
        latent_group2, ids_group2 = analyzer.encode_images(paths_group2)
        print(f"Encoded {len(latent_group2)} images to latent space")

        all_latent.append(latent_group2)
        all_ids.append(ids_group2)
        all_markers.append(['^'] * len(latent_group2))
        all_names.append(["group2"] * len(latent_group2))

    # Combine all data
    combined_latent = np.concatenate(all_latent)
    combined_ids = sum(all_ids, [])
    combined_markers = sum(all_markers, [])
    combined_names = sum(all_names, [])

    # t-SNE reduction
    print("\n⏳ Computing t-SNE projection...")
    print("(This may take a few minutes...)")
    tsne_combined = analyzer.reduce_dimensionality_tsne(
        combined_latent,
        perplexity=args.perplexity,
        random_state=args.seed
    )
    print(f"t-SNE projection computed: {tsne_combined.shape}")

    # Split back into groups
    split_idx = len(latent_group1)
    tsne_group1 = tsne_combined[:split_idx]

    projections = [(tsne_group1, ids_group1, 'o', "group1")]

    if args.folder_group2:
        tsne_group2 = tsne_combined[split_idx:]
        projections.append((tsne_group2, ids_group2, '^', "group2"))

    # Visualization
    print("\n📊 Creating visualizations...")

    # Determine title
    group1_name = Path(args.folder_group1).name
    if args.folder_group2:
        group2_name = Path(args.folder_group2).name
        title = f"t-SNE: {group1_name} (o) vs {group2_name} (^)"
    else:
        title = f"t-SNE: {group1_name}"

    if args.color_by_exam:
        title += " — colored by exam"
    else:
        title += " — colored by group"

    # Plot
    save_path = output_dir / "tsne_projection.png"
    analyzer.plot_projection_2d(
        projections=projections,
        output_path=str(save_path),
        title=title,
        color_by_exam=args.color_by_exam,
        show_labels=args.color_by_exam
    )
    print(f"✅ Plot saved to {save_path}")

    # Save color legend if coloring by exam
    if args.color_by_exam:
        all_ids = ids_group1
        if args.folder_group2:
            all_ids = all_ids + ids_group2
        exam_to_id, exam_to_color = analyzer.create_exam_colormap(all_ids)
        legend_path = output_dir / "color_legend.txt"
        analyzer.save_color_legend(exam_to_id, exam_to_color, legend_path)
        print(f"✅ Color legend saved to {legend_path}")

    # Compute statistics if two groups
    if args.folder_group2:
        print("\n📈 Computing group statistics...")
        projection_data = [
            (tsne_group1, ids_group1, group1_name),
            (tsne_group2, ids_group2, group2_name)
        ]
        latent_data = [
            (latent_group1, ids_group1, group1_name),
            (latent_group2, ids_group2, group2_name)
        ]
        analyzer.compute_group_statistics(projection_data, latent_data, output_dir)
        print(f"✅ Statistics saved to {output_dir}/distance_metrics.txt")
        print(f"✅ Sorted exams saved to {output_dir}/exams_sorted_by_distance.txt")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()