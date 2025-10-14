"""
Image comparison and metrics computation for evaluating generated medical images.

This module provides comprehensive metrics for comparing ground truth and generated images:
- Reconstruction metrics: MSE, SSIM, PSNR
- Segmentation metrics: Dice coefficient, IoU
- Feature-based metrics: VGG16 cosine similarity, euclidean distance
- Morphological metrics: height and width measurements at different levels
"""

import os
import random
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from scipy.spatial.distance import cityblock, chebyshev, euclidean, minkowski
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from torchvision import models
from torchvision.models import VGG16_Weights
import tifffile as tiff


class ImageComparison:
    """
    Comprehensive image comparison tool for medical image analysis.

    Computes various metrics between ground truth and generated images,
    including reconstruction quality, segmentation accuracy, and morphological
    measurements.

    Args:
        apply_otsu_mask: Whether to apply Otsu thresholding for masking (default: False)
    """

    def __init__(self, apply_otsu_mask: bool = False):
        self.apply_otsu_mask = apply_otsu_mask
        self.weights = VGG16_Weights.IMAGENET1K_V1
        self.model = models.vgg16(weights=self.weights).features
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.worst_metrics: Dict[str, Tuple[float, Optional[np.ndarray]]] = {}

    def _to_2d(self, img: np.ndarray) -> np.ndarray:
        """
        Convert image to 2D array by squeezing dimensions.

        Args:
            img: Input image array

        Returns:
            2D float32 array

        Raises:
            ValueError: If image cannot be converted to 2D
        """
        img = np.asarray(img)
        img = np.squeeze(img)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        if img.ndim != 2:
            raise ValueError(f"Image must be 2D after squeeze, got shape {img.shape}")
        return img.astype(np.float32)

    def get_image_pair(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, None]:
        """
        Load ground truth and prediction image pair.

        Expects directory structure with 'edente' (GT) and 'edente_synth' (prediction)
        folders containing matching filenames.

        Args:
            image_path: Path to either GT or prediction image

        Returns:
            Tuple of (ground_truth, prediction, None)

        Raises:
            ValueError: If path doesn't contain 'edente' or 'edente_synth'
            FileNotFoundError: If paired image is missing
        """
        norm = os.path.normpath(image_path)
        parts = norm.split(os.sep)

        if "edente_synth" in parts:
            idx = parts.index("edente_synth")
            pred_path = norm
            parts[idx] = "edente"
            gt_path = os.path.join(*parts)
        elif "edente" in parts:
            idx = parts.index("edente")
            gt_path = norm
            parts[idx] = "edente_synth"
            pred_path = os.path.join(*parts)
        else:
            raise ValueError("get_image_pair expects path containing 'edente' or 'edente_synth'.")

        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Ground truth file missing: {gt_path}")
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(f"Prediction file missing: {pred_path}")

        gt = self._to_2d(tiff.imread(gt_path))
        pred = self._to_2d(tiff.imread(pred_path))
        return gt, pred, None

    def get_all_files_from_folders(
        self,
        folder_paths: List[str],
        file_selection_mode: str = "all",
        n: Optional[int] = None
    ) -> List[str]:
        """
        Get list of file paths from folders with optional sampling.

        Args:
            folder_paths: List of folder paths to search
            file_selection_mode: "all", "random_n", or "last_n"
            n: Number of files to select (for random_n or last_n modes)

        Returns:
            List of file paths
        """
        all_file_paths = []

        for folder_path in folder_paths:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    all_file_paths.append(os.path.join(root, file))

        if file_selection_mode == "random_n" and n is not None:
            all_file_paths = random.sample(all_file_paths, min(n, len(all_file_paths)))
        elif file_selection_mode == "last_n" and n is not None:
            all_file_paths = all_file_paths[-n:]

        return all_file_paths

    def generate_clean_mask(self, image: np.ndarray, kind: str = "gt") -> np.ndarray:
        """
        Generate binary mask from image.

        For GT: non-zero pixels become 1
        For prediction: pixels outside [-0.2, 0.2] become 1, with morphological cleaning

        Args:
            image: Input float32 image
            kind: "gt" or "pred"

        Returns:
            Binary mask as uint8 (0 or 255)
        """
        if kind == "gt":
            mask = (image != 0).astype(np.uint8)
        elif kind == "pred":
            mask = ((image > 0.2) | (image < -0.2)).astype(np.uint8)
            # Keep only largest connected component
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                biggest = max(contours, key=cv2.contourArea)
                cleaned = np.zeros_like(mask)
                cv2.drawContours(cleaned, [biggest], -1, color=1, thickness=-1)
                mask = cleaned

        return (mask * 255).astype(np.uint8)

    def dice_coefficient(
        self,
        prediction: np.ndarray,
        gt: np.ndarray,
        smooth: float = 1e-6
    ) -> float:
        """
        Compute Dice coefficient between prediction and ground truth masks.

        Args:
            prediction: Predicted image
            gt: Ground truth image
            smooth: Smoothing factor to avoid division by zero

        Returns:
            Dice coefficient in [0, 1]
        """
        pred_bin = self.generate_clean_mask(prediction, kind="pred")
        gt_bin = self.generate_clean_mask(gt, kind="gt")
        image1_flat = (pred_bin.flatten() / 255).astype(np.float32)
        image2_flat = (gt_bin.flatten() / 255).astype(np.float32)
        intersection = np.sum(image1_flat * image2_flat)
        union = np.sum(image1_flat) + np.sum(image2_flat)

        return (2. * intersection + smooth) / (union + smooth)

    def iou(self, prediction: np.ndarray, gt: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU) between prediction and GT masks.

        Args:
            prediction: Predicted image
            gt: Ground truth image

        Returns:
            IoU score in [0, 1]
        """
        pred_bin = self.generate_clean_mask(prediction, kind="pred")
        gt_bin = self.generate_clean_mask(gt, kind="gt")
        image1_flat = (pred_bin.flatten() / 255).astype(np.float32)
        image2_flat = (gt_bin.flatten() / 255).astype(np.float32)
        intersection = np.sum(image1_flat * image2_flat)
        union = np.sum((image1_flat + image2_flat) > 0)
        if union == 0:
            return 1.0

        return intersection / union

    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract VGG16 features from image for perceptual comparison.

        Note: Converts float32 to uint8 for VGG16 compatibility.

        Args:
            image: Input grayscale image

        Returns:
            Feature vector as torch.Tensor
        """
        image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)

        with torch.no_grad():
            features = self.model(image_tensor).view(-1)

        return features

    def align_images_by_bottom_20_center(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        verbosity: bool = False
    ) -> np.ndarray:
        """
        Align image2 to image1 based on bottom 20% center alignment.

        Args:
            image1: Reference image
            image2: Image to align
            verbosity: Print debug information

        Returns:
            Aligned image2

        Raises:
            ValueError: If images have different shapes or no white pixels found
        """
        if image1.shape != image2.shape:
            raise ValueError("Images do not have the same dimensions. Resize them to match.")

        def get_bottom_20_center(image):
            height = image.shape[0]
            bottom_20_height = int(height * 0.2)

            binary_image = self.generate_clean_mask(image, kind="gt")
            bottom_20_region = binary_image[-bottom_20_height:, :]

            white_pixel_indices = np.column_stack(np.where(bottom_20_region == 255))

            if len(white_pixel_indices) == 0:
                return None

            center_x = int(np.mean(white_pixel_indices[:, 1]))
            return center_x

        center1 = get_bottom_20_center(image1)
        center2 = get_bottom_20_center(image2)

        if center1 is None or center2 is None:
            raise ValueError("Could not find white pixels in bottom 20% of one or both images.")

        shift = center1 - center2

        if shift > 0:
            aligned_image2 = np.zeros_like(image2)
            aligned_image2[:, shift:] = image2[:, :-shift]
        elif shift < 0:
            aligned_image2 = np.zeros_like(image2)
            aligned_image2[:, :shift] = image2[:, -shift:]
        else:
            aligned_image2 = image2.copy()

        return aligned_image2

    def straighten_image(self, image: np.ndarray, verbosity: bool = False) -> np.ndarray:
        """
        Rotate image to straighten the main object using ellipse fitting.

        Args:
            image: Input image
            verbosity: Print debug information

        Returns:
            Rotated image

        Raises:
            ValueError: If no contours found or insufficient points for ellipse
        """
        binary_image = self.generate_clean_mask(image, kind="gt")
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]
                if angle > 90:
                    angle -= 180

                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_image = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated_image
            else:
                raise ValueError("Not enough points to fit an ellipse.")
        else:
            raise ValueError("No contours found in the image.")

    def compute_object_dimensions(
        self,
        binary_image: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """
        Compute height and widths of object at different vertical levels.

        Args:
            binary_image: Binary mask of object

        Returns:
            Tuple of (height, width_upper, width_middle, width_lower)

        Raises:
            ValueError: If no contours found
        """
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            raise ValueError("No contours found in the binary image.")

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        object_height = h

        # Compute widths at upper third, middle, and lower third
        upper_third_y = y + h // 4
        middle_y = y + h // 2
        lower_third_y = y + 3 * h // 4

        width_upper_third = np.sum(binary_image[upper_third_y, x:x+w] == 255)
        width_middle = np.sum(binary_image[middle_y, x:x+w] == 255)
        width_lower_third = np.sum(binary_image[lower_third_y, x:x+w] == 255)

        return object_height, width_upper_third, width_middle, width_lower_third

    def compute_height_width_metrics(
        self,
        gt_img: np.ndarray,
        gen_img: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute normalized height and width metrics with absolute differences.

        Args:
            gt_img: Ground truth image
            gen_img: Generated image

        Returns:
            Dictionary with normalized metrics and absolute differences
        """
        gt_binary = self.generate_clean_mask(gt_img, kind="gt")
        gen_binary = self.generate_clean_mask(gen_img, kind="pred")

        gt_height, gt_width_upper, gt_width_middle, gt_width_lower = self.compute_object_dimensions(gt_binary)
        gen_height, gen_width_upper, gen_width_middle, gen_width_lower = self.compute_object_dimensions(gen_binary)

        # Normalized metrics
        height_metric = min(gt_height, gen_height) / max(gt_height, gen_height)
        width_metric_middle = min(gt_width_middle, gen_width_middle) / max(gt_width_middle, gen_width_middle)
        width_metric_upper = min(gt_width_upper, gen_width_upper) / max(gt_width_upper, gen_width_upper)
        width_metric_lower = min(gt_width_lower, gen_width_lower) / max(gt_width_lower, gen_width_lower)

        # Absolute differences
        abs_height_diff = abs(gt_height - gen_height)
        abs_width_upper_diff = abs(gt_width_upper - gen_width_upper)
        abs_width_middle_diff = abs(gt_width_middle - gen_width_middle)
        abs_width_lower_diff = abs(gt_width_lower - gen_width_lower)

        return {
            'height_metric': height_metric,
            'width_metric_upper': width_metric_upper,
            'width_metric_middle': width_metric_middle,
            'width_metric_lower': width_metric_lower,
            'abs_height_diff': abs_height_diff,
            'abs_width_upper_diff': abs_width_upper_diff,
            'abs_width_middle_diff': abs_width_middle_diff,
            'abs_width_lower_diff': abs_width_lower_diff
        }

    def calculate_psnr(self, gt_img: np.ndarray, gen_img: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).

        Args:
            gt_img: Ground truth image
            gen_img: Generated image

        Returns:
            PSNR value in dB
        """
        mse = mean_squared_error(gt_img, gen_img)
        if mse == 0:
            return float('inf')
        PIXEL_MAX = max(np.max(gt_img), np.max(gen_img))
        psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
        return psnr

    def compare_images_and_display_metrics(
        self,
        gt_img: np.ndarray,
        gen_img: np.ndarray,
        original_image: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics between ground truth and generated image.

        Args:
            gt_img: Ground truth image
            gen_img: Generated image
            original_image: Optional original image for tracking worst cases

        Returns:
            Dictionary of all computed metrics

        Raises:
            ValueError: If images have different dimensions
        """
        if gen_img.shape != gt_img.shape:
            raise ValueError("Images do not have the same dimensions. Resize them to match.")

        mse_value = mean_squared_error(gen_img, gt_img)
        ssim_value, _ = ssim(gen_img, gt_img, data_range=gt_img.max() - gt_img.min(), full=True)
        psnr_value = self.calculate_psnr(gt_img, gen_img)

        dice_value = self.dice_coefficient(gen_img, gt_img)
        dice_loss = 1 - dice_value

        iou_value = self.iou(gen_img, gt_img)

        features_gen = self.extract_features(gen_img)
        features_gt = self.extract_features(gt_img)

        cosine_similarity = nn.CosineSimilarity(dim=0)
        similarity_score = cosine_similarity(features_gen, features_gt).item()
        euclidean_distance = round(euclidean(features_gen, features_gt), 2)
        manhattan_distance = cityblock(features_gen, features_gt)
        chebyshev_distance = chebyshev(features_gen, features_gt)
        minkowski_distance = minkowski(features_gen, features_gt, p=3)

        height_width_metrics = self.compute_height_width_metrics(gt_img, gen_img)

        metrics = {
            "MSE": mse_value,
            "SSIM": ssim_value,
            "PSNR": psnr_value,
            "Dice Coefficient": dice_value,
            "Dice Loss": dice_loss,
            "IoU": iou_value,
            "Cosine Similarity": similarity_score,
            "Euclidean Distance": euclidean_distance,
            "Height Metric": height_width_metrics['height_metric'],
            "Width Metric Upper": height_width_metrics['width_metric_upper'],
            "Width Metric Middle": height_width_metrics['width_metric_middle'],
            "Width Metric Lower": height_width_metrics['width_metric_lower'],
            "Absolute Height Difference": height_width_metrics['abs_height_diff'],
            "Absolute Width Upper Difference": height_width_metrics['abs_width_upper_diff'],
            "Absolute Width Middle Difference": height_width_metrics['abs_width_middle_diff'],
            "Absolute Width Lower Difference": height_width_metrics['abs_width_lower_diff']
        }

        # Track worst metrics
        for metric_name, value in metrics.items():
            if metric_name not in self.worst_metrics:
                self.worst_metrics[metric_name] = (value, original_image)
            else:
                # Higher is better for these metrics
                if metric_name in ["SSIM", "PSNR", "Dice Coefficient", "Cosine Similarity",
                                  "Height Metric", "Width Metric Upper", "Width Metric Middle",
                                  "Width Metric Lower", "IoU"]:
                    if value < self.worst_metrics[metric_name][0]:
                        self.worst_metrics[metric_name] = (value, original_image)
                else:  # Lower is better
                    if value > self.worst_metrics[metric_name][0]:
                        self.worst_metrics[metric_name] = (value, original_image)

        return metrics

    def count_outliers(
        self,
        all_metrics: List[Dict[str, float]],
        metrics_avg: Dict[str, float],
        metrics_ci95: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Dict[str, int]]:
        """
        Count outliers using multiple methods (CI, IQR, Z-score).

        Args:
            all_metrics: List of metric dictionaries for all images
            metrics_avg: Average metrics
            metrics_ci95: 95% confidence intervals

        Returns:
            Dictionary of outlier counts for each method
        """
        outlier_counts = {
            "outside_1_ci": {},
            "outside_2_ci": {},
            "outside_3_ci": {},
            "outside_iqr": {},
            "outside_z": {}
        }

        for key in metrics_avg.keys():
            data = [m[key] for m in all_metrics]
            mean = metrics_avg[key]
            std_dev = np.std(data)
            ci_lower, ci_upper = metrics_ci95[key]
            margin_of_error = (ci_upper - ci_lower) / 2

            # Z-Score outliers (|z| > 3)
            z_scores = [(x - mean) / std_dev for x in data]
            outliers_z = np.sum([1 for z in z_scores if abs(z) > 3])

            # IQR outliers
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_iqr = q1 - 1.5 * iqr
            upper_iqr = q3 + 1.5 * iqr
            outliers_iqr = np.sum([1 for x in data if x < lower_iqr or x > upper_iqr])

            # CI outliers
            outliers_1 = np.sum([1 for m in all_metrics if (m[key] < ci_lower or m[key] > ci_upper)])
            outliers_2 = np.sum([1 for m in all_metrics if (m[key] < mean - 2 * margin_of_error or m[key] > mean + 2 * margin_of_error)])
            outliers_3 = np.sum([1 for m in all_metrics if (m[key] < mean - 3 * margin_of_error or m[key] > mean + 3 * margin_of_error)])

            outlier_counts["outside_1_ci"][key] = outliers_1
            outlier_counts["outside_2_ci"][key] = outliers_2
            outlier_counts["outside_3_ci"][key] = outliers_3
            outlier_counts["outside_iqr"][key] = outliers_iqr
            outlier_counts["outside_z"][key] = outliers_z

        return outlier_counts

    def plot_metric_distributions_with_ci(
        self,
        all_metrics: List[Dict[str, float]],
        metrics_avg: Dict[str, float],
        metrics_ci95: Dict[str, Tuple[float, float]],
        save_path: Optional[str] = None
    ):
        """
        Plot distributions of all metrics with confidence intervals and outliers.

        Args:
            all_metrics: List of metric dictionaries
            metrics_avg: Average metrics
            metrics_ci95: 95% confidence intervals
            save_path: Path to save plot (optional)
        """
        exclude_metrics = ['Euclidean Distance', 'Manhattan Distance',
                          'Chebyshev Distance', 'Minkowski Distance']

        filtered_metrics = {key: value for key, value in metrics_avg.items()
                           if key not in exclude_metrics}

        num_metrics = len(filtered_metrics)
        num_cols = 3
        num_rows = (num_metrics + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))
        axes = axes.flatten()

        for idx, key in enumerate(filtered_metrics.keys()):
            data = [m[key] for m in all_metrics]
            mean = metrics_avg[key]
            ci_lower, ci_upper = metrics_ci95[key]
            std_dev = np.std(data)
            margin_of_error = (ci_upper - ci_lower) / 2

            # Z-score calculation
            z_scores = [(x - mean) / std_dev for x in data]
            z_outliers = [x for i, x in enumerate(data) if abs(z_scores[i]) > 3]

            # IQR calculation
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_iqr = q1 - 1.5 * iqr
            upper_iqr = q3 + 1.5 * iqr
            iqr_outliers = [x for x in data if x < lower_iqr or x > upper_iqr]

            ax = axes[idx]
            ax.hist(data, bins=20, color='lightblue', edgecolor='black', alpha=0.7)

            # Mean line
            ax.axvline(mean, color='red', linestyle='--', label='Mean', lw=2)

            # IQR boundaries
            ax.axvline(lower_iqr, color='orange', linestyle='-', label='IQR Lower', lw=2)
            ax.axvline(upper_iqr, color='orange', linestyle='-', label='IQR Upper', lw=2)

            # Z-score boundaries
            ax.axvline(mean - 3 * std_dev, color='red', linestyle='-', label='Z-Score -3', lw=2)
            ax.axvline(mean + 3 * std_dev, color='red', linestyle='-', label='Z-Score +3', lw=2)

            # Outliers
            ax.scatter(iqr_outliers, [0] * len(iqr_outliers), color='orange',
                      label='IQR Outliers', marker='o')
            ax.scatter(z_outliers, [0] * len(z_outliers), color='red',
                      label='Z-Score Outliers', marker='x')

            ax.set_title(f'Distribution of {key}', fontsize=12)
            ax.set_xlabel(f'{key} values', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(loc='upper left', fontsize=8)

        # Remove empty subplots
        for i in range(num_metrics, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)

        plt.show()

    def process_all_images(
        self,
        folder_paths: List[str],
        file_selection_mode: str = "all",
        n: Optional[int] = None,
        verbose: bool = False,
        save_csv: bool = False
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
        """
        Process all images and compute aggregate statistics.

        Args:
            folder_paths: List of folders containing images
            file_selection_mode: "all", "random_n", or "last_n"
            n: Number of samples for random_n or last_n
            verbose: Print detailed progress
            save_csv: Save results to CSV files

        Returns:
            Tuple of (average_metrics, confidence_intervals)
        """
        print("Processing images ...")
        all_metrics = []
        all_file_paths = self.get_all_files_from_folders(
            folder_paths,
            file_selection_mode=file_selection_mode,
            n=n
        )
        num_images = len(all_file_paths)

        # Counters for specific thresholds
        exams_with_97_height_metric = 0
        exams_with_95_height_metric = 0
        exams_with_90_height_metric = 0
        exams_with_97_width_metric = 0
        exams_with_90_width_metric = 0
        exams_with_95_width_metric = 0

        dimensions_df = pd.DataFrame(columns=[
            "Image Path", "GT Height", "GT Width Upper", "GT Width Middle", "GT Width Lower",
            "Gen Height", "Gen Width Upper", "Gen Width Middle", "Gen Width Lower"
        ])

        for path in all_file_paths:
            try:
                ground_truth, prediction, original_image = self.get_image_pair(path)

                # Clean prediction using mask
                prediction_mask = self.generate_clean_mask(prediction, kind="pred")
                prediction = prediction * (prediction_mask > 0)

                # Straighten and align
                rotated_gt = self.straighten_image(ground_truth, verbosity=verbose)
                rotated_gen = self.straighten_image(prediction, verbosity=verbose)
                realigned_gen = self.align_images_by_bottom_20_center(
                    rotated_gt, rotated_gen, verbosity=verbose
                )

                # Get dimensions
                binary_gt = self.generate_clean_mask(rotated_gt, kind="gt")
                binary_gen = self.generate_clean_mask(realigned_gen, kind="pred")

                gt_height, gt_width_upper, gt_width_middle, gt_width_lower = \
                    self.compute_object_dimensions(binary_gt)
                gen_height, gen_width_upper, gen_width_middle, gen_width_lower = \
                    self.compute_object_dimensions(binary_gen)

                row = [os.path.basename(path), gt_height, gt_width_upper,
                      gt_width_middle, gt_width_lower, gen_height, gen_width_upper,
                      gen_width_middle, gen_width_lower]
                dimensions_df.loc[len(dimensions_df)] = row

                # Compute metrics
                metrics = self.compare_images_and_display_metrics(
                    rotated_gt, realigned_gen, original_image=None
                )
                all_metrics.append(metrics)

                # Count threshold passes
                if metrics["Height Metric"] > 0.95:
                    exams_with_95_height_metric += 1
                if metrics["Width Metric Middle"] > 0.95:
                    exams_with_95_width_metric += 1
                if metrics["Height Metric"] > 0.97:
                    exams_with_97_height_metric += 1
                if metrics["Width Metric Middle"] > 0.97:
                    exams_with_97_width_metric += 1
                if metrics["Height Metric"] > 0.90:
                    exams_with_90_height_metric += 1
                if metrics["Width Metric Middle"] > 0.90:
                    exams_with_90_width_metric += 1

                if verbose:
                    print(f"Metrics for {path}:")
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")
                    print()

            except Exception as e:
                num_images -= 1
                if verbose:
                    print(f"Failed to process image {path}: {traceback.format_exc()}")
                continue

        # Compute statistics
        metrics_avg = {key: np.mean([m[key] for m in all_metrics])
                      for key in all_metrics[0].keys()}
        metrics_std = {key: np.std([m[key] for m in all_metrics])
                      for key in all_metrics[0].keys()}
        metrics_ci95 = {
            key: (metrics_avg[key] - 1.96 * (metrics_std[key] / np.sqrt(num_images)),
                  metrics_avg[key] + 1.96 * (metrics_std[key] / np.sqrt(num_images)))
            for key in metrics_avg.keys()
        }
        outlier_counts = self.count_outliers(all_metrics, metrics_avg, metrics_ci95)

        height_diff = abs(dimensions_df["Gen Height"] - dimensions_df["GT Height"])
        width_diff_middle = abs(dimensions_df["Gen Width Middle"] - dimensions_df["GT Width Middle"])
        width_diff_lower = abs(dimensions_df["Gen Width Lower"] - dimensions_df["GT Width Lower"])

        if save_csv:
            filename = f"{folder_paths[0]}/_metrics.csv"

            output_data = []
            for key in metrics_avg.keys():
                output_data.append({
                    'Metric': key,
                    'Average': round(metrics_avg[key], 3),
                    'Worst Value': round(self.worst_metrics[key][0], 3),
                    'Confidence Interval Lower (95%)': round(metrics_ci95[key][0], 3),
                    'Confidence Interval Upper (95%)': round(metrics_ci95[key][1], 3),
                    'Number of Images Processed': num_images,
                    'Outside 1 CI': outlier_counts["outside_1_ci"][key],
                    'Outside 2 CI': outlier_counts["outside_2_ci"][key],
                    'Outside 3 CI': outlier_counts["outside_3_ci"][key],
                    'IQR Outliers': int(outlier_counts["outside_iqr"][key]),
                    'Z-Score Outliers': int(outlier_counts["outside_z"][key])
                })

            # Add threshold statistics
            threshold_stats = [
                ('Exams with Height Metric > 0.95', exams_with_95_height_metric),
                ('Exams with Width Metric > 0.95', exams_with_95_width_metric),
                ('Exams with Height Metric > 0.97', exams_with_97_height_metric),
                ('Exams with Width Metric > 0.97', exams_with_97_width_metric),
                ('Exams with Height Metric > 0.90', exams_with_90_height_metric),
                ('Exams with Width Metric > 0.90', exams_with_90_width_metric),
                ('Exams with Absolute Height Difference < 5', (height_diff < 5).sum()),
                ('Exams with Absolute Middle Width Difference < 5', (width_diff_middle < 5).sum()),
                ('Exams with Absolute Lower Width Difference < 5', (width_diff_lower < 5).sum()),
                ('Exams with Absolute Height Difference < 10', (height_diff < 10).sum()),
                ('Exams with Absolute Middle Width Difference < 10', (width_diff_middle < 10).sum()),
                ('Exams with Absolute Lower Width Difference < 10', (width_diff_lower < 10).sum()),
            ]

            for metric_name, count in threshold_stats:
                output_data.append({
                    'Metric': metric_name,
                    'Count': count,
                    'Percentage': round((count / num_images) * 100, 2)
                })

            df = pd.DataFrame(output_data)
            df.to_csv(filename, index=False, sep=';')
            if verbose:
                print(f"Metrics and confidence intervals saved to {filename}")

            dimensions_df.to_csv(f"{folder_paths[0]}/_dimensions.csv", index=False, sep=';')
            if verbose:
                print(f"Dimensions saved to {folder_paths[0]}/_dimensions.csv")

        if verbose:
            print("Final Average Metrics:")
            for key, value in metrics_avg.items():
                print(f"{key}: {value}")

            print("\n95% Confidence Intervals:")
            for key, value in metrics_ci95.items():
                print(f"{key}: {value}")

        plot_filename = f"{folder_paths[0]}/_metrics_distribution.png"
        self.plot_metric_distributions_with_ci(
            all_metrics, metrics_avg, metrics_ci95, save_path=plot_filename
        )

        return metrics_avg, metrics_ci95