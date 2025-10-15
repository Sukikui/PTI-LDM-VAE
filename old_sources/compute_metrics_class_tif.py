import os
import random
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.spatial.distance import chebyshev, cityblock, euclidean, minkowski
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from torchvision import models
from torchvision.models import VGG16_Weights


class ImageComparison:
    def __init__(self, apply_otsu_mask=False):
        self.apply_otsu_mask = apply_otsu_mask
        self.weights = VGG16_Weights.IMAGENET1K_V1
        self.model = models.vgg16(weights=self.weights).features
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.worst_metrics = {}  # To track worst metrics

    def _to_2d(self, img):
        img = np.asarray(img)
        # squeeze tous les axes de taille 1
        img = np.squeeze(img)
        # si c'est encore 3D de type (H, W, 1), on prend le canal
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        if img.ndim != 2:
            raise ValueError(f"Image must be 2D after squeeze, got shape {img.shape}")
        return img.astype(np.float32)

    def get_image_pair(self, image_path):
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
            raise ValueError("get_image_pair attend un chemin dans 'edente' ou 'edente_synth'.")

        if not os.path.isfile(gt_path):
            raise FileNotFoundError(f"Fichier GT manquant: {gt_path}")
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(f"Fichier prédiction manquant: {pred_path}")

        gt = self._to_2d(tiff.imread(gt_path))
        pred = self._to_2d(tiff.imread(pred_path))
        return gt, pred, None

    def get_all_files_from_folders(self, folder_paths, file_selection_mode="all", n=None):
        all_file_paths = []

        for folder_path in folder_paths:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    relative_path = os.path.relpath(os.path.join(root, file), os.path.dirname(folder_path))
                    all_file_paths.append(os.path.join(root, file))

        # Select files based on the chosen mode
        if file_selection_mode == "random_n" and n is not None:
            all_file_paths = random.sample(all_file_paths, min(n, len(all_file_paths)))
        elif file_selection_mode == "last_n" and n is not None:
            all_file_paths = all_file_paths[-n:]  # Take the last n files

        return all_file_paths

    def generate_clean_mask(self, image, kind="gt"):
        """Convertit une image float32 en masque binaire :
        - pour la GT : tout ce qui est différent de 0 devient 1
        - pour la prédiction : valeurs entre -0.2 et 0.2 considérées comme fond
        """
        if kind == "gt":
            mask = (image != 0).astype(np.uint8)
        elif kind == "pred":
            mask = ((image > 0.2) | (image < -0.2)).astype(np.uint8)
            # nettoyer le bruit : on garde la plus grosse région blanche uniquement
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                biggest = max(contours, key=cv2.contourArea)
                cleaned = np.zeros_like(mask)
                cv2.drawContours(cleaned, [biggest], -1, color=1, thickness=-1)
                mask = cleaned

        return (mask * 255).astype(np.uint8)

    def dice_coefficient(self, prediction, gt, smooth=1e-6):
        pred_bin = self.generate_clean_mask(prediction, kind="pred")
        gt_bin = self.generate_clean_mask(gt, kind="gt")
        image1_flat = (pred_bin.flatten() / 255).astype(np.float32)
        image2_flat = (gt_bin.flatten() / 255).astype(np.float32)
        intersection = np.sum(image1_flat * image2_flat)
        union = np.sum(image1_flat) + np.sum(image2_flat)

        return (2.0 * intersection + smooth) / (union + smooth)

    def iou(self, prediction, gt):
        pred_bin = self.generate_clean_mask(prediction, kind="pred")
        gt_bin = self.generate_clean_mask(gt, kind="gt")
        image1_flat = (pred_bin.flatten() / 255).astype(np.float32)
        image2_flat = (gt_bin.flatten() / 255).astype(np.float32)
        intersection = np.sum(image1_flat * image2_flat)
        union = np.sum((image1_flat + image2_flat) > 0)
        if union == 0:
            return 1.0

        return intersection / union

    def extract_features(self, image):
        # ⚠️ Conversion temporaire en 8-bit pour VGG16 uniquement
        # Ne reflète pas la richesse réelle des données float32
        image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)  # Shape: [1, 3, 224, 224]

        with torch.no_grad():
            features = self.model(image_tensor).view(-1)

        return features

    def align_images_by_bottom_20_center(self, image1, image2, verbosity=False):
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
            raise ValueError("Could not find white pixels in the bottom 20% of one or both images.")

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

    def straighten_image(self, image, verbosity=False):
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
                rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return rotated_image
            raise ValueError("Not enough points to fit an ellipse.")
        raise ValueError("No contours found in the image.")

    def compute_object_dimensions(self, binary_image):
        """Compute the height of the object and the width at the upper, middle, and lower thirds."""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imwrite(f"debug_thresholded.png", binary_image)

        if len(contours) == 0:
            raise ValueError("No contours found in the binary image.")

        contour = max(contours, key=cv2.contourArea)  # Use the largest contour
        x, y, w, h = cv2.boundingRect(contour)

        object_height = h

        # Compute widths at the upper third, middle, and lower third
        upper_third_y = y + h // 4
        middle_y = y + h // 2
        lower_third_y = y + 3 * h // 4

        width_upper_third = np.sum(binary_image[upper_third_y, x : x + w] == 255)
        width_middle = np.sum(binary_image[middle_y, x : x + w] == 255)
        width_lower_third = np.sum(binary_image[lower_third_y, x : x + w] == 255)

        return object_height, width_upper_third, width_middle, width_lower_third

    def compute_height_width_metrics(self, gt_img, gen_img):
        """Compute the height and width metrics based on Otsu-binarized images and return absolute differences."""
        gt_binary = self.generate_clean_mask(gt_img, kind="gt")
        gen_binary = self.generate_clean_mask(gen_img, kind="pred")

        # Compute object dimensions for ground truth and generated images
        gt_height, gt_width_upper, gt_width_middle, gt_width_lower = self.compute_object_dimensions(gt_binary)
        gen_height, gen_width_upper, gen_width_middle, gen_width_lower = self.compute_object_dimensions(gen_binary)

        # Calculate the metrics (normalized values)
        height_metric = min(gt_height, gen_height) / max(gt_height, gen_height)
        width_metric_middle = min(gt_width_middle, gen_width_middle) / max(gt_width_middle, gen_width_middle)
        width_metric_upper = min(gt_width_upper, gen_width_upper) / max(gt_width_upper, gen_width_upper)
        width_metric_lower = min(gt_width_lower, gen_width_lower) / max(gt_width_lower, gen_width_lower)

        # Calculate absolute differences between ground truth and generated image dimensions
        abs_height_diff = abs(gt_height - gen_height)
        abs_width_upper_diff = abs(gt_width_upper - gen_width_upper)
        abs_width_middle_diff = abs(gt_width_middle - gen_width_middle)
        abs_width_lower_diff = abs(gt_width_lower - gen_width_lower)

        return {
            "height_metric": height_metric,
            "width_metric_upper": width_metric_upper,
            "width_metric_middle": width_metric_middle,
            "width_metric_lower": width_metric_lower,
            "abs_height_diff": abs_height_diff,
            "abs_width_upper_diff": abs_width_upper_diff,
            "abs_width_middle_diff": abs_width_middle_diff,
            "abs_width_lower_diff": abs_width_lower_diff,
        }

    def calculate_psnr(self, gt_img, gen_img):
        mse = mean_squared_error(gt_img, gen_img)
        if mse == 0:
            return float("inf")
        PIXEL_MAX = max(np.max(gt_img), np.max(gen_img))  # Ou une constante comme 6.0
        psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
        return psnr

    # def compare_images_and_display_metrics(self, gen_img, gt_img, original_image=None):
    #     print("[DEBUG] compare_images_and_display_metrics CALLED")
    #     print("→ gen_img dtype:", gen_img.dtype, "shape:", gen_img.shape)
    #     print("→ gt_img dtype:", gt_img.dtype, "shape:", gt_img.shape)

    #     if gen_img.shape != gt_img.shape:
    #         raise ValueError("Images do not have the same dimensions. Resize them to match.")

    #     try:
    #         mse_value = mean_squared_error(gen_img, gt_img)
    #     except Exception as e:
    #         print("[ERROR] MSE failed:", e)
    #         return {}

    #     try:
    #         ssim_value, _ = ssim_value, _ = ssim(gen_img, gt_img, data_range=gt_img.max() - gt_img.min(), full=True)

    #     except Exception as e:
    #         print("[ERROR] SSIM failed:", e)
    #         return {}

    #     try:
    #         psnr_value = self.calculate_psnr(gt_img, gen_img)
    #     except Exception as e:
    #         print("[ERROR] PSNR failed:", e)
    #         return {}

    #     try:
    #         dice_value = self.dice_coefficient(gen_img, gt_img)
    #         dice_loss = 1 - dice_value
    #     except Exception as e:
    #         print("[ERROR] Dice failed:", e)
    #         return {}

    #     try:
    #         iou_value = self.iou(gen_img, gt_img)
    #     except Exception as e:
    #         print("[ERROR] IoU failed:", e)
    #         return {}

    #     try:
    #         features_gen = self.extract_features(gen_img)
    #         features_gt = self.extract_features(gt_img)
    #     except Exception as e:
    #         print("[ERROR] Feature extraction failed:", e)
    #         return {}

    #     try:
    #         cosine_similarity = nn.CosineSimilarity(dim=0)
    #         similarity_score = cosine_similarity(features_gen, features_gt).item()
    #         euclidean_distance = round(euclidean(features_gen, features_gt), 2)
    #         # distances facultatives :
    #         # manhattan_distance = cityblock(features_gen, features_gt)
    #         # chebyshev_distance = chebyshev(features_gen, features_gt)
    #         # minkowski_distance = minkowski(features_gen, features_gt, p=3)
    #     except Exception as e:
    #         print("[ERROR] Similarity metric failed:", e)
    #         return {}

    #     try:
    #         height_width_metrics = self.compute_height_width_metrics(gt_img, gen_img)
    #     except Exception as e:
    #         print("[ERROR] Height/width metric failed:", e)
    #         return {}

    #     metrics = {
    #         "MSE": mse_value,
    #         "SSIM": ssim_value,
    #         "PSNR": psnr_value,
    #         "Dice Coefficient": dice_value,
    #         "Dice Loss": dice_loss,
    #         "IoU": iou_value,
    #         "Cosine Similarity": similarity_score,
    #         "Euclidean Distance": euclidean_distance,
    #         "Height Metric": height_width_metrics['height_metric'],
    #         "Width Metric Upper": height_width_metrics['width_metric_upper'],
    #         "Width Metric Middle": height_width_metrics['width_metric_middle'],
    #         "Width Metric Lower": height_width_metrics['width_metric_lower'],
    #         "Absolute Height Difference": height_width_metrics['abs_height_diff'],
    #         "Absolute Width Upper Difference": height_width_metrics['abs_width_upper_diff'],
    #         "Absolute Width Middle Difference": height_width_metrics['abs_width_middle_diff'],
    #         "Absolute Width Lower Difference": height_width_metrics['abs_width_lower_diff']
    #     }

    #     for metric_name, value in metrics.items():
    #         if metric_name not in self.worst_metrics:
    #             self.worst_metrics[metric_name] = (value, original_image)
    #         else:
    #             if metric_name in ["SSIM", "PSNR", "Dice Coefficient", "Cosine Similarity", "Height Metric", "Width Metric Upper", "Width Metric Middle", "Width Metric Lower", "IoU"]:
    #                 if value < self.worst_metrics[metric_name][0]:
    #                     self.worst_metrics[metric_name] = (value, original_image)
    #             else:
    #                 if value > self.worst_metrics[metric_name][0]:
    #                     self.worst_metrics[metric_name] = (value, original_image)

    #     print("✅ METRICS GENERATED:")
    #     for key, value in metrics.items():
    #         print(f"{key}: {value}")

    #     return metrics

    def compare_images_and_display_metrics(self, gt_img, gen_img, original_image=None):
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

        # Compute height and width metrics, including absolute differences
        height_width_metrics = self.compute_height_width_metrics(gt_img, gen_img)

        # Add the metrics and absolute differences to the final output
        metrics = {
            "MSE": mse_value,
            "SSIM": ssim_value,
            "PSNR": psnr_value,
            "Dice Coefficient": dice_value,
            "Dice Loss": dice_loss,
            "IoU": iou_value,
            "Cosine Similarity": similarity_score,
            "Euclidean Distance": euclidean_distance,
            # "Manhattan Distance": manhattan_distance,
            # "Chebyshev Distance": chebyshev_distance,
            # "Minkowski Distance": minkowski_distance,
            # Add the height and width metrics
            "Height Metric": height_width_metrics["height_metric"],
            "Width Metric Upper": height_width_metrics["width_metric_upper"],
            "Width Metric Middle": height_width_metrics["width_metric_middle"],
            "Width Metric Lower": height_width_metrics["width_metric_lower"],
            # Add the absolute differences
            "Absolute Height Difference": height_width_metrics["abs_height_diff"],
            "Absolute Width Upper Difference": height_width_metrics["abs_width_upper_diff"],
            "Absolute Width Middle Difference": height_width_metrics["abs_width_middle_diff"],
            "Absolute Width Lower Difference": height_width_metrics["abs_width_lower_diff"],
        }

        # Update worst metrics and store images if needed
        for metric_name, value in metrics.items():
            if metric_name not in self.worst_metrics:
                self.worst_metrics[metric_name] = (value, original_image)
            else:
                if metric_name in [
                    "SSIM",
                    "PSNR",
                    "Dice Coefficient",
                    "Cosine Similarity",
                    "Height Metric",
                    "Width Metric Upper",
                    "Width Metric Middle",
                    "Width Metric Lower",
                    "IoU",
                ]:
                    if value < self.worst_metrics[metric_name][0]:
                        self.worst_metrics[metric_name] = (value, original_image)
                else:
                    if value > self.worst_metrics[metric_name][0]:
                        self.worst_metrics[metric_name] = (value, original_image)

        return metrics

    def count_outliers(self, all_metrics, metrics_avg, metrics_ci95):
        """Count how many values are outside the CI, 2xCI, 3xCI, IQR, and 3x Z-score based on margin of error."""
        outlier_counts = {
            "outside_1_ci": {},
            "outside_2_ci": {},
            "outside_3_ci": {},
            "outside_iqr": {},
            "outside_z": {},
        }

        for key in metrics_avg.keys():
            data = [m[key] for m in all_metrics]
            mean = metrics_avg[key]
            std_dev = np.std(data)
            ci_lower, ci_upper = metrics_ci95[key]
            margin_of_error = (ci_upper - ci_lower) / 2  # This is the margin of error

            # Z-Score
            z_scores = [(x - mean) / std_dev for x in data]
            outliers_z = np.sum([1 for z in z_scores if abs(z) > 3])

            # IQR
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_iqr = q1 - 1.5 * iqr
            upper_iqr = q3 + 1.5 * iqr
            outliers_iqr = np.sum([1 for x in data if x < lower_iqr or x > upper_iqr])

            # CI Outliers
            outliers_1 = np.sum([1 for m in all_metrics if (m[key] < ci_lower or m[key] > ci_upper)])
            outliers_2 = np.sum(
                [1 for m in all_metrics if (m[key] < mean - 2 * margin_of_error or m[key] > mean + 2 * margin_of_error)]
            )
            outliers_3 = np.sum(
                [1 for m in all_metrics if (m[key] < mean - 3 * margin_of_error or m[key] > mean + 3 * margin_of_error)]
            )

            outlier_counts["outside_1_ci"][key] = outliers_1
            outlier_counts["outside_2_ci"][key] = outliers_2
            outlier_counts["outside_3_ci"][key] = outliers_3
            outlier_counts["outside_iqr"][key] = outliers_iqr
            outlier_counts["outside_z"][key] = outliers_z

        return outlier_counts

    def plot_metric_distributions_with_ci(self, all_metrics, metrics_avg, metrics_ci95, save_path=None):
        exclude_metrics = ["Euclidean Distance", "Manhattan Distance", "Chebyshev Distance", "Minkowski Distance"]

        # Filter out excluded metrics
        filtered_metrics = {key: value for key, value in metrics_avg.items() if key not in exclude_metrics}

        num_metrics = len(filtered_metrics)
        num_cols = 3  # Number of columns for subplots
        num_rows = (num_metrics + num_cols - 1) // num_cols  # Calculate number of rows

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

            # CI calculation
            """lower_1xCI = ci_lower
            upper_1xCI = ci_upper
            lower_2xCI = mean - 2 * margin_of_error
            upper_2xCI = mean + 2 * margin_of_error
            lower_3xCI = mean - 3 * margin_of_error
            upper_3xCI = mean + 3 * margin_of_error"""

            ax = axes[idx]
            ax.hist(data, bins=20, color="lightblue", edgecolor="black", alpha=0.7)

            # Add vertical lines for the mean and confidence intervals
            ax.axvline(mean, color="red", linestyle="--", label="Mean", lw=2)
            """ax.axvline(lower_1xCI, color='green', linestyle='--', label='1xCI Lower', lw=2) ax.axvline(upper_1xCI,
            color='green', linestyle='--', label='1xCI Upper', lw=2) ax.axvline(lower_2xCI, color='orange',
            linestyle='--', label='2xCI Lower', lw=2) ax.axvline(upper_2xCI, color='orange', linestyle='--', label='2xCI
            Upper', lw=2) ax.axvline(lower_3xCI, color='purple', linestyle='--', label='3xCI Lower', lw=2)
            ax.axvline(upper_3xCI, color='purple', linestyle='--', label='3xCI Upper', lw=2)"""

            # Add full lines for IQR boundaries
            ax.axvline(lower_iqr, color="orange", linestyle="-", label="IQR Lower", lw=2)
            ax.axvline(upper_iqr, color="orange", linestyle="-", label="IQR Upper", lw=2)

            # Add full lines for Z-score boundaries (+/-3 standard deviations)
            ax.axvline(mean - 3 * std_dev, color="red", linestyle="-", label="Z-Score -3", lw=2)
            ax.axvline(mean + 3 * std_dev, color="red", linestyle="-", label="Z-Score +3", lw=2)

            # Highlight the outliers (IQR and Z-score)
            ax.scatter(iqr_outliers, [0] * len(iqr_outliers), color="orange", label="IQR Outliers", marker="o")
            ax.scatter(z_outliers, [0] * len(z_outliers), color="red", label="Z-Score Outliers", marker="x")

            ax.set_title(f"Distribution of {key}", fontsize=12)
            ax.set_xlabel(f"{key} values", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend(loc="upper left", fontsize=8)

        # Remove any empty subplots
        for i in range(num_metrics, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            # print(f"Figure saved to {save_path}")

        plt.show()

    def process_all_images(self, folder_paths, file_selection_mode="all", n=None, verbose=False, save_csv=False):
        print("Processing images ...")
        all_metrics = []
        all_file_paths = self.get_all_files_from_folders(folder_paths, file_selection_mode=file_selection_mode, n=n)
        num_images = len(all_file_paths)
        exams_with_97_height_metric = 0
        exams_with_95_height_metric = 0
        exams_with_90_height_metric = 0
        exams_with_97_width_metric = 0
        exams_with_90_width_metric = 0
        exams_with_95_width_metric = 0
        dimensions_df = pd.DataFrame(
            columns=[
                "Image Path",
                "GT Height",
                "GT Width Upper",
                "GT Width Middle",
                "GT Width Lower",
                "Gen Height",
                "Gen Width Upper",
                "Gen Width Middle",
                "Gen Width Lower",
            ]
        )

        for path in all_file_paths:
            # print(f"Processing image: {path}")

            try:
                ground_truth, prediction, original_image = self.get_image_pair(path)
                # cv2.imwrite("debug/pred_before.png", cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                prediction_mask = self.generate_clean_mask(prediction, kind="pred")
                prediction = prediction * (prediction_mask > 0)
                # cv2.imwrite("debug/pred_after.png", cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

                # os.makedirs("debug", exist_ok=True)

                # cv2.imwrite("debug/ground_truth.png", cv2.normalize(ground_truth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                # cv2.imwrite("debug/prediction.png", cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                # cv2.imwrite("debug/original_image.png", cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

                rotated_gt = self.straighten_image(ground_truth, verbosity=verbose)
                rotated_gen = self.straighten_image(prediction, verbosity=verbose)

                # cv2.imwrite("debug/rotated_gt.png", cv2.normalize(rotated_gt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                # cv2.imwrite("debug/rotated_gen.png", cv2.normalize(rotated_gen, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

                realigned_gen = self.align_images_by_bottom_20_center(rotated_gt, rotated_gen, verbosity=verbose)

                # cv2.imwrite("debug/realigned_gen.png", cv2.normalize(realigned_gen, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

                binary_gt = self.generate_clean_mask(rotated_gt, kind="gt")
                binary_gen = self.generate_clean_mask(realigned_gen, kind="pred")

                # cv2.imwrite("debug/binary_gt.png", cv2.normalize(binary_gt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                # cv2.imwrite("debug/binary_gen.png", cv2.normalize(binary_gen, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

                gt_height, gt_width_upper, gt_width_middle, gt_width_lower = self.compute_object_dimensions(binary_gt)
                gen_height, gen_width_upper, gen_width_middle, gen_width_lower = self.compute_object_dimensions(
                    binary_gen
                )

                # print("Dimensions GT :")
                # print(gt_height, gt_width_upper, gt_width_middle, gt_width_lower)
                # print("Dimensions Pred :")
                # print(gen_height, gen_width_upper, gen_width_middle, gen_width_lower)

                row = [
                    os.path.basename(path),
                    gt_height,
                    gt_width_upper,
                    gt_width_middle,
                    gt_width_lower,
                    gen_height,
                    gen_width_upper,
                    gen_width_middle,
                    gen_width_lower,
                ]
                dimensions_df.loc[len(dimensions_df)] = row
                metrics = self.compare_images_and_display_metrics(rotated_gt, realigned_gen, original_image=None)
                all_metrics.append(metrics)
                # Check if the height or width metric is greater than 0.95
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

            except Exception:
                num_images -= 1
                if verbose:
                    print(f"Failed to process image {path}: {traceback.format_exc()}")
                continue

        metrics_avg = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
        metrics_std = {key: np.std([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
        metrics_ci95 = {
            key: (
                metrics_avg[key] - 1.96 * (metrics_std[key] / np.sqrt(num_images)),
                metrics_avg[key] + 1.96 * (metrics_std[key] / np.sqrt(num_images)),
            )
            for key in metrics_avg
        }
        outlier_counts = self.count_outliers(all_metrics, metrics_avg, metrics_ci95)

        height_diff = abs(dimensions_df["Gen Height"] - dimensions_df["GT Height"])
        width_diff_middle = abs(dimensions_df["Gen Width Middle"] - dimensions_df["GT Width Middle"])
        width_diff_lower = abs(dimensions_df["Gen Width Lower"] - dimensions_df["GT Width Lower"])

        if save_csv:
            filename = f"{folder_paths[0]}/_metrics.csv"

            # Combine metrics and confidence intervals for output
            output_data = []
            for key in metrics_avg:
                output_data.append(
                    {
                        "Metric": key,
                        "Average": round(metrics_avg[key], 3),
                        "Worst Value": round(self.worst_metrics[key][0], 3),
                        "Confidence Interval Lower (95%)": round(metrics_ci95[key][0], 3),
                        "Confidence Interval Upper (95%)": round(metrics_ci95[key][1], 3),
                        "Number of Images Processed": num_images,
                        "Outside 1 CI": outlier_counts["outside_1_ci"][key],
                        "Outside 2 CI": outlier_counts["outside_2_ci"][key],
                        "Outside 3 CI": outlier_counts["outside_3_ci"][key],
                        "IQR Outliers": int(outlier_counts["outside_iqr"][key]),  # Added IQR outliers
                        "Z-Score Outliers": int(outlier_counts["outside_z"][key]),  # Added Z-score outliers
                    }
                )
            output_data.append(
                {
                    "Metric": "Exams with Height Metric > 0.95",
                    "Count": exams_with_95_height_metric,
                    "Percentage": round((exams_with_95_height_metric / num_images) * 100, 2),
                }
            )

            output_data.append(
                {
                    "Metric": "Exams with Width Metric > 0.95",
                    "Count": exams_with_95_width_metric,
                    "Percentage": round((exams_with_95_width_metric / num_images) * 100, 2),
                }
            )
            output_data.append(
                {
                    "Metric": "Exams with Height Metric > 0.97",
                    "Count": exams_with_97_height_metric,
                    "Percentage": round((exams_with_97_height_metric / num_images) * 100, 2),
                }
            )
            output_data.append(
                {
                    "Metric": "Exams with Width Metric > 0.97",
                    "Count": exams_with_97_width_metric,
                    "Percentage": round((exams_with_97_width_metric / num_images) * 100, 2),
                }
            )
            output_data.append(
                {
                    "Metric": "Exams with Height Metric > 0.90",
                    "Count": exams_with_90_height_metric,
                    "Percentage": round((exams_with_90_height_metric / num_images) * 100, 2),
                }
            )
            output_data.append(
                {
                    "Metric": "Exams with Width Metric > 0.90",
                    "Count": exams_with_90_width_metric,
                    "Percentage": round((exams_with_90_width_metric / num_images) * 100, 2),
                }
            )
            output_data.append(
                {
                    "Metric": "Exams with Absolute Height Difference < 5",
                    "Count": (height_diff < 5).sum(),
                    "Percentage": round((height_diff < 5).sum() / num_images * 100, 2),
                }
            )
            output_data.append(
                {
                    "Metric": "Exams with Absolute Middle Width Difference < 5",
                    "Count": (width_diff_middle < 5).sum(),
                    "Percentage": round((width_diff_middle < 5).sum() / num_images * 100, 2),
                }
            )
            output_data.append(
                {
                    "Metric": "Exams with Absolute Lower Width Difference < 5",
                    "Count": (width_diff_lower < 5).sum(),
                    "Percentage": round((width_diff_lower < 5).sum() / num_images * 100, 2),
                }
            )
            output_data.append(
                {
                    "Metric": "Exams with Absolute Height Difference < 10",
                    "Count": (height_diff < 10).sum(),
                    "Percentage": round((height_diff < 10).sum() / num_images * 100, 2),
                }
            )
            output_data.append(
                {
                    "Metric": "Exams with Absolute Middle Width Difference < 10",
                    "Count": (width_diff_middle < 10).sum(),
                    "Percentage": round((width_diff_middle < 10).sum() / num_images * 100, 2),
                }
            )
            output_data.append(
                {
                    "Metric": "Exams with Absolute Lower Width Difference < 10",
                    "Count": (width_diff_lower < 10).sum(),
                    "Percentage": round((width_diff_lower < 10).sum() / num_images * 100, 2),
                }
            )

            # Convert to DataFrame and save to CSV
            df = pd.DataFrame(output_data)
            df.to_csv(filename, index=False, sep=";")
            if verbose:
                print(f"Metrics and confidence intervals saved to {filename}")
            dimensions_df.to_csv(f"{folder_paths[0]}/_dimensions.csv", index=False, sep=";")
            if verbose:
                print(f"Dimensions saved to {models_dir}_{folder_name}_dimensions.csv")
        if verbose:
            # Save the worst images for specified metrics
            for metric_name, (worst_value, worst_image) in self.worst_metrics.items():
                if metric_name in ["Height Metric", "Width Metric", "Dice Coefficient", "Cosine Similarity"]:
                    worst_filename = f"{models_dir}_{folder_name}_worst_{metric_name.replace(' ', '_')}.png"
                    cv2.imwrite(worst_filename, worst_image)
                    if verbose:
                        print(f"Worst image for {metric_name} saved as {worst_filename}")
        if verbose:
            print("Final Average Metrics:")
            for key, value in metrics_avg.items():
                print(f"{key}: {value}")

            print("\n95% Confidence Intervals:")
            for key, value in metrics_ci95.items():
                print(f"{key}: {value}")
        plot_filename = f"{folder_paths[0]}/_metrics_distribution.png"
        # Call the plot function and save the figure to a file
        self.plot_metric_distributions_with_ci(all_metrics, metrics_avg, metrics_ci95, save_path=plot_filename)

        return metrics_avg, metrics_ci95


if __name__ == "__main__":
    comparer = ImageComparison(apply_otsu_mask=True)
    # Changer le chemin vers le dossier contenant les images à analyser pour le train et la validation
    folder_path = "05_07_2025_ldm_dente_edente_zdente_cond"
    num_epoch = "50"

    # folder_path_train = f"{folder_path}/train/epoch_{num_epoch}"

    # average_metrics_train, confidence_intervals_train = comparer.process_all_images(
    #     [folder_path_train],
    #     file_selection_mode="last_n",
    #     n=1000,
    #     verbose=False,
    #     save_csv=True
    # )
    folder_path_validation = f"{folder_path}/validation_samples/epoch_{num_epoch}"

    average_metrics_validation, confidence_intervals_validation = comparer.process_all_images(
        [f"{folder_path_validation}/edente"], file_selection_mode="all", n="", verbose=False, save_csv=True
    )
    # for i in range(78, 80):
    #     folder_path_train = f"05_02_2025/data_augmentation_2/results/train/epoch_{i}"

    #     average_metrics_train, confidence_intervals_train = comparer.process_all_images(
    #         [folder_path_train],
    #         file_selection_mode="last_n",
    #         n=1000,
    #         verbose=False,
    #         save_csv=True

    #     )

    #     folder_path_validation = f"05_02_2025/data_augmentation_2/results/validation/epoch_{i}"

    #     average_metrics_validation, confidence_intervals_validation = comparer.process_all_images(
    #         [folder_path_validation],
    #         file_selection_mode="last_n",
    #         n=1000,
    #         verbose=False,
    #         save_csv=True
    #     )
    # folder_paths = [r"transpix2pix_2025_gan_vor/11_12_2024/grad_cam_Encoders_2_3/results/validation/epoch_41"]
    # average_metrics, confidence_intervals = comparer.process_all_images(folder_paths, file_selection_mode="last_n", n=1000, verbose=False, save_csv=True)
    # print("Average Metrics:", average_metrics)
    # print("Confidence Intervals:", confidence_intervals)

# "all" il prends toutes les images du ou des dossiers
# "random_n" où il prends n coupes au hasard dans les dossiers
# "last_n" où il prends les n dernieres coupes
