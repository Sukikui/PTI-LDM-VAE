from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@contextmanager
def limit_threadpools(num_threads: int) -> Iterator[None]:
    """Limit native thread pools to reduce OpenMP-related crashes.

    Args:
        num_threads (int): Maximum number of threads for BLAS/OpenMP pools.

    Yields:
        None: Context manager for scoped thread limiting.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return

    with threadpool_limits(limits=num_threads):
        yield


def _set_umap_thread_env() -> None:
    """Set conservative thread limits for UMAP/numba workloads.

    This reduces the risk of OpenMP/numba-related crashes on some environments.
    """
    thread_limits = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "NUMBA_THREADING_LAYER": "workqueue",
    }
    for key, value in thread_limits.items():
        os.environ.setdefault(key, value)


def save_color_legend(patient_to_id: dict[str, int], patient_to_color: dict[str, str], output_path: Path) -> None:
    """Save a color legend mapping to a text file.

    Args:
        patient_to_id (dict[str, int]): Mapping from patient ID to integer index.
        patient_to_color (dict[str, str]): Mapping from patient ID to hex color.
        output_path (Path): Destination text file path.
    """
    with open(output_path, "w") as file:
        file.write("Color Legend for Patients\n")
        file.write("=" * 60 + "\n\n")
        for patient in sorted(patient_to_id.keys(), key=lambda key: patient_to_id[key]):
            file.write(f"{patient_to_id[patient]}: {patient} - {patient_to_color[patient]}\n")


def compute_distance_metrics(points1: np.ndarray, points2: np.ndarray) -> tuple[float, float, float, float] | None:
    """Compute distance metrics between two point clouds.

    Args:
        points1 (np.ndarray): Points of shape ``[N, D]``.
        points2 (np.ndarray): Points of shape ``[M, D]``.

    Returns:
        tuple[float, float, float, float] | None: (center_distance, std1, std2, mean_cross_distance),
        or ``None`` if one of the point sets is empty.
    """
    if len(points1) == 0 or len(points2) == 0:
        return None

    try:
        from scipy.spatial.distance import cdist
    except ImportError as exc:
        raise ImportError("scipy is required to compute distance metrics") from exc

    points1 = np.asarray(points1)
    points2 = np.asarray(points2)

    mean1 = np.mean(points1, axis=0)
    mean2 = np.mean(points2, axis=0)
    center_distance = float(np.linalg.norm(mean1 - mean2))

    std1 = float(np.mean(np.std(points1, axis=0))) if len(points1) > 1 else 0.0
    std2 = float(np.mean(np.std(points2, axis=0))) if len(points2) > 1 else 0.0

    mean_cross_distance = float(np.mean(cdist(points1, points2)))

    return center_distance, std1, std2, mean_cross_distance


def compute_group_statistics(
    *,
    projection_group1: np.ndarray,
    projection_group2: np.ndarray,
    latent_group1: np.ndarray,
    latent_group2: np.ndarray,
    ids_group1: list[str],
    ids_group2: list[str],
    group1_name: str,
    group2_name: str,
    output_dir: Path,
) -> None:
    """Compute and save group-level statistics per patient.

    Args:
        projection_group1 (np.ndarray): Projection for group 1 (``[N, 2]``).
        projection_group2 (np.ndarray): Projection for group 2 (``[M, 2]``).
        latent_group1 (np.ndarray): Latents for group 1 (``[N, D]``).
        latent_group2 (np.ndarray): Latents for group 2 (``[M, D]``).
        ids_group1 (list[str]): Patient IDs for group 1.
        ids_group2 (list[str]): Patient IDs for group 2.
        group1_name (str): Group 1 name.
        group2_name (str): Group 2 name.
        output_dir (Path): Output directory for reports.
    """
    exam_data_proj: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {group1_name: [], group2_name: []}
    )
    exam_data_lat: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {group1_name: [], group2_name: []}
    )

    for idx, exam_id in enumerate(ids_group1):
        exam_data_proj[exam_id][group1_name].append(projection_group1[idx])
        exam_data_lat[exam_id][group1_name].append(latent_group1[idx])

    for idx, exam_id in enumerate(ids_group2):
        exam_data_proj[exam_id][group2_name].append(projection_group2[idx])
        exam_data_lat[exam_id][group2_name].append(latent_group2[idx])

    results_file = output_dir / "distance_metrics.txt"
    sorted_file = output_dir / "exams_sorted_by_distance.txt"
    exam_distances: list[tuple[str, float]] = []

    with open(results_file, "w") as file:
        file.write("Distance Metrics per Patient (Latent Space and Projection)\n")
        file.write("=" * 60 + "\n\n")

        for exam in sorted(exam_data_lat.keys()):
            lat_group1 = exam_data_lat[exam][group1_name]
            lat_group2 = exam_data_lat[exam][group2_name]
            proj_group1 = exam_data_proj[exam][group1_name]
            proj_group2 = exam_data_proj[exam][group2_name]

            if len(lat_group1) == 0 or len(lat_group2) == 0:
                continue

            metrics_lat = compute_distance_metrics(np.array(lat_group1), np.array(lat_group2))
            metrics_proj = compute_distance_metrics(np.array(proj_group1), np.array(proj_group2))

            if metrics_lat is None or metrics_proj is None:
                continue

            file.write(f"{exam}\n")
            file.write(f"  - n_{group1_name}: {len(lat_group1)}, n_{group2_name}: {len(lat_group2)}\n")
            file.write(
                f"  - [Latent] center_dist: {metrics_lat[0]:.3f}, "
                f"std_{group1_name}: {metrics_lat[1]:.3f}, std_{group2_name}: {metrics_lat[2]:.3f}, "
                f"mean_cross_dist: {metrics_lat[3]:.3f}\n"
            )
            file.write(
                f"  - [Projection] center_dist: {metrics_proj[0]:.3f}, "
                f"std_{group1_name}: {metrics_proj[1]:.3f}, std_{group2_name}: {metrics_proj[2]:.3f}, "
                f"mean_cross_dist: {metrics_proj[3]:.3f}\n\n"
            )

            exam_distances.append((exam, metrics_lat[0]))

    exam_distances.sort(key=lambda item: item[1])
    with open(sorted_file, "w") as file:
        file.write("Patients sorted by latent space center distance\n")
        file.write("=" * 60 + "\n\n")
        for exam, dist in exam_distances:
            file.write(f"{exam}: {dist:.3f}\n")

def extract_patient_id_from_filename(filename: str) -> str:
    """Extract a patient ID from a filename.

    The dataset filenames typically follow ``..._<patient_id>.tif`` and the patient ID is
    stored as the last underscore-separated token (without extension).

    Args:
        filename (str): Filename (e.g. ``1000_HA_2021_02_545.tif``).

    Returns:
        str: Extracted patient identifier.
    """
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    parts = stem.split("_")
    return parts[-1] if parts else stem


def list_image_paths(folder_path: str, *, max_images: int | None = None) -> list[str]:
    """List image paths from a folder.

    Args:
        folder_path (str): Directory containing images.
        max_images (int | None): Optional cap on the number of returned files.

    Returns:
        list[str]: Sorted list of image paths.

    Raises:
        FileNotFoundError: If the folder does not exist or contains no supported images.
    """
    base = Path(folder_path)
    if not base.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    paths = sorted(base.glob("*.tif")) + sorted(base.glob("*.tiff"))
    if not paths:
        raise FileNotFoundError(f"No .tif/.tiff images found in: {folder_path}")

    if max_images is not None:
        paths = paths[:max_images]

    return [str(path) for path in paths]


def latent_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute Euclidean distance between two latent vectors.

    Args:
        vec_a (np.ndarray): Latent vector of shape ``[D]``.
        vec_b (np.ndarray): Latent vector of shape ``[D]``.

    Returns:
        float: Euclidean distance.
    """
    if vec_a.ndim != 1 or vec_b.ndim != 1:
        raise ValueError(f"Expected 1D latent vectors, got shapes {vec_a.shape} and {vec_b.shape}")
    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Latent vectors must have the same shape, got {vec_a.shape} and {vec_b.shape}")
    return float(np.linalg.norm(vec_a - vec_b))


def latent_distance_cross(latents_a: np.ndarray, idx_a: int, latents_b: np.ndarray, idx_b: int) -> float:
    """Compute Euclidean distance between latents from two groups using indices.

    Args:
        latents_a (np.ndarray): Latents of shape ``[N, D]``.
        idx_a (int): Index in ``latents_a``.
        latents_b (np.ndarray): Latents of shape ``[M, D]``.
        idx_b (int): Index in ``latents_b``.

    Returns:
        float: Euclidean distance.
    """
    if latents_a.ndim != 2 or latents_b.ndim != 2:
        raise ValueError(f"Expected 2D latents, got shapes {latents_a.shape} and {latents_b.shape}")
    if latents_a.shape[1] != latents_b.shape[1]:
        raise ValueError(
            f"Latent dims must match between groups, got {latents_a.shape[1]} and {latents_b.shape[1]}"
        )
    if not (0 <= idx_a < latents_a.shape[0]):
        raise ValueError(f"idx_a must be in [0, {latents_a.shape[0] - 1}], got {idx_a}")
    if not (0 <= idx_b < latents_b.shape[0]):
        raise ValueError(f"idx_b must be in [0, {latents_b.shape[0] - 1}], got {idx_b}")
    return latent_distance(latents_a[idx_a], latents_b[idx_b])


def _color_from_index(index: int, total: int) -> str:
    """Generate a deterministic color from an index.

    Args:
        index (int): Color index.
        total (int): Total number of colors needed.

    Returns:
        str: Hex color string (e.g. ``#a1b2c3``).
    """
    import colorsys

    if total <= 0:
        total = 1
    hue = (index % total) / total
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


class LatentSpaceAnalyzer:
    """Encode images with a VAE and compute 2D projections (UMAP/t-SNE)."""

    def __init__(self, vae_model: torch.nn.Module, device: torch.device, transform) -> None:
        """Initialize the analyzer.

        Args:
            vae_model (torch.nn.Module): VAE model exposing ``encode_deterministic``.
            device (torch.device): Target device.
            transform: MONAI transform pipeline for preprocessing.
        """
        self.vae = vae_model
        self.device = device
        self.transform = transform
        self.vae.eval()

    def encode_images(
        self,
        image_paths: list[str],
        *,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """Encode images to deterministic latents.

        Args:
            image_paths (list[str]): Input image paths.
            batch_size (int): Batch size used for VAE encoding.
            show_progress (bool): Whether to show a progress bar if tqdm is available.

        Returns:
            tuple[np.ndarray, list[str]]: Flattened latents ``[N, D]`` and patient IDs.
        """
        if not image_paths:
            raise ValueError("image_paths cannot be empty")

        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Encoding images", unit="batch")
            except ImportError:
                pass

        latent_batches: list[torch.Tensor] = []
        patient_ids: list[str] = []

        with torch.no_grad():
            for start in iterator:
                batch_paths = image_paths[start : start + batch_size]
                batch_imgs: list[torch.Tensor] = [self.transform(path) for path in batch_paths]
                batch_tensor = torch.stack(batch_imgs, dim=0).to(self.device)

                if not hasattr(self.vae, "encode_deterministic"):
                    raise AttributeError("VAE model must expose encode_deterministic for analysis.")

                z = self.vae.encode_deterministic(batch_tensor)
                latent_batches.append(z.cpu().flatten(start_dim=1))

                for path in batch_paths:
                    patient_ids.append(extract_patient_id_from_filename(Path(path).name))

        latents = torch.cat(latent_batches, dim=0).numpy()
        return latents, patient_ids

    @staticmethod
    def create_patient_colormap(patient_ids: list[str]) -> tuple[dict[str, int], dict[str, str]]:
        """Create a deterministic mapping from patient IDs to colors.

        Args:
            patient_ids (list[str]): List of patient identifiers.

        Returns:
            tuple[dict[str, int], dict[str, str]]: (patient_to_id, patient_to_color).
        """
        unique = sorted(set(patient_ids))
        patient_to_id = {patient: idx for idx, patient in enumerate(unique)}
        patient_to_color = {patient: _color_from_index(idx, len(unique)) for patient, idx in patient_to_id.items()}
        return patient_to_id, patient_to_color

    @staticmethod
    def reduce_dimensionality_umap(
        latent_vectors: np.ndarray,
        *,
        n_neighbors: int,
        min_dist: float,
        random_state: int | None = None,
        pca_components: int = 50,
    ) -> tuple[np.ndarray, object, PCA]:
        """Reduce dimensionality using PCA + UMAP.

        Args:
            latent_vectors (np.ndarray): Latents ``[N, D]``.
            n_neighbors (int): UMAP ``n_neighbors``.
            min_dist (float): UMAP ``min_dist``.
            random_state (int | None): Optional random seed passed to PCA and UMAP.
            pca_components (int): PCA components count before UMAP.

        Returns:
            tuple[np.ndarray, object, PCA]: UMAP embedding ``[N, 2]``, fitted UMAP model, fitted PCA model.
        """
        if latent_vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {latent_vectors.shape}")

        n_samples = latent_vectors.shape[0]
        pca_components = int(min(pca_components, n_samples))
        if pca_components < 2:
            raise ValueError(f"Need at least 2 samples for PCA, got {n_samples}")
        if n_neighbors >= n_samples:
            raise ValueError(f"n_neighbors ({n_neighbors}) must be < n_samples ({n_samples})")

        _set_umap_thread_env()
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise ImportError("UMAP requires umap-learn. Install with: pip install umap-learn") from exc

        try:
            import numba  # type: ignore

            numba.set_num_threads(1)
        except Exception:
            pass

        pca = PCA(n_components=pca_components, random_state=random_state)
        vectors_pca = pca.fit_transform(latent_vectors)
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            n_components=2,
            n_jobs=1,
            low_memory=True,
        )
        embedding = umap_model.fit_transform(vectors_pca)
        return embedding, umap_model, pca

    @staticmethod
    def reduce_dimensionality_tsne(
        latent_vectors: np.ndarray,
        *,
        perplexity: int,
        random_state: int | None = None,
        pca_components: int = 50,
    ) -> np.ndarray:
        """Reduce dimensionality using PCA + t-SNE.

        Args:
            latent_vectors (np.ndarray): Latents ``[N, D]``.
            perplexity (int): t-SNE perplexity.
            random_state (int | None): Optional random seed passed to PCA/t-SNE.
            pca_components (int): PCA components count before t-SNE.

        Returns:
            np.ndarray: t-SNE embedding ``[N, 2]``.
        """
        if latent_vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {latent_vectors.shape}")

        n_samples = latent_vectors.shape[0]
        pca_components = int(min(pca_components, n_samples))
        if pca_components < 2:
            raise ValueError(f"Need at least 2 samples for PCA, got {n_samples}")
        if perplexity >= n_samples:
            raise ValueError(f"perplexity ({perplexity}) must be < n_samples ({n_samples})")

        pca = PCA(n_components=pca_components, random_state=random_state)
        vectors_pca = pca.fit_transform(latent_vectors)
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state)
        return tsne.fit_transform(vectors_pca)


class LatentCache:
    """Disk cache for per-image latents (keyed by checkpoint + patch size + file mtime)."""

    def __init__(self, cache_root: Path) -> None:
        """Initialize cache storage.

        Args:
            cache_root (Path): Root directory for cache artifacts.
        """
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def clear_cache(self, model_signature: str | None = None) -> None:
        """Clear cache files.

        Args:
            model_signature (str | None): If provided, clears only that model signature cache.
                If ``None``, clears all cached signatures under ``cache_root``.
        """
        import shutil

        if model_signature is None:
            if self.cache_root.exists():
                shutil.rmtree(self.cache_root)
            self.cache_root.mkdir(parents=True, exist_ok=True)
            print(f"[LatentCache] Cleared all cache under: {self.cache_root}")
            return

        model_dir = self.cache_root / model_signature
        if model_dir.exists():
            shutil.rmtree(model_dir)
        print(f"[LatentCache] Cleared cache for signature: {model_signature}")

    def get_cache_stats(self) -> dict[str, dict[str, object]]:
        """Return basic statistics about the cache.

        Returns:
            dict[str, dict[str, object]]: Mapping signature -> stats.
        """
        stats: dict[str, dict[str, object]] = {}
        if not self.cache_root.exists():
            return stats

        for model_dir in sorted([path for path in self.cache_root.iterdir() if path.is_dir()]):
            metadata = self._load_metadata(model_dir.name)
            total_size = sum(path.stat().st_size for path in model_dir.glob("*.npz") if path.is_file())
            stats[model_dir.name] = {
                "model": metadata.get("model", "unknown"),
                "patch_size": metadata.get("patch_size", []),
                "num_images": len(metadata.get("images", {})),
                "cache_size_mb": total_size / (1024 * 1024),
            }

        return stats

    def _model_signature(self, checkpoint_path: str, patch_size: tuple[int, int]) -> str:
        """Compute a short model signature for cache grouping.

        Args:
            checkpoint_path (str): Path to the VAE checkpoint.
            patch_size (tuple[int, int]): Preprocess patch size (H, W).

        Returns:
            str: 8-char hex signature.
        """
        signature = f"{Path(checkpoint_path).resolve()}_{patch_size}"
        return hashlib.md5(signature.encode()).hexdigest()[:8]

    @staticmethod
    def _image_cache_key(image_path: str) -> str:
        """Compute a cache key for an image path including its modification time.

        Args:
            image_path (str): Image path.

        Returns:
            str: 12-char hex key.
        """
        resolved = Path(image_path).resolve()
        mtime = resolved.stat().st_mtime if resolved.exists() else 0
        key = f"{resolved}_{mtime}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _cache_file(self, image_path: str, model_signature: str) -> Path:
        """Build cache filename for a single image.

        Args:
            image_path (str): Image path.
            model_signature (str): Signature returned by ``_model_signature``.

        Returns:
            Path: Cache file path.
        """
        model_dir = self.cache_root / model_signature
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{self._image_cache_key(image_path)}.npz"

    def _metadata_path(self, model_signature: str) -> Path:
        """Return metadata path for a signature.

        Args:
            model_signature (str): Signature returned by ``_model_signature``.

        Returns:
            Path: Metadata JSON path.
        """
        return self.cache_root / model_signature / "_metadata.json"

    def _load_metadata(self, model_signature: str) -> dict:
        """Load metadata JSON for a signature.

        Args:
            model_signature (str): Signature returned by ``_model_signature``.

        Returns:
            dict: Metadata mapping.
        """
        metadata_path = self._metadata_path(model_signature)
        if metadata_path.exists():
            with open(metadata_path) as file:
                return json.load(file)
        return {"images": {}}

    def _save_metadata(self, model_signature: str, metadata: dict) -> None:
        """Persist metadata JSON for a signature.

        Args:
            model_signature (str): Signature returned by ``_model_signature``.
            metadata (dict): Metadata mapping to write.
        """
        metadata_path = self._metadata_path(model_signature)
        with open(metadata_path, "w") as file:
            json.dump(metadata, file, indent=2)

    def get_or_encode_batch(
        self,
        *,
        image_paths: list[str],
        analyzer: LatentSpaceAnalyzer,
        checkpoint_path: str,
        patch_size: tuple[int, int],
        group_name: str,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Return latents for a batch of images using cache when possible.

        Args:
            image_paths (list[str]): Input image paths.
            analyzer (LatentSpaceAnalyzer): Encoder wrapper.
            checkpoint_path (str): VAE checkpoint path (part of cache signature).
            patch_size (tuple[int, int]): Preprocess patch size (part of cache signature).
            group_name (str): Name used for logging.
            batch_size (int): Encoding batch size for cache misses.
            show_progress (bool): Whether to show progress when encoding cache misses.

        Returns:
            tuple[np.ndarray, list[str], list[str]]: Latents ``[N, D]``, patient IDs, and paths.
        """
        if not image_paths:
            raise ValueError("image_paths cannot be empty")

        model_signature = self._model_signature(checkpoint_path, patch_size)
        metadata = self._load_metadata(model_signature)

        latents: list[np.ndarray | None] = [None] * len(image_paths)
        ids: list[str | None] = [None] * len(image_paths)
        miss_indices: list[int] = []
        miss_paths: list[str] = []

        for idx, img_path in enumerate(image_paths):
            cache_file = self._cache_file(img_path, model_signature)
            abs_img_path = str(Path(img_path).resolve())
            expected_key = self._image_cache_key(img_path)

            meta = metadata.get("images", {}).get(abs_img_path, {})
            is_valid = cache_file.exists() and meta.get("cache_key") == expected_key

            if is_valid:
                try:
                    data = np.load(cache_file)
                    latents[idx] = data["latent"]
                    ids[idx] = str(data["patient_id"])
                    continue
                except Exception:
                    pass

            miss_indices.append(idx)
            miss_paths.append(img_path)

        cached_count = len(image_paths) - len(miss_paths)
        encoded_count = len(miss_paths)

        if miss_paths:
            print(f"[LatentCache] Encoding {len(miss_paths)} / {len(image_paths)} images for '{group_name}'")
            new_latents, new_ids = analyzer.encode_images(miss_paths, batch_size=batch_size, show_progress=show_progress)

            for offset, idx in enumerate(miss_indices):
                img_path = miss_paths[offset]
                latent = new_latents[offset]
                patient_id = new_ids[offset]

                cache_file = self._cache_file(img_path, model_signature)
                np.savez(cache_file, latent=latent, patient_id=patient_id)

                abs_img_path = str(Path(img_path).resolve())
                metadata.setdefault("images", {})[abs_img_path] = {
                    "cache_key": self._image_cache_key(img_path),
                    "patient_id": patient_id,
                }

                latents[idx] = latent
                ids[idx] = patient_id

            metadata["model"] = str(Path(checkpoint_path).name)
            metadata["patch_size"] = list(patch_size)
            self._save_metadata(model_signature, metadata)

        print(
            f"[LatentCache] {group_name}: ✅ {cached_count} from cache, 🔄 {encoded_count} encoded "
            f"(sig: {model_signature})"
        )

        if any(item is None for item in latents) or any(item is None for item in ids):
            raise RuntimeError("Latent cache produced incomplete results.")

        latents_out = np.stack([latent for latent in latents if latent is not None], axis=0)
        ids_out = [str(pid) for pid in ids if pid is not None]
        return latents_out, ids_out, list(image_paths)
