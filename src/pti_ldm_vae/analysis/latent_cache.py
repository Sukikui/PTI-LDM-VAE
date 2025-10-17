import hashlib
import json
from pathlib import Path

import numpy as np


class LatentCache:
    """Automatic caching system for encoded latents with per-image granularity.

    The cache system:
    - Stores each encoded image separately
    - Organizes cache by VAE model signature (weights + params)
    - Automatically detects new images and encodes only those
    - Preserves caches from different VAE models
    - Tracks image modification times to detect changes

    Cache structure:
        cache/latents/
        ├── {model_sig1}/
        │   ├── _metadata.json
        │   ├── {image_hash1}.npz
        │   └── {image_hash2}.npz
        └── {model_sig2}/
            ├── _metadata.json
            └── ...

    Attributes:
        cache_root: Root directory for all cached latents
    """

    def __init__(self, cache_root: Path = Path("cache/latents")) -> None:
        """Initialize latent cache system.

        Args:
            cache_root: Root directory for cache storage (default: cache/latents)
        """
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _get_model_signature(self, vae_weights: str, patch_size: tuple[int, int]) -> str:
        """Generate unique signature for VAE model + encoding parameters.

        Args:
            vae_weights: Path to VAE weights file
            patch_size: Image patch size as (height, width)

        Returns:
            8-character hex string uniquely identifying this model configuration
        """
        # Use absolute path to avoid confusion with relative paths
        abs_weights_path = Path(vae_weights).resolve()
        signature_str = f"{abs_weights_path}_{patch_size}"
        return hashlib.md5(signature_str.encode()).hexdigest()[:8]

    def _get_image_cache_key(self, image_path: str) -> str:
        """Generate cache key for a specific image.

        The key includes both the image path and its modification time
        to automatically invalidate cache if the image file changes.

        Args:
            image_path: Path to image file

        Returns:
            12-character hex string uniquely identifying this image version
        """
        abs_image_path = Path(image_path).resolve()

        # Include mtime to detect if image has been modified
        mtime = abs_image_path.stat().st_mtime if abs_image_path.exists() else 0

        key_str = f"{abs_image_path}_{mtime}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def _get_cache_file_path(self, image_path: str, model_signature: str) -> Path:
        """Get cache file path for specific image + model combination.

        Args:
            image_path: Path to source image
            model_signature: Model signature from _get_model_signature()

        Returns:
            Path to cache file (.npz format)
        """
        image_key = self._get_image_cache_key(image_path)
        model_dir = self.cache_root / model_signature
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{image_key}.npz"

    def _get_metadata_path(self, model_signature: str) -> Path:
        """Get path to metadata file for a model.

        Args:
            model_signature: Model signature from _get_model_signature()

        Returns:
            Path to metadata JSON file
        """
        model_dir = self.cache_root / model_signature
        return model_dir / "_metadata.json"

    def _load_metadata(self, model_signature: str) -> dict:
        """Load metadata mapping cache files to original images.

        Args:
            model_signature: Model signature from _get_model_signature()

        Returns:
            Dictionary with metadata structure:
            {
                "model": "path/to/weights.pth",
                "patch_size": [256, 256],
                "images": {
                    "/abs/path/img.png": {
                        "cache_key": "abc123def456",
                        "patient_id": "P001"
                    }
                }
            }
        """
        metadata_path = self._get_metadata_path(model_signature)
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {"images": {}}

    def _save_metadata(self, model_signature: str, metadata: dict) -> None:
        """Save metadata mapping.

        Args:
            model_signature: Model signature from _get_model_signature()
            metadata: Metadata dictionary to save
        """
        metadata_path = self._get_metadata_path(model_signature)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_or_encode_batch(
        self,
        image_paths: list[str],
        encoder_fn,
        vae_weights: str,
        patch_size: tuple[int, int],
        group_name: str,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Get latents for a batch of images, using cache when available.

        This is the main method for batch processing. It:
        1. Checks cache for each image
        2. Encodes only images not in cache
        3. Saves newly encoded images to cache
        4. Returns all latents, IDs, and paths in order

        Args:
            image_paths: List of paths to images to process
            encoder_fn: Function that takes (image_path) and returns (latent, patient_id)
            vae_weights: Path to VAE weights file (for cache key)
            patch_size: Image patch size as (height, width)
            group_name: Name for logging (e.g., "edente", "dente")

        Returns:
            Tuple of:
                - latents: numpy array of shape (n_images, latent_dim)
                - ids: list of patient IDs
                - paths: list of image paths (same order as input)
        """
        model_signature = self._get_model_signature(vae_weights, patch_size)
        metadata = self._load_metadata(model_signature)

        print(f"📂 Processing {group_name} ({len(image_paths)} images)")
        print(f"   Model: {Path(vae_weights).name} (cache sig: {model_signature})")

        latents_list: list[np.ndarray] = []
        ids_list: list[str] = []
        paths_list: list[str] = []

        cached_count = 0
        encoded_count = 0

        for img_path in image_paths:
            cache_file_path = self._get_cache_file_path(img_path, model_signature)
            abs_img_path = str(Path(img_path).resolve())
            cache_key = self._get_image_cache_key(img_path)

            # Check if this exact image version is in cache
            image_metadata = metadata["images"].get(abs_img_path, {})
            cache_valid = (
                cache_file_path.exists()
                and image_metadata.get("cache_key") == cache_key
            )

            if cache_valid:
                # Cache hit - load from cache
                try:
                    data = np.load(cache_file_path)
                    latent = data["latent"]
                    patient_id = str(data["patient_id"])
                    cached_count += 1
                except Exception as e:
                    print(f"   ⚠️  Cache corrupted for {Path(img_path).name}, re-encoding: {e}")
                    # Cache corrupted, re-encode
                    latent, patient_id = encoder_fn(img_path)
                    np.savez(cache_file_path, latent=latent, patient_id=patient_id)
                    encoded_count += 1
            else:
                # Cache miss - encode image
                latent, patient_id = encoder_fn(img_path)

                # Save to cache
                np.savez(cache_file_path, latent=latent, patient_id=patient_id)

                # Update metadata
                metadata["images"][abs_img_path] = {
                    "cache_key": cache_key,
                    "patient_id": patient_id,
                }

                encoded_count += 1

            latents_list.append(latent)
            ids_list.append(patient_id)
            paths_list.append(img_path)

        # Save updated metadata if any new images were encoded
        if encoded_count > 0:
            metadata["model"] = str(Path(vae_weights).name)
            metadata["patch_size"] = list(patch_size)
            self._save_metadata(model_signature, metadata)

        print(f"   ✅ {cached_count} from cache, 🔄 {encoded_count} newly encoded")

        return np.array(latents_list), ids_list, paths_list

    def clear_cache(self, model_signature: str | None = None) -> None:
        """Clear cache files.

        Args:
            model_signature: If provided, clear only this model's cache.
                           If None, clear entire cache.
        """
        if model_signature is None:
            # Clear all cache
            import shutil

            if self.cache_root.exists():
                shutil.rmtree(self.cache_root)
                self.cache_root.mkdir(parents=True, exist_ok=True)
            print(f"🗑️  Cleared all cache in {self.cache_root}")
        else:
            # Clear specific model cache
            model_dir = self.cache_root / model_signature
            if model_dir.exists():
                import shutil

                shutil.rmtree(model_dir)
            print(f"🗑️  Cleared cache for model {model_signature}")

    def get_cache_stats(self) -> dict[str, dict]:
        """Get statistics about cache usage.

        Returns:
            Dictionary mapping model signatures to their stats:
            {
                "abc12345": {
                    "model": "epoch73.pth",
                    "patch_size": [256, 256],
                    "num_images": 150,
                    "cache_size_mb": 45.2
                }
            }
        """
        stats = {}

        if not self.cache_root.exists():
            return stats

        for model_dir in self.cache_root.iterdir():
            if not model_dir.is_dir():
                continue

            model_sig = model_dir.name
            metadata = self._load_metadata(model_sig)

            # Calculate cache size
            total_size = sum(
                f.stat().st_size for f in model_dir.glob("*.npz") if f.is_file()
            )

            stats[model_sig] = {
                "model": metadata.get("model", "unknown"),
                "patch_size": metadata.get("patch_size", []),
                "num_images": len(metadata.get("images", {})),
                "cache_size_mb": total_size / (1024 * 1024),
            }

        return stats