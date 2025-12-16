"""Data augmentation utilities for medical image training."""

try:
    import albumentations as A
except ImportError as e:
    raise ImportError(
        "albumentations is required for data augmentation. Install it with: pip install albumentations"
    ) from e


def get_albumentations_transform():
    """Get albumentations transform for medical image augmentation.

    Returns:
        albumentations.Compose: Augmentation pipeline that applies the same
            transforms to both 'image' and 'condition_image' (for paired images)

    Example:
        >>> import numpy as np
        >>> from pti_ldm_vae.data.augmentation import get_albumentations_transform
        >>>
        >>> transform = get_albumentations_transform()
        >>> image = np.random.rand(256, 256).astype(np.float32)
        >>> condition = np.random.rand(256, 256).astype(np.float32)
        >>>
        >>> # For single image
        >>> augmented = transform(image=image)
        >>>
        >>> # For paired images (e.g., conditioned tasks)
        >>> augmented = transform(image=image, condition_image=condition)
    """
    return A.Compose(
        [
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
            A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, border_mode=0, p=0.3),
        ],
        additional_targets={"condition_image": "image"},
    )
