"""
PhysioRANO Preprocessing Pipeline using MONAI Transforms.

Provides dictionary-based MONAI transform compositions for reading 3D NIfTI images,
channel-first conversion, multi-modal stacking (T1n, T1c, T2w, T2f into a single 4-channel tensor),
resampling, foreground cropping, intensity normalization, resizing, and spatial/intensity augmentations.
"""

from typing import Sequence, Tuple, Union
import numpy as np

from monai.transforms import (
    Compose,
    ConcatItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandScaleIntensityd,
    RandSpatialCropd,
    Resized,
    Spacingd,
)


def get_base_load_transforms(
    keys: Sequence[str] = ("t1n", "t1c", "t2w", "t2f"),
    label_key: str = "label",
) -> Compose:
    """Constructs the initial dictionary loading and modal stacking transforms.

    Loads 4 NIfTI modalities (T1n, T1c, T2w, T2f) and segmentation label, converts to
    channel-first, and stacks the 4 modalities into a single 4-channel tensor under key 'image'.

    Args:
        keys (Sequence[str]): Modal file keys in sample dictionary.
        label_key (str): Segmentation label key in sample dictionary.

    Returns:
        Compose: MONAI Loading transform composition.
    """
    load_keys = list(keys)
    all_keys = load_keys + [label_key]

    return Compose([
        LoadImaged(keys=all_keys, image_only=False),
        EnsureChannelFirstd(keys=all_keys),
        ConcatItemsd(keys=load_keys, name="image", dim=0),
    ])


def get_val_transforms(
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    keys: Sequence[str] = ("t1n", "t1c", "t2w", "t2f"),
    label_key: str = "label",
) -> Compose:
    """Constructs validation MONAI preprocessing pipeline without random data augmentations.

    Pipeline operations:
    1. Read NIfTI images & segmentation mask
    2. Convert to channel-first representation
    3. Stack T1n, T1c, T2w, T2f into a single 4-channel 3D volume [4, H, W, D]
    4. Resample to isotropic physical voxel spacing (pixdim)
    5. Crop non-zero brain foreground region
    6. Normalize intensity (nonzero channel-wise Z-score)
    7. Resize volume to target spatial size [128, 128, 128]

    Args:
        spatial_size (Tuple[int, int, int]): Target volume dimensions [H, W, D].
        pixdim (Tuple[float, float, float]): Target voxel resolution in mm.
        keys (Sequence[str]): Modal file keys.
        label_key (str): Label key name.

    Returns:
        Compose: Validation MONAI transform pipeline.
    """
    load_keys = list(keys)
    all_keys = load_keys + [label_key]

    transforms = [
        LoadImaged(keys=all_keys, image_only=False, allow_missing_keys=True),
        EnsureChannelFirstd(keys=all_keys, allow_missing_keys=True),
        ConcatItemsd(keys=load_keys, name="image", dim=0),
        Spacingd(
            keys=["image", label_key],
            pixdim=pixdim,
            mode=("bilinear", "nearest"),
            allow_missing_keys=True,
        ),
        CropForegroundd(
            keys=["image", label_key],
            source_key="image",
            allow_missing_keys=True,
        ),
        NormalizeIntensityd(
            keys="image",
            nonzero=True,
            channel_wise=True,
        ),
        Resized(
            keys=["image", label_key],
            spatial_size=spatial_size,
            mode=("trilinear", "nearest"),
            allow_missing_keys=True,
        ),
        EnsureTyped(keys=["image", label_key], allow_missing_keys=True),
    ]

    return Compose(transforms)


def get_train_transforms(
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    keys: Sequence[str] = ("t1n", "t1c", "t2w", "t2f"),
    label_key: str = "label",
) -> Compose:
    """Constructs training MONAI preprocessing pipeline including data augmentations.

    Deterministic Preprocessing:
    1. NIfTI loading & channel-first conversion
    2. Modality stacking into 4-channel 3D volume [4, H, W, D]
    3. Spatial resampling (pixdim)
    4. Non-zero foreground cropping
    5. Z-score intensity normalization
    6. Resizing to spatial dimensions

    Training Data Augmentations:
    - Random spatial crop (RandCropByPosNegLabeld or RandSpatialCropd)
    - Random axis flips (RandFlipd along spatial dimensions)
    - Random 3D rotation (RandRotated)
    - Gaussian noise addition (RandGaussianNoised)
    - Random intensity scaling (RandScaleIntensityd)

    Args:
        spatial_size (Tuple[int, int, int]): Target volume dimensions.
        pixdim (Tuple[float, float, float]): Isotropic voxel resolution.
        keys (Sequence[str]): Modal file keys.
        label_key (str): Label key name.

    Returns:
        Compose: Training MONAI transform pipeline.
    """
    load_keys = list(keys)
    all_keys = load_keys + [label_key]

    transforms = [
        LoadImaged(keys=all_keys, image_only=False, allow_missing_keys=True),
        EnsureChannelFirstd(keys=all_keys, allow_missing_keys=True),
        ConcatItemsd(keys=load_keys, name="image", dim=0),
        Spacingd(
            keys=["image", label_key],
            pixdim=pixdim,
            mode=("bilinear", "nearest"),
            allow_missing_keys=True,
        ),
        CropForegroundd(
            keys=["image", label_key],
            source_key="image",
            allow_missing_keys=True,
        ),
        NormalizeIntensityd(
            keys="image",
            nonzero=True,
            channel_wise=True,
        ),
        Resized(
            keys=["image", label_key],
            spatial_size=spatial_size,
            mode=("trilinear", "nearest"),
            allow_missing_keys=True,
        ),
        # Training Augmentations
        RandFlipd(
            keys=["image", label_key],
            prob=0.5,
            spatial_axis=0,
            allow_missing_keys=True,
        ),
        RandFlipd(
            keys=["image", label_key],
            prob=0.5,
            spatial_axis=1,
            allow_missing_keys=True,
        ),
        RandFlipd(
            keys=["image", label_key],
            prob=0.5,
            spatial_axis=2,
            allow_missing_keys=True,
        ),
        RandRotated(
            keys=["image", label_key],
            range_x=np.pi / 12,
            range_y=np.pi / 12,
            range_z=np.pi / 12,
            mode=("bilinear", "nearest"),
            prob=0.3,
            allow_missing_keys=True,
        ),
        RandGaussianNoised(
            keys="image",
            prob=0.2,
            mean=0.0,
            std=0.1,
        ),
        RandScaleIntensityd(
            keys="image",
            factors=0.1,
            prob=0.3,
        ),
        EnsureTyped(keys=["image", label_key], allow_missing_keys=True),
    ]

    return Compose(transforms)
