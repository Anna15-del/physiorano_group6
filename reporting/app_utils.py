"""
PhysioRANO Streamlit Application Visualization Utilities.

Provides figure generation and quantitative volumetric calculation functions for Streamlit dashboard.
"""

from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch


def render_4channel_slice(
    image: Union[np.ndarray, torch.Tensor],
    label: Optional[Union[np.ndarray, torch.Tensor]] = None,
    pred_label: Optional[Union[np.ndarray, torch.Tensor]] = None,
    plane: str = "axial",
    slice_idx: Optional[int] = None,
    alpha: float = 0.5,
) -> plt.Figure:
    """Generates a multi-modal 4-channel MRI slice figure for Streamlit display.

    Args:
        image: 4-channel 3D volume [4, H, W, D].
        label: Ground truth segmentation mask [1, H, W, D] or [H, W, D].
        pred_label: Predicted segmentation mask [1, H, W, D] or [H, W, D].
        plane: View orientation plane ('axial', 'coronal', 'sagittal').
        slice_idx: Slice index along selected orientation plane.
        alpha: Transparency factor for segmentation mask overlays.

    Returns:
        plt.Figure: Matplotlib figure instance.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if label is not None and isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    if pred_label is not None and isinstance(pred_label, torch.Tensor):
        pred_label = pred_label.detach().cpu().numpy()

    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    num_channels, H, W, D = image.shape

    # Determine slice extraction based on orientation plane
    if plane == "axial":
        max_slices = D
        s_idx = slice_idx if slice_idx is not None else D // 2
        extract = lambda data, c: data[c, :, :, s_idx]
        extract_mask = lambda mask: mask[0, :, :, s_idx] if mask.ndim == 4 else mask[:, :, s_idx]
    elif plane == "coronal":
        max_slices = W
        s_idx = slice_idx if slice_idx is not None else W // 2
        extract = lambda data, c: data[c, :, s_idx, :]
        extract_mask = lambda mask: mask[0, :, s_idx, :] if mask.ndim == 4 else mask[:, s_idx, :]
    else:  # sagittal
        max_slices = H
        s_idx = slice_idx if slice_idx is not None else H // 2
        extract = lambda data, c: data[c, s_idx, :, :]
        extract_mask = lambda mask: mask[0, s_idx, :, :] if mask.ndim == 4 else mask[s_idx, :, :]

    modality_names = ["T1n (Native)", "T1c (Contrast)", "T2w (Weighted)", "T2f (FLAIR)"]
    extra_cols = (1 if label is not None else 0) + (1 if pred_label is not None else 0)
    total_cols = num_channels + extra_cols

    fig, axes = plt.subplots(1, total_cols, figsize=(3.8 * total_cols, 3.8))
    if total_cols == 1:
        axes = [axes]

    # Plot 4 Modal Slices
    for c in range(min(num_channels, 4)):
        img_slice = extract(image, c)
        axes[c].imshow(img_slice, cmap="gray")
        axes[c].set_title(f"{modality_names[c]}", fontsize=11, fontweight="bold")
        axes[c].axis("off")

    col_ptr = num_channels

    # Plot Ground Truth Overlay
    if label is not None:
        lbl_slice = extract_mask(label)
        bg_slice = extract(image, 1)  # T1c background
        axes[col_ptr].imshow(bg_slice, cmap="gray")
        masked_gt = np.ma.masked_where(lbl_slice == 0, lbl_slice)
        axes[col_ptr].imshow(masked_gt, cmap="jet", alpha=alpha)
        axes[col_ptr].set_title("Ground Truth Mask", fontsize=11, fontweight="bold", color="darkgreen")
        axes[col_ptr].axis("off")
        col_ptr += 1

    # Plot Model Prediction Overlay
    if pred_label is not None:
        pred_slice = extract_mask(pred_label)
        bg_slice = extract(image, 1)
        axes[col_ptr].imshow(bg_slice, cmap="gray")
        masked_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
        axes[col_ptr].imshow(masked_pred, cmap="hot", alpha=alpha)
        axes[col_ptr].set_title("SegResNet Prediction", fontsize=11, fontweight="bold", color="darkred")
        axes[col_ptr].axis("off")

    plt.tight_layout()
    return fig


def compute_tumor_volumes(
    mask: Union[np.ndarray, torch.Tensor],
    voxel_size_mm3: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Calculates voxel counts and physical volumes (cm³) for brain tumor regions.

    Args:
        mask: 3D or 4D integer segmentation mask array.
        voxel_size_mm3: Volume per voxel in cubic millimeters.

    Returns:
        Dict[str, Dict[str, float]]: Quantitative metrics per tumor sub-region.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if mask.ndim == 4:
        mask = mask[0]

    # BraTS Tumor Sub-regions:
    # 1 = Enhancing Tumor (ET) / Necrosis
    # 2 = Edema / Whole Tumor
    # 3 = Contrast Enhancing
    total_voxels = np.sum(mask > 0)
    et_voxels = np.sum(mask == 3) + np.sum(mask == 1)
    tc_voxels = np.sum(mask == 1) + np.sum(mask == 3)
    wt_voxels = total_voxels

    # Convert mm³ to cm³ (1 cm³ = 1000 mm³)
    mm3_to_cm3 = voxel_size_mm3 / 1000.0

    return {
        "Whole Tumor (WT)": {
            "voxels": float(wt_voxels),
            "volume_cm3": float(wt_voxels * mm3_to_cm3),
        },
        "Tumor Core (TC)": {
            "voxels": float(tc_voxels),
            "volume_cm3": float(tc_voxels * mm3_to_cm3),
        },
        "Enhancing Tumor (ET)": {
            "voxels": float(et_voxels),
            "volume_cm3": float(et_voxels * mm3_to_cm3),
        },
    }
