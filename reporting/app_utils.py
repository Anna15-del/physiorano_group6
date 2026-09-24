"""
PhysioRANO Streamlit Application Visualization Utilities.

Provides figure generation and quantitative volumetric calculation functions for Streamlit dashboard.
"""

from typing import Dict, List, Optional, Tuple, Union
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


def render_cgan_comparison_slice(
    base_image: Union[np.ndarray, torch.Tensor],
    future_image: Union[np.ndarray, torch.Tensor],
    plane: str = "axial",
    slice_idx: Optional[int] = None,
    modality_idx: int = 1,
) -> plt.Figure:
    """Renders a side-by-side comparison between baseline and cGAN forecasted future 3D MRI scans.

    Args:
        base_image: Baseline 4-channel MRI volume [4, H, W, D] or [1, 4, H, W, D].
        future_image: Synthesized future 4-channel MRI volume [4, H, W, D] or [1, 4, H, W, D].
        plane: Orientation plane ('axial', 'coronal', 'sagittal').
        slice_idx: Slice index along selected orientation plane.
        modality_idx: Channel index (0=T1n, 1=T1c, 2=T2w, 3=T2f).

    Returns:
        plt.Figure: Matplotlib figure with baseline, future, and difference map panels.
    """
    if isinstance(base_image, torch.Tensor):
        base_image = base_image.detach().cpu().numpy()
    if isinstance(future_image, torch.Tensor):
        future_image = future_image.detach().cpu().numpy()

    if base_image.ndim == 5:
        base_image = base_image[0]
    if future_image.ndim == 5:
        future_image = future_image[0]

    _, H, W, D = base_image.shape

    if plane == "axial":
        s_idx = slice_idx if slice_idx is not None else D // 2
        extract = lambda data, c: data[c, :, :, s_idx]
    elif plane == "coronal":
        s_idx = slice_idx if slice_idx is not None else W // 2
        extract = lambda data, c: data[c, :, s_idx, :]
    else:  # sagittal
        s_idx = slice_idx if slice_idx is not None else H // 2
        extract = lambda data, c: data[c, s_idx, :, :]

    modality_names = ["T1n (Native)", "T1c (Contrast-Enhancing)", "T2w (Weighted)", "T2f (FLAIR)"]
    m_name = modality_names[modality_idx] if modality_idx < len(modality_names) else f"Modality {modality_idx}"

    base_slice = extract(base_image, modality_idx)
    future_slice = extract(future_image, modality_idx)
    diff_slice = future_slice - base_slice

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor="#0f172a")

    # Panel 1: Baseline Scan (t0)
    axes[0].imshow(base_slice, cmap="gray")
    axes[0].set_title(f"Baseline Scan (t₀)\n{m_name}", fontsize=11, fontweight="bold", color="#38bdf8")
    axes[0].axis("off")

    # Panel 2: cGAN Synthesized Future Scan (t1)
    axes[1].imshow(future_slice, cmap="gray")
    axes[1].set_title(f"cGAN Forecasted Scan (t₁)\n{m_name}", fontsize=11, fontweight="bold", color="#818cf8")
    axes[1].axis("off")

    # Panel 3: Change Heatmap (Delta = Future - Baseline)
    vmax = max(0.15, float(np.percentile(np.abs(diff_slice), 99)))
    im_diff = axes[2].imshow(diff_slice, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[2].set_title(f"Predicted Focal Change (Δ)\n[Blue: Regression | Red: Growth]", fontsize=11, fontweight="bold", color="#34d399")
    axes[2].axis("off")

    cbar = fig.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors="#cbd5e1")

    plt.tight_layout()
    return fig


def plot_forecast_trajectory(trajectory_points: List[Dict], selected_days: float = 90.0) -> plt.Figure:
    """Plots longitudinal tumor sub-region volumetric growth/regression trajectories.

    Args:
        trajectory_points (List[Dict]): List of point dicts with keys 'days', 'wt_vol_cm3', 'tc_vol_cm3', 'et_vol_cm3'.
        selected_days (float): Highlighted target forecasting horizon.

    Returns:
        plt.Figure: Matplotlib trajectory curve figure.
    """
    days = [pt["days"] for pt in trajectory_points]
    wt = [pt["wt_vol_cm3"] for pt in trajectory_points]
    tc = [pt["tc_vol_cm3"] for pt in trajectory_points]
    et = [pt["et_vol_cm3"] for pt in trajectory_points]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")

    ax.plot(days, wt, marker="o", linewidth=2.5, color="#38bdf8", label="Whole Tumor (WT)")
    ax.plot(days, tc, marker="s", linewidth=2.2, color="#fbbf24", label="Tumor Core (TC)")
    ax.plot(days, et, marker="^", linewidth=2.2, color="#f43f5e", label="Enhancing Tumor (ET)")

    # Highlight forecast point
    ax.axvline(x=selected_days, color="#a855f7", linestyle="--", linewidth=1.8, label=f"Forecast Horizon (t₁ = {selected_days:.0f}d)")

    ax.set_title("Longitudinal Volumetric Trajectory Forecast (t₀ → 180 Days)", fontsize=12, fontweight="bold", color="#f8fafc", pad=12)
    ax.set_xlabel("Elapsed Time (Days Post-Baseline)", fontsize=10, fontweight="bold", color="#cbd5e1")
    ax.set_ylabel("Tumor Volume (cm³)", fontsize=10, fontweight="bold", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle=":", alpha=0.35, color="#64748b")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=9)

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    return fig

