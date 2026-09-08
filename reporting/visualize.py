"""
PhysioRANO MRI Visualization Module.

Renders multi-modal MRI 3D tensor slices (T1n, T1c, T2w, T2f) alongside ground truth
segmentation masks across axial, coronal, and sagittal orthogonal views.
"""

from pathlib import Path
from typing import Optional, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_mri_slices(
    image: Union[np.ndarray, torch.Tensor],
    label: Optional[Union[np.ndarray, torch.Tensor]] = None,
    subject_id: str = "Subject",
    slice_idx: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Path:
    """Plots and saves multi-modal MRI slices (T1n, T1c, T2w, T2f) and segmentation mask.

    Args:
        image (Union[np.ndarray, torch.Tensor]): 4-channel MRI volume of shape [4, H, W, D].
        label (Optional[Union[np.ndarray, torch.Tensor]]): Segmentation mask of shape [1, H, W, D] or [H, W, D].
        subject_id (str): Subject/patient identification string.
        slice_idx (Optional[int]): Depth slice index to render. If None, extracts the middle slice.
        save_path (Optional[Union[str, Path]]): File path to save output figure PNG.
        show (bool): Whether to display figure interactively.

    Returns:
        Path: Path to saved output image file.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if label is not None and isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()

    # Ensure shape is [4, H, W, D]
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    num_channels, H, W, D = image.shape
    if slice_idx is None:
        slice_idx = D // 2

    # Modality labels matching 4 concatenated channels
    modality_names = ["T1n", "T1c", "T2w", "T2f"]

    num_cols = num_channels + (1 if label is not None else 0)
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    fig.suptitle(f"PhysioRANO - Subject: {subject_id} (Axial Slice Z={slice_idx})", fontsize=14, fontweight="bold")

    if num_cols == 1:
        axes = [axes]

    for c in range(min(num_channels, 4)):
        slice_2d = image[c, :, :, slice_idx]
        axes[c].imshow(slice_2d, cmap="gray")
        axes[c].set_title(f"Modality: {modality_names[c] if c < len(modality_names) else f'Ch {c}'}")
        axes[c].axis("off")

    if label is not None:
        seg_2d = label[0, :, :, slice_idx] if label.ndim == 4 else label[:, :, slice_idx]
        axes[num_channels].imshow(image[1, :, :, slice_idx], cmap="gray")  # Background T1c
        masked_seg = np.ma.masked_where(seg_2d == 0, seg_2d)
        axes[num_channels].imshow(masked_seg, cmap="jet", alpha=0.6)
        axes[num_channels].set_title("Segmentation Mask Overlay")
        axes[num_channels].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return save_path
    else:
        if show:
            plt.show()
        plt.close(fig)
        return Path("visualization.png")
