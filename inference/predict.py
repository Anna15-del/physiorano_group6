"""
PhysioRANO 3D Brain Tumor Segmentation Prediction & Inference Engine.

Runs 3D sliding-window inference on subject MRI volumes using trained model checkpoints
(e.g., best_model.pth) and exports segmentation prediction masks in NIfTI format.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import nibabel as nib
import numpy as np
import torch

from monai.inferers import SlidingWindowInferer
from models import get_segmentation_model
from utils.logger import get_logger

logger = get_logger("PhysioRANO.Predict")


def predict_subject(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 2,
    overlap: float = 0.25,
    use_amp: bool = True,
) -> np.ndarray:
    """Performs 3D sliding window inference on a 4-channel MRI image tensor.

    Args:
        model (torch.nn.Module): Trained segmentation model.
        image_tensor (torch.Tensor): Input volume tensor of shape [4, H, W, D] or [1, 4, H, W, D].
        device (torch.device): Compute device.
        roi_size (Tuple[int, int, int]): Sliding window ROI dimensions.
        sw_batch_size (int): Sliding window batch size.
        overlap (float): Sliding window tile overlap.
        use_amp (bool): Whether to use mixed precision.

    Returns:
        np.ndarray: Predicted 3D segmentation class mask of shape [H, W, D].
    """
    model.eval()
    if image_tensor.ndim == 4:
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dim [1, 4, H, W, D]

    image_tensor = image_tensor.to(device)
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = inferer(image_tensor, model)
            preds = torch.argmax(logits, dim=1).squeeze(0)  # Argmax -> [H, W, D]

    return preds.cpu().numpy().astype(np.int16)


def save_nifti_prediction(
    pred_mask: np.ndarray,
    output_path: Union[str, Path],
    reference_nifti_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Saves a 3D segmentation prediction numpy array as a NIfTI file (.nii.gz).

    Args:
        pred_mask (np.ndarray): 3D integer numpy array [H, W, D].
        output_path (Union[str, Path]): Target output file path.
        reference_nifti_path (Optional[Union[str, Path]]): Reference NIfTI file to copy affine matrix from.

    Returns:
        Path: Path to saved NIfTI prediction file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if reference_nifti_path and Path(reference_nifti_path).exists():
        ref_nifti = nib.load(reference_nifti_path)
        affine = ref_nifti.affine
        header = ref_nifti.header
    else:
        affine = np.eye(4)
        header = None

    nifti_img = nib.Nifti1Image(pred_mask, affine=affine, header=header)
    nib.save(nifti_img, output_path)
    logger.info(f"Saved 3D prediction NIfTI mask to {output_path}")

    return output_path
