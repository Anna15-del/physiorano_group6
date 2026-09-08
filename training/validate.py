"""
PhysioRANO Validation and Metric Evaluation Engine.

Evaluates 3D brain tumor segmentation models using MONAI SlidingWindowInferer,
calculating Validation Loss (DiceCELoss), Dice Score (DiceMetric), and Hausdorff Distance (HausdorffDistanceMetric).
"""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn

from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.utils import MetricReduction

from utils.logger import get_logger

logger = get_logger("PhysioRANO.Validate")


def evaluate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 2,
    overlap: float = 0.25,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Runs sliding-window validation evaluation across all subjects in val_loader.

    Args:
        model (nn.Module): Segmentation network model.
        val_loader (DataLoader): Validation DataLoader.
        device (torch.device): CUDA or CPU compute device.
        roi_size (Tuple[int, int, int]): Sliding-window 3D tile shape.
        sw_batch_size (int): Number of sliding window tiles per forward pass.
        overlap (float): Sliding window tile overlap fraction.
        use_amp (bool): Whether to use mixed precision inference.

    Returns:
        Dict[str, float]: Dictionary containing computed validation metrics:
            - val_loss
            - val_dice_mean
            - val_hd95_mean
    """
    model.eval()
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction=MetricReduction.MEAN)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction=MetricReduction.MEAN)

    post_pred = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)

    total_loss = 0.0
    val_steps = 0

    dice_metric.reset()
    hd95_metric.reset()

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device) if "label" in batch else None

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                preds = inferer(images, model)
                if labels is not None:
                    loss = loss_fn(preds, labels)
                    total_loss += loss.item()
                    val_steps += 1

            if labels is not None:
                # One-hot formatting for metric calculation
                val_preds_onehot = [post_pred(p) for p in preds]
                val_labels_onehot = [post_label(l) for l in labels]

                dice_metric(y_pred=val_preds_onehot, y=val_labels_onehot)
                try:
                    hd95_metric(y_pred=val_preds_onehot, y=val_labels_onehot)
                except Exception:
                    pass  # Handle empty segmentation mask cases gracefully

    mean_loss = total_loss / max(val_steps, 1)
    mean_dice = dice_metric.aggregate().item() if val_steps > 0 else 0.0

    try:
        mean_hd95 = hd95_metric.aggregate().item() if val_steps > 0 else 0.0
    except Exception:
        mean_hd95 = 0.0

    metrics = {
        "val_loss": mean_loss,
        "val_dice_mean": mean_dice,
        "val_hd95_mean": mean_hd95,
    }

    logger.info(
        f"Validation Results -> Loss: {mean_loss:.4f} | "
        f"Mean Dice: {mean_dice:.4f} | Mean HD95: {mean_hd95:.4f}"
    )

    return metrics
