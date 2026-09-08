"""
PhysioRANO 3D Brain Tumor Segmentation Training Engine.

Handles full training execution:
    - 80/20 Train/Validation DataLoader iteration across all patient subjects
    - MONAI SegResNet model initialization
    - Combined DiceCELoss (Dice + CrossEntropy Loss)
    - Automatic Mixed Precision (AMP) with GradScaler
    - Cosine Annealing Learning Rate Scheduling
    - Gradient Clipping & Early Stopping
    - TensorBoard logging
    - Checkpoint persistence (best_model.pth & latest_checkpoint.pth)
    - Loss & Dice metric curve visualization generation
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from monai.losses import DiceCELoss
from datasets import get_train_val_dataloaders
from models import get_segmentation_model
from preprocessing import get_train_transforms, get_val_transforms
from training.validate import evaluate_model
from utils.helpers import set_seed
from utils.logger import get_logger

logger = get_logger("PhysioRANO.Train")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_dices: List[float],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Generates and saves Training/Validation Loss and Dice metric curve plots.

    Args:
        train_losses (List[float]): Training loss per epoch.
        val_losses (List[float]): Validation loss per epoch.
        val_dices (List[float]): Validation mean Dice score per epoch.
        output_dir (Path): Directory where output PNG plots will be saved.

    Returns:
        Tuple[Path, Path]: Saved paths for (loss_curve_path, dice_curve_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", color="red", linestyle="--", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("DiceCELoss")
    plt.title("PhysioRANO - Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = output_dir / "loss_curve.png"
    plt.savefig(loss_path, bbox_inches="tight", dpi=150)
    plt.close()

    # 2. Dice Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_dices, label="Validation Mean Dice", color="green", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("PhysioRANO - Validation Dice Score Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    dice_path = output_dir / "dice_curve.png"
    plt.savefig(dice_path, bbox_inches="tight", dpi=150)
    plt.close()

    return loss_path, dice_path


def train_segmentation_model(
    data_dir: Path,
    checkpoint_dir: Path,
    report_dir: Path,
    log_dir: Path,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    batch_size: int = 2,
    train_ratio: float = 0.8,
    use_amp: bool = True,
    max_grad_norm: float = 1.0,
    patience: int = 10,
    seed: int = 42,
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    """Executes full training pipeline for 3D brain tumor segmentation.

    Returns:
        Path: Path to saved best model checkpoint (best_model.pth).
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initiating training pipeline on compute device: {device}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    # 1. Transforms & Dataloaders
    train_transform = get_train_transforms(spatial_size=spatial_size, pixdim=pixdim)
    val_transform = get_val_transforms(spatial_size=spatial_size, pixdim=pixdim)

    train_dataset, train_loader, val_dataset, val_loader = get_train_val_dataloaders(
        data_dir=data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        train_ratio=train_ratio,
        batch_size=batch_size,
        seed=seed,
    )

    logger.info(f"Training split: {len(train_dataset)} subjects | Validation split: {len(val_dataset)} subjects.")

    # 2. Model, Loss, Optimizer, & AMP Scaler
    model = get_segmentation_model(in_channels=4, out_channels=4).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    best_val_dice = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = checkpoint_dir / "best_model.pth"

    train_epoch_losses: List[float] = []
    val_epoch_losses: List[float] = []
    val_epoch_dices: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = batch["image"].to(device)
            labels = batch["label"].to(device) if "label" in batch else None

            if labels is None:
                continue

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                preds = model(images)
                loss = loss_fn(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            train_batches += 1

        avg_train_loss = running_train_loss / max(train_batches, 1)
        scheduler.step()

        # 3. Validation Evaluation
        val_metrics = evaluate_model(
            model=model,
            val_loader=val_loader,
            device=device,
            roi_size=spatial_size,
            use_amp=use_amp,
        )

        val_loss = val_metrics["val_loss"]
        val_dice = val_metrics["val_dice_mean"]
        val_hd95 = val_metrics["val_hd95_mean"]

        train_epoch_losses.append(avg_train_loss)
        val_epoch_losses.append(val_loss)
        val_epoch_dices.append(val_dice)

        # TensorBoard Logs
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Dice/Val_Mean", val_dice, epoch)
        writer.add_scalar("HD95/Val_Mean", val_hd95, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] -> Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val HD95: {val_hd95:.4f}"
        )

        # 4. Checkpoint & Early Stopping
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_loss": val_loss,
                },
                best_model_path,
            )
            logger.info(f"--> Saved new BEST checkpoint (Dice: {val_dice:.4f}) to {best_model_path}")
        else:
            patience_counter += 1

        # Always save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_dir / "latest_checkpoint.pth",
        )

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs (Patience={patience}).")
            break

    writer.close()

    # 5. Generate Curves & Summary Reports
    loss_curve_path, dice_curve_path = plot_training_curves(
        train_losses=train_epoch_losses,
        val_losses=val_epoch_losses,
        val_dices=val_epoch_dices,
        output_dir=report_dir,
    )

    summary_data = {
        "best_val_dice": best_val_dice,
        "best_val_loss": best_val_loss,
        "total_epochs_trained": len(train_epoch_losses),
        "training_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "best_checkpoint": str(best_model_path.resolve()),
        "loss_curve": str(loss_curve_path.resolve()),
        "dice_curve": str(dice_curve_path.resolve()),
    }

    with open(report_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    logger.info(f"Training completed! Best Model Checkpoint saved at: {best_model_path.resolve()}")
    return best_model_path
