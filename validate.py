"""
PhysioRANO Model Validation Execution Tool.

Loads a trained model checkpoint (e.g. best_model.pth) and runs validation metrics evaluation
(Loss, Dice Score, HD95) over the validation split of the BraTS dataset.

Usage:
    python validate.py
"""

from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig

from datasets import get_train_val_dataloaders
from models import get_segmentation_model
from preprocessing import get_val_transforms
from training.validate import evaluate_model
from utils.logger import setup_logger

logger = setup_logger("PhysioRANO.ValidateCLI")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra validation evaluation task entrypoint.

    Args:
        cfg (DictConfig): Main Hydra config hierarchy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # 1. Load Data
    data_dir = Path(cfg.dataset.download.cache_dir)
    val_transform = get_val_transforms(
        spatial_size=tuple(cfg.dataset.preprocessing.spatial_size),
        pixdim=tuple(cfg.dataset.preprocessing.pixdim),
    )

    _, _, val_dataset, val_loader = get_train_val_dataloaders(
        data_dir=data_dir,
        val_transform=val_transform,
        train_ratio=cfg.training.train_val_split,
        seed=cfg.project.seed,
    )

    logger.info(f"Loaded validation set: {len(val_dataset)} subjects.")

    # 2. Instantiate Model & Load Weights
    model = get_segmentation_model(in_channels=4, out_channels=4).to(device)

    checkpoint_path = Path(cfg.paths.checkpoint_dir) / "best_model.pth"
    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Evaluating uninitialized model weights.")

    # 3. Execute Validation Evaluation
    metrics = evaluate_model(
        model=model,
        val_loader=val_loader,
        device=device,
        roi_size=tuple(cfg.training.inference.roi_size),
        sw_batch_size=cfg.training.inference.sw_batch_size,
        overlap=cfg.training.inference.overlap,
        use_amp=cfg.training.use_amp,
    )

    logger.info("Validation complete.")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
