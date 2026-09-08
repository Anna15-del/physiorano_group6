"""
PhysioRANO 3D Segmentation Training CLI Tool.

Loads Hydra configurations, initiates 80/20 train/validation dataset splitting,
and executes 3D SegResNet brain tumor segmentation model training with AMP,
TensorBoard logging, learning rate scheduling, early stopping, and checkpoint saving.

Usage:
    python train.py
"""

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from training import train_segmentation_model
from utils.logger import setup_logger

logger = setup_logger("PhysioRANO.TrainCLI")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra training task entrypoint.

    Args:
        cfg (DictConfig): Main Hydra configuration hierarchy.
    """
    logger.info(f"=== Initiating PhysioRANO Phase 3 Training Pipeline v{cfg.project.version} ===")

    data_dir = Path(cfg.dataset.download.cache_dir)
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    report_dir = Path(cfg.paths.report_dir)
    log_dir = Path(cfg.paths.log_dir)

    best_checkpoint = train_segmentation_model(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        report_dir=report_dir,
        log_dir=log_dir,
        epochs=cfg.training.epochs,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        batch_size=cfg.dataset.loader.batch_size,
        train_ratio=cfg.training.train_val_split,
        use_amp=cfg.training.use_amp,
        max_grad_norm=cfg.training.max_grad_norm,
        patience=cfg.training.early_stopping.patience,
        seed=cfg.project.seed,
        spatial_size=tuple(cfg.dataset.preprocessing.spatial_size),
        pixdim=tuple(cfg.dataset.preprocessing.pixdim),
    )

    logger.info(f"PhysioRANO Phase 3 Training Successful! Model saved to: {best_checkpoint}")


if __name__ == "__main__":
    main()
