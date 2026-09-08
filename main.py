"""
PhysioRANO Main Entry Point.

Loads Hydra configurations, sets up logging, initializes random seed,
and verifies Phase 2 MONAI preprocessing transform pipelines (training with augmentations
and validation without augmentations) alongside BraTS dataset loaders.
"""

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.logger import setup_logger, get_logger
from utils.helpers import set_seed
from datasets import download_brats_dataset, get_brats_dataloader, get_burdenko_dataloader
from preprocessing import get_train_transforms, get_val_transforms


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra main execution entrypoint.

    Args:
        cfg (DictConfig): Parsed Hydra configuration hierarchy object.
    """
    # 1. Initialize Logger
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger = setup_logger(
        name="PhysioRANO",
        log_file=output_dir / "physiorano.log",
    )

    logger.info(f"=== Starting PhysioRANO Phase 2 Pipeline v{cfg.project.version} ===")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 2. Set Seed for Reproducibility
    set_seed(cfg.project.seed)
    logger.info(f"Random seed set to {cfg.project.seed}")

    # 3. Build MONAI Preprocessing Pipelines
    spatial_size = tuple(cfg.dataset.preprocessing.spatial_size)
    pixdim = tuple(cfg.dataset.preprocessing.pixdim)

    logger.info(f"Building Phase 2 MONAI Preprocessing Pipelines (Target Size: {spatial_size}, Spacing: {pixdim})...")
    train_transforms = get_train_transforms(spatial_size=spatial_size, pixdim=pixdim)
    val_transforms = get_val_transforms(spatial_size=spatial_size, pixdim=pixdim)

    logger.info("Validation transforms initialized (NIfTI Read -> Channel First -> Modal Concatenation -> Spacing -> Crop -> Z-Score Normalize -> Resize).")
    logger.info("Training transforms initialized (Validation pipeline + Spatial Flips, 3D Rotation, Gaussian Noise, & Intensity Scaling).")

    # 4. Verify Dataset Configuration
    if cfg.dataset.name == "brats2024_gli":
        logger.info("Initializing BraTS 2024 GLI Dataset pipeline...")
        data_dir = Path(cfg.dataset.download.cache_dir)

        # Trigger automatic download if enabled
        if cfg.dataset.download.auto_download:
            logger.info("Checking Hugging Face automatic download...")
            download_brats_dataset(
                repo_id=cfg.dataset.repo_id,
                subfolder=cfg.dataset.subfolder,
                cache_dir=data_dir,
            )

        # Initialize MONAI Dataset and DataLoader with Validation Transforms
        dataset, dataloader = get_brats_dataloader(
            data_dir=data_dir,
            transform=val_transforms,
            batch_size=cfg.dataset.loader.batch_size,
            num_workers=cfg.dataset.loader.num_workers,
            shuffle=cfg.dataset.loader.shuffle,
            pin_memory=cfg.dataset.loader.pin_memory,
        )

        logger.info(f"BraTS Dataset initialized with Phase 2 transforms. Total subjects found: {len(dataset)}")
        logger.info(f"DataLoader initialized with batch size {cfg.dataset.loader.batch_size}")

    elif cfg.dataset.name == "burdenko_gbm_progression":
        logger.info("Initializing Burdenko Longitudinal Dataset template...")
        dataset, dataloader = get_burdenko_dataloader(
            data_dir=cfg.dataset.data_dir,
            batch_size=cfg.dataset.loader.batch_size,
        )
        logger.info(f"Burdenko Template Dataset initialized. Total samples: {len(dataset)}")

    else:
        logger.warning(f"Unknown dataset configuration: {cfg.dataset.name}")

    logger.info("PhysioRANO Phase 2 Preprocessing & Data Pipeline initialized successfully.")


if __name__ == "__main__":
    main()
