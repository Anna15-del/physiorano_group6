"""
PhysioRANO MRI Visualization Tool.

Loads preprocessed 3D MRI volume batches (T1n, T1c, T2w, T2f + Segmentation mask)
and outputs slice visualization plots.

Usage:
    python visualize.py
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from datasets import get_brats_dataloader
from preprocessing import get_val_transforms
from reporting import plot_mri_slices
from utils.logger import setup_logger

logger = setup_logger("PhysioRANO.Visualize")


def create_synthetic_sample() -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a synthetic 4-channel MRI volume and segmentation mask for visualization testing.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Synthetic image tensor [4, 128, 128, 128] and label tensor [1, 128, 128, 128].
    """
    H, W, D = 128, 128, 128
    image = torch.zeros(4, H, W, D)

    # Sphere grid representing head/brain tissue
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        torch.linspace(-1, 1, D),
        indexing="ij",
    )
    dist_from_center = torch.sqrt(x**2 + y**2 + z**2)
    brain_mask = dist_from_center < 0.75

    # Modality synthetic contrasts
    image[0] = torch.where(brain_mask, 0.4 + 0.1 * torch.randn(H, W, D), 0.0)  # T1n
    image[1] = torch.where(brain_mask, 0.6 + 0.1 * torch.randn(H, W, D), 0.0)  # T1c
    image[2] = torch.where(brain_mask, 0.8 + 0.1 * torch.randn(H, W, D), 0.0)  # T2w
    image[3] = torch.where(brain_mask, 0.5 + 0.1 * torch.randn(H, W, D), 0.0)  # T2f

    # Synthetic tumor core
    tumor_mask = dist_from_center < 0.25
    image[1, tumor_mask] += 0.4  # T1c hyperintensity
    image[3, tumor_mask] += 0.3  # T2f hyperintensity

    label = torch.zeros(1, H, W, D, dtype=torch.int64)
    label[0, tumor_mask] = 1

    return image, label


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra-configured visualization task entry point.

    Args:
        cfg (DictConfig): Main Hydra configuration dictionary.
    """
    output_dir = Path(cfg.paths.output_dir) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(cfg.dataset.download.cache_dir)
    val_transforms = get_val_transforms(
        spatial_size=tuple(cfg.dataset.preprocessing.spatial_size),
        pixdim=tuple(cfg.dataset.preprocessing.pixdim),
    )

    dataset, dataloader = get_brats_dataloader(
        data_dir=data_dir,
        transform=val_transforms,
        batch_size=1,
        shuffle=False,
    )

    if len(dataset) > 0:
        logger.info(f"Loading real subject from BraTS dataset ({len(dataset)} subjects available)...")
        batch = next(iter(dataloader))
        image = batch["image"][0]
        label = batch.get("label", None)
        if label is not None:
            label = label[0]
        subject_id = batch.get("subject_id", ["BraTS-GLI-Sample"])[0]
    else:
        logger.info("No local BraTS subject files found. Rendering synthetic multi-modal MRI volume...")
        image, label = create_synthetic_sample()
        subject_id = "Synthetic-BraTS-GLI"

    save_path = output_dir / f"{subject_id}_slice_plot.png"
    result_path = plot_mri_slices(
        image=image,
        label=label,
        subject_id=subject_id,
        save_path=save_path,
        show=False,
    )

    logger.info(f"Successfully generated MRI slice plot: {result_path.resolve()}")


if __name__ == "__main__":
    main()
