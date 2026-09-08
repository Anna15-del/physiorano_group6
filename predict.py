"""
PhysioRANO Single Subject Inference CLI Tool.

Loads a trained model checkpoint and performs 3D brain tumor segmentation prediction
on a target subject volume, saving output NIfTI predictions.

Usage:
    python predict.py
"""

from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig

from datasets import get_brats_dataloader
from models import get_segmentation_model
from preprocessing import get_val_transforms
from inference import predict_subject, save_nifti_prediction
from utils.logger import setup_logger

logger = setup_logger("PhysioRANO.PredictCLI")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra prediction task entrypoint.

    Args:
        cfg (DictConfig): Hydra configuration tree.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    output_dir = Path(cfg.paths.output_dir) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Model Checkpoint
    model = get_segmentation_model(in_channels=4, out_channels=4).to(device)
    checkpoint_path = Path(cfg.paths.checkpoint_dir) / "best_model.pth"

    if checkpoint_path.exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Using uninitialized model.")

    # 2. Load Subject Data
    data_dir = Path(cfg.dataset.download.cache_dir)
    val_transform = get_val_transforms(
        spatial_size=tuple(cfg.dataset.preprocessing.spatial_size),
        pixdim=tuple(cfg.dataset.preprocessing.pixdim),
    )

    dataset, dataloader = get_brats_dataloader(
        data_dir=data_dir,
        transform=val_transform,
        batch_size=1,
        shuffle=False,
    )

    if len(dataset) > 0:
        batch = next(iter(dataloader))
        image = batch["image"][0]
        subject_id = batch.get("subject_id", ["BraTS-Subject"])[0]
        ref_nifti = batch.get("t1c", [None])[0]
    else:
        logger.info("No local BraTS subject found. Generating synthetic subject volume for prediction test...")
        from visualize import create_synthetic_sample
        image, _ = create_synthetic_sample()
        subject_id = "Synthetic-BraTS-Subject"
        ref_nifti = None

    # 3. Predict & Export
    logger.info(f"Running 3D sliding-window prediction for {subject_id}...")
    pred_mask = predict_subject(
        model=model,
        image_tensor=image,
        device=device,
        roi_size=tuple(cfg.training.inference.roi_size),
        sw_batch_size=cfg.training.inference.sw_batch_size,
        overlap=cfg.training.inference.overlap,
        use_amp=cfg.training.use_amp,
    )

    save_path = output_dir / f"{subject_id}_predict_seg.nii.gz"
    saved_file = save_nifti_prediction(
        pred_mask=pred_mask,
        output_path=save_path,
        reference_nifti_path=ref_nifti,
    )

    logger.info(f"Prediction successful! Output written to {saved_file.resolve()}")


if __name__ == "__main__":
    main()
