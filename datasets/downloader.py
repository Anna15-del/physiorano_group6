"""
Hugging Face Dataset Downloader for BraTS 2024.

Handles automated downloading of the BraTS-GLI dataset from Hugging Face Hub
(Aff77/BraTS-2024-Complete) into a local cache/data directory.
"""

import os
from pathlib import Path
from typing import Optional
from huggingface_hub import snapshot_download
from utils.logger import get_logger

logger = get_logger("PhysioRANO.Downloader")


def download_brats_dataset(
    repo_id: str = "Aff77/BraTS-2024-Complete",
    subfolder: str = "BraTS-GLI",
    cache_dir: Optional[Path] = None,
) -> Path:
    """Downloads the BraTS 2024 GLI dataset from Hugging Face Hub.

    Args:
        repo_id (str): Hugging Face repository identifier.
        subfolder (str): Dataset subfolder name within the repository.
        cache_dir (Optional[Path]): Directory where the dataset will be saved.

    Returns:
        Path: Local path to the downloaded BraTS-GLI dataset directory.
    """
    if cache_dir is None:
        cache_dir = Path("./data/brats2024_gli").resolve()
    else:
        cache_dir = Path(cache_dir).resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Checking dataset repository: {repo_id} (Target subfolder: {subfolder})")
    logger.info(f"Target local directory: {cache_dir}")

    try:
        # Download files matching NIfTI patterns in the specified subfolder
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=cache_dir,
            allow_patterns=[f"*{subfolder}*/*.nii.gz", "*.nii.gz"],
            ignore_patterns=["*.git*", "*.md"],
        )
        logger.info(f"Successfully verified/downloaded BraTS dataset to {downloaded_path}")
        return Path(downloaded_path)
    except Exception as e:
        logger.error(f"Failed to download BraTS dataset from Hugging Face: {e}")
        logger.info(
            "If automatic download fails (e.g. due to missing internet or HF auth), "
            "ensure local data directory is populated manually."
        )
        return cache_dir
