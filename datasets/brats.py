"""
MONAI BraTS 2024 GLI Dataset and DataLoader Loader with Train/Val Split Support.

Scans the local filesystem for patient subject directories, automatically locating
the required modalities:
    - t1n.nii.gz (T1 native)
    - t1c.nii.gz (T1 contrast-enhanced)
    - t2w.nii.gz (T2 weighted)
    - t2f.nii.gz (T2 FLAIR)
    - seg.nii.gz (Segmentation ground truth)

Supports deterministic 80/20 train/validation dataset splitting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from monai.data import DataLoader, Dataset
from utils.logger import get_logger

logger = get_logger("PhysioRANO.BraTSDataset")


def find_brats_samples(data_dir: Union[str, Path]) -> List[Dict[str, Union[List[str], str]]]:
    """Scans dataset directory and constructs sample dictionaries matching required file patterns.

    Args:
        data_dir (Union[str, Path]): Root path of the BraTS dataset.

    Returns:
        List[Dict[str, Union[List[str], str]]]: List of sample dictionaries containing modality file paths.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.warning(f"Data directory {data_dir} does not exist.")
        return []

    # Search for all subdirectories or patient folders
    all_subdirs = [p for p in data_dir.rglob("*") if p.is_dir()]
    patient_dirs = all_subdirs if all_subdirs else [data_dir]

    samples = []
    seen_subjects = set()

    for p_dir in patient_dirs:
        # Avoid processing root folder if subdirectories are present
        if p_dir == data_dir and len(all_subdirs) > 0:
            continue

        t1n_files = list(p_dir.glob("*t1n.nii.gz"))
        t1c_files = list(p_dir.glob("*t1c.nii.gz"))
        t2w_files = list(p_dir.glob("*t2w.nii.gz"))
        t2f_files = list(p_dir.glob("*t2f.nii.gz"))
        seg_files = list(p_dir.glob("*seg.nii.gz"))

        # Verify that all modalities are present for this patient
        if t1n_files and t1c_files and t2w_files and t2f_files:
            subject_id = p_dir.name
            if subject_id in seen_subjects:
                continue
            seen_subjects.add(subject_id)

            sample = {
                "subject_id": subject_id,
                "t1n": str(t1n_files[0]),
                "t1c": str(t1c_files[0]),
                "t2w": str(t2w_files[0]),
                "t2f": str(t2f_files[0]),
            }
            if seg_files:
                sample["label"] = str(seg_files[0])

            samples.append(sample)

    logger.info(f"Located {len(samples)} complete BraTS subjects in {data_dir}")
    return samples


class BraTSDataset(Dataset):
    """MONAI Dataset wrapper for BraTS 2024 GLI dataset.

    Maps file paths to MONAI transform pipelines for loading, modality stacking,
    intensity normalization, spatial resampling, and augmentations.
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        samples: Optional[List[Dict]] = None,
        transform=None,
    ):
        """
        Args:
            data_dir (Optional[Union[str, Path]]): Path to directory containing BraTS subject data.
            samples (Optional[List[Dict]]): Pre-filtered list of subject sample dictionaries.
            transform (Optional): MONAI transform pipeline composition.
        """
        if samples is not None:
            self.samples = samples
        elif data_dir is not None:
            self.data_dir = Path(data_dir)
            self.samples = find_brats_samples(self.data_dir)
        else:
            self.samples = []

        super().__init__(data=self.samples, transform=transform)


def split_brats_samples(
    samples: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Splits subject sample list into deterministic train and validation subsets.

    Args:
        samples (List[Dict]): Full list of subject dictionaries.
        train_ratio (float): Fraction of subjects to allocate to training (default 0.8).
        seed (int): Seed for shuffling reproducibility.

    Returns:
        Tuple[List[Dict], List[Dict]]: Train samples list and Validation samples list.
    """
    if not samples:
        return [], []

    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(len(samples))

    num_train = int(len(samples) * train_ratio)
    train_indices = shuffled_indices[:num_train]
    val_indices = shuffled_indices[num_train:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    logger.info(
        f"Dataset split ({train_ratio*100:.0f}%/{100-train_ratio*100:.0f}%): "
        f"{len(train_samples)} training cases, {len(val_samples)} validation cases."
    )
    return train_samples, val_samples


def get_brats_dataloader(
    data_dir: Union[str, Path],
    transform=None,
    batch_size: int = 2,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = False,
) -> Tuple[BraTSDataset, DataLoader]:
    """Factory function returning MONAI BraTS Dataset and DataLoader.

    Args:
        data_dir (Union[str, Path]): Directory path of BraTS data.
        transform (Optional): MONAI transform pipeline.
        batch_size (int): DataLoader batch size.
        num_workers (int): DataLoader parallel worker processes.
        shuffle (bool): Whether to shuffle samples during iteration.
        pin_memory (bool): Whether to pin memory for CUDA transfers.

    Returns:
        Tuple[BraTSDataset, DataLoader]: Initialized MONAI Dataset and DataLoader.
    """
    dataset = BraTSDataset(data_dir=data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    return dataset, dataloader


def get_train_val_dataloaders(
    data_dir: Union[str, Path],
    train_transform=None,
    val_transform=None,
    train_ratio: float = 0.8,
    batch_size: int = 2,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[BraTSDataset, DataLoader, BraTSDataset, DataLoader]:
    """Factory function creating split 80% Train and 20% Validation Datasets and DataLoaders.

    Args:
        data_dir (Union[str, Path]): Path to dataset root directory.
        train_transform (Optional): Preprocessing and augmentation transform for training.
        val_transform (Optional): Preprocessing transform for validation.
        train_ratio (float): Split ratio (default 0.8).
        batch_size (int): Batch size for training.
        num_workers (int): Parallel worker processes.
        seed (int): Random split seed.

    Returns:
        Tuple[BraTSDataset, DataLoader, BraTSDataset, DataLoader]:
            (train_dataset, train_loader, val_dataset, val_loader)
    """
    all_samples = find_brats_samples(data_dir)
    train_samples, val_samples = split_brats_samples(all_samples, train_ratio=train_ratio, seed=seed)

    train_dataset = BraTSDataset(samples=train_samples, transform=train_transform)
    val_dataset = BraTSDataset(samples=val_samples, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Validation evaluated sample-by-sample
        num_workers=num_workers,
        shuffle=False,
    )

    return train_dataset, train_loader, val_dataset, val_loader
