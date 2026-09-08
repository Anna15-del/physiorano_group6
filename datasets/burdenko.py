"""
Burdenko-GBM-Progression Longitudinal Clinical Dataset Loader.

Parses patient demographics, molecular status (IDH1/2, MGMT), treatment timeline
(Surgery, RT start/end dates), longitudinal follow-up scan dates, and ground truth
recurrence outcomes (pseudo progression vs. progression vs. response/stable).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from monai.data import DataLoader, Dataset
from utils.logger import get_logger

logger = get_logger("PhysioRANO.BurdenkoDataset")


def load_burdenko_clinical_df(
    csv_path: Union[str, Path] = "data/burdenko_gbm/burdenko_clinical.csv"
) -> pd.DataFrame:
    """Loads and preprocesses the Burdenko clinical CSV dataset.

    Args:
        csv_path (Union[str, Path]): Path to Burdenko clinical CSV file.

    Returns:
        pd.DataFrame: Cleaned pandas DataFrame with extracted numerical features,
                      temporal deltas (days), and encoded outcome labels.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        # Search fallback relative to repo root
        root_dir = Path(__file__).parent.parent
        csv_path = root_dir / "data" / "burdenko_gbm" / "burdenko_clinical.csv"

    if not csv_path.exists():
        logger.warning(f"Burdenko clinical CSV not found at {csv_path}. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded Burdenko clinical CSV with {len(df)} patient records.")

    # Standardize column whitespace
    df.columns = [c.strip() for c in df.columns]

    # Convert Age to numeric
    df["AgeAtStudyDate"] = pd.to_numeric(df["AgeAtStudyDate"], errors="coerce").fillna(df["AgeAtStudyDate"].median())

    # Encode Sex (1 for M, 0 for F)
    df["Sex_encoded"] = df["Sex"].astype(str).str.upper().map({"M": 1, "F": 0}).fillna(0).astype(int)

    # Encode IDH1/2 (1 for positive, 0 for negative, 0.5 for unknown/missing)
    def encode_idh(val):
        s = str(val).strip().lower()
        if "pos" in s:
            return 1.0
        elif "neg" in s:
            return 0.0
        return 0.5

    df["IDH_encoded"] = df["IDH1/2"].apply(encode_idh)

    # Encode MGMT (1 for positive, 0 for negative, 0.5 for unknown/missing)
    def encode_mgmt(val):
        s = str(val).strip().lower()
        if "pos" in s:
            return 1.0
        elif "neg" in s:
            return 0.0
        return 0.5

    df["MGMT_encoded"] = df["MGMT"].apply(encode_mgmt)

    # Parse key treatment timeline dates to compute temporal deltas (in days)
    date_cols = [
        "AnonymStudyDate", "AnonymDate of surgery", "AnonymStart of RT",
        "AnonymEnd of RT", "AnonymDate of topometric MRI", "Anonym1st_fup",
        "Anonym2nd_fup", "Anonym3rd_fup", "Anonym4th_fup", "Anonym5th_fup", "AnonymDeath"
    ]
    for col in date_cols:
        if col in df.columns:
            df[col + "_dt"] = pd.to_datetime(df[col], errors="coerce")

    # Temporal delta: Days between Surgery and 1st follow-up
    if "AnonymDate of surgery_dt" in df.columns and "Anonym1st_fup_dt" in df.columns:
        df["days_surgery_to_fup1"] = (df["Anonym1st_fup_dt"] - df["AnonymDate of surgery_dt"]).dt.days.fillna(90)
    else:
        df["days_surgery_to_fup1"] = 90.0

    # Temporal delta: Days between RT End and 1st follow-up
    if "AnonymEnd of RT_dt" in df.columns and "Anonym1st_fup_dt" in df.columns:
        df["days_rt_end_to_fup1"] = (df["Anonym1st_fup_dt"] - df["AnonymEnd of RT_dt"]).dt.days.fillna(45)
    else:
        df["days_rt_end_to_fup1"] = 45.0

    # Number of active follow-up scans recorded
    fup_cols = [c for c in df.columns if c.startswith("Response_") and c.endswith("_fup")]
    df["num_followup_scans"] = df[fup_cols].apply(lambda row: row.notna().sum(), axis=1)

    # Outcome Label Extraction:
    # 0 = Response / Stable Disease
    # 1 = True Tumor Progression
    # 2 = Pseudoprogression (Treatment Effect)
    def extract_recurrence_target(row):
        all_responses = " ".join([str(row[c]).lower() for c in fup_cols if pd.notna(row[c])])
        if "pseudo" in all_responses:
            return 2  # Pseudoprogression
        elif "progression" in all_responses:
            return 1  # True Progression
        return 0  # Response / Stable Disease

    df["target_label"] = df.apply(extract_recurrence_target, axis=1)
    df["target_name"] = df["target_label"].map({0: "Response / Stable", 1: "True Progression", 2: "Pseudoprogression"})

    return df


class BurdenkoLongitudinalDataset(Dataset):
    """MONAI Dataset wrapper for Burdenko longitudinal patient records."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/burdenko_gbm",
        csv_filename: str = "burdenko_clinical.csv",
        transform=None,
    ):
        """
        Args:
            data_dir (Union[str, Path]): Root path of Burdenko dataset.
            csv_filename (str): Name of clinical CSV file.
            transform (Optional): MONAI transform pipeline.
        """
        self.data_dir = Path(data_dir)
        csv_path = self.data_dir / csv_filename
        self.df = load_burdenko_clinical_df(csv_path)

        # Convert DataFrame rows into MONAI sample dictionary list
        self.samples = self.df.to_dict(orient="records") if not self.df.empty else []
        super().__init__(data=self.samples, transform=transform)

    def __len__(self) -> int:
        return len(self.samples)


def get_burdenko_dataloader(
    data_dir: Union[str, Path] = "data/burdenko_gbm",
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle: bool = True,
) -> Tuple[BurdenkoLongitudinalDataset, DataLoader]:
    """Factory function returning Burdenko Dataset and DataLoader.

    Args:
        data_dir (Union[str, Path]): Root dataset directory.
        batch_size (int): Batch size.
        num_workers (int): Parallel worker count.
        shuffle (bool): Shuffle switch.

    Returns:
        Tuple[BurdenkoLongitudinalDataset, DataLoader]: Dataset and DataLoader instances.
    """
    dataset = BurdenkoLongitudinalDataset(data_dir=data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)) if len(dataset) > 0 else batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    return dataset, dataloader


if __name__ == "__main__":
    df_test = load_burdenko_clinical_df()
    print("Burdenko Dataset Summary:")
    print(f"Total Patients: {len(df_test)}")
    if not df_test.empty:
        print("Target Distribution:")
        print(df_test["target_name"].value_counts())
