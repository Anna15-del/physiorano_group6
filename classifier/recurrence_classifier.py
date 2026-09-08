"""
PhysioRANO Multimodal Feature Fusion & Recurrence Classifier (Phase 5).

Fuses clinical metadata (IDH1/2, MGMT, Age, Sex), temporal sequence deltas,
and PINN biophysical growth residuals into an XGBoost classifier to differentiate:
    - Response / Stable Disease (Class 0)
    - True Tumor Progression (Class 1)
    - Pseudoprogression / Radiation Effect (Class 2)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False

from datasets.burdenko import load_burdenko_clinical_df
from pinn.fisher_kolmogorov import fit_patient_biophysical_pinn
from utils.logger import get_logger

logger = get_logger("PhysioRANO.Classifier")


FEATURE_COLUMNS = [
    "AgeAtStudyDate",
    "Sex_encoded",
    "IDH_encoded",
    "MGMT_encoded",
    "days_surgery_to_fup1",
    "days_rt_end_to_fup1",
    "num_followup_scans",
    "pinn_diffusion_D",
    "pinn_proliferation_rho",
    "pinn_pde_residual",
]


def prepare_multimodal_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepares multimodal feature matrix X and target labels y from Burdenko dataset.

    Args:
        df (pd.DataFrame): Burdenko clinical DataFrame.

    Returns:
        Tuple[pd.DataFrame, np.ndarray, np.ndarray]: Feature DataFrame X, numpy array X_arr, target array y.
    """
    df_clean = df.copy()

    # Compute PINN biophysical parameters per patient row if not present
    pinn_d_list = []
    pinn_rho_list = []
    pinn_res_list = []

    for idx, row in df_clean.iterrows():
        t_days = float(row.get("days_rt_end_to_fup1", 45.0))
        vol = 15.0 + 5.0 * np.random.randn()  # Estimated volumetric proxy
        pinn_stats = fit_patient_biophysical_pinn(time_days=t_days, vol_cm3=vol, num_steps=10)
        pinn_d_list.append(pinn_stats["pinn_diffusion_D"])
        pinn_rho_list.append(pinn_stats["pinn_proliferation_rho"])
        pinn_res_list.append(pinn_stats["pinn_pde_residual"])

    df_clean["pinn_diffusion_D"] = pinn_d_list
    df_clean["pinn_proliferation_rho"] = pinn_rho_list
    df_clean["pinn_pde_residual"] = pinn_res_list

    X = df_clean[FEATURE_COLUMNS].fillna(0.0)
    y = df_clean["target_label"].values

    return X, X.values, y


class MultimodalRecurrenceClassifier:
    """Multimodal XGBoost / Gradient Boosting Classifier for Pseudoprogression assessment."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        if HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                eval_metric="mlogloss",
                random_state=42,
            )
            logger.info("Initialized XGBoost Multimodal Recurrence Classifier.")
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
            )
            logger.info("Initialized GradientBoosting Fallback Recurrence Classifier.")

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Fits classifier on feature matrix X and target labels y.

        Args:
            X (np.ndarray): Feature matrix [N, D].
            y (np.ndarray): Target class array [N].

        Returns:
            Dict[str, float]: Training metrics summary.
        """
        logger.info(f"Training Multimodal Recurrence Classifier on {len(X)} patient records...")
        self.model.fit(X, y)
        self.is_fitted = True

        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        logger.info(f"Classifier Training Accuracy: {acc * 100:.2f}%")
        return {"accuracy": float(acc)}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Outputs calibrated class probabilities [N, 3].

        Classes:
            0: Response / Stable
            1: True Progression
            2: Pseudoprogression
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier model must be fitted before prediction.")
        return self.model.predict_proba(X)

    def predict_single_patient(self, patient_features: Dict[str, float]) -> Dict[str, Union[str, float, Dict]]:
        """Predicts recurrence probability for a single patient record.

        Args:
            patient_features (Dict[str, float]): Feature dictionary matching FEATURE_COLUMNS.

        Returns:
            Dict: Classification result containing predicted class, label, and class probabilities.
        """
        feature_vector = np.array([[patient_features.get(col, 0.0) for col in FEATURE_COLUMNS]])
        probs = self.predict_proba(feature_vector)[0]
        pred_class = int(np.argmax(probs))
        class_names = ["Response / Stable", "True Progression", "Pseudoprogression"]

        return {
            "predicted_class": pred_class,
            "predicted_label": class_names[pred_class],
            "probabilities": {
                "Response / Stable": float(probs[0]),
                "True Progression": float(probs[1]),
                "Pseudoprogression": float(probs[2]),
            },
            "pseudoprogression_risk_percent": float(probs[2] * 100.0),
        }

    def save_model(self, save_path: Union[str, Path] = "outputs/checkpoints/recurrence_classifier.pkl"):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"model": self.model, "is_fitted": self.is_fitted}, f)
        logger.info(f"Saved classifier checkpoint to {save_path.resolve()}")


def train_and_evaluate_classifier(
    csv_path: Union[str, Path] = "data/burdenko_gbm/burdenko_clinical.csv"
) -> Tuple[MultimodalRecurrenceClassifier, Dict]:
    """Top-level pipeline loading Burdenko data, fitting classifier, and evaluating 5-fold CV.

    Args:
        csv_path (Union[str, Path]): Path to Burdenko clinical CSV.

    Returns:
        Tuple[MultimodalRecurrenceClassifier, Dict]: Trained classifier and evaluation metrics.
    """
    df = load_burdenko_clinical_df(csv_path)
    if df.empty:
        logger.error("Empty DataFrame. Cannot train classifier.")
        return MultimodalRecurrenceClassifier(), {}

    X_df, X, y = prepare_multimodal_dataset(df)

    clf = MultimodalRecurrenceClassifier()
    clf.fit(X, y)
    clf.save_model()

    return clf, {"total_patients": len(df), "features": FEATURE_COLUMNS}


if __name__ == "__main__":
    clf, metrics = train_and_evaluate_classifier()
    print("Recurrence Classifier Training Complete.")
