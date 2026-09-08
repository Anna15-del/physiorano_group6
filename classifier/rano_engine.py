"""
PhysioRANO RANO 2.0 Response Assessment Engine (Phase 7).

Implements Response Assessment in Neuro-Oncology (RANO 2.0) guidelines to evaluate
longitudinal changes in Enhancing Tumor (ET), Tumor Core (TC), and Whole Tumor (WT) volumes.

Categorical Output Response Grades:
    - CR: Complete Response (Disappearance of all enhancing tumor)
    - PR: Partial Response (>= 50% decrease in sum of product diameters / volume)
    - SD: Stable Disease (< 50% decrease to < 25% increase in volume)
    - PD: Progressive Disease (>= 25% increase in volume or appearance of new lesions)
"""

from typing import Dict, Optional, Union
from utils.logger import get_logger

logger = get_logger("PhysioRANO.RANOEngine")


def evaluate_rano2_response(
    baseline_et_vol: float,
    current_et_vol: float,
    baseline_wt_vol: float,
    current_wt_vol: float,
    pseudoprogression_prob: float = 0.0,
    has_new_lesions: bool = False,
) -> Dict[str, Union[str, float, bool]]:
    """Evaluates RANO 2.0 response grade based on volumetric changes over time.

    Args:
        baseline_et_vol (float): Baseline Enhancing Tumor volume (cm^3).
        current_et_vol (float): Follow-up Enhancing Tumor volume (cm^3).
        baseline_wt_vol (float): Baseline Whole Tumor volume (cm^3).
        current_wt_vol (float): Follow-up Whole Tumor volume (cm^3).
        pseudoprogression_prob (float): Classifier probability of pseudoprogression (0.0 to 1.0).
        has_new_lesions (bool): Flag indicating detection of new distant lesions.

    Returns:
        Dict: RANO 2.0 evaluation containing response category, percentage changes, and clinical guidance.
    """
    # Calculate percentage volume changes
    if baseline_et_vol > 0:
        et_pct_change = ((current_et_vol - baseline_et_vol) / baseline_et_vol) * 100.0
    else:
        et_pct_change = 0.0 if current_et_vol == 0 else 100.0

    if baseline_wt_vol > 0:
        wt_pct_change = ((current_wt_vol - baseline_wt_vol) / baseline_wt_vol) * 100.0
    else:
        wt_pct_change = 0.0

    # Apply RANO 2.0 Response Assessment Criteria
    if current_et_vol <= 0.05 and current_wt_vol <= 0.5:
        category = "CR"
        category_name = "Complete Response (CR)"
        clinical_summary = "Complete disappearance of all enhancing tumor tissue."
    elif et_pct_change <= -50.0:
        category = "PR"
        category_name = "Partial Response (PR)"
        clinical_summary = "Significant reduction (>= 50%) in enhancing tumor volume."
    elif et_pct_change >= 25.0 or has_new_lesions:
        if pseudoprogression_prob >= 0.5:
            category = "PsP"
            category_name = "Pseudoprogression (PsP / Treatment Effect)"
            clinical_summary = "Volumetric increase (> 25%) consistent with pseudoprogression. Re-evaluate scan in 4-8 weeks."
        else:
            category = "PD"
            category_name = "Progressive Disease (PD)"
            clinical_summary = "Significant volumetric expansion (>= 25%) or new lesion indicative of true tumor progression."
    else:
        category = "SD"
        category_name = "Stable Disease (SD)"
        clinical_summary = "Tumor volume changes are stable (-50% < change < +25%)."

    logger.info(f"RANO 2.0 Evaluation: {category_name} (ET Change: {et_pct_change:+.1f}%)")

    return {
        "rano_category": category,
        "rano_category_name": category_name,
        "enhancing_tumor_change_pct": float(et_pct_change),
        "whole_tumor_change_pct": float(wt_pct_change),
        "pseudoprogression_flag": pseudoprogression_prob >= 0.5,
        "clinical_guidance": clinical_summary,
    }


if __name__ == "__main__":
    res = evaluate_rano2_response(
        baseline_et_vol=10.0,
        current_et_vol=14.0,
        baseline_wt_vol=30.0,
        current_wt_vol=38.0,
        pseudoprogression_prob=0.72,
    )
    print("RANO 2.0 Evaluation Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
