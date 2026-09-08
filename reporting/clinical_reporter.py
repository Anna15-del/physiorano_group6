"""
PhysioRANO Automated Clinical Report Generator (Phase 7).

Generates structured JSON metric reports and clinical PDF export documents
summarizing patient demographics, PINN biophysical growth parameters, XGBoost pseudoprogression probability,
and RANO 2.0 response grade.
"""

from pathlib import Path
import json
from typing import Dict, Optional, Union
import matplotlib.pyplot as plt
import numpy as np

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from utils.logger import get_logger

logger = get_logger("PhysioRANO.ClinicalReporter")


def generate_patient_clinical_report(
    patient_id: str,
    demographics: Dict[str, Union[str, float]],
    volumetrics: Dict[str, Dict[str, float]],
    pinn_params: Dict[str, float],
    classifier_result: Dict,
    rano_result: Dict,
    output_dir: Union[str, Path] = "outputs/reports",
) -> Dict[str, str]:
    """Generates patient JSON metric summary and clinical PDF export report.

    Args:
        patient_id (str): Patient Subject Identifier.
        demographics (Dict): Patient metadata (Age, Sex, IDH, MGMT).
        volumetrics (Dict): Tumor sub-region volumes (WT, TC, ET).
        pinn_params (Dict): PINN biophysical parameters (diffusion, proliferation, residual).
        classifier_result (Dict): Pseudoprogression classifier probabilities.
        rano_result (Dict): RANO 2.0 evaluation result.
        output_dir (Union[str, Path]): Target output directory.

    Returns:
        Dict[str, str]: Paths to generated JSON and PDF report files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{patient_id}_clinical_report.json"
    pdf_path = output_dir / f"{patient_id}_clinical_report.pdf"

    # Assemble complete clinical report dictionary
    report_data = {
        "patient_id": patient_id,
        "demographics": demographics,
        "volumetrics": volumetrics,
        "pinn_biophysical_modeling": pinn_params,
        "recurrence_classification": classifier_result,
        "rano2_evaluation": rano_result,
    }

    # Save JSON Report
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Saved patient JSON clinical report: {json_path.resolve()}")

    # Generate PDF Report if reportlab is available
    if HAS_REPORTLAB:
        try:
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            title_style = ParagraphStyle(
                "DocTitle",
                parent=styles["Heading1"],
                fontSize=20,
                leading=24,
                textColor=colors.HexColor("#1a365d"),
                spaceAfter=12,
            )
            h2_style = ParagraphStyle(
                "SectionH2",
                parent=styles["Heading2"],
                fontSize=14,
                leading=18,
                textColor=colors.HexColor("#2b6cb0"),
                spaceBefore=10,
                spaceAfter=6,
            )

            story.append(Paragraph(f"🧠 PhysioRANO Clinical Evaluation Report", title_style))
            story.append(Paragraph(f"<b>Patient Identifier:</b> {patient_id}", styles["Normal"]))
            story.append(Spacer(1, 10))

            # Demographics Table
            story.append(Paragraph("📋 Patient Clinical Metadata", h2_style))
            demo_table_data = [
                ["Age", "Sex", "IDH1/2 Mutation", "MGMT Methylation"],
                [
                    str(demographics.get("Age", "N/A")),
                    str(demographics.get("Sex", "N/A")),
                    str(demographics.get("IDH", "N/A")),
                    str(demographics.get("MGMT", "N/A")),
                ],
            ]
            t_demo = Table(demo_table_data, colWidths=[100, 100, 150, 150])
            t_demo.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ebf8ff")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2c5282")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
            ]))
            story.append(t_demo)
            story.append(Spacer(1, 10))

            # RANO 2.0 Evaluation Section
            story.append(Paragraph("🎯 RANO 2.0 Assessment", h2_style))
            rano_cat = rano_result.get("rano_category_name", "N/A")
            story.append(Paragraph(f"<b>Category:</b> {rano_cat}", styles["Normal"]))
            story.append(Paragraph(f"<b>Guidance:</b> {rano_result.get('clinical_guidance', '')}", styles["Normal"]))
            story.append(Spacer(1, 10))

            # Multimodal Classifier & PINN Section
            story.append(Paragraph("🧬 Pseudoprogression Risk & Biophysical PINN", h2_style))
            psp_risk = classifier_result.get("pseudoprogression_risk_percent", 0.0)
            story.append(Paragraph(f"<b>Pseudoprogression Probability:</b> {psp_risk:.1f}%", styles["Normal"]))
            story.append(Paragraph(f"<b>PINN Diffusion (D):</b> {pinn_params.get('pinn_diffusion_D', 0.0):.4f} mm²/day", styles["Normal"]))
            story.append(Paragraph(f"<b>PINN Proliferation (rho):</b> {pinn_params.get('pinn_proliferation_rho', 0.0):.4f} 1/day", styles["Normal"]))

            doc.build(story)
            logger.info(f"Generated PDF clinical export report: {pdf_path.resolve()}")
        except Exception as e:
            logger.warning(f"PDF report generation note: {e}")

    return {
        "json_report": str(json_path),
        "pdf_report": str(pdf_path) if pdf_path.exists() else "",
    }


if __name__ == "__main__":
    rep = generate_patient_clinical_report(
        patient_id="Burdenko-GBM-007",
        demographics={"Age": 40, "Sex": "F", "IDH": "Positive", "MGMT": "Positive"},
        volumetrics={"Whole Tumor (WT)": {"volume_cm3": 24.5}},
        pinn_params={"pinn_diffusion_D": 0.12, "pinn_proliferation_rho": 0.04, "pinn_pde_residual": 0.002},
        classifier_result={"predicted_label": "Pseudoprogression", "pseudoprogression_risk_percent": 78.4},
        rano_result={"rano_category_name": "Pseudoprogression (PsP)", "clinical_guidance": "Re-evaluate scan in 4-8 weeks."},
    )
    print("Clinical Report Generated:")
    print(rep)
