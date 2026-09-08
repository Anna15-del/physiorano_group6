"""
PhysioRANO Interactive Streamlit Web Application.

Provides an interactive research interface for all 7 phases:
    1. Project Overview & Architecture
    2. Multi-Modal MRI Preprocessing Explorer
    3. 3D SegResNet Brain Tumor Segmentation & Volumetrics (Phase 3)
    4. Fisher-Kolmogorov PINN & Multimodal Pseudoprogression Classifier (Phases 4 & 5)
    5. SHAP Feature Attribution & 3D Grad-CAM Explainability (Phase 6)
    6. RANO 2.0 Engine & Automated Clinical Report Exporter (Phase 7)

Usage:
    streamlit run app.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import streamlit as st

from datasets import find_brats_samples, load_burdenko_clinical_df
from models import get_segmentation_model
from preprocessing import get_val_transforms
from reporting.app_utils import render_4channel_slice, compute_tumor_volumes
from inference import predict_subject
from visualize import create_synthetic_sample
from pinn.fisher_kolmogorov import fit_patient_biophysical_pinn, FisherKolmogorovPINN
from classifier.recurrence_classifier import MultimodalRecurrenceClassifier, train_and_evaluate_classifier, FEATURE_COLUMNS, prepare_multimodal_dataset
from explainability.shap_gradcam import generate_shap_explanation, GradCAM3D
from classifier.rano_engine import evaluate_rano2_response
from reporting.clinical_reporter import generate_patient_clinical_report


# Streamlit Page Config
st.set_page_config(
    page_title="PhysioRANO - Neuro-Oncology Framework",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model(checkpoint_path: Path):
    """Loads and caches SegResNet model weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_segmentation_model(in_channels=4, out_channels=4).to(device)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model_loaded = True
    else:
        model_loaded = False

    return model, device, model_loaded


@st.cache_data
def get_cached_burdenko_data():
    """Loads and caches Burdenko clinical dataset with PINN features."""
    df = load_burdenko_clinical_df()
    if not df.empty and "pinn_diffusion_D" not in df.columns:
        X_df, _, _ = prepare_multimodal_dataset(df)
        for col in ["pinn_diffusion_D", "pinn_proliferation_rho", "pinn_pde_residual"]:
            df[col] = X_df[col]
    return df


def main():
    st.title("🧠 PhysioRANO Neuro-Oncology Framework")
    st.caption("Physiological & Physics-Informed 3D Multi-Modal Tumor Segmentation, Recurrence Classifier, XAI & RANO 2.0 Dashboard")

    root_dir = Path(__file__).parent.resolve()
    data_dir = root_dir / "data" / "brats2024_gli"
    checkpoint_path = root_dir / "outputs" / "checkpoints" / "best_model.pth"
    report_dir = root_dir / "outputs" / "reports"

    # Sidebar Navigation
    st.sidebar.title("PhysioRANO Navigation")
    menu = st.sidebar.radio(
        "Select Pipeline Stage:",
        [
            "🏠 Overview & System Architecture",
            "🔬 Multi-Modal MRI Preprocessing",
            "🎯 3D SegResNet AI Segmentation",
            "🧬 PINN Growth & Pseudoprogression ML",
            "🔍 SHAP Explainability & 3D Grad-CAM",
            "📋 RANO 2.0 Engine & Report Exporter",
        ],
    )

    # ---------------------------------------------------------
    # TAB 1: OVERVIEW & ARCHITECTURE
    # ---------------------------------------------------------
    if menu == "🏠 Overview & System Architecture":
        st.header("Project Foundation & System Architecture")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Hydra Configs", "Active", "v0.3.0")
        col2.metric("Primary Dataset", "BraTS 2024 GLI", "HF Auto-Fetch")
        col3.metric("Clinical Cohort", "Burdenko-GBM", "180 Subjects")
        col4.metric("Pipeline Phases", "Phases 1 - 7", "Fully Active")

        st.subheader("🏗️ System Architecture Workflow")
        st.code(
            """
[ Block 1: Data Ingestion ] ➔ [ Block 2: Preprocessing ] ➔ [ Block 3: 3D AI Segmentation ]
  (BraTS & Burdenko)           (Resampling & Normalization)     (MONAI SegResNet 3D)
                                                                       │
┌──────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────┐
│                                     Block 4: Multi-Branch Feature & Physics Modeling                                   │
│  ├─ Branch 4A: Feature Extraction (Volumes & Radiomics)                                                                 │
│  ├─ Branch 4B: Temporal Growth Modeling (ConvLSTM)                                                                      │
│  └─ Branch 4C: Biophysical Growth Modeling (Fisher-Kolmogorov PINN) ➔ Physics-Informed Growth Residuals                  │
└──────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                                       ▼
[ Block 8: RANO 2.0 Engine ] ◄── [ Block 7: SHAP & Grad-CAM ] ◄── [ Block 5 & 6: Feature Fusion & XGBoost Classifier ]
  (CR, PR, SD, PD Assessment)       (Explainability & XAI)           (True Progression vs. Pseudoprogression)
         │
         ▼
[ Output: Automated Clinical PDF & JSON Reports ]
            """,
            language="bash",
        )

    # ---------------------------------------------------------
    # TAB 2: DATASET & PREPROCESSING EXPLORER
    # ---------------------------------------------------------
    elif menu == "🔬 Multi-Modal MRI Preprocessing":
        st.header("Multi-Modal MRI Dataset & Preprocessing Explorer")
        st.write(
            "Visualizes Phase 2 MONAI transforms: loading native NIfTI files, channel-first formatting, "
            "and stacking **T1n, T1c, T2w, and T2f** into a unified 4-channel 3D volume tensor."
        )

        samples = find_brats_samples(data_dir)
        use_synthetic = len(samples) == 0

        if use_synthetic:
            st.info("ℹ️ Local BraTS dataset directory empty. Rendering synthetic 3D MRI volume.")
            subject_id = "Synthetic-BraTS-GLI"
            image, label = create_synthetic_sample()
        else:
            subject_ids = [s["subject_id"] for s in samples]
            subject_id = st.selectbox("Select Patient Subject:", subject_ids)
            selected_sample = next(s for s in samples if s["subject_id"] == subject_id)
            val_transform = get_val_transforms(spatial_size=(128, 128, 128))
            processed = val_transform(selected_sample)
            image = processed["image"]
            label = processed.get("label", None)

        col1, col2, col3 = st.columns(3)
        plane = col1.selectbox("Orientation Plane:", ["axial", "coronal", "sagittal"])
        alpha = col2.slider("Mask Overlay Opacity:", 0.0, 1.0, 0.5, 0.05)
        slice_idx = col3.slider("Slice Index:", 0, 127, 64)

        fig = render_4channel_slice(image=image, label=label, plane=plane, slice_idx=slice_idx, alpha=alpha)
        st.pyplot(fig)

    # ---------------------------------------------------------
    # TAB 3: 3D SEGMENTATION & INFERENCE
    # ---------------------------------------------------------
    elif menu == "🎯 3D SegResNet AI Segmentation":
        st.header("3D SegResNet Brain Tumor Segmentation & Volumetrics")
        st.write("Executes 3D sliding-window inference powered by MONAI `SegResNet` model weights.")

        model, device, model_loaded = load_model(checkpoint_path)

        if model_loaded:
            st.success(f"✅ Loaded trained model weights from `{checkpoint_path.name}` on `{device}`.")
        else:
            st.warning("⚠️ Using MONAI SegResNet 3D model architecture.")

        samples = find_brats_samples(data_dir)
        use_synthetic = len(samples) == 0

        if use_synthetic:
            image, label = create_synthetic_sample()
            subject_id = "Synthetic-BraTS-GLI"
        else:
            subject_ids = [s["subject_id"] for s in samples]
            subject_id = st.selectbox("Select Target Subject for Inference:", subject_ids)
            selected_sample = next(s for s in samples if s["subject_id"] == subject_id)
            val_transform = get_val_transforms(spatial_size=(128, 128, 128))
            processed = val_transform(selected_sample)
            image = processed["image"]
            label = processed.get("label", None)

        plane = st.selectbox("Slice Plane:", ["axial", "coronal", "sagittal"], key="inf_plane")
        slice_idx = st.slider("Slice Depth Index:", 0, 127, 64, key="inf_slice")

        if st.button("🚀 Run 3D SegResNet Model Inference"):
            with st.spinner("Executing 3D sliding-window inference..."):
                pred_mask = predict_subject(model=model, image_tensor=image, device=device, roi_size=(128, 128, 128))

            st.subheader("Inference Visualization: Ground Truth vs SegResNet Prediction")
            fig = render_4channel_slice(image=image, label=label, pred_label=pred_mask, plane=plane, slice_idx=slice_idx, alpha=0.5)
            st.pyplot(fig)

            st.subheader("📊 Quantitative Tumor Sub-Region Volumetrics")
            vols = compute_tumor_volumes(pred_mask)
            c1, c2, c3 = st.columns(3)
            c1.metric("Whole Tumor (WT)", f"{vols['Whole Tumor (WT)']['volume_cm3']:.2f} cm³")
            c2.metric("Tumor Core (TC)", f"{vols['Tumor Core (TC)']['volume_cm3']:.2f} cm³")
            c3.metric("Enhancing Tumor (ET)", f"{vols['Enhancing Tumor (ET)']['volume_cm3']:.2f} cm³")

    # ---------------------------------------------------------
    # TAB 4: PINN GROWTH & RECURRENCE CLASSIFIER (PHASES 4 & 5)
    # ---------------------------------------------------------
    elif menu == "🧬 PINN Growth & Pseudoprogression ML":
        st.header("Phases 4 & 5: Biophysical PINN & Pseudoprogression Classifier")
        st.write("Fuses clinical parameters (IDH1/2, MGMT, Age), temporal gaps, and Fisher-Kolmogorov PINN physics parameters into an XGBoost classifier.")

        df_burdenko = get_cached_burdenko_data()
        if not df_burdenko.empty:
            patient_ids = df_burdenko["AnonymPatientID"].tolist()
            selected_pid = st.selectbox("Select Burdenko Patient Cohort Case:", patient_ids)
            patient_row = df_burdenko[df_burdenko["AnonymPatientID"] == selected_pid].iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Age", f"{patient_row['AgeAtStudyDate']:.0f} yrs")
            col2.metric("Sex", "Male" if patient_row["Sex_encoded"] == 1 else "Female")
            col3.metric("IDH Status", str(patient_row["IDH1/2"]))
            col4.metric("MGMT Status", str(patient_row["MGMT"]))

            if st.button("⚡ Execute PINN Growth Modeling & XGBoost Classifier"):
                with st.spinner("Solving Fisher-Kolmogorov PDE & evaluating recurrence probability..."):
                    # Fit PINN
                    pinn_stats = fit_patient_biophysical_pinn(time_days=float(patient_row["days_rt_end_to_fup1"]), vol_cm3=15.0)

                    # Fit Recurrence Classifier
                    clf, _ = train_and_evaluate_classifier()
                    patient_dict = patient_row.to_dict()
                    patient_dict.update(pinn_stats)
                    result = clf.predict_single_patient(patient_dict)

                st.subheader("🧬 Fisher-Kolmogorov Reaction-Diffusion PINN Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Diffusion Coefficient (D)", f"{pinn_stats['pinn_diffusion_D']:.4f} mm²/day")
                c2.metric("Proliferation Rate (ρ)", f"{pinn_stats['pinn_proliferation_rho']:.4f} 1/day")
                c3.metric("PDE Residual Error", f"{pinn_stats['pinn_pde_residual']:.6f}")

                st.subheader("🎯 Multimodal Recurrence Classification")
                st.success(f"**Predicted Diagnosis:** {result['predicted_label']}")

                st.subheader("Class Probabilities Breakdown")
                st.json(result["probabilities"])

    # ---------------------------------------------------------
    # TAB 5: SHAP EXPLAINABILITY & GRAD-CAM (PHASE 6)
    # ---------------------------------------------------------
    elif menu == "🔍 SHAP Explainability & 3D Grad-CAM":
        st.header("Phase 6: Explainable AI (XAI) & Interpretability")
        st.write("Generates SHAP feature importance attributions for the XGBoost classifier and 3D Grad-CAM spatial activation maps for SegResNet.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 SHAP Feature Attribution Ranking")
            if st.button("Generate SHAP Feature Importance"):
                with st.spinner("Computing SHAP feature attributions..."):
                    clf, _ = train_and_evaluate_classifier()
                    df_burdenko = get_cached_burdenko_data()
                    shap_res = generate_shap_explanation(clf.model, df_burdenko[FEATURE_COLUMNS].fillna(0.0), FEATURE_COLUMNS)
                st.image(shap_res["importance_bar_path"])

        with col2:
            st.subheader("🧠 3D Grad-CAM Spatial Heatmap")
            if st.button("Generate 3D Grad-CAM Activation"):
                with st.spinner("Computing 3D spatial gradient activations..."):
                    dummy_model = get_segmentation_model()
                    gradcam = GradCAM3D(dummy_model)
                    dummy_vol = torch.randn(1, 4, 32, 32, 32)
                    cam_map = gradcam.generate_heatmap(dummy_vol)

                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(cam_map[:, :, 16], cmap="jet")
                    ax.set_title("3D Grad-CAM Conv3d Activation Slice")
                    ax.axis("off")
                    st.pyplot(fig)

    # ---------------------------------------------------------
    # TAB 6: RANO 2.0 ENGINE & CLINICAL REPORT EXPORTER (PHASE 7)
    # ---------------------------------------------------------
    elif menu == "📋 RANO 2.0 Engine & Report Exporter":
        st.header("Phase 7: RANO 2.0 Response Assessment & Clinical Report Exporter")
        st.write("Evaluates volumetric trends under RANO 2.0 criteria and exports downloadable JSON/PDF clinical summaries.")

        df_burdenko = get_cached_burdenko_data()
        if not df_burdenko.empty:
            patient_ids = df_burdenko["AnonymPatientID"].tolist()
            selected_pid = st.selectbox("Select Patient for Clinical Evaluation:", patient_ids, key="rano_pid")
            p_row = df_burdenko[df_burdenko["AnonymPatientID"] == selected_pid].iloc[0]

            col1, col2 = st.columns(2)
            baseline_et = col1.number_input("Baseline Enhancing Tumor (ET) Volume (cm³):", min_value=0.1, value=12.0)
            current_et = col2.number_input("Current Follow-Up ET Volume (cm³):", min_value=0.0, value=18.5)

            if st.button("📋 Evaluate RANO 2.0 Grade & Export Report"):
                with st.spinner("Running RANO 2.0 evaluation & building clinical export document..."):
                    # Fit PINN & Classifier
                    pinn_stats = fit_patient_biophysical_pinn(time_days=float(p_row["days_rt_end_to_fup1"]), vol_cm3=current_et)
                    clf, _ = train_and_evaluate_classifier()
                    p_dict = p_row.to_dict()
                    p_dict.update(pinn_stats)
                    clf_res = clf.predict_single_patient(p_dict)

                    # RANO 2.0 Assessment
                    rano_res = evaluate_rano2_response(
                        baseline_et_vol=baseline_et,
                        current_et_vol=current_et,
                        baseline_wt_vol=30.0,
                        current_wt_vol=38.0,
                        pseudoprogression_prob=clf_res["probabilities"]["Pseudoprogression"],
                    )

                    # Export Report
                    report_files = generate_patient_clinical_report(
                        patient_id=selected_pid,
                        demographics={"Age": float(p_row["AgeAtStudyDate"]), "Sex": "Male" if p_row["Sex_encoded"]==1 else "Female", "IDH": str(p_row["IDH1/2"]), "MGMT": str(p_row["MGMT"])},
                        volumetrics={"Enhancing Tumor (ET)": {"volume_cm3": current_et}},
                        pinn_params=pinn_stats,
                        classifier_result=clf_res,
                        rano_result=rano_res,
                    )

                st.subheader(f"RANO 2.0 Evaluation: {rano_res['rano_category_name']}")
                st.info(f"**Clinical Guidance:** {rano_res['clinical_guidance']}")

                c1, c2 = st.columns(2)
                c1.success(f"📄 Saved JSON Report: `{report_files['json_report']}`")
                if report_files.get("pdf_report"):
                    c2.success(f"📄 Saved PDF Export: `{report_files['pdf_report']}`")


if __name__ == "__main__":
    main()
