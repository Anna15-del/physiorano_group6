# 🧠 PhysioRANO: Physiological & Physics-Informed Neuro-Oncology Framework

PhysioRANO is a PyTorch and MONAI-based research framework designed for neuro-oncology medical imaging analysis (BraTS 2024 GLI and longitudinal Burdenko datasets), 3D tumor segmentation, physics-informed biophysical growth modeling, explainable AI (XAI), and automated RANO 2.0 clinical response assessment.

---

## 🏗️ System Architecture Overview

```
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
```

---

## 🗺️ Phase-by-Phase Implementation Roadmap

### 📍 Phase 1: Project Foundation & System Architecture *(Completed)*
* **Objective**: Build a modular, scalable project framework with configuration management and reproducible pipeline entry points.
* **Key Deliverables**:
  * **Hydra Configuration Engine**: Hierarchical YAML configuration (`configs/config.yaml`, `configs/dataset/`) for flexible experiment setup.
  * **Modular Repository Architecture**: Organized directory structure decoupling data loading, modeling, training, inference, and reporting.
  * **Logging & Seeds**: Deterministic seed utility and structured color logging across all execution modules.

### 📍 Phase 2: Multi-Modal Data Ingestion & Preprocessing Pipeline *(Completed)*
* **Objective**: Ingest multi-modal MRI sequences and apply MONAI 3D dictionary-based preprocessing transformations.
* **Key Deliverables**:
  * **BraTS 2024 GLI Auto-Ingestion**: `BraTSDataset` automatically pairs 4 MRI modalities (`T1n`, `T1c`, `T2w`, `T2f`) + ground truth segmentation mask (`seg.nii.gz`) with automated Hugging Face download integration.
  * **Longitudinal Cohort Data Structure**: `BurdenkoLongitudinalDataset` template for multi-timepoint patient scan series.
  * **3D MONAI Transforms**:
    * Isotropic voxel spatial resampling (`Spacingd` to 1.0mm³).
    * Non-zero brain tissue foreground cropping (`CropForegroundd`).
    * Channel-wise Z-score intensity normalization (`NormalizeIntensityd`).
    * Modality tensor stacking into 4-channel 3D volumes (`ConcatItemsd` $\to$ `[4, H, W, D]`).

### 📍 Phase 3: 3D AI Tumor Segmentation & Volumetric Extraction *(Completed)*
* **Objective**: Develop deep learning 3D segmentation models to identify and quantify glioblastoma sub-regions.
* **Key Deliverables**:
  * **MONAI SegResNet 3D Architecture**: 3D convolutional neural network optimized for multi-modal brain tumor segmentation.
  * **Sliding-Window Inference**: `SlidingWindowInferer` engine for memory-efficient dense 3D volume prediction.
  * **Tumor Sub-Region Quantification**: Extraction of physical volumes ($cm^3$) and voxel counts for:
    * Whole Tumor (WT)
    * Tumor Core (TC)
    * Enhancing Tumor (ET)
    * Edema (ED)

### 📍 Phase 4: Multi-Branch Feature Analysis & Biophysical Growth Modeling *(In Progress)*
* **Objective**: Analyze temporal tumor trajectory by combining radiomic features, temporal deep learning, and physics-informed neural networks.
* **Key Deliverables**:
  * **Branch 4A (Radiomics & Volumetrics)**: Extraction of PyRadiomics texture, shape, and intensity features from segmented sub-regions.
  * **Branch 4B (Temporal ConvLSTM)**: ConvLSTM network for spatio-temporal modeling of longitudinal follow-up scan series.
  * **Branch 4C (Fisher-Kolmogorov PINN)**: Physics-Informed Neural Network (PINN) enforcing reaction-diffusion PDEs ($\frac{\partial u}{\partial t} = D \nabla^2 u + \rho u (1 - u)$) to model biophysical tumor invasion and compute physics-informed growth residuals.

### 📍 Phase 5: Multimodal Feature Fusion & Recurrence Classifier *(Planned)*
* **Objective**: Differentiate **True Tumor Progression** from **Pseudoprogression** (radiation necrosis) by fusing imaging features with clinical parameters.
* **Key Deliverables**:
  * **Multimodal Fusion Layer**: Merging radiomics, ConvLSTM embeddings, PINN growth residuals, and patient clinical/molecular data (IDH status, MGMT promoter methylation, age).
  * **XGBoost Recurrence Classifier**: Supervised machine learning model outputting calibrated probabilities for True Progression vs. Pseudoprogression.

### 📍 Phase 6: Explainable AI (XAI) & Interpretability *(Planned)*
* **Objective**: Provide visual and quantitative explanations for clinical decision support.
* **Key Deliverables**:
  * **3D Grad-CAM**: Spatial heatmaps illustrating regional 3D MRI activation areas driving SegResNet model predictions.
  * **SHAP (SHapley Additive exPlanations)**: Global and local feature attribution ranking clinical and radiomic predictors in the XGBoost classifier.

### 📍 Phase 7: RANO 2.0 Engine & Clinical Reporting *(Planned / Dashboard Live)*
* **Objective**: Automate clinical response evaluation based on updated RANO 2.0 (Response Assessment in Neuro-Oncology) guidelines and generate exportable reports.
* **Key Deliverables**:
  * **RANO 2.0 Rules Engine**: Automated categorical classification into:
    * **CR** (Complete Response)
    * **PR** (Partial Response)
    * **SD** (Stable Disease)
    * **PD** (Progressive Disease)
  * **Automated Clinical Reporting**: Generation of downloadable PDF clinical summaries and structured JSON metric files.
  * **Interactive Streamlit Web Dashboard**: Live multi-tab UI (`app.py`) for visual slice inspection, model inference, volumetrics, and training analytics.

---

## 📁 Repository Layout

```
PhysioRANO/
├── configs/                # Hydra YAML configuration files
│   ├── config.yaml         # Main configuration hierarchy
│   └── dataset/            # Dataset configs (brats.yaml, burdenko.yaml)
├── datasets/               # MONAI Dataset loaders & HF downloader
│   ├── brats.py            # BraTS 2024 GLI dataset & loader
│   ├── burdenko.py         # Burdenko longitudinal dataset template
│   └── downloader.py       # Automated Hugging Face downloader
├── models/                 # Neural network architectures (SegResNet 3D)
├── preprocessing/          # MONAI 4-modal transforms, Z-score, & resampling
├── inference/              # 3D Sliding-window inferer & NIfTI prediction
├── longitudinal/           # Temporal alignment & ConvLSTM tracking (Phase 4)
├── pinn/                   # Fisher-Kolmogorov reaction-diffusion PINN (Phase 4)
├── classifier/             # XGBoost multimodal recurrence classifier (Phase 5)
├── explainability/         # 3D Grad-CAM & SHAP attribution (Phase 6)
├── reporting/              # Slice renderers, volumetrics, & RANO reports (Phase 7)
├── training/               # Training engine, loss functions, & AMP
├── utils/                  # Structured logging & seed determinism
├── outputs/                # Checkpoints, logs, and generated reports
├── app.py                  # Interactive Streamlit Web Application
├── train.py                # Top-level Training CLI
├── validate.py             # Top-level Validation CLI
└── predict.py              # Top-level Inference CLI
```

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Interactive Web Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

### 3. Model Training & Inference CLI

Run model training:
```bash
python train.py
```

Execute 3D segmentation inference:
```bash
python predict.py
```
