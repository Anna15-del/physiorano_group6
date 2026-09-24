"""
Generates visual demonstration figures of the 3D cGAN Tumor Growth & Regression Forecaster.
Saves PNG figures to the artifact directory so they can be viewed immediately.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
import torch

from visualize import create_synthetic_sample
from generative.tumor_forecast_gan import predict_future_mri_scan
from reporting.app_utils import render_cgan_comparison_slice, plot_forecast_trajectory

artifact_dir = repo_root / "outputs" / "reports"
artifact_dir.mkdir(parents=True, exist_ok=True)

print("1. Creating baseline 3D MRI volume...")
base_img, _ = create_synthetic_sample()

print("2. Running 3D cGAN Forecaster for Delta_t = 90 days...")
res = predict_future_mri_scan(
    baseline_image=base_img,
    delta_t_days=90.0,
    pinn_diffusion_D=0.1245,
    pinn_proliferation_rho=0.0412,
    clinical_scenario="Standard (Slow Growth / PsP)",
)

print("3. Generating Side-by-Side MRI comparison figure...")
fig_comp = render_cgan_comparison_slice(
    base_image=base_img,
    future_image=res["future_image"],
    plane="axial",
    slice_idx=64,
    modality_idx=1,  # T1c contrast-enhancing
)
comp_path = artifact_dir / "cgan_mri_comparison.png"
fig_comp.savefig(comp_path, dpi=180, bbox_inches="tight")
plt.close(fig_comp)
print(f"Saved: {comp_path}")

print("4. Generating Longitudinal Trajectory Curve figure...")
fig_traj = plot_forecast_trajectory(res["trajectory"], selected_days=90.0)
traj_path = artifact_dir / "cgan_trajectory_curve.png"
fig_traj.savefig(traj_path, dpi=180, bbox_inches="tight")
plt.close(fig_traj)
print(f"Saved: {traj_path}")

print("Done! Visual demonstration figures created successfully.")
