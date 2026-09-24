"""
Verification test suite for 3D Conditional GAN (cGAN) Tumor Growth & Regression Forecaster.
"""

import sys
from pathlib import Path
import torch

# Ensure repository root is in sys.path
repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from generative.tumor_forecast_gan import (
    ConditionEmbeddingNetwork,
    TumorForecastGenerator3D,
    TumorForecastDiscriminator3D,
    TumorForecastLoss,
    predict_future_mri_scan,
)
from visualize import create_synthetic_sample


def test_condition_embedding():
    print("Testing ConditionEmbeddingNetwork...")
    cond_net = ConditionEmbeddingNetwork(condition_dim=3, embed_dim=64)
    c = torch.tensor([[90.0, 0.12, 0.04]])
    out = cond_net(c)
    assert out.shape == (1, 64), f"Unexpected embedding shape: {out.shape}"
    print("[PASSED] ConditionEmbeddingNetwork passed!")


def test_generator_forward():
    print("Testing TumorForecastGenerator3D forward pass...")
    gen = TumorForecastGenerator3D(in_channels=4, out_channels=4, init_filters=8, embed_dim=64)
    # Test with small 3D volume for rapid CPU test
    x = torch.randn(1, 4, 32, 32, 32)
    c = torch.tensor([[60.0, 0.15, 0.05]])
    y_pred = gen(x, c)
    assert y_pred.shape == (1, 4, 32, 32, 32), f"Unexpected generator output shape: {y_pred.shape}"
    print(f"[PASSED] TumorForecastGenerator3D passed! Output shape: {y_pred.shape}")


def test_discriminator_forward():
    print("Testing TumorForecastDiscriminator3D forward pass...")
    disc = TumorForecastDiscriminator3D(in_channels=4, init_filters=8)
    x = torch.randn(1, 4, 32, 32, 32)
    y = torch.randn(1, 4, 32, 32, 32)
    c = torch.tensor([[60.0, 0.15, 0.05]])
    patch_scores = disc(x, y, c)
    assert patch_scores.ndim == 5, f"Unexpected discriminator patch shape: {patch_scores.shape}"
    print(f"[PASSED] TumorForecastDiscriminator3D passed! Patch output shape: {patch_scores.shape}")


def test_loss_computation():
    print("Testing TumorForecastLoss...")
    loss_fn = TumorForecastLoss(lambda_l1=100.0)
    d_fake = torch.zeros(1, 1, 4, 4, 4)
    d_real = torch.ones(1, 1, 4, 4, 4)
    y_pred = torch.randn(1, 4, 16, 16, 16)
    y_real = torch.randn(1, 4, 16, 16, 16)

    g_loss, g_metrics = loss_fn.generator_loss(d_fake, y_pred, y_real)
    d_loss, d_metrics = loss_fn.discriminator_loss(d_real, d_fake)

    assert g_loss.item() > 0, "Generator loss should be positive"
    assert d_loss.item() >= 0, "Discriminator loss should be non-negative"
    print(f"[PASSED] TumorForecastLoss passed! G-Loss: {g_loss.item():.4f}, D-Loss: {d_loss.item():.4f}")


def test_predict_future_mri():
    print("Testing predict_future_mri_scan inference API...")
    synth_img, synth_lbl = create_synthetic_sample()
    # Crop to 64x64x64 for fast CPU test
    cropped_img = synth_img[:, 32:96, 32:96, 32:96]

    res = predict_future_mri_scan(
        baseline_image=cropped_img,
        delta_t_days=90.0,
        pinn_diffusion_D=0.1245,
        pinn_proliferation_rho=0.0412,
        clinical_scenario="Standard (Slow Growth / PsP)",
    )

    assert "future_image" in res
    assert "baseline_volumes" in res
    assert "forecasted_volumes" in res
    assert "volume_changes_pct" in res
    assert "trajectory" in res
    assert len(res["trajectory"]) == 6

    print(f"Baseline WT: {res['baseline_volumes']['Whole Tumor (WT)']['volume_cm3']} cm3")
    print(f"Forecasted WT (90 days): {res['forecasted_volumes']['Whole Tumor (WT)']['volume_cm3']} cm3")
    print(f"Percentage WT Change: {res['volume_changes_pct']['Whole Tumor (WT)']:+.1f}%")
    print(f"Trajectory timepoints: {[pt['days'] for pt in res['trajectory']]}")
    print("[PASSED] predict_future_mri_scan passed successfully!")


if __name__ == "__main__":
    test_condition_embedding()
    test_generator_forward()
    test_discriminator_forward()
    test_loss_computation()
    test_predict_future_mri()
    print("\nALL 3D cGAN TESTS PASSED SUCCESSFULLY!")
