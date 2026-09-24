"""
PhysioRANO Generative Deep Learning Engine.

Provides 3D Conditional Generative Adversarial Network (cGAN) modules
for longitudinal brain tumor growth and treatment regression forecasting.
"""

from generative.tumor_forecast_gan import (
    ConditionEmbeddingNetwork,
    TumorForecastGenerator3D,
    TumorForecastDiscriminator3D,
    TumorForecastLoss,
    predict_future_mri_scan,
)

__all__ = [
    "ConditionEmbeddingNetwork",
    "TumorForecastGenerator3D",
    "TumorForecastDiscriminator3D",
    "TumorForecastLoss",
    "predict_future_mri_scan",
]
