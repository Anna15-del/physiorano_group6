"""
PhysioRANO Inference Module.

Provides 3D sliding-window prediction and NIfTI file export utilities.
"""

from .predict import predict_subject, save_nifti_prediction

__all__ = ["predict_subject", "save_nifti_prediction"]
