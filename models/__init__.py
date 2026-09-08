"""
PhysioRANO Models Module.

Provides neural network model architectures for 3D brain tumor segmentation.
"""

from .segmentation import get_segmentation_model

__all__ = ["get_segmentation_model"]
