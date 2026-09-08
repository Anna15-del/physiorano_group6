"""
PhysioRANO 3D Brain Tumor Segmentation Model Definitions.

Constructs MONAI SegResNet / SegResNetDS 3D convolutional network architecture
optimized for multi-modal BraTS 2024 GLI tumor segmentation.
"""

from typing import Optional
import torch
import torch.nn as nn
from monai.networks.nets import SegResNet
from utils.logger import get_logger

logger = get_logger("PhysioRANO.Models")


def get_segmentation_model(
    spatial_dims: int = 3,
    in_channels: int = 4,
    out_channels: int = 4,
    init_filters: int = 16,
    dropout_prob: float = 0.2,
) -> nn.Module:
    """Instantiates MONAI SegResNet 3D network for multi-modal brain tumor segmentation.

    Args:
        spatial_dims (int): Spatial dimensions (3 for 3D volumes).
        in_channels (int): Input channel count (4 for T1n, T1c, T2w, T2f modalities).
        out_channels (int): Output channel count (4 segmentation classes).
        init_filters (int): Initial feature filters for convolution layers.
        dropout_prob (float): Dropout probability for regularization.

    Returns:
        nn.Module: PyTorch MONAI SegResNet model.
    """
    logger.info(
        f"Instantiating MONAI SegResNet (spatial_dims={spatial_dims}, "
        f"in_channels={in_channels}, out_channels={out_channels}, init_filters={init_filters})"
    )

    model = SegResNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        dropout_prob=dropout_prob,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
    )

    return model
