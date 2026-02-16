"""
Custom model builders for non-vision domains.
"""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn
import torchvision.models as models


def build_ember_mlp(
    input_dim: int = 2381,
    hidden_dims: Iterable[int] = (512, 256),
    num_classes: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Simple MLP for EMBER vectorized features.

    Requires a pretrained checkpoint for real evaluation.
    """
    dims = [int(input_dim)] + [int(d) for d in hidden_dims]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
    layers.append(nn.Linear(dims[-1], int(num_classes)))
    return nn.Sequential(*layers)


def build_nuscenes_resnet18(
    num_classes: int = 10,
    pretrained: bool = False,
) -> nn.Module:
    """
    ResNet-18 backbone for nuScenes image classification.

    Requires a pretrained checkpoint for real evaluation.
    """
    model = models.resnet18(weights="DEFAULT" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, int(num_classes))
    return model


__all__ = ["build_ember_mlp", "build_nuscenes_resnet18"]
