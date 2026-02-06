"""
Model and dataset loading helpers.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from ..evaluation.datasets import DatasetFactory
from ..models.factory import ModelFactory
from ..models.loader import load_model as load_rb_model


def load_model(
    model_name: str,
    *,
    dataset: str = "cifar10",
    threat_model: str = "Linf",
    device: str = "cpu",
    **kwargs: Any,
) -> torch.nn.Module:
    if "domain" in kwargs or "training_type" in kwargs:
        domain = kwargs.pop("domain", "vision")
        training_type = kwargs.pop("training_type", "standard")
        return ModelFactory.load_model(
            domain=domain,
            model_name=model_name,
            training_type=training_type,
            dataset=dataset,
            device=device,
            **kwargs,
        )
    return load_rb_model(
        model_name=model_name,
        dataset=dataset,
        threat_model=threat_model,
        device=device,
        model_dir=kwargs.get("model_dir"),
        model_factory=kwargs.get("model_factory"),
        model_kwargs=kwargs.get("model_kwargs"),
        checkpoint_path=kwargs.get("checkpoint_path"),
    )


def load_dataset(dataset_name: str, **kwargs: Any) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
    return DatasetFactory.get_dataset(dataset_name, **kwargs)
