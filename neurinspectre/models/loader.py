"""
Model loading utilities for reproducible evaluations.

Currently supports robustbench models (recommended for CIFAR-10/ImageNet).
"""

from __future__ import annotations

from typing import Any


def _import_robustbench():
    try:
        from robustbench.utils import load_model as rb_load_model
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "robustbench is required to load pretrained robustness models. "
            "Install it with: pip install robustbench"
        ) from exc
    return rb_load_model


def _parse_factory(factory: str):
    import importlib

    if ":" not in factory:
        raise ValueError("model_factory must be 'module:function'")
    mod_name, fn_name = factory.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise ValueError("model_factory does not reference a callable")
    return fn


def load_model(
    model_name: str | None = None,
    *,
    dataset: str = "cifar10",
    threat_model: str = "Linf",
    device: str = "cpu",
    model_dir: str | None = None,
    model_factory: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    checkpoint_path: str | None = None,
):
    """
    Load a model for the requested dataset/threat model.

    Preferred path: robustbench pretrained models.
    Alternative path: user-provided model factory + checkpoint.
    """
    if model_factory:
        fn = _parse_factory(model_factory)
        kwargs = model_kwargs or {}
        model = fn(**kwargs)
        if checkpoint_path:
            import torch

            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
        elif model_name is None:
            raise ValueError("checkpoint_path is required when using model_factory without model_name.")
        model = model.to(device)
        model.eval()
        return model

    if not model_name:
        raise ValueError("model_name is required unless model_factory is provided.")

    rb_load_model = _import_robustbench()
    model = rb_load_model(
        model_name,
        dataset=dataset,
        threat_model=threat_model,
        model_dir=model_dir,
    )
    model = model.to(device)
    model.eval()
    return model
