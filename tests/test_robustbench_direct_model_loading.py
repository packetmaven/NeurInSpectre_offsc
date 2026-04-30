import pytest


def test_model_factory_robustbench_direct_calls_rb_loader(monkeypatch):
    rb = pytest.importorskip("robustbench")

    import torch.nn as nn

    from neurinspectre.models.factory import ModelFactory

    calls = []

    class Dummy(nn.Module):
        def forward(self, x):
            return x

    def fake_load_model(*, model_name, dataset, threat_model, model_dir=None):
        calls.append(
            {
                "model_name": model_name,
                "dataset": dataset,
                "threat_model": threat_model,
                "model_dir": model_dir,
            }
        )
        return Dummy()

    # Patch robustbench loader to avoid any network/model download.
    import robustbench.utils as rb_utils

    monkeypatch.setattr(rb_utils, "load_model", fake_load_model, raising=True)

    m = ModelFactory.load_model(
        domain="vision",
        model_name="Zhang2019Theoretically",
        training_type="robustbench",
        dataset="cifar10",
        device="cpu",
        threat_model="Linf",
        model_dir="/tmp/rb_models",
    )

    assert isinstance(m, nn.Module)
    assert m.training is False
    assert calls == [
        {
            "model_name": "Zhang2019Theoretically",
            "dataset": "cifar10",
            "threat_model": "Linf",
            "model_dir": "/tmp/rb_models",
        }
    ]

