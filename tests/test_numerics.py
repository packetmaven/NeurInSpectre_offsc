import torch

from neurinspectre.attacks.numerics import (
    clamp_to_range,
    check_grad_sanity,
    infer_data_range,
    safe_flat_norm,
    transformed_gradient,
)


def test_infer_data_range_and_clamp():
    x = torch.tensor([[[-1.0, 0.5], [2.0, 3.0]]])
    x_min, x_max = infer_data_range(x)
    assert x_min == -1.0
    assert x_max == 3.0
    clamped = clamp_to_range(x, x_min, 1.0)
    assert float(clamped.max()) == 1.0


def test_safe_flat_norm():
    delta = torch.zeros(4, 3, 2, 2)
    norms = safe_flat_norm(delta, p=2)
    assert norms.shape == (4, 1)
    assert float(norms.min()) > 0.0


def test_check_grad_sanity_nan_inf():
    grad = torch.tensor([1.0, float("nan"), 2.0])
    assert not check_grad_sanity(grad, "nan_test")
    grad = torch.tensor([1.0, float("inf")])
    assert not check_grad_sanity(grad, "inf_test")
    grad = torch.tensor([1.0, 2.0])
    assert check_grad_sanity(grad, "ok_test")


def test_transformed_gradient_shape():
    g = torch.randn(2, 3, 4, 4)
    out = transformed_gradient(g)
    assert out.shape == g.shape
