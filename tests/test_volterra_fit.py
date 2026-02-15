import numpy as np

from neurinspectre.mathematical.volterra import (
    fit_volterra_kernel,
    predict_volterra_power_law,
)


def test_fit_volterra_kernel_power_law_returns_finite_estimate():
    n = 24
    alpha_true = 0.72
    c_true = 0.35
    y = predict_volterra_power_law(1.0, alpha=alpha_true, c=c_true, n=n, dt=1.0)
    # Build a small multi-dimensional sequence with mild perturbations.
    grad = np.stack([y, y * 0.95, y * 1.05], axis=1)
    grad += np.random.default_rng(0).normal(scale=1e-3, size=grad.shape)

    kernel, rmse, info = fit_volterra_kernel(
        grad,
        kernel_type="power_law",
        method="L-BFGS-B",
        verbose=False,
    )

    assert np.isfinite(rmse)
    assert np.isfinite(float(kernel.alpha))
    assert 0.1 <= float(kernel.alpha) <= 0.99
    assert isinstance(info, dict)


def test_fit_volterra_kernel_exponential_is_stable():
    n = 20
    t = np.arange(n, dtype=np.float64)
    y = np.exp(-0.2 * t)
    grad = np.stack([y, y * 0.8], axis=1)

    _kernel, rmse, info = fit_volterra_kernel(
        grad,
        kernel_type="exponential",
        method="L-BFGS-B",
        verbose=False,
    )
    assert np.isfinite(rmse)
    assert isinstance(info, dict)
