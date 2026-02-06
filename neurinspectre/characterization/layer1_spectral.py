"""Layer 1 spectral-temporal characterization."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_spectral_features(
    seq: np.ndarray,
    *,
    fs: float = 1.0,
    hf_ratio: float = 0.25,
) -> dict[str, Any]:
    """
    Compute spectral entropy and high-frequency energy ratio.
    """
    arr = np.asarray(seq, dtype=np.float64)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    arr = arr.reshape(-1)
    n = int(arr.size)
    if n < 2:
        raise ValueError("Need at least 2 samples for spectral features.")

    g = np.fft.rfft(arr)
    psd = (np.abs(g) ** 2) / float(n)

    psd_sum = float(np.sum(psd)) if psd.size else 0.0
    if psd_sum <= 0.0:
        return {"spectral_entropy": 0.0, "spectral_entropy_norm": 0.0, "high_freq_ratio": 0.0}

    psd_norm = psd / psd_sum
    spectral_entropy = -float(np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
    spectral_entropy_norm = float(spectral_entropy / np.log2(psd_norm.size))

    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    cutoff = float(fs) * float(hf_ratio)
    hf_mask = freqs >= cutoff
    hf_energy = float(np.sum(psd[hf_mask]))
    high_freq_ratio = float(hf_energy / psd_sum)

    return {
        "spectral_entropy": spectral_entropy,
        "spectral_entropy_norm": spectral_entropy_norm,
        "high_freq_ratio": high_freq_ratio,
    }
