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

    # Draft Section 3.1: Morlet CWT energy at octave-ish scales {2,4,8,16}.
    #
    # We implement a small, dependency-free approximation via discrete convolution
    # (Gaussian-windowed cosine). This is intentionally lightweight: Layer-1 uses
    # short gradient sequences (T~50-200) and the goal is an auditable scalar
    # signature rather than a full time-frequency map.
    def _morlet_energy_1d(x: np.ndarray, *, scales: list[float], w0: float = 5.0) -> dict[str, float]:
        out: dict[str, float] = {}
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        for s in scales:
            scale = float(s)
            sigma = scale / float(fs)
            t_max = max(int(np.ceil(4.0 * sigma * float(fs))), 1)
            t = np.arange(-t_max, t_max + 1, dtype=np.float64) / float(fs)
            kernel = np.exp(-(t**2) / (2.0 * sigma**2)) * np.cos(float(w0) * t / scale)
            kernel = kernel / (float(np.linalg.norm(kernel)) + 1e-12)
            # "Same" convolution with stable centering even when len(kernel) > len(x).
            full = np.convolve(x, kernel, mode="full")
            k = int(kernel.size)
            start = (k - 1) // 2
            conv = full[start : start + x.size]
            out[f"scale_{int(scale)}"] = float(np.mean(np.abs(conv) ** 2))
        return out

    wavelet_energy = _morlet_energy_1d(arr, scales=[2.0, 4.0, 8.0, 16.0], w0=5.0)

    return {
        "spectral_entropy": spectral_entropy,
        "spectral_entropy_norm": spectral_entropy_norm,
        "high_freq_ratio": high_freq_ratio,
        "wavelet_energy": wavelet_energy,
    }
