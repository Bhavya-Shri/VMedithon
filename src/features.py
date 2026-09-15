"""Log band-power features. 10 channels x 3 bands = 30-D. No raw time-series into the classifier."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BANDS, LOG_BP_EPS, N_FEATURES, SFREQ_TGT, SHARED_CH, WELCH_NPERSEG


def feature_names() -> list[str]:
    """F3_theta, F3_alpha, F3_beta, ... same order as bandpower_vector."""
    names = []
    for ch in SHARED_CH:
        for band in BANDS:
            names.append(f"{ch}_{band}")
    return names


def bandpower_vector(x: np.ndarray, sfreq: float = SFREQ_TGT) -> np.ndarray:
    """One epoch (10, T) -> (30,) log10 mean PSD in theta/alpha/beta."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (n_ch, n_times), got {x.shape}")
    freqs, psd = welch(x, fs=sfreq, nperseg=min(WELCH_NPERSEG, x.shape[-1]), axis=-1)
    vec = np.empty(x.shape[0] * len(BANDS), dtype=np.float32)
    k = 0
    for ch_i in range(x.shape[0]):
        for f0, f1 in BANDS.values():
            mask = (freqs >= f0) & (freqs < f1)
            if not np.any(mask):
                raise ValueError(f"no Welch bins in band {(f0, f1)} Hz")
            vec[k] = np.log10(float(psd[ch_i, mask].mean()) + LOG_BP_EPS)
            k += 1
    return vec


def transform_epochs(X: np.ndarray, sfreq: float = SFREQ_TGT) -> np.ndarray:
    """X (N, 10, T) -> (N, 30) float32. Vectorized Welch; order matches bandpower_vector."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"expected (N, ch, T), got {X.shape}")
    nperseg = min(WELCH_NPERSEG, X.shape[-1])
    freqs, psd = welch(X, fs=sfreq, nperseg=nperseg, axis=-1)
    parts = []
    for f0, f1 in BANDS.values():
        mask = (freqs >= f0) & (freqs < f1)
        if not np.any(mask):
            raise ValueError(f"no Welch bins in band {(f0, f1)} Hz")
        parts.append(np.log10(psd[..., mask].mean(axis=-1) + LOG_BP_EPS))
    stacked = np.stack(parts, axis=-1)  # (N, 10, 3) theta/alpha/beta
    return stacked.reshape(X.shape[0], N_FEATURES).astype(np.float32)


def _self_check() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(256) / 128.0
    x = np.stack([np.sin(2 * np.pi * 10 * t) for _ in range(10)], axis=0)
    vec = bandpower_vector(x)
    assert vec.shape == (N_FEATURES,)
    names = feature_names()
    assert len(names) == N_FEATURES
    assert names[0] == "F3_theta" and names[1] == "F3_alpha" and names[2] == "F3_beta"
    F = transform_epochs(rng.normal(size=(4, 10, 256)))
    assert F.shape == (4, 30)
    assert np.allclose(transform_epochs(x[None, ...])[0], vec, atol=1e-5)
    print("features: OK", names[:6], "...", vec[:3])


if __name__ == "__main__":
    _self_check()
