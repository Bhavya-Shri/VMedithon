"""Train-time fake-Emotiv and the Streamlit degrade slider. One function family so sim and trainer cannot drift."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import NOISE_P, NOISE_STD_UV, QUANTIZE_UV, SFREQ_TGT


def fake_emotiv(
    epoch: np.ndarray,
    rng: np.random.Generator,
    std_uv: float = NOISE_STD_UV,
    lsb: float = QUANTIZE_UV,
) -> np.ndarray:
    """epoch (10, T) in µV -> noisy 14-bit quantized copy. Used only while training, and by the slider."""
    epoch = np.asarray(epoch, dtype=np.float64)
    noise = rng.normal(0.0, std_uv, size=epoch.shape)
    q = np.round((epoch + noise) / lsb) * lsb
    return q.astype(np.float32)


def maybe_degrade_batch(
    X: np.ndarray,
    rng: np.random.Generator,
    p: float = NOISE_P,
    std_uv: float = NOISE_STD_UV,
    lsb: float = QUANTIZE_UV,
) -> np.ndarray:
    """Replace each epoch with fake_emotiv with probability p. Train only."""
    out = np.array(X, copy=True)
    for i in range(out.shape[0]):
        if rng.random() < p:
            out[i] = fake_emotiv(out[i], rng, std_uv=std_uv, lsb=lsb)
    return out


def slider_degrade(epoch: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Hardware-gap slider. 0 = aligned clinical epoch, 1 = full fake EPOC.

    0.00–0.25 identity (already 10 ch / 128 Hz)
    0.25–0.50 Gaussian noise 0 → 8 µV
    0.50–0.75 14-bit quantize
    0.75–1.00 extra noise + 4–8 Hz jitter
    """
    x = np.array(epoch, dtype=np.float64, copy=True)
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.25:
        return x.astype(np.float32)

    noise_mix = min(1.0, (amount - 0.25) / 0.25)
    x = x + rng.normal(0.0, noise_mix * 8.0, size=x.shape)

    if amount > 0.50:
        x = np.round(x / QUANTIZE_UV) * QUANTIZE_UV

    if amount > 0.75:
        extra = (amount - 0.75) / 0.25
        x = x + rng.normal(0.0, extra * 4.0, size=x.shape)
        t = np.arange(x.shape[-1]) / SFREQ_TGT
        freq = float(rng.uniform(4.0, 8.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        jitter = extra * 3.0 * np.sin(2.0 * np.pi * freq * t + phase)
        x = x + jitter[None, :]

    return x.astype(np.float32)


def _self_check() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(10, 256)).astype(np.float32) * 20
    y = fake_emotiv(x, rng)
    assert y.shape == x.shape
    z0 = slider_degrade(x, 0.0, rng)
    assert np.allclose(z0, x)
    z1 = slider_degrade(x, 1.0, rng)
    assert z1.shape == x.shape
    print("degrade: OK")


if __name__ == "__main__":
    _self_check()
