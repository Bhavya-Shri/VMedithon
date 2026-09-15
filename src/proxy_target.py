"""Wearable proxy when STEW txt files are not on disk.

This is a hardware-shift simulation of EEGMAT (same neural activity, Emotiv-like SNR /
gain / mixing). It is NOT STEW. Official EEGMAT→STEW numbers require IEEE DataPort files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_PROC, N_CH, NOISE_STD_UV, QUANTIZE_UV, RANDOM_SEED, SHARED_CH
from src.degrade import fake_emotiv


def build_proxy_from_eegmat(
    eegmat_path: Path | None = None,
    out_path: Path | None = None,
    seed: int = RANDOM_SEED,
) -> Path:
    eegmat_path = Path(eegmat_path) if eegmat_path else DATA_PROC / "eegmat_epochs.npz"
    out_path = Path(out_path) if out_path else DATA_PROC / "stew_epochs.npz"
    z = np.load(eegmat_path, allow_pickle=False)
    X, y, subject = z["X"], z["y"], z["subject"]
    rng = np.random.default_rng(seed + 7)
    Xp = np.empty_like(X)
    for sub in np.unique(subject):
        mask = subject == sub
        gain = rng.uniform(0.45, 2.3, size=(N_CH, 1)).astype(np.float32)
        bias = rng.normal(0.0, 14.0, size=(N_CH, 1)).astype(np.float32)
        mix = np.eye(N_CH, dtype=np.float32) + rng.normal(0.0, 0.16, size=(N_CH, N_CH)).astype(np.float32)
        idx = np.where(mask)[0]
        for i in idx:
            x = mix @ (gain * X[i] + bias)
            x = fake_emotiv(x, rng, std_uv=NOISE_STD_UV * 1.35, lsb=QUANTIZE_UV)
            Xp[i] = x
    sub_out = np.array([f"w{s}" for s in subject], dtype="U8")
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=Xp,
        y=y.astype(np.int8),
        subject=sub_out,
        ch=np.array(SHARED_CH) if "ch" not in z.files else z["ch"],
        sfreq=z["sfreq"],
        proxy=np.array("eegmat_degraded_proxy"),
    )
    print(f"proxy npz {out_path}  X={Xp.shape}  (NOT STEW)")
    return out_path
