"""Unlabeled adapters. Labels are forbidden inside fit."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import EA_MIN_EPOCHS, EA_SHRINK, N_CH, ZSCORE_EPS


class ChannelZScore:
    """Per-channel mean/std over epochs and time. Fit never sees y."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=(0, 2))
        self.std_ = X.std(axis=(0, 2)) + ZSCORE_EPS
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ChannelZScore.fit has not been called")
        X = np.asarray(X)
        out = (X - self.mean_[None, :, None]) / self.std_[None, :, None]
        return out.astype(X.dtype, copy=False)


class EuclideanAlign:
    """Whitening by mean covariance: X <- R^{-1/2} X. Fit never sees y."""

    def __init__(self) -> None:
        self.R_: np.ndarray | None = None
        self.R_inv_sqrt_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        if n < EA_MIN_EPOCHS:
            self.R_ = np.eye(X.shape[1])
            self.R_inv_sqrt_ = np.eye(X.shape[1])
            return self
        covs = np.stack([_epoch_cov(x) for x in X], axis=0)
        self.R_ = covs.mean(axis=0)
        self.R_inv_sqrt_ = _inv_sqrt(self.R_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.R_inv_sqrt_ is None:
            raise RuntimeError("EuclideanAlign.fit has not been called")
        X = np.asarray(X)
        out = np.einsum("ij,njt->nit", self.R_inv_sqrt_, X)
        return out.astype(X.dtype, copy=False)


def _epoch_cov(x: np.ndarray) -> np.ndarray:
    t = x.shape[-1]
    c = (x @ x.T) / max(t, 1)
    return c + EA_SHRINK * np.eye(x.shape[0])


def _inv_sqrt(R: np.ndarray) -> np.ndarray:
    u, s, _vh = np.linalg.svd(R, full_matrices=True)
    s = np.clip(s, 1e-12, None)
    return (u * (s ** -0.5)) @ u.T


def apply_gap(
    X: np.ndarray,
    subjects: np.ndarray | None = None,
    *,
    zscore: bool = False,
    ea: bool = False,
) -> np.ndarray:
    """Unlabeled z-score and/or EA. Per-subject if `subjects` is given (transductive)."""
    X = np.asarray(X)
    if not zscore and not ea:
        return X
    if subjects is None:
        Y = X
        if zscore:
            Y = ChannelZScore().fit(Y).transform(Y)
        if ea:
            Y = EuclideanAlign().fit(Y).transform(Y)
        return Y

    out = np.empty_like(X)
    for sub in np.unique(subjects):
        mask = subjects == sub
        Yi = X[mask]
        if zscore:
            Yi = ChannelZScore().fit(Yi).transform(Yi)
        if ea:
            if Yi.shape[0] < EA_MIN_EPOCHS:
                pass
            else:
                Yi = EuclideanAlign().fit(Yi).transform(Yi)
        out[mask] = Yi
    return out


def _self_check() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(loc=3.0, scale=4.0, size=(20, N_CH, 256)).astype(np.float32)
    z = ChannelZScore().fit(X)
    Xz = z.transform(X)
    assert Xz.mean() < 1e-5
    assert abs(Xz.std() - 1.0) < 0.05
    ea = EuclideanAlign().fit(X)
    Xa = ea.transform(X)
    assert Xa.shape == X.shape
    subs = np.array(["a"] * 10 + ["b"] * 10)
    Y = apply_gap(X, subs, zscore=True, ea=True)
    assert Y.shape == X.shape
    print("alignment: OK")


if __name__ == "__main__":
    _self_check()
