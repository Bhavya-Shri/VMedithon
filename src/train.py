"""Frozen source classifier. Trained on EEGMAT only. Never fits on STEW y."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    ART,
    LR_C,
    LR_CLASS_WEIGHT,
    LR_MAX_ITER,
    LR_SOLVER,
    RANDOM_SEED,
    SFREQ_TGT,
    SHARED_CH,
)
from src.alignment import apply_gap
from src.degrade import maybe_degrade_batch
from src.features import feature_names, transform_epochs


def make_logreg() -> LogisticRegression:
    return LogisticRegression(
        solver=LR_SOLVER,
        class_weight=LR_CLASS_WEIGHT,
        max_iter=LR_MAX_ITER,
        C=LR_C,
        random_state=RANDOM_SEED,
    )


def make_linearsvc():
    base = LinearSVC(
        class_weight="balanced",
        C=1.0,
        random_state=RANDOM_SEED,
        max_iter=8000,
        dual="auto",
    )
    return CalibratedClassifierCV(base, cv=3, method="sigmoid")


def _prepare_features(
    X: np.ndarray,
    subjects: np.ndarray,
    *,
    ea: bool,
    degrade: bool,
    rng: np.random.Generator | None,
) -> np.ndarray:
    if degrade:
        if rng is None:
            raise ValueError("degrade=True requires an rng")
        X = maybe_degrade_batch(X, rng)
    if ea:
        X = apply_gap(X, subjects, zscore=False, ea=True)
    return transform_epochs(X)


def loso_eegmat(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    ea: bool = True,
    degrade: bool = False,
    model: str = "logreg",
) -> dict:
    """Leave-one-subject-out on EEGMAT. Chooses nothing using STEW.

    When degrade is off, per-subject EA + band-power are computed once. That is not
    leakage: each subject's whitener uses only that subject's unlabeled epochs.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    unique = np.unique(subjects)
    y_true_all = []
    y_pred_all = []
    per_subject = []
    if not degrade:
        feats = _prepare_features(X, subjects, ea=ea, degrade=False, rng=None)
    for held in unique:
        tr = subjects != held
        te = subjects == held
        if te.sum() == 0 or tr.sum() == 0:
            continue
        if degrade:
            Ftr = _prepare_features(X[tr], subjects[tr], ea=ea, degrade=True, rng=rng)
            Fte = _prepare_features(X[te], subjects[te], ea=ea, degrade=False, rng=None)
        else:
            Ftr, Fte = feats[tr], feats[te]
        scaler = StandardScaler()
        Ftr_s = scaler.fit_transform(Ftr)
        Fte_s = scaler.transform(Fte)
        clf = make_logreg() if model == "logreg" else make_linearsvc()
        clf.fit(Ftr_s, y[tr])
        pred = clf.predict(Fte_s)
        acc = float(accuracy_score(y[te], pred))
        f1 = float(f1_score(y[te], pred, average="macro", zero_division=0))
        per_subject.append({"subject": str(held), "acc": acc, "f1": f1, "n": int(te.sum())})
        y_true_all.append(y[te])
        y_pred_all.append(pred)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return {
        "model": model,
        "ea": ea,
        "degrade": degrade,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_subjects": int(len(unique)),
        "n_epochs": int(len(y_true)),
        "per_subject": per_subject,
    }


def pick_source_model(X, y, subjects) -> dict:
    """Keep whichever of logreg / LinearSVC is better on EEGMAT LOSO, not on STEW."""
    candidates = []
    for model in ("logreg", "linearsvc"):
        for ea in (True, False):
            try:
                result = loso_eegmat(X, y, subjects, ea=ea, degrade=False, model=model)
            except Exception as exc:
                print(f"  LOSO {model} ea={ea} failed: {exc}")
                continue
            candidates.append(result)
            print(
                f"  LOSO {model:10s} ea={str(ea):5s}  acc={result['acc']:.3f}  f1={result['f1']:.3f}"
            )
    if not candidates:
        raise RuntimeError("All EEGMAT LOSO candidates failed")
    best = max(candidates, key=lambda r: (r["f1"], r["acc"]))
    print(f"  winner on EEGMAT LOSO: {best['model']} ea={best['ea']} acc={best['acc']:.3f}")
    return best


def fit_source(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    ea: bool = True,
    degrade: bool = False,
    model: str = "logreg",
) -> tuple[StandardScaler, object, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    feats = _prepare_features(X, subjects, ea=ea, degrade=degrade, rng=rng)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(feats)
    clf = make_logreg() if model == "logreg" else make_linearsvc()
    clf.fit(Xs, y)
    return scaler, clf, feats


def save_model(scaler, clf, meta: dict) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, ART / "scaler.joblib")
    joblib.dump(clf, ART / "clf.joblib")
    meta = {
        **meta,
        "channels": list(SHARED_CH),
        "sfreq": SFREQ_TGT,
        "seed": RANDOM_SEED,
        "feature_names": feature_names(),
        "C": LR_C,
    }
    (ART / "train_meta.json").write_text(json.dumps(meta, indent=2))


def load_model():
    scaler = joblib.load(ART / "scaler.joblib")
    clf = joblib.load(ART / "clf.joblib")
    meta = json.loads((ART / "train_meta.json").read_text())
    return scaler, clf, meta


def predict_features(feats: np.ndarray, scaler, clf) -> tuple[np.ndarray, np.ndarray]:
    Xs = scaler.transform(feats)
    pred = clf.predict(Xs)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xs)
    else:
        # Should not happen with CalibratedClassifierCV / LogisticRegression
        scores = clf.decision_function(Xs)
        p1 = 1.0 / (1.0 + np.exp(-scores))
        proba = np.stack([1 - p1, p1], axis=1)
    return pred, proba


def predict_epochs(
    X: np.ndarray,
    subjects: np.ndarray | None,
    scaler,
    clf,
    *,
    zscore: bool = False,
    ea: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    Xa = apply_gap(X, subjects, zscore=zscore, ea=ea)
    feats = transform_epochs(Xa)
    return predict_features(feats, scaler, clf)
