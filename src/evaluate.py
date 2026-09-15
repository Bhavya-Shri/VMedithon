"""STEW (or proxy) evaluation. Frozen EEGMAT model. Adapter fit never uses y."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ART, N_TIMES, RANDOM_SEED, SHARED_CH
from src.features import feature_names, transform_epochs
from src.train import fit_source, predict_epochs, predict_features


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "cm": cm.tolist(),
        "n": int(len(y_true)),
        "n_rest": int((y_true == 0).sum()),
        "n_load": int((y_true == 1).sum()),
        "both_classes_predicted": bool(len(np.unique(y_pred)) == 2),
    }


def _plot_cm(cm: list, title: str, path: Path) -> None:
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1], ["Rest", "Load"])
    ax.set_yticks([0, 1], ["Rest", "Load"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(arr[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_ablation(
    X_src,
    y_src,
    sub_src,
    X_tgt,
    y_tgt,
    sub_tgt,
    *,
    source_spec: dict,
    target_source: str,
) -> dict:
    """A = naive port. C = z-score. D = z-score+EA. E = train-time fake Emotiv + D adapter.

    y_tgt is evaluation-only. Never passed into fit_source or apply_gap.
    """
    ART.mkdir(parents=True, exist_ok=True)
    scaler, clf, _ = fit_source(
        X_src,
        y_src,
        sub_src,
        ea=source_spec["ea"],
        degrade=False,
        model=source_spec["model"],
    )

    table = {}
    # Pipeline A — channel+rate+filter only (already in npz)
    pred_a, proba_a = predict_epochs(X_tgt, sub_tgt, scaler, clf, zscore=False, ea=False)
    table["A"] = {
        **metrics(y_tgt, pred_a),
        "train_extras": "none",
        "test_adapter": "none",
        "proves": "naive port / collapse",
    }

    pred_c, proba_c = predict_epochs(X_tgt, sub_tgt, scaler, clf, zscore=True, ea=False)
    table["C"] = {
        **metrics(y_tgt, pred_c),
        "train_extras": "P3 reref in EEGMAT preprocess",
        "test_adapter": "target z-score (unlabeled, per subject)",
        "proves": "gain / impedance",
    }

    pred_d, proba_d = predict_epochs(X_tgt, sub_tgt, scaler, clf, zscore=True, ea=True)
    table["D"] = {
        **metrics(y_tgt, pred_d),
        "train_extras": "same as C",
        "test_adapter": "z-score + Euclidean Alignment (unlabeled, per subject)",
        "proves": "spatial mixing",
    }

    scaler_e, clf_e, _ = fit_source(
        X_src,
        y_src,
        sub_src,
        ea=source_spec["ea"],
        degrade=True,
        model=source_spec["model"],
    )
    pred_e, proba_e = predict_epochs(X_tgt, sub_tgt, scaler_e, clf_e, zscore=True, ea=True)
    table["E"] = {
        **metrics(y_tgt, pred_e),
        "train_extras": "train-time fake_emotiv",
        "test_adapter": "z-score + EA",
        "proves": "SNR gap",
    }

    table["F"] = {
        "skipped": True,
        "reason": "Tent needs BatchNorm layers; logistic regression / linear SVM does not use it.",
    }

    # Winner = highest macro-F1 among A/C/D/E provided both classes appear
    ranked = []
    for key in ("A", "C", "D", "E"):
        row = table[key]
        if row.get("both_classes_predicted"):
            ranked.append((key, row["f1"], row))
        else:
            print(f"  {key} predicted one class only — not eligible as winner")
    if not ranked:
        ranked = [(k, table[k]["f1"], table[k]) for k in ("A", "C", "D", "E")]
    winner_id, _, winner_row = max(ranked, key=lambda t: t[1])
    table["winner"] = winner_id
    table["before"] = "A"
    table["target_source"] = target_source
    table["ghost_baseline_scvcnet"] = 0.629

    packs = {
        "A": (pred_a, proba_a, scaler, clf),
        "C": (pred_c, proba_c, scaler, clf),
        "D": (pred_d, proba_d, scaler, clf),
        "E": (pred_e, proba_e, scaler_e, clf_e),
    }
    after_pred, after_proba, after_scaler, after_clf = packs[winner_id]

    np.savez_compressed(
        ART / "stew_preds_before.npz",
        y=y_tgt,
        pred=pred_a,
        proba=proba_a,
        subject=sub_tgt,
    )
    np.savez_compressed(
        ART / "stew_preds_after.npz",
        y=y_tgt,
        pred=after_pred,
        proba=after_proba,
        subject=sub_tgt,
        pipeline=np.array(winner_id),
    )

    _plot_cm(table["A"]["cm"], "Before GAP-Align (pipeline A)", ART / "cm_before.png")
    _plot_cm(winner_row["cm"], f"After GAP-Align (pipeline {winner_id})", ART / "cm_after.png")

    (ART / "ablation.json").write_text(json.dumps(table, indent=2))
    return {
        "table": table,
        "scaler": scaler,
        "clf": clf,
        "after_scaler": after_scaler,
        "after_clf": after_clf,
        "pred_a": pred_a,
        "proba_a": proba_a,
        "after_pred": after_pred,
        "after_proba": after_proba,
        "winner_id": winner_id,
        "winner_zscore": winner_id in ("C", "D", "E"),
        "winner_ea": winner_id in ("D", "E"),
    }


def save_demo_windows(
    X_src,
    y_src,
    sub_src,
    X_tgt,
    y_tgt,
    sub_tgt,
    pack: dict,
    *,
    n_each: int = 10,
) -> Path:
    rng = np.random.default_rng(RANDOM_SEED)
    from src.alignment import apply_gap

    X_after = apply_gap(
        X_tgt,
        sub_tgt,
        zscore=pack["winner_zscore"],
        ea=pack["winner_ea"],
    )
    clinical = _pick_clinical(X_src, y_src, sub_src, pack["scaler"], pack["clf"], n_each, rng)
    wearable = _pick_wearable(
        X_tgt,
        X_after,
        y_tgt,
        sub_tgt,
        pack,
        n_each,
        rng,
    )
    payload = {
        "ch": list(SHARED_CH),
        "sfreq": 128.0,
        "n_times": N_TIMES,
        "feature_names": feature_names(),
        "clinical": clinical,
        "wearable": wearable,
        "winner": pack["winner_id"],
    }
    path = ART / "demo_windows.json"
    path.write_text(json.dumps(payload))
    return path


def _balanced_indices(y, n_each, rng) -> np.ndarray:
    chosen = []
    for label in (0, 1):
        idx = np.where(y == label)[0]
        if len(idx) == 0:
            continue
        take = rng.choice(idx, size=min(max(n_each // 2, 1), len(idx)), replace=False)
        chosen.extend(int(i) for i in take)
    return np.array(chosen, dtype=int)


def _pick_clinical(X, y, subjects, scaler, clf, n_each, rng) -> list[dict]:
    chosen = []
    for i in _balanced_indices(y, n_each, rng):
        epoch = X[i]
        pred, proba = predict_features(transform_epochs(epoch[None, ...]), scaler, clf)
        chosen.append(
            {
                "subject": str(subjects[i]),
                "y": int(y[i]),
                "x": epoch.astype(float).tolist(),
                "pred": int(pred[0]),
                "proba": [float(proba[0, 0]), float(proba[0, 1])],
                "features": transform_epochs(epoch[None, ...])[0].astype(float).tolist(),
            }
        )
    return chosen


def _pick_wearable(X, X_after, y, subjects, pack, n_each, rng) -> list[dict]:
    chosen = []
    for i in _balanced_indices(y, n_each, rng):
        epoch = X[i]
        feat_b = transform_epochs(epoch[None, ...])[0]
        feat_a = transform_epochs(X_after[i][None, ...])[0]
        chosen.append(
            {
                "subject": str(subjects[i]),
                "y": int(y[i]),
                "x": epoch.astype(float).tolist(),
                "features_before": feat_b.astype(float).tolist(),
                "features_after": feat_a.astype(float).tolist(),
                "pred_before": int(pack["pred_a"][i]),
                "proba_before": [float(pack["proba_a"][i, 0]), float(pack["proba_a"][i, 1])],
                "pred_after": int(pack["after_pred"][i]),
                "proba_after": [float(pack["after_proba"][i, 0]), float(pack["after_proba"][i, 1])],
            }
        )
    return chosen


def print_markdown_table(table: dict) -> None:
    print("\n| Pipeline | Acc | Macro-F1 | Kappa | Both classes |")
    print("|---|---:|---:|---:|---|")
    for key in ("A", "C", "D", "E"):
        r = table[key]
        flag = "yes" if r.get("both_classes_predicted") else "NO"
        print(f"| {key} | {r['acc']:.3f} | {r['f1']:.3f} | {r['kappa']:.3f} | {flag} |")
    print(f"\nWinner (macro-F1): {table['winner']}   before = A   target = {table['target_source']}")
    print("Ghost baseline SCVCNet EEGMAT→STEW (no target labels): 0.629 acc")
    if table["target_source"] != "stew":
        print("WARNING: target is NOT STEW. Do not quote these as STEW numbers.")
