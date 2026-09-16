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
        "ea_on_source": bool(source_spec.get("ea", True)),
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
    """Build Page 1 pairs. Traces stay sensor µV. Features/preds use the same path as the frozen model."""
    rng = np.random.default_rng(RANDOM_SEED)
    from src.alignment import apply_gap

    n_rest = max(n_each // 2, 1)
    n_load = max(n_each - n_rest, 1)
    ea_src = bool(pack.get("ea_on_source", True))
    X_src_model = apply_gap(X_src, sub_src, zscore=False, ea=True) if ea_src else np.asarray(X_src)
    X_after = apply_gap(
        X_tgt,
        sub_tgt,
        zscore=pack["winner_zscore"],
        ea=pack["winner_ea"],
    )
    rest_c = _pick_clinical_indices(X_src_model, y_src, sub_src, pack["scaler"], pack["clf"], 0, n_rest, rng)
    load_c = _pick_clinical_indices(X_src_model, y_src, sub_src, pack["scaler"], pack["clf"], 1, n_load, rng)
    rest_w = _pick_wearable_indices(y_tgt, sub_tgt, pack, 0, n_rest, rng)
    load_w = _pick_wearable_indices(y_tgt, sub_tgt, pack, 1, n_load, rng)
    clinical = [_clinical_record(X_src, X_src_model, y_src, sub_src, pack, i) for i in rest_c + load_c]
    wearable = [_wearable_record(X_tgt, X_after, y_tgt, sub_tgt, pack, i) for i in rest_w + load_w]
    payload = {
        "ch": list(SHARED_CH),
        "sfreq": 128.0,
        "n_times": N_TIMES,
        "feature_names": feature_names(),
        "clinical": clinical,
        "wearable": wearable,
        "winner": pack["winner_id"],
        "ea_on_source": ea_src,
        "pairing": "slots 1-N rest, then load; different people; clinical feats include source EA if trained with it",
    }
    path = ART / "demo_windows.json"
    path.write_text(json.dumps(payload))
    return path


def _take_unique(indices, subjects, k, rng) -> list[int]:
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0 or k <= 0:
        return []
    order = np.arange(len(indices))
    rng.shuffle(order)
    picked, seen = [], set()
    for j in order:
        i = int(indices[j])
        s = str(subjects[i])
        if s in seen:
            continue
        seen.add(s)
        picked.append(i)
        if len(picked) >= k:
            return picked
    for j in order:
        i = int(indices[j])
        if i not in picked:
            picked.append(i)
        if len(picked) >= k:
            break
    return picked


def _pick_clinical_indices(X_model, y, subjects, scaler, clf, label: int, k: int, rng) -> list[int]:
    feats = transform_epochs(X_model)
    pred, _ = predict_features(feats, scaler, clf)
    y = np.asarray(y)
    ok = np.where((y == label) & (pred == label))[0]
    any_lab = np.where(y == label)[0]
    picked = _take_unique(ok, subjects, k, rng)
    if len(picked) < k:
        extra = _take_unique(any_lab, subjects, k - len(picked), rng)
        picked.extend(i for i in extra if i not in picked)
    return picked[:k]


def _pick_wearable_indices(y, subjects, pack, label: int, k: int, rng) -> list[int]:
    y = np.asarray(y)
    pred_b = np.asarray(pack["pred_a"])
    pred_a = np.asarray(pack["after_pred"])
    if label == 0:
        prefer = np.where((y == 0) & (pred_b == 1) & (pred_a == 0))[0]
        fallback = np.where((y == 0) & (pred_a == 0))[0]
        last = np.where(y == 0)[0]
    else:
        prefer = np.where((y == 1) & (pred_a == 1))[0]
        fallback = np.where(y == 1)[0]
        last = fallback
    picked = _take_unique(prefer, subjects, k, rng)
    if len(picked) < k:
        picked.extend(i for i in _take_unique(fallback, subjects, k - len(picked), rng) if i not in picked)
    if len(picked) < k:
        picked.extend(i for i in _take_unique(last, subjects, k - len(picked), rng) if i not in picked)
    return picked[:k]


def _clinical_record(X_raw, X_model, y, subjects, pack, i: int) -> dict:
    feat = transform_epochs(X_model[i][None, ...])
    pred, proba = predict_features(feat, pack["scaler"], pack["clf"])
    return {
        "subject": str(subjects[i]),
        "y": int(y[i]),
        "x": np.asarray(X_raw[i], dtype=float).tolist(),
        "pred": int(pred[0]),
        "proba": [float(proba[0, 0]), float(proba[0, 1])],
        "features": feat[0].astype(float).tolist(),
        "model_path": "source_EA+logBP+scaler" if pack.get("ea_on_source", True) else "logBP+scaler",
    }


def _wearable_record(X_raw, X_after, y, subjects, pack, i: int) -> dict:
    feat_b = transform_epochs(np.asarray(X_raw[i])[None, ...])[0]
    feat_a = transform_epochs(np.asarray(X_after[i])[None, ...])[0]
    return {
        "subject": str(subjects[i]),
        "y": int(y[i]),
        "x": np.asarray(X_raw[i], dtype=float).tolist(),
        "features_before": feat_b.astype(float).tolist(),
        "features_after": feat_a.astype(float).tolist(),
        "pred_before": int(pack["pred_a"][i]),
        "proba_before": [float(pack["proba_a"][i, 0]), float(pack["proba_a"][i, 1])],
        "pred_after": int(pack["after_pred"][i]),
        "proba_after": [float(pack["after_proba"][i, 0]), float(pack["after_proba"][i, 1])],
        "model_path_before": "logBP+scaler (pipeline A)",
        "model_path_after": "unlabeled z-score+logBP+scaler (winner C)"
        if pack["winner_id"] == "C"
        else f"adapter {pack['winner_id']}+logBP+scaler",
    }


def rebuild_demo_windows_from_artifacts(*, n_each: int = 10) -> Path:
    """Rebuild demo_windows.json from frozen npz/joblib/preds. Does not refit the classifier."""
    import joblib

    from config import DATA_PROC

    src = np.load(DATA_PROC / "eegmat_epochs.npz")
    tgt = np.load(DATA_PROC / "stew_epochs.npz")
    before = np.load(ART / "stew_preds_before.npz")
    after = np.load(ART / "stew_preds_after.npz")
    meta = json.loads((ART / "train_meta.json").read_text())
    table = json.loads((ART / "ablation.json").read_text())
    winner = table["winner"]
    pack = {
        "scaler": joblib.load(ART / "scaler.joblib"),
        "clf": joblib.load(ART / "clf.joblib"),
        "pred_a": before["pred"],
        "proba_a": before["proba"],
        "after_pred": after["pred"],
        "after_proba": after["proba"],
        "winner_id": winner,
        "winner_zscore": winner in ("C", "D", "E"),
        "winner_ea": winner in ("D", "E"),
        "ea_on_source": bool(meta.get("ea_on_source", True)),
    }
    if len(tgt["y"]) != len(before["pred"]) or len(tgt["y"]) != len(after["pred"]):
        raise RuntimeError("stew_preds_*.npz length does not match stew_epochs.npz — rerun python -m src.run_all")
    path = save_demo_windows(
        src["X"], src["y"], src["subject"], tgt["X"], tgt["y"], tgt["subject"], pack, n_each=n_each
    )
    print(f"wrote {path}")
    return path


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


if __name__ == "__main__":
    rebuild_demo_windows_from_artifacts()
