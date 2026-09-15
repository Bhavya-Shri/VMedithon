"""Rebuild processed epochs, train the frozen EEGMAT model, run unlabeled ablation, dump artifacts."""
from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ART, DATA_PROC, EEGMAT_DIR, LOSO_EXIT, LOSO_PASS, LOSO_WARN, N_CH, N_TIMES, SHARED_CH
from src.io_eegmat import list_edf_paths
from src.io_stew import list_txt_paths
from src.preprocess import main as preprocess_eegmat
from src.preprocess import main_stew as preprocess_stew
from src.proxy_target import build_proxy_from_eegmat
from src.train import fit_source, pick_source_model, save_model
from src.evaluate import print_markdown_table, run_ablation, save_demo_windows

EEGMAT_ZIP = (
    "https://physionet.org/static/published-projects/eegmat/"
    "eeg-during-mental-arithmetic-tasks-1.0.0.zip"
)
EEGMAT_FILE = "https://physionet.org/files/eegmat/1.0.0/{name}"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def ensure_eegmat(min_files: int = 6) -> None:
    paths = list_edf_paths()
    if len(paths) >= min_files:
        print(f"EEGMAT on disk: {len(paths)} EDFs")
        return
    EEGMAT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = EEGMAT_DIR / "eegmat.zip"
    try:
        _download(EEGMAT_ZIP, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if member.lower().endswith(".edf"):
                    target = EEGMAT_DIR / Path(member).name
                    if not target.exists():
                        target.write_bytes(zf.read(member))
        zip_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"zip download failed ({exc}); trying individual EDFs")
        for i in range(36):
            for cond in (1, 2):
                name = f"Subject{i:02d}_{cond}.edf"
                dest = EEGMAT_DIR / name
                if dest.exists() and dest.stat().st_size > 0:
                    continue
                try:
                    _download(EEGMAT_FILE.format(name=name), dest)
                except Exception as e2:
                    print(f"  skip {name}: {e2}")
    paths = list_edf_paths()
    print(f"EEGMAT files after download: {len(paths)}")
    if not paths:
        raise SystemExit("No EEGMAT EDFs. Place files in data/raw/eegmat/")


def _load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {
        "X": z["X"],
        "y": z["y"],
        "subject": z["subject"],
        "ch": list(z["ch"]),
        "sfreq": float(z["sfreq"]),
        "proxy": "proxy" in z.files,
    }


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    ensure_eegmat()

    eegmat_npz = DATA_PROC / "eegmat_epochs.npz"
    stew_npz = DATA_PROC / "stew_epochs.npz"
    if not eegmat_npz.exists():
        preprocess_eegmat()
    src = _load_npz(eegmat_npz)
    assert src["ch"] == SHARED_CH
    assert src["X"].shape[1:] == (N_CH, N_TIMES)

    stew_files = list_txt_paths()
    target_source = "stew"
    if stew_files:
        if not stew_npz.exists() or "proxy" in np.load(stew_npz).files:
            preprocess_stew()
        tgt = _load_npz(stew_npz)
        if tgt.get("proxy"):
            raise SystemExit("STEW txt present but npz still marked proxy — delete data/processed/stew_epochs.npz")
    else:
        print("STEW txt files not found under data/raw/stew/.")
        print("Building an Emotiv-like PROXY from degraded EEGMAT so the dashboard can run.")
        print("These metrics are NOT EEGMAT→STEW. Do not quote them as STEW.")
        build_proxy_from_eegmat(eegmat_npz, stew_npz)
        tgt = _load_npz(stew_npz)
        target_source = "eegmat_degraded_proxy"

    assert tgt["ch"] == SHARED_CH
    assert tgt["X"].shape[1:] == src["X"].shape[1:] == (N_CH, N_TIMES)

    print("\n=== EEGMAT LOSO (source sanity) ===")
    best = pick_source_model(src["X"], src["y"], src["subject"])
    acc = best["acc"]
    if acc < LOSO_EXIT:
        raise SystemExit(f"EEGMAT LOSO acc={acc:.3f} < {LOSO_EXIT}. Debug channels/units/labels. Not running STEW.")
    if acc < LOSO_WARN:
        print(f"WARN LOSO acc={acc:.3f} < {LOSO_WARN}")
    if acc < LOSO_PASS:
        print(
            f"NOTE LOSO acc={acc:.3f} < {LOSO_PASS} hard gate. "
            f"n_subjects={best['n_subjects']}. Smoke test only if this is a sample, not the full 36."
        )

    scaler, clf, _feats = fit_source(
        src["X"], src["y"], src["subject"], ea=best["ea"], degrade=False, model=best["model"]
    )
    save_model(
        scaler,
        clf,
        {
            "loso_acc": best["acc"],
            "loso_f1": best["f1"],
            "loso_per_subject": best["per_subject"],
            "model": best["model"],
            "ea_on_source": best["ea"],
            "n_train_epochs": int(len(src["y"])),
            "n_train_subjects": int(len(np.unique(src["subject"]))),
            "target_source": target_source,
            "y_counts_src": {str(int(k)): int((src["y"] == k).sum()) for k in np.unique(src["y"])},
            "disclaimer": "Classifier fitted on EEGMAT only. Adapter fit never uses target labels.",
        },
    )

    print("\n=== Unlabeled target ablation ===")
    pack = run_ablation(
        src["X"],
        src["y"],
        src["subject"],
        tgt["X"],
        tgt["y"],
        tgt["subject"],
        source_spec=best,
        target_source=target_source,
    )
    save_model(
        pack["after_scaler"] if pack["winner_id"] == "E" else scaler,
        pack["after_clf"] if pack["winner_id"] == "E" else clf,
        json.loads((ART / "train_meta.json").read_text())
        | {
            "winner": pack["winner_id"],
            "target_source": target_source,
        },
    )
    # Keep the frozen A/C/D scaler+clf as scaler.joblib (no train-time noise).
    # If E wins, also store the noisy-train model separately.
    import joblib

    joblib.dump(pack["scaler"], ART / "scaler.joblib")
    joblib.dump(pack["clf"], ART / "clf.joblib")
    if pack["winner_id"] == "E":
        joblib.dump(pack["after_scaler"], ART / "scaler_E.joblib")
        joblib.dump(pack["after_clf"], ART / "clf_E.joblib")

    save_demo_windows(
        src["X"], src["y"], src["subject"], tgt["X"], tgt["y"], tgt["subject"], pack
    )
    print_markdown_table(pack["table"])
    print(f"\nartifacts in {ART}")
    print("Next: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
