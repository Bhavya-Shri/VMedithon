"""Preprocess: physics match for EEGMAT (Phase 6) and STEW (Phase 7).

Filter the whole recording, then window -- never filter per-epoch (Guide §20).
STEW: no P3 re-reference. lo/hi labels are stored for EVALUATION only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    BANDPASS,
    CLINICAL_REF,
    DATA_PROC,
    EPOCH_SEC,
    EPOCH_STRIDE_SEC,
    N_CH,
    N_TIMES,
    NOTCH,
    PTP_DROP_FRAC,
    PTP_LOOSEN_UV,
    PTP_REJECT_UV,
    RANDOM_SEED,
    SFREQ_TGT,
    SHARED_CH,
)
from src.channels import canonicalize, stew_column_index
from src.io_eegmat import label_from_path, list_edf_paths, load_raw, subject_id_from_path
from src.io_stew import eval_label_from_path, list_txt_paths, load_txt
from src.io_stew import subject_id_from_path as stew_subject_id

# Names that are not scalp EEG. Checked on the RAW file labels before rename.
_NON_EEG_TOKENS = ("ECG", "STI", "STIM", "STATUS", "TIME", "MARKER", "RESP")


def _drop_non_eeg(raw) -> list[str]:
    """Step 2: drop ECG/stim/markers/time. Why: they are not 10-20 scalp sites."""
    drop = []
    for ch in raw.ch_names:
        u = ch.upper()
        if any(tok in u for tok in _NON_EEG_TOKENS):
            drop.append(ch)
    if drop:
        raw.drop_channels(drop)
    return drop


def _rename_canonical(raw) -> dict[str, str]:
    """Strip EEG prefixes and T3->T7 so P3/SHARED_CH lookups succeed."""
    mapping = {}
    for old in raw.ch_names:
        new = canonicalize(old)
        if new != old:
            mapping[old] = new
    if mapping:
        raw.rename_channels(mapping)
    return mapping


def reref_p3(raw):
    """Imitate Emotiv CMS: subtract P3 from every channel, then drop P3.

    After this, P3 is ~0. Do not average-reference as a fallback.
    """
    if CLINICAL_REF not in raw.ch_names:
        raise ValueError(f"{CLINICAL_REF} missing; cannot re-reference. Skip this subject.")
    raw.set_eeg_reference(ref_channels=[CLINICAL_REF], projection=False, verbose="ERROR")
    p3 = raw.get_data(picks=[CLINICAL_REF])
    p3_max_uv = float(np.max(np.abs(p3))) * 1e6  # still Volts here
    raw.drop_channels([CLINICAL_REF])
    return raw, p3_max_uv


def filter_resample(raw):
    """Notch 50 Hz (if sfreq>100), band-pass 1-40 Hz firwin zero-phase, resample 128 Hz.

    Anti-alias lives inside resample: you cannot downsample 500->128 without it
    or 40-64 Hz would fold back into the band we keep.
    """
    sfreq = float(raw.info["sfreq"])
    if sfreq > 100:
        raw.notch_filter(NOTCH, verbose="ERROR")
    raw.filter(
        l_freq=BANDPASS[0],
        h_freq=BANDPASS[1],
        fir_design="firwin",
        phase="zero",
        verbose="ERROR",
    )
    if abs(sfreq - SFREQ_TGT) > 0.1:
        raw.resample(SFREQ_TGT, npad="auto", verbose="ERROR")
    return raw


def make_epochs(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Sliding windows. data (n_ch, n_times) -> (n_epochs, n_ch, n_times_win).

    Incomplete last window is dropped. Filter already applied on the full take.
    """
    n_ch, n_times = data.shape
    win = int(round(sfreq * EPOCH_SEC))
    step = int(round(sfreq * EPOCH_STRIDE_SEC))
    if win != N_TIMES:
        raise ValueError(f"epoch length {win} != N_TIMES {N_TIMES} at sfreq={sfreq}")
    starts = range(0, n_times - win + 1, step)
    if not starts:
        return np.zeros((0, n_ch, win), dtype=np.float32)
    stacked = np.stack([data[:, s : s + win] for s in starts], axis=0)
    return stacked.astype(np.float32, copy=False)


def reject_ptp(X: np.ndarray, thresh_uv: float) -> tuple[np.ndarray, np.ndarray]:
    """Drop epoch if any channel peak-to-peak exceeds thresh_uv.

    X: (n_epochs, n_ch, T) in microvolts.
    Returns kept X and boolean keep mask.
    """
    if X.size == 0:
        return X, np.zeros((0,), dtype=bool)
    ptp = X.max(axis=-1) - X.min(axis=-1)  # (n_epochs, n_ch)
    keep = ptp.max(axis=1) <= thresh_uv
    return X[keep], keep


def _ptp_with_loosen(data_uv: np.ndarray, sfreq: float) -> tuple[np.ndarray, int, float, float]:
    """Epoch then PTP-reject. Loosen 200->300 uV if more than 40% of windows drop."""
    X = make_epochs(data_uv, sfreq)
    n_before = int(X.shape[0])
    thresh = PTP_REJECT_UV
    X, _keep = reject_ptp(X, thresh)
    drop_frac = 1.0 - (X.shape[0] / n_before) if n_before else 0.0
    if n_before and drop_frac > PTP_DROP_FRAC:
        X, _keep = reject_ptp(make_epochs(data_uv, sfreq), PTP_LOOSEN_UV)
        thresh = PTP_LOOSEN_UV
        drop_frac = 1.0 - (X.shape[0] / n_before)
    return X, n_before, thresh, drop_frac


def process_eegmat_file(path, label: int | None = None) -> dict | None:
    """One EDF -> epochs. Returns None if P3 or SHARED_CH missing (skip, do not impute).

    Out: epochs (n, 10, 256) float32 uV at 128 Hz, y (n,) int8, subject_id, meta.
    """
    path = Path(path)
    if label is None:
        label = label_from_path(path)
    subject_id = subject_id_from_path(path)
    raw = load_raw(path, preload=True)
    dropped_non_eeg = _drop_non_eeg(raw)
    mapping = _rename_canonical(raw)
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    # Montage after rename (Guide §6.2 step 3). A2-A1 has no 10-20 site -- warn, continue.
    raw.set_montage("standard_1020", on_missing="warn", verbose="ERROR")

    if CLINICAL_REF not in raw.ch_names:
        print(f"SKIP {path.name}: no {CLINICAL_REF}. Not using average reference.")
        return None
    missing = [ch for ch in SHARED_CH if ch not in raw.ch_names]
    if missing:
        print(f"SKIP {path.name}: missing SHARED_CH {missing}. Not imputing.")
        return None

    raw, p3_max_uv = reref_p3(raw)
    raw.pick(SHARED_CH)  # error if any missing; order becomes SHARED_CH
    if list(raw.ch_names) != list(SHARED_CH):
        raise RuntimeError(f"pick did not lock order: {raw.ch_names} vs {SHARED_CH}")

    sfreq_before = float(raw.info["sfreq"])
    raw = filter_resample(raw)
    sfreq = float(raw.info["sfreq"])
    data_uv = raw.get_data() * 1e6
    X, n_before, thresh, drop_frac = _ptp_with_loosen(data_uv, sfreq)

    y = np.full((X.shape[0],), int(label), dtype=np.int8)
    meta = {
        "path": str(path),
        "dropped_non_eeg": dropped_non_eeg,
        "rename": mapping,
        "p3_max_abs_uV_after_reref": p3_max_uv,
        "sfreq_before": sfreq_before,
        "sfreq": sfreq,
        "n_epochs_before_ptp": n_before,
        "n_epochs": int(X.shape[0]),
        "ptp_thresh_uV": thresh,
        "ptp_drop_frac": drop_frac,
        "data_min_uV": float(X.min()) if X.size else float("nan"),
        "data_max_uV": float(X.max()) if X.size else float("nan"),
        "ch": list(raw.ch_names),
    }
    return {"epochs": X, "y": y, "subject_id": subject_id, "meta": meta}


def process_stew_file(path) -> dict | None:
    """One STEW txt -> epochs. NO P3 re-reference (CMS already in hardware).

    y from lo/hi is EVALUATION-ONLY. Never pass it into z-score/EA/classifier fit.
    Native values are uV with a large DC offset -- do NOT *1e6.
    MNE RawArray wants Volts, so we /1e6 for the filter and *1e6 after -- same FIR as EEGMAT.
    """
    path = Path(path)
    arr = load_txt(path)
    if arr.ndim != 2 or arr.shape[1] < max(stew_column_index()) + 1:
        print(f"SKIP {path.name}: expected 14 columns, got {arr.shape}. Not imputing.")
        return None

    cols = stew_column_index()
    picked_uv = arr[:, cols].T.astype(np.float64)  # (10, n_times), SHARED_CH order
    n_times = picked_uv.shape[1]
    # 19200 samples -> 150 s at 128 Hz, 75 s at 256 Hz (EPOC X). Detect from length if needed.
    sfreq_native = float(SFREQ_TGT)
    info = mne.create_info(list(SHARED_CH), sfreq=sfreq_native, ch_types="eeg")
    raw = mne.io.RawArray(picked_uv / 1e6, info, verbose="ERROR")
    raw.set_montage("standard_1020", on_missing="warn", verbose="ERROR")
    raw = filter_resample(raw)
    sfreq = float(raw.info["sfreq"])
    data_uv = raw.get_data() * 1e6
    X, n_before, thresh, drop_frac = _ptp_with_loosen(data_uv, sfreq)
    # eval_label_from_path: stored in npz for final metrics only
    y = np.full((X.shape[0],), eval_label_from_path(path), dtype=np.int8)
    meta = {
        "path": str(path),
        "reref": False,
        "sfreq": sfreq,
        "n_times_raw": int(n_times),
        "n_epochs_before_ptp": n_before,
        "n_epochs": int(X.shape[0]),
        "ptp_thresh_uV": thresh,
        "ptp_drop_frac": drop_frac,
        "data_min_uV": float(X.min()) if X.size else float("nan"),
        "data_max_uV": float(X.max()) if X.size else float("nan"),
        "ch": list(SHARED_CH),
        "y_is_eval_only": True,
    }
    return {"epochs": X, "y": y, "subject_id": stew_subject_id(path), "meta": meta}


def load_eegmat_recording(path, label: int) -> dict:
    """Guide §6.1 name. Same as process_eegmat_file (Guide §25)."""
    out = process_eegmat_file(path, label=label)
    if out is None:
        raise ValueError(f"Skipped {path}")
    return out


def build_npz(results: list[dict], out_path: Path) -> Path:
    """Stack recordings into one npz. X float32 uV, y int8, subject per epoch."""
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    X = np.concatenate([r["epochs"] for r in results], axis=0).astype(np.float32)
    y = np.concatenate([r["y"] for r in results], axis=0).astype(np.int8)
    subject = np.concatenate(
        [np.array([r["subject_id"]] * len(r["y"]), dtype="U8") for r in results]
    )
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        subject=subject,
        ch=np.array(SHARED_CH),
        sfreq=np.float32(SFREQ_TGT),
    )
    return out_path


def main() -> None:
    np.random.seed(RANDOM_SEED)
    mne.set_log_level("ERROR")
    paths = list_edf_paths()
    print(f"EEGMAT files: {len(paths)}")
    results = []
    skipped = []
    for path in tqdm(paths, desc="EEGMAT preprocess"):
        rec = process_eegmat_file(path)
        if rec is None:
            skipped.append(path.name)
            continue
        m = rec["meta"]
        print(
            f"  {path.name} sub={rec['subject_id']} y={int(rec['y'][0]) if len(rec['y']) else 'empty'} "
            f"epochs={m['n_epochs']}/{m['n_epochs_before_ptp']} "
            f"ptp={m['ptp_thresh_uV']}uV drop={m['ptp_drop_frac']:.2%} "
            f"P3~0 max={m['p3_max_abs_uV_after_reref']:.3f}uV "
            f"range=[{m['data_min_uV']:.1f},{m['data_max_uV']:.1f}]uV "
            f"ch={m['ch']}"
        )
        results.append(rec)

    if not results:
        print("No EEGMAT recordings survived. Not writing npz.")
        return

    out = DATA_PROC / "eegmat_epochs.npz"
    build_npz(results, out)
    z = np.load(out, allow_pickle=False)
    X, y, sub, ch, sfreq = z["X"], z["y"], z["subject"], z["ch"], z["sfreq"]
    print("--- npz ---")
    print("path     ", out)
    print("X        ", X.shape, X.dtype, "min/max", float(X.min()), float(X.max()))
    print("y        ", y.shape, y.dtype, "counts", {int(k): int((y == k).sum()) for k in np.unique(y)})
    print("subject  ", sub.shape, list(np.unique(sub)))
    print("ch       ", list(ch))
    print("sfreq    ", float(sfreq))
    print("skipped  ", skipped)
    assert list(ch) == SHARED_CH
    assert X.shape[1:] == (N_CH, N_TIMES)
    assert X.dtype == np.float32
    assert y.dtype == np.int8
    assert abs(float(sfreq) - SFREQ_TGT) < 0.1
    # Units: after *1e6, scalp EEG is tens to hundreds of uV, not 1e-4.
    peak = max(abs(float(X.min())), abs(float(X.max())))
    print("unit_ok  ", "uV-scale" if 1.0 < peak < 5000 else f"CHECK peak={peak}")
    print("P3 in ch ", CLINICAL_REF in list(ch), "(must be False -- dropped after reref)")


def _print_npz(out: Path, skipped: list[str], *, stew: bool = False) -> None:
    z = np.load(out, allow_pickle=False)
    X, y, sub, ch, sfreq = z["X"], z["y"], z["subject"], z["ch"], z["sfreq"]
    print("--- npz ---")
    print("path     ", out)
    print("X        ", X.shape, X.dtype, "min/max", float(X.min()), float(X.max()))
    print("y        ", y.shape, y.dtype, "counts", {int(k): int((y == k).sum()) for k in np.unique(y)})
    if stew:
        print("y note   EVALUATION ONLY -- do not fit adapter/classifier on this")
    print("subject  ", sub.shape, "n_subj", len(np.unique(sub)))
    print("ch       ", list(ch))
    print("sfreq    ", float(sfreq))
    print("skipped  ", skipped)
    assert list(ch) == SHARED_CH
    assert X.shape[1:] == (N_CH, N_TIMES)
    assert X.dtype == np.float32
    assert y.dtype == np.int8
    assert abs(float(sfreq) - SFREQ_TGT) < 0.1
    peak = max(abs(float(X.min())), abs(float(X.max())))
    print("unit_ok  ", "uV-scale" if 1.0 < peak < 5000 else f"CHECK peak={peak}")
    print("P3 in ch ", CLINICAL_REF in list(ch), "(must be False)")


def main_stew() -> None:
    np.random.seed(RANDOM_SEED)
    mne.set_log_level("ERROR")
    paths = list_txt_paths()
    print(f"STEW files: {len(paths)}")
    results = []
    skipped = []
    n_loosen = 0
    for path in tqdm(paths, desc="STEW preprocess"):
        rec = process_stew_file(path)
        if rec is None:
            skipped.append(path.name)
            continue
        if rec["epochs"].shape[0] == 0:
            skipped.append(f"{path.name} (0 epochs after PTP)")
            continue
        m = rec["meta"]
        if m["ptp_thresh_uV"] != PTP_REJECT_UV:
            n_loosen += 1
        if m["ptp_drop_frac"] > 0.10 or m["n_epochs"] == 0:
            print(
                f"  {path.name} sub={rec['subject_id']} y_eval={int(rec['y'][0]) if len(rec['y']) else 'empty'} "
                f"epochs={m['n_epochs']}/{m['n_epochs_before_ptp']} "
                f"ptp={m['ptp_thresh_uV']}uV drop={m['ptp_drop_frac']:.2%} "
                f"range=[{m['data_min_uV']:.1f},{m['data_max_uV']:.1f}]uV"
            )
        results.append(rec)

    if not results:
        print("No STEW recordings survived. Not writing npz.")
        return

    out = DATA_PROC / "stew_epochs.npz"
    build_npz(results, out)
    _print_npz(out, skipped, stew=True)
    print("ptp loosened files", n_loosen)

    eegmat_path = DATA_PROC / "eegmat_epochs.npz"
    if eegmat_path.exists():
        em = np.load(eegmat_path, allow_pickle=False)
        st = np.load(out, allow_pickle=False)
        assert list(em["ch"]) == list(st["ch"]) == SHARED_CH
        assert em["X"].shape[1:] == st["X"].shape[1:] == (N_CH, N_TIMES)
        print("assert    EEGMAT ch/shape == STEW ch/shape == SHARED_CH (10, 256) OK")
    else:
        print("assert    skipped -- eegmat_epochs.npz missing")


if __name__ == "__main__":
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "eegmat"
    if which in ("stew", "target"):
        main_stew()
    else:
        main()
