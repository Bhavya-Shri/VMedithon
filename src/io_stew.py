"""STEW (IEEE DataPort / Emotiv EPOC) I/O. Audit and load only -- no filter/epoch here.

STEW lo/hi in the filename is an EVALUATION label only. Never pass it into adapter fit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SFREQ_TGT, STEW_CH_FILE, STEW_DIR

_SUB_RE = re.compile(r"sub(\d+)_(lo|hi)\.txt$", re.IGNORECASE)


def subject_id_from_path(path: Path | str) -> str:
    """'sub01_lo.txt' -> '01'."""
    path = Path(path)
    m = _SUB_RE.search(path.name)
    if not m:
        raise ValueError(f"Unrecognized STEW filename: {path.name}")
    return m.group(1)


def eval_label_from_path(path: Path | str) -> int:
    """lo=0 rest, hi=1 load. FOR EVALUATION METRICS ONLY. Do not fit on this."""
    path = Path(path)
    m = _SUB_RE.search(path.name)
    if not m:
        raise ValueError(f"Unrecognized STEW filename: {path.name}")
    return 0 if m.group(2).lower() == "lo" else 1


def list_txt_paths(root: Path | None = None) -> list[Path]:
    root = Path(root) if root is not None else STEW_DIR
    return sorted(p for p in root.rglob("*.txt") if _SUB_RE.search(p.name))


def list_subjects(root: Path | None = None) -> list[str]:
    return sorted({subject_id_from_path(p) for p in list_txt_paths(root)})


def load_txt(path: Path | str) -> np.ndarray:
    """Load one STEW txt to float64 array (n_times, n_cols).

    Tries no header first; if that fails or first row looks non-numeric, skiprows=1.
    """
    path = Path(path)
    try:
        arr = np.loadtxt(path)
    except ValueError:
        arr = np.loadtxt(path, skiprows=1)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _unit_guess(arr: np.ndarray) -> str:
    peak = float(max(abs(np.nanmin(arr)), abs(np.nanmax(arr))))
    # Guide §5.3: +/-1e-4 -> Volts; +/-50 -> uV
    if peak < 1e-1:
        return "Volts (values look like +/-1e-4 scale; multiply by 1e6)"
    return "uV (values look like +/-50 scale; do not multiply)"


def audit_one(path: Path) -> dict:
    arr = load_txt(path)
    n_times, n_cols = arr.shape
    sfreq = float(SFREQ_TGT)  # STEW txt usually has no header rate; native EPOC is 128 Hz
    return {
        "path": str(path),
        "subject": subject_id_from_path(path),
        "eval_label": eval_label_from_path(path),
        "shape": arr.shape,
        "n_cols": n_cols,
        "sfreq_assumed": sfreq,
        "duration_s": n_times / sfreq if sfreq else float("nan"),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "first_row": arr[0].tolist() if n_times else [],
        "unit_guess": _unit_guess(arr),
        "documented_ch": list(STEW_CH_FILE),
    }


def main() -> None:
    paths = list_txt_paths()
    print(f"STEW dir:   {STEW_DIR}")
    print(f"txt files:  {len(paths)}")
    if not paths:
        print("NO STEW FILES.")
        print("Download from IEEE DataPort (account required):")
        print("  https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset")
        print("  DOI 10.21227/44r8-ya50")
        print("Extract sub##_lo.txt / sub##_hi.txt into data/raw/stew/")
        print("Do NOT substitute another Emotiv dataset.")
        return

    subjects = list_subjects()
    print(f"subjects:   {len(subjects)} -> {subjects}")

    probe = paths[0]
    info = audit_one(probe)
    print(f"probe file: {Path(info['path']).name}")
    print(f"shape:      {info['shape']}  (expect n_times, 14)")
    print(f"n_cols:     {info['n_cols']}")
    print(f"sfreq:      {info['sfreq_assumed']} Hz assumed (EPOC native; confirm if 256)")
    print(f"duration:   {info['duration_s']:.3f} s")
    print(f"min/max:    {info['min']:.6g} / {info['max']:.6g}")
    print(f"unit_guess: {info['unit_guess']}")
    print(f"first_row:  {info['first_row']}")
    print(f"documented: {info['documented_ch']}")
    if info["n_cols"] == 13:
        print("WARNING: 13 columns -- likely a header/index column (Guide §20).")
    if info["n_cols"] != 14:
        print(f"WARNING: expected 14 columns, got {info['n_cols']}")


if __name__ == "__main__":
    main()
