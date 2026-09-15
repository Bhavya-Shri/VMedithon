"""EEGMAT (PhysioNet / Neurocom) I/O. Audit and load only -- no filter/epoch here."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import EEGMAT_DIR

_SUBJECT_RE = re.compile(r"Subject(\d+)_([12])\.edf$", re.IGNORECASE)


def subject_id_from_path(path: Path | str) -> str:
    """'Subject00_1.edf' -> '00'. Same string for rest (_1) and task (_2)."""
    path = Path(path)
    m = _SUBJECT_RE.search(path.name)
    if not m:
        raise ValueError(f"Unrecognized EEGMAT filename: {path.name}")
    return m.group(1)


def label_from_path(path: Path | str) -> int:
    """Filename label only: _1 rest=0, _2 arithmetic=1. Not used for STEW adaptation."""
    path = Path(path)
    m = _SUBJECT_RE.search(path.name)
    if not m:
        raise ValueError(f"Unrecognized EEGMAT filename: {path.name}")
    return 0 if m.group(2) == "1" else 1


def list_edf_paths(root: Path | None = None) -> list[Path]:
    root = Path(root) if root is not None else EEGMAT_DIR
    return sorted(root.rglob("Subject*_*.edf"))


def list_subjects(root: Path | None = None) -> list[str]:
    """Unique subject ids, sorted, e.g. ['00', '01', ...]."""
    ids = {subject_id_from_path(p) for p in list_edf_paths(root)}
    return sorted(ids)


def load_raw(path: Path | str, *, preload: bool = True):
    """Read one EDF. Returns MNE Raw. Channel names are as in the file (not canonicalized).

    MNE data are in Volts. Convert to microvolts at preprocess time, not here.
    """
    path = Path(path)
    raw = mne.io.read_raw_edf(path, preload=preload, verbose="ERROR")
    return raw


def _volts_to_uv_stats(raw) -> dict:
    """Min/max in Volts and uV so we catch a missing *1e6 later."""
    data = raw.get_data()  # (n_ch, n_times), Volts
    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))
    return {
        "n_ch": int(data.shape[0]),
        "n_times": int(data.shape[1]),
        "min_V": vmin,
        "max_V": vmax,
        "min_uV": vmin * 1e6,
        "max_uV": vmax * 1e6,
        "unit_guess": "Volts" if max(abs(vmin), abs(vmax)) < 0.1 else "likely already uV-scale",
    }


def audit_one(path: Path) -> dict:
    raw = load_raw(path)
    stats = _volts_to_uv_stats(raw)
    sfreq = float(raw.info["sfreq"])
    duration = stats["n_times"] / sfreq if sfreq else float("nan")
    return {
        "path": str(path),
        "subject": subject_id_from_path(path),
        "file_label": label_from_path(path),
        "sfreq": sfreq,
        "ch_names": list(raw.ch_names),
        "duration_s": duration,
        **stats,
    }


def main() -> None:
    paths = list_edf_paths()
    print(f"EEGMAT dir: {EEGMAT_DIR}")
    print(f"EDF files:  {len(paths)}")
    if not paths:
        print("NO EEGMAT FILES. Run wfdb.dl_database('eegmat', 'data/raw/eegmat')")
        return

    subjects = list_subjects()
    print(f"subjects:   {len(subjects)} -> {subjects}")

    probe = None
    for p in paths:
        if p.name.lower().startswith("subject00_1"):
            probe = p
            break
    if probe is None:
        probe = paths[0]

    info = audit_one(probe)
    print(f"probe file: {Path(info['path']).name}")
    print(f"sfreq:      {info['sfreq']} Hz  (header, not assumed 500)")
    print(f"duration:   {info['duration_s']:.3f} s")
    print(f"n_ch:       {info['n_ch']}  n_times: {info['n_times']}")
    print(f"ch_names:   {info['ch_names']}")
    print(f"min/max V:  {info['min_V']:.6e} / {info['max_V']:.6e}")
    print(f"min/max uV: {info['min_uV']:.3f} / {info['max_uV']:.3f}")
    print(f"unit_guess: {info['unit_guess']}")

    sfreqs = set()
    n_ch_set = set()
    for p in paths:
        raw = mne.io.read_raw_edf(p, preload=False, verbose="ERROR")
        sfreqs.add(float(raw.info["sfreq"]))
        n_ch_set.add(len(raw.ch_names))
    print(f"all sfreq:  {sorted(sfreqs)}")
    print(f"all n_ch:   {sorted(n_ch_set)}")


if __name__ == "__main__":
    main()
