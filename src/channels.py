"""Channel-name translation. No EEG arrays here -- strings only.

Why this file exists: Neurocom EDFs say T3/EEG F3; Emotiv txt says T7/F3.
If we do not lock names AND order, train and test are different montages.
"""
from __future__ import annotations

import sys
from pathlib import Path

# python -m src.channels must see the repo-root config.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CH_ALIAS, CLINICAL_REF, N_CH, SHARED_CH, STEW_CH_FILE

# Proper-case 10-20 / Emotiv labels we expect. Lookup is case-insensitive.
# Unknown names are NOT in this dict: canonicalize keeps them (Guide §4).
_CANON_CASE = {
    n.upper(): n
    for n in [
        "Fp1", "Fp2", "Fpz",
        "AF3", "AF4",
        "F7", "F3", "Fz", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6",
        "T7", "C3", "Cz", "C4", "T8",
        "CP5", "CP6",
        "P7", "P3", "Pz", "P4", "P8",
        "O1", "Oz", "O2",
        "A1", "A2",
        *SHARED_CH,
        CLINICAL_REF,
        *STEW_CH_FILE,
        *CH_ALIAS.values(),
    ]
}


def canonicalize(name: str) -> str:
    """Map one channel label to Emotiv-style 10-20.

    Steps (Guide §4): strip -> drop EEG prefix -> alias T3/T4/T5/T6 -> canonical case.
    Unknown names are returned as stripped text; crash later at pick time, not here.
    """
    raw = str(name).strip()
    if not raw:
        return raw

    raw = _strip_eeg_prefix(raw)
    key = raw.upper()

    if key in CH_ALIAS:
        return CH_ALIAS[key]
    if key in _CANON_CASE:
        return _CANON_CASE[key]
    return raw


def _strip_eeg_prefix(name: str) -> str:
    """Remove 'EEG ', 'EEG-', then bare 'EEG'. Delimited forms first so 'EEG F3' works."""
    upper = name.upper()
    for prefix in ("EEG ", "EEG-", "EEG_"):
        if upper.startswith(prefix):
            return name[len(prefix) :].strip()
    if upper.startswith("EEG") and len(name) > 3:
        return name[3:].strip()
    return name


def pick_shared(ch_names: list[str], needed: list[str] | None = None) -> list[int]:
    """Indices into ch_names for `needed` channels, in `needed` order.

    ch_names: labels as stored in the file (prefixes/aliases ok).
    needed: defaults to SHARED_CH (10 names). Pass [CLINICAL_REF] + SHARED_CH
            before P3 re-reference.
    Returns: list of ints, len == len(needed).
    Raises ValueError if any required channel is missing. Does not impute.
    """
    if needed is None:
        needed = list(SHARED_CH)

    canon = [canonicalize(n) for n in ch_names]
    first_idx = {}
    for i, n in enumerate(canon):
        if n not in first_idx:
            first_idx[n] = i

    missing = [ch for ch in needed if ch not in first_idx]
    if missing:
        raise ValueError(
            f"Missing required channels {missing}. "
            f"Have (canonical): {canon}. Do not impute -- skip this subject."
        )
    return [first_idx[ch] for ch in needed]


def stew_column_index() -> list[int]:
    """Column indices in the 14-col STEW txt for SHARED_CH, same order as EEGMAT.

    Returns: list of 10 ints into columns of shape (n_times, 14).
    """
    file_canon = [canonicalize(n) for n in STEW_CH_FILE]
    idx_by_name = {n: i for i, n in enumerate(file_canon)}
    missing = [ch for ch in SHARED_CH if ch not in idx_by_name]
    if missing:
        raise ValueError(f"STEW_CH_FILE is missing SHARED_CH entries {missing}")
    return [idx_by_name[ch] for ch in SHARED_CH]


def _self_check() -> None:
    """String-only tests. No EDF/txt IO -- that is Phase 5."""
    assert canonicalize("EEG F3") == "F3"
    assert canonicalize("eeg-t3") == "T7"
    assert canonicalize("T4") == "T8"
    assert canonicalize("T5") == "P7"
    assert canonicalize("EEG_T6") == "P8"
    assert canonicalize("EEGF7") == "F7"
    assert canonicalize("P3") == CLINICAL_REF
    assert canonicalize("STI 014") == "STI 014"  # unknown: keep, do not crash

    fake_eegmat = [
        "EEG Fp1", "EEG F3", "EEG F4", "EEG F7", "EEG F8",
        "EEG T3", "EEG T4", "EEG T5", "EEG T6",
        "EEG P3", "EEG O1", "EEG O2", "ECG",
    ]
    idx_with_p3 = pick_shared(fake_eegmat, needed=[CLINICAL_REF, *SHARED_CH])
    picked = [canonicalize(fake_eegmat[i]) for i in idx_with_p3]
    assert picked[0] == CLINICAL_REF
    assert picked[1:] == SHARED_CH

    idx10 = pick_shared(fake_eegmat)
    assert [canonicalize(fake_eegmat[i]) for i in idx10] == SHARED_CH
    assert len(idx10) == N_CH

    try:
        pick_shared(["F3", "F4"])
        raise AssertionError("should have failed on missing channels")
    except ValueError:
        pass

    cols = stew_column_index()
    assert len(cols) == N_CH
    assert [STEW_CH_FILE[i] for i in cols] == SHARED_CH
    # Documented file: AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4
    assert cols == [2, 11, 1, 12, 4, 9, 5, 8, 6, 7]

    print("canonicalize / pick_shared / stew_column_index: OK")
    print("SHARED_CH     ", SHARED_CH)
    print("STEW col idx  ", cols)
    print("STEW -> shared", [STEW_CH_FILE[i] for i in cols])
    print("synthetic EEGMAT map:")
    for raw in fake_eegmat:
        print(f"  {raw!r:16s} -> {canonicalize(raw)}")


if __name__ == "__main__":
    _self_check()
