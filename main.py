"""
EEG Cross-Device Workload Pipeline
==================================
Reads your real EEGMAT (.edf) and STEW (.txt) recordings, computes the
features the dashboard visualizes, and writes a single `report.json`
next to this script. Point the dashboard at that file (see the README
block at the bottom of this file) and it will render your real data
instead of the illustrative placeholder numbers.

Dependencies (all lightweight, no mne required — EDF is parsed by hand):
    pip install numpy scipy scikit-learn

Usage:
    python main.py --root "C:\\Users\\nethr\\Desktop\\Dataset"

Expected layout (matches what you showed me):
    Dataset/
      EEGMAT_dataset/
        Subject00_1.edf   <- rest / background   (bigger file)
        Subject00_2.edf   <- task / arithmetic    (smaller file)
        ... Subject01..04
      STEW_dataset/
        sub01_lo.txt       <- rest / low workload
        sub01_hi.txt       <- task / high workload
        ... sub02..48
        ratings.txt
"""

import argparse
import glob
import json
import os
import re
import struct
import sys

import numpy as np
from scipy.signal import welch
from scipy.linalg import fractional_matrix_power
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

_trapz = getattr(np, "trapezoid", None) or np.trapz

# ------------------------------------------------------------------ #
# 1. MONTAGE DEFINITIONS (must match the HTML's MONTAGES exactly —
#    schematic 10-20 layout used only for the 3D visualization; band
#    powers below are the real, computed numbers)
# ------------------------------------------------------------------ #
MONTAGES = {
    "stew": [
        ("AF3", "frontal"), ("F7", "frontal"), ("F3", "frontal"), ("FC5", "frontal"),
        ("T7", "temporal"), ("P7", "parietal"), ("O1", "occipital"), ("O2", "occipital"),
        ("P8", "parietal"), ("T8", "temporal"), ("FC6", "frontal"), ("F4", "frontal"),
        ("F8", "frontal"), ("AF4", "frontal"),
    ],
    "eegmat": [
        ("Fp1", "frontal"), ("Fp2", "frontal"), ("F7", "frontal"), ("F3", "frontal"),
        ("Fz", "frontal"), ("F4", "frontal"), ("F8", "frontal"), ("T7", "temporal"),
        ("C3", "central"), ("Cz", "central"), ("C4", "central"), ("T8", "temporal"),
        ("P7", "parietal"), ("P3", "parietal"), ("Pz", "parietal"), ("P4", "parietal"),
        ("P8", "parietal"), ("O1", "occipital"), ("O2", "occipital"),
    ],
}

# Old <-> modern 10-20 naming, so EEGMAT edf labels (often T3/T4/T5/T6)
# line up with STEW's modern Emotiv naming (T7/T8/P7/P8).
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def clean_label(raw):
    lbl = raw.strip()
    lbl = re.sub(r"^EEG\s*", "", lbl, flags=re.I)
    lbl = re.sub(r"[-_ ]?(REF|A1|A2|LE|M1|M2)$", "", lbl, flags=re.I)
    lbl = lbl.strip()
    # normalize case: Fp1, Fz, Cz, T7 etc. -> title-case each alpha run, keep digits
    m = re.match(r"([A-Za-z]+)(\d*)", lbl)
    if m:
        letters, digits = m.groups()
        lbl = letters.capitalize() + digits
    lbl = ALIASES.get(lbl.upper(), lbl)
    # re-fix casing after alias (aliases are stored upper e.g. T7)
    for canon in ("T7", "T8", "P7", "P8", "Fp1", "Fp2", "Fz", "Cz", "Pz"):
        if lbl.upper() == canon.upper():
            lbl = canon
    return lbl


# ------------------------------------------------------------------ #
# 2. MINIMAL EDF READER (no mne dependency)
# ------------------------------------------------------------------ #
def read_edf(path):
    with open(path, "rb") as f:
        main_header = f.read(256)
        n_records = int(main_header[236:244].decode("ascii").strip())
        record_dur = float(main_header[244:252].decode("ascii").strip() or 1)
        ns = int(main_header[252:256].decode("ascii").strip())

        def read_field(width):
            return [f.read(width).decode("ascii", "ignore").strip() for _ in range(ns)]

        labels = read_field(16)
        _transducer = read_field(80)
        _phys_dim = read_field(8)
        phys_min = [float(x) for x in read_field(8)]
        phys_max = [float(x) for x in read_field(8)]
        dig_min = [int(x) for x in read_field(8)]
        dig_max = [int(x) for x in read_field(8)]
        _prefilt = read_field(80)
        samples_per_record = [int(x) for x in read_field(8)]
        _reserved = read_field(32)

        gains = [(phys_max[i] - phys_min[i]) / (dig_max[i] - dig_min[i] or 1) for i in range(ns)]
        offsets = [phys_min[i] - dig_min[i] * gains[i] for i in range(ns)]

        data = [np.empty(n_records * samples_per_record[i], dtype=np.float32) for i in range(ns)]
        for r in range(n_records):
            for i in range(ns):
                n = samples_per_record[i]
                raw = f.read(n * 2)
                if len(raw) < n * 2:
                    n_records = r
                    break
                vals = np.frombuffer(raw, dtype="<i2").astype(np.float32)
                data[i][r * n:(r + 1) * n] = vals * gains[i] + offsets[i]

    fs_list = [samples_per_record[i] / record_dur for i in range(ns)]
    clean = [clean_label(l) for l in labels]
    return clean, data, fs_list


def load_eegmat_subject(rest_path, task_path, wanted_channels):
    out = {}
    for state, path in (("rest", rest_path), ("task", task_path)):
        labels, data, fs_list = read_edf(path)
        idx_by_label = {l: i for i, l in enumerate(labels)}
        sig = {}
        fs_used = None
        for ch in wanted_channels:
            if ch in idx_by_label:
                i = idx_by_label[ch]
                sig[ch] = data[i]
                fs_used = fs_list[i]
        out[state] = (sig, fs_used or fs_list[0])
    return out


# ------------------------------------------------------------------ #
# 3. STEW TEXT READER
# ------------------------------------------------------------------ #
STEW_FS = 128.0  # Emotiv EPOC sampling rate used for the STEW dataset


def load_stew_file(path):
    try:
        arr = np.loadtxt(path, delimiter=",")
    except ValueError:
        arr = np.loadtxt(path)
    channels = [c for c, _ in MONTAGES["stew"]]
    sig = {ch: arr[:, i] for i, ch in enumerate(channels) if i < arr.shape[1]}
    return sig


# ------------------------------------------------------------------ #
# 4. FEATURES: band power + PSD
# ------------------------------------------------------------------ #
def bandpowers(x, fs):
    nper = int(min(len(x), max(fs * 2, 256)))
    f, p = welch(x, fs=fs, nperseg=nper)
    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (f >= lo) & (f <= hi)
        out[name] = float(_trapz(p[mask], f[mask])) if mask.any() else 1e-12
    return out, f, p


def workload_index(bp):
    # frontal-theta / posterior-alpha style engagement index, log-scaled
    return float(np.log(bp["theta"] + 1e-12) + np.log(bp["beta"] + 1e-12) - np.log(bp["alpha"] + 1e-12))


def minmax(d):
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in d.items()}


# ------------------------------------------------------------------ #
# 5. DOMAIN ADAPTATION HELPERS
# ------------------------------------------------------------------ #
def coral(source, target):
    """CORAL: recolor source covariance to match target covariance."""
    eps = 1e-5
    cs = np.cov(source, rowvar=False) + eps * np.eye(source.shape[1])
    ct = np.cov(target, rowvar=False) + eps * np.eye(target.shape[1])
    cs_inv_sqrt = fractional_matrix_power(cs, -0.5).real
    ct_sqrt = fractional_matrix_power(ct, 0.5).real
    return source @ cs_inv_sqrt @ ct_sqrt


def rbf_mmd(x, y, gamma=None):
    if gamma is None:
        gamma = 1.0 / x.shape[1]

    def k(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d2)

    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())


# ------------------------------------------------------------------ #
# 6. MAIN PIPELINE
# ------------------------------------------------------------------ #
def main(root):
    eegmat_dir = os.path.join(root, "EEGMAT_dataset")
    stew_dir = os.path.join(root, "STEW_dataset")

    stew_channels = [c for c, _ in MONTAGES["stew"]]
    eegmat_channels = [c for c, _ in MONTAGES["eegmat"]]
    overlap = [c for c in stew_channels if c in eegmat_channels]
    print("Overlap channels used for cross-device features:", overlap)

    # ---- load EEGMAT ----
    eegmat_files = sorted(glob.glob(os.path.join(eegmat_dir, "Subject*_1.edf")))
    eegmat_subjects = []
    for rest_path in eegmat_files:
        sid = re.match(r"Subject(\d+)_1\.edf", os.path.basename(rest_path)).group(1)
        task_path = os.path.join(eegmat_dir, f"Subject{sid}_2.edf")
        if not os.path.exists(task_path):
            continue
        print("Reading EEGMAT subject", sid)
        rec = load_eegmat_subject(rest_path, task_path, eegmat_channels)
        eegmat_subjects.append((sid, rec))

    # ---- load STEW ----
    stew_lo_files = sorted(glob.glob(os.path.join(stew_dir, "sub*_lo.txt")))
    stew_subjects = []
    for lo_path in stew_lo_files:
        sid = re.match(r"sub(\d+)_lo\.txt", os.path.basename(lo_path)).group(1)
        hi_path = os.path.join(stew_dir, f"sub{sid}_hi.txt")
        if not os.path.exists(hi_path):
            continue
        print("Reading STEW subject", sid)
        rest_sig = load_stew_file(lo_path)
        task_sig = load_stew_file(hi_path)
        stew_subjects.append((sid, {"rest": (rest_sig, STEW_FS), "task": (task_sig, STEW_FS)}))

    if not eegmat_subjects or not stew_subjects:
        print("ERROR: no subjects loaded — check --root path.", file=sys.stderr)
        sys.exit(1)

    # ---- per-channel workload index, averaged across subjects ----
    def device_activity(subjects, channels):
        acc = {"rest": {ch: [] for ch in channels}, "task": {ch: [] for ch in channels}}
        psd_acc = {"rest": [], "task": []}
        for sid, rec in subjects:
            for state in ("rest", "task"):
                sig, fs = rec[state]
                for ch in channels:
                    if ch not in sig:
                        continue
                    bp, f, p = bandpowers(sig[ch], fs)
                    acc[state][ch].append(workload_index(bp))
                    psd_acc[state].append(np.interp(np.arange(1, 45, 1.0), f, p))
        activity = {
            state: minmax({ch: float(np.mean(v)) for ch, v in acc[state].items() if v})
            for state in ("rest", "task")
        }
        psd = {
            state: [[float(fr), float(pw)] for fr, pw in
                     zip(np.arange(1, 45, 1.0), np.mean(psd_acc[state], axis=0))]
            for state in ("rest", "task") if psd_acc[state]
        }
        return activity, psd

    stew_activity, stew_psd = device_activity(stew_subjects, stew_channels)
    eegmat_activity, eegmat_psd = device_activity(eegmat_subjects, eegmat_channels)

    # ---- connectivity: correlation of band-power envelopes across channels ----
    def device_connectivity(subjects, channels):
        conn = {"rest": [], "task": []}
        for state in ("rest", "task"):
            mats = []
            for sid, rec in subjects:
                sig, fs = rec[state]
                present = [ch for ch in channels if ch in sig]
                if len(present) < 2:
                    continue
                feats = np.array([sig[ch][: min(len(sig[ch]) for ch in present)] for ch in present])
                mats.append((present, np.corrcoef(feats)))
            if not mats:
                continue
            present = mats[0][0]
            avg = np.mean([m for _, m in mats if len(_) == len(present)], axis=0)
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    w = float(abs(avg[i, j]))
                    if w > 0.3:
                        conn[state].append([present[i], present[j], round(w, 3)])
        return conn

    stew_conn = device_connectivity(stew_subjects, stew_channels)
    eegmat_conn = device_connectivity(eegmat_subjects, eegmat_channels)

    # ---- cross-device features on overlap channels (trial = subject x state) ----
    def build_feature_table(subjects, channels):
        X, y, groups = [], [], []
        for sid, rec in subjects:
            for label, state in ((0, "rest"), (1, "task")):
                sig, fs = rec[state]
                row = []
                ok = True
                for ch in overlap:
                    if ch not in sig:
                        ok = False
                        break
                    bp, _, _ = bandpowers(sig[ch], fs)
                    row.extend([np.log(bp["theta"] + 1e-12), np.log(bp["alpha"] + 1e-12), np.log(bp["beta"] + 1e-12)])
                if ok:
                    X.append(row)
                    y.append(label)
                    groups.append(sid)
        return np.array(X), np.array(y), np.array(groups)

    Xs, ys, gs = build_feature_table(stew_subjects, stew_channels)
    Xe, ye, ge = build_feature_table(eegmat_subjects, eegmat_channels)

    scaler = StandardScaler().fit(Xs)
    Xs_n, Xe_n = scaler.transform(Xs), scaler.transform(Xe)

    # accuracy before adaptation: train on all STEW, test on EEGMAT (LOSO avg)
    clf = SVC(kernel="rbf").fit(Xs_n, ys)
    acc_before = accuracy_score(ye, clf.predict(Xe_n)) * 100

    # CORAL: align EEGMAT (target) features toward STEW (source) space, retrain
    Xe_aligned = coral(Xe_n, Xs_n)
    clf2 = SVC(kernel="rbf").fit(Xs_n, ys)
    acc_after = accuracy_score(ye, clf2.predict(Xe_aligned)) * 100

    mmd_before = rbf_mmd(Xs_n, Xe_n)
    mmd_after = rbf_mmd(Xs_n, Xe_aligned)

    # 2D projection for the alignment scatter panel
    pca_before = PCA(n_components=2).fit(np.vstack([Xs_n, Xe_n]))
    before_2d = pca_before.transform(np.vstack([Xs_n, Xe_n]))
    pca_after = PCA(n_components=2).fit(np.vstack([Xs_n, Xe_aligned]))
    after_2d = pca_after.transform(np.vstack([Xs_n, Xe_aligned]))

    def scale_to_canvas(pts, w=460, h=230, pad=40):
        pts = np.array(pts)
        lo, hi = pts.min(0), pts.max(0)
        span = np.where(hi - lo == 0, 1, hi - lo)
        norm = (pts - lo) / span
        out = norm * [w - 2 * pad, h - 2 * pad] + pad
        return out

    before_scaled = scale_to_canvas(before_2d)
    after_scaled = scale_to_canvas(after_2d)

    points = []
    n_s = len(ys)
    for i in range(n_s):
        points.append({
            "device": "stew", "label": int(ys[i]),
            "before": {"x": float(before_scaled[i, 0]), "y": float(before_scaled[i, 1])},
            "after": {"x": float(after_scaled[i, 0]), "y": float(after_scaled[i, 1])},
        })
    for j in range(len(ye)):
        points.append({
            "device": "eegmat", "label": int(ye[j]),
            "before": {"x": float(before_scaled[n_s + j, 0]), "y": float(before_scaled[n_s + j, 1])},
            "after": {"x": float(after_scaled[n_s + j, 0]), "y": float(after_scaled[n_s + j, 1])},
        })

    report = {
        "generated_from": "real recordings (see paths below)",
        "paths": {"eegmat": eegmat_dir, "stew": stew_dir},
        "n_subjects": {"eegmat": len(eegmat_subjects), "stew": len(stew_subjects)},
        "overlap_channels": overlap,
        "activity": {"stew": stew_activity, "eegmat": eegmat_activity},
        "psd": {"stew": stew_psd, "eegmat": eegmat_psd},
        "connectivity": {"stew": stew_conn, "eegmat": eegmat_conn},
        "alignment": {"points": points},
        "metrics": {
            "accuracy_before": round(acc_before, 1),
            "accuracy_after": round(acc_after, 1),
            "mmd_before": round(mmd_before, 5),
            "mmd_after": round(mmd_after, 5),
        },
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"EEGMAT subjects: {len(eegmat_subjects)}  |  STEW subjects: {len(stew_subjects)}")
    print(f"Cross-device accuracy: {acc_before:.1f}% -> {acc_after:.1f}% after CORAL")
    print(f"MMD: {mmd_before:.5f} -> {mmd_after:.5f} after CORAL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Dataset", help="Path to the folder containing EEGMAT_dataset/ and STEW_dataset/")
    args = ap.parse_args()
    main(args.root)

# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------
# 1. pip install numpy scipy scikit-learn
# 2. python main.py --root "C:\Users\nethr\Desktop\Dataset"
#    (takes a few minutes — STEW is 48 subjects x 2 files x ~4.3MB)
# 3. Copy the generated report.json into the same folder as
#    eeg_workload_dashboard.html
# 4. Serve both files from a local server (fetch() of local JSON is blocked
#    under file:// in Chrome/Edge):
#      python -m http.server 8000
#    then open http://localhost:8000/eeg_workload_dashboard.html
#    (Firefox can usually open the HTML directly via file:// too.)
# ---------------------------------------------------------------------------
