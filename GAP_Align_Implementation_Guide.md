# GAP-Align — Complete Implementation Guide

**Upload this file into the IDE and follow it in order.**  
Do not skip Phase 0. Do not train on STEW. Do not test on EEGMAT as the “success” number.

This document is the build manual. Pair it with `GAP_Align_Project_Design.md` for the pitch. If the two disagree, **this file wins for code**.

---

## 0. What you are building (read once)

A frozen classifier trained **only** on clinical EEG (Neurocom / EEGMAT).  
The same classifier is run on a commercial wearable (Emotiv EPOC / STEW).  
It will fail. That failure is the demo.

Then an unlabeled adapter (**GAP-Align**) is applied on the wearable side.  
You show before vs after on a dashboard, plus a scalp simulation of why the two headsets differ.

**Direction (non-negotiable)**

```
TRAIN  = EEGMAT  = Neurocom 23-ch wet clinical   (source)
TEST   = STEW    = Emotiv EPOC 14-ch saline      (target)
LABEL  = rest (0) vs cognitive-load (1)
NO STEW LABELS may be used for fitting the adapter.
```

**Ghost baseline to beat:** SCVCNet EEGMAT→STEW ≈ **63%** accuracy with no target labels.

**Stack**

| Layer | Library |
|---|---|
| EEG I/O + filter + resample | `mne` |
| Arrays | `numpy`, `scipy` |
| Classifier | `scikit-learn` |
| Dashboard | `streamlit` + `plotly` |
| Optional plots | `matplotlib` |

**Do not** start with PyTorch, EEGNet, Tent, or a live Bluetooth headset.

---

## 1. Repository layout (create this first)

```
Bhav_VMedithon/
├── GAP_Align_Project_Design.md
├── GAP_Align_Implementation_Guide.md   ← this file
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── data/
│   ├── raw/
│   │   ├── eegmat/          # PhysioNet EDFs
│   │   └── stew/            # IEEE DataPort txt files
│   └── processed/
│       ├── eegmat_epochs.npz
│       └── stew_epochs.npz
├── artifacts/
│   ├── scaler.joblib
│   ├── clf.joblib
│   ├── ablation.json
│   ├── stew_preds_before.npz
│   ├── stew_preds_after.npz
│   └── demo_windows.json
├── src/
│   ├── __init__.py
│   ├── channels.py
│   ├── io_eegmat.py
│   ├── io_stew.py
│   ├── preprocess.py
│   ├── features.py
│   ├── alignment.py
│   ├── degrade.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_all.py
├── app/
│   ├── streamlit_app.py
│   └── components.py
└── recordings/
    └── demo_backup.mp4      # 15s screen capture, judging insurance
```

`.gitignore` must include `data/raw/`, `data/processed/`, `artifacts/*.joblib`, `__pycache__/`, `.venv/`.

---

## 2. Environment

### 2.1 Python

Use **Python 3.10 or 3.11** (64-bit). Create a venv in the project root.

PowerShell:

```powershell
cd C:\Users\mail2_tdwzs1y\Desktop\bhavya_cursor\Bhav_VMedithon
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If execution policy blocks the venv:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2.2 `requirements.txt`

Pin reasonably; do not chase latest-nightly.

```
numpy>=1.24
scipy>=1.11
mne>=1.6
scikit-learn>=1.3
joblib>=1.3
pandas>=2.0
matplotlib>=3.8
plotly>=5.18
streamlit>=1.32
tqdm>=4.66
```

### 2.3 Sanity check

```powershell
python -c "import mne, sklearn, streamlit, plotly; print(mne.__version__, sklearn.__version__)"
```

---

## 3. `config.py` — single source of truth

Create this before any processing. Every script imports from here. Do not hard-code paths in five files.

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"

EEGMAT_DIR = DATA_RAW / "eegmat"
STEW_DIR = DATA_RAW / "stew"

SFREQ_SRC = 500.0          # expected EEGMAT; always overwrite from EDF header
SFREQ_TGT = 128.0          # Emotiv EPOC
BANDPASS = (1.0, 40.0)     # Hz
NOTCH = 50.0               # EEGMAT is Ukraine/EU line noise; STEW may be 50 or 60
EPOCH_SEC = 2.0
EPOCH_STRIDE_SEC = 1.0     # 50% overlap
RANDOM_SEED = 42

# Canonical 10 shared 10-20 sites (Emotiv names)
SHARED_CH = ["F3", "F4", "F7", "F8", "T7", "T8", "P7", "P8", "O1", "O2"]

# Clinical-only sites we DROP after using P3 as reference
CLINICAL_REF = "P3"

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta":  (13.0, 30.0),
}

# Train-time fake-Emotiv noise (pipeline E)
NOISE_P = 0.5
NOISE_STD_UV = 5.0         # start here; tune only if source CV collapses
QUANTIZE_UV = 0.51         # Emotiv 14-bit LSB

# Adapter unlabeled batch
ADAPTER_SECONDS = 30.0     # unlabeled STEW used to estimate z-score / EA
```

---

## 4. Channel maps (`src/channels.py`)

Emotiv and Neurocom use different 10–20 aliases. Normalize **everything** to Emotiv-style names.

```python
# Old 10-20  →  Emotiv / modern 10-20
T3 → T7
T4 → T8
T5 → P7
T6 → P8
```

Implement `canonicalize(name: str) -> str`:

1. Strip, upper-case for matching, then restore canonical casing from a dict.
2. Remove prefixes: `"EEG "`, `"EEG"`, `"EEG-"`.
3. Apply the T3/T4/T5/T6 map.
4. If the name is unknown, keep it; do not crash until pick time.

**EEGMAT pick order after canonicalize**

Must include `P3` **before** re-reference. After re-reference, drop `P3` and keep `SHARED_CH` only.

**STEW column order in the .txt files (documented)**

```
AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
```

For the classifier you **discard** `AF3, AF4, FC5, FC6` so train and test have the same 10 channels.

If a STEW file is missing a shared channel, skip that subject and log it. Do not impute channels.

Print `raw.ch_names` for `Subject00_1.edf` on first run and paste the mapping table into `artifacts/channel_audit.txt`. **Do this before writing the rest of preprocess.** EEGMAT EDF labels vary (`EEG F3` vs `F3`).

---

## 5. Download the data (do this today, not during the hackathon)

### 5.1 EEGMAT (open, easy)

https://physionet.org/content/eegmat/1.0.0/

```powershell
mkdir -p data\raw\eegmat
# from project root, with venv on
pip install wfdb
python -c "import wfdb; wfdb.dl_database('eegmat', 'data/raw/eegmat')"
```

If `wfdb.dl_database` fails, zip download from PhysioNet “Download ZIP” and extract EDFs into `data/raw/eegmat/`.

**Files**

- `SubjectXX_1.edf` = rest / background → label **0**
- `SubjectXX_2.edf` = mental arithmetic → label **1**
- ~36 subjects. Each file is about **60 s**, already ICA-cleaned by the authors.
- Expected `sfreq` ≈ **500 Hz**. Always read `raw.info["sfreq"]`.

README mentions a “30 Hz high-pass”; that is almost certainly a translation issue (they notch 50 Hz and the files still contain theta/alpha — people publish PSD papers on this set). You still apply **1–40 Hz** yourself.

### 5.2 STEW (needs IEEE DataPort account)

https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset  
DOI: `10.21227/44r8-ya50`

1. Create a free IEEE account.
2. Download the dataset.
3. Extract into `data/raw/stew/`.

**Files (typical)**

- `sub01_lo.txt` = rest → label **0**
- `sub01_hi.txt` = SIMKAP multitask → label **1**
- ~48 subjects, **128 Hz**, **14 columns**, ~2.5 minutes each.

**Load rule:** `numpy.loadtxt`. If the file has a header, `skiprows=1`. Print `shape` and first row. Confirm 14 columns.

If DataPort is blocked, you cannot invent STEW. Stop and get the files. Do not substitute a random Emotiv dataset — labels and montage will not match the pitch.

### 5.3 Data audit script (run immediately after download)

`src/io_eegmat.py` and `src/io_stew.py` should each have a `main()` that prints:

- number of subjects
- sampling rate(s)
- channel names
- duration in seconds
- min/max µV (catch unit mistakes: MNE often returns **Volts**, not µV)

**Units:** convert to µV as soon as you load (`data * 1e6` if MNE SI volts). STEW txt is usually already µV. If STEW values look like `±1e-4`, they are volts; multiply. If they look like `±50`, they are µV. Log which.

---

## 6. Preprocess — clinical source (`src/preprocess.py` + `src/io_eegmat.py`)

Process **one recording** as a function so train and the degrade-simulator share it.

### 6.1 Function signature

```python
def load_eegmat_recording(path, label: int) -> dict:
    """
    Returns:
      epochs: np.ndarray  (n_epochs, 10, n_times)  at 128 Hz
      y:      np.ndarray  (n_epochs,)
      subject_id: str
      meta: dict
    """
```

### 6.2 Steps, in this exact order

1. `mne.io.read_raw_edf(path, preload=True, verbose="ERROR")`
2. Drop non-EEG (ECG, stim, markers, time) if present.
3. Set channel types to `eeg`. Set montage `standard_1020` after renaming. Missing montage positions: warn, continue.
4. `canonicalize` channel names; `rename_channels`.
5. Confirm `P3` exists. If a subject has no P3, skip that subject and log. **Do not** fall back to average reference silently.
6. **Re-reference to P3** (this imitates Emotiv CMS):

   ```python
   raw.set_eeg_reference(ref_channels=["P3"])
   ```

   After this, P3 is ~zero. Drop P3 from picks.
7. `raw.pick(SHARED_CH)` — error if any shared channel missing.
8. Notch 50 Hz if `sfreq > 100`.
9. Band-pass `1–40 Hz`, `firwin`, `phase="zero"`.
10. `raw.resample(128, npad="auto")`.
11. Convert to µV.
12. Epoch with a sliding window: `EPOCH_SEC=2`, stride `1 s`.
    - `n_times = int(128 * 2) = 256`.
    - Drop the last incomplete window.
13. Optional artifact reject: drop epoch if peak-to-peak on any channel > **200 µV**. Log how many you drop. If you drop >40% of a subject, loosen to 300 µV.
14. Attach `label` to every epoch from that file (`_1` → 0, `_2` → 1).

### 6.3 Subject-level arrays

Save one npz:

```python
np.savez_compressed(
    "data/processed/eegmat_epochs.npz",
    X=X,                # (N, 10, 256)
    y=y,                # (N,)
    subject=subject,    # (N,) unicode
    ch=np.array(SHARED_CH),
    sfreq=128.0,
)
```

`X` must be `float32`. `y` `int8`. Same subject string for rest and task files of that person (`"00"`, `"01"`, …).

---

## 7. Preprocess — commercial target (`src/io_stew.py`)

Same epoch length and channel order. **No P3 re-reference** (Emotiv is already CMS-referenced in hardware).

1. Load txt → `(n_times, 14)`.
2. Map columns to names. Pick `SHARED_CH` in **the same order as EEGMAT**.
3. Band-pass 1–40 Hz (`scipy.signal.butter` sosfiltfilt, order 4, or create an MNE RawArray and reuse the same filter function).
4. Already 128 Hz: do not resample unless `sfreq` is 256 (EPOC X). If 256, resample to 128.
5. Epoch 2 s, stride 1 s.
6. Same 200 µV reject.
7. Labels from filename `lo`/`hi` are **only for evaluation**, never for fitting z-score, EA, or the classifier.

Save `data/processed/stew_epochs.npz` with the same keys.

**Critical:** `SHARED_CH` order in both npz files must be identical. Write a unit check:

```python
assert list(eegmat["ch"]) == list(stew["ch"]) == SHARED_CH
assert eegmat["X"].shape[1:] == stew["X"].shape[1:] == (10, 256)
```

---

## 8. Features (`src/features.py`)

Do not feed raw time-series into SVM as a first model. Use log band-power.

For one epoch `x` shape `(10, 256)`, `sfreq=128`:

```python
from scipy.signal import welch

freqs, psd = welch(x, fs=128, nperseg=256, axis=-1)  # psd: (10, n_freqs)
```

For each channel and each band in `BANDS`, mean PSD in that frequency range, then `log10(mean + 1e-12)`.

Output vector length = `10 channels × 3 bands = 30`.

Stack to `(N, 30)`. Feature names like `F3_theta`, `F3_alpha`, … — save them. The dashboard will use them for bar charts.

**Optional extra (only if 30-D is too weak on source CV):** add theta/alpha ratio per channel (10 more dims). Do not add dozens of features.

---

## 9. Alignment operators (`src/alignment.py`)

Each operator is a class with `fit` / `transform`. Labels are **forbidden** inside `fit`.

### 9.1 Per-channel z-score

```python
class ChannelZScore:
    def fit(self, X):
        # X: (N, 10, T)
        self.mean_ = X.mean(axis=(0, 2))      # (10,)
        self.std_  = X.std(axis=(0, 2)) + 1e-8
        return self
    def transform(self, X):
        return (X - self.mean_[None, :, None]) / self.std_[None, :, None]
```

**When to fit**

- Source (train): fit on **EEGMAT train epochs only**, or skip z-score on source if you z-score features later with `StandardScaler`. Recommended: **epoch z-score on target only**; on source, rely on `StandardScaler` on the 30-D features.
- Target (test adapter): fit on an **unlabeled** STEW subset (first `ADAPTER_SECONDS` of mixed epochs **without using y**). Then transform **all** STEW epochs.

Do not fit target z-score on the evaluation epochs and then also test on those same epochs without stating it. For the hackathon, transductive (fit on all unlabeled STEW) is acceptable if you write “unlabeled transductive calibration” on the slide. Prefer: fit on 30 s per subject, transform the rest of that subject (still unlabeled).

### 9.2 Euclidean Alignment (EA)

For each epoch, covariance `C = (x @ x.T) / T` with `x` shape `(10, T)`, plus shrinkage:

```python
C = (x @ x.T) / T
C = C + 1e-6 * np.eye(10)
```

Mean covariance `R = mean_i C_i`.  
Whitener `R_inv_sqrt` via SVD: `R = U diag(s) U.T`, then `U diag(s**-0.5) U.T`.

```python
X_aligned[i] = R_inv_sqrt @ X[i]
```

**Fit R**

- Train: per EEGMAT **subject**, compute R on that subject’s unlabeled epochs, apply to that subject (standard EA). If too slow, one global R on all EEGMAT train epochs is OK; log which.
- Test: per STEW **subject**, R from that subject’s unlabeled epochs, apply to that subject. This is still unlabeled.

If a subject has < 5 epochs, skip EA for that subject (identity).

### 9.3 Pipeline flags

```python
def apply_gap(X, *, zscore=False, ea=False, z_fit=None, ea_fit=None):
    ...
```

The ablation is just different flags. Do not copy-paste five scripts.

---

## 10. Train-time device randomization (`src/degrade.py`)

Used **only while training** (pipeline E). Never on the official STEW test path except in the simulator UI.

```python
def fake_emotiv(epoch, rng, std_uv=5.0, lsb=0.51):
    # epoch: (10, T) in µV
    noise = rng.normal(0, std_uv, size=epoch.shape)
    q = np.round((epoch + noise) / lsb) * lsb
    return q
```

In `train.py`, with probability `NOISE_P`, replace the epoch with `fake_emotiv` **before** features.

The Streamlit degrade slider should call this same function so the sim and the trainer cannot drift apart.

---

## 11. Train the frozen classifier (`src/train.py`)

### 11.1 Source sanity (must pass before any STEW number)

Leave-one-**subject**-out (LOSO) on EEGMAT:

1. Features on EEGMAT (optionally EA per subject).
2. `StandardScaler` + `LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)`.
3. LOSO accuracy and macro-F1.

**Pass bar:** LOSO accuracy **≥ 0.70**. If you are at ~0.55, stop. Debug channels, units, labels, filter. Do not jump to STEW.

Also try `LinearSVC`. Keep whichever is better on **EEGMAT LOSO**, not on STEW.

### 11.2 Final source model

Fit scaler + clf on **all** EEGMAT epochs. Save:

```
artifacts/scaler.joblib
artifacts/clf.joblib
artifacts/train_meta.json   # LOSO scores, C, seed, channel list, sfreq
```

`train_meta.json` is what you quote if a judge asks “did you overfit STEW?”

---

## 12. Evaluation and ablation (`src/evaluate.py`)

### 12.1 Metrics (always all four)

- Accuracy
- Macro-F1
- Cohen’s kappa
- Confusion matrix (2×2)

If F1 is far below accuracy, you collapsed to the majority class. That is a failed adapter even if accuracy went up.

### 12.2 Ablation table (required)

Evaluate the **frozen** EEGMAT model on STEW under:

| ID | Train extras | Test adapter | What it proves |
|---|---|---|---|
| A | none | none (channel+rate+filter only) | naive port / collapse |
| B | P3 re-ref already in EEGMAT preprocess | none | reference gap (B vs a no-P3 control if you have time) |
| C | same as B | target z-score | gain / impedance |
| D | same | z-score + EA | spatial mixing |
| E | train-time `fake_emotiv` | z-score + EA | SNR gap |
| F | same as E | + Tent | optional; drop if worse |

**Pipeline B note:** P3 re-reference is on the **source** recording, not a test-time flag. To isolate it, also train a control `A0` **without** P3 re-ref (average or linked-ear as in the file) and compare A0 vs B on STEW. If you only have time for one source preprocess, **keep P3** and skip A0.

Write `artifacts/ablation.json`:

```json
{
  "A": {"acc": 0.00, "f1": 0.00, "kappa": 0.00, "cm": [[0,0],[0,0]]},
  "C": {"acc": 0.00, "f1": 0.00, "kappa": 0.00, "cm": [[0,0],[0,0]]},
  "D": {"acc": 0.00, "f1": 0.00, "kappa": 0.00, "cm": [[0,0],[0,0]]},
  "E": {"acc": 0.00, "f1": 0.00, "kappa": 0.00, "cm": [[0,0],[0,0]]}
}
```

**Winner** = highest **macro-F1** among A–E, provided confusion matrix shows both classes. Freeze winner as `after`. Pipeline A (or A0) is `before`.

### 12.3 Save demo tensors

For the dashboard, pick **8–12** clean STEW windows (mix rest and load) plus **8–12** EEGMAT windows. Save waveforms `(10, 256)`, features `(30,)`, labels, before-proba, after-proba.

```
artifacts/demo_windows.json
artifacts/stew_preds_before.npz
artifacts/stew_preds_after.npz
```

The Streamlit app should read **artifacts**, not retrain.

---

## 13. Optional Tent (pipeline F) — implement last, expect failure

Only if you switched the classifier to a tiny PyTorch net with BatchNorm.

Tent = one SGD step minimizing prediction entropy on a STEW batch, updating BN stats only.

If you stayed with logistic regression (recommended), **there is no Tent**. Skip F. Write on the slide: “Tent needs BN layers; our linear adapter does not use it.”

---

## 14. End-to-end runner (`src/run_all.py`)

One command that rebuilds everything from processed npz (or from raw if npz missing):

```powershell
python -m src.run_all
```

Order inside `run_all`:

1. If npz missing → preprocess EEGMAT, preprocess STEW.
2. EEGMAT LOSO → print and fail if < 0.65 (warn) / < 0.55 (exit).
3. Fit final source model.
4. Run ablation A, C, D, E (and B/A0 if coded).
5. Dump `ablation.json`, confusion plots as PNG under `artifacts/`, demo windows.
6. Print a markdown table to stdout you can paste into slides.

Never require a GPU.

---

## 15. Dashboard (`app/streamlit_app.py`)

```powershell
streamlit run app/streamlit_app.py
```

Four pages. Nothing else.

### Page 1 — Live compare (the 90-second demo)

Layout: three columns.

**Left — Clinical (Neurocom / EEGMAT)**

- Channel traces (2 s), 10 shared sites
- θ/α/β bars
- Prediction: Rest | Cognitive load
- Probability bar
- Caption: “Hospital-grade stand-in · 19–23 ch wet · 500 Hz → aligned 10 ch 128 Hz”

**Right — Commercial (Emotiv / STEW)**

- Same traces for a STEW window
- Toggle: **Before GAP-Align** / **After GAP-Align**
- Prediction + probability for the selected mode
- Caption: “Wearable stand-in · 14 ch saline · native 128 Hz”

**Center**

- Source LOSO accuracy (from `train_meta.json`)
- STEW accuracy before (pipeline A)
- STEW accuracy after (winner)
- Delta in percentage points
- Kappa before / after
- Badge: “Unlabeled adaptation · no STEW labels used to train the classifier”

Controls: subject dropdown, window slider, play button (advance window every 0.4 s to fake streaming).

### Page 2 — Hardware simulation

Two 2D scalps (do not start in 3D).

Use 10–20 coordinates (hard-code a dict of x,y for Fp, F, C, T, P, O).

- **Neurocom:** plot 19 sites. Colour shared sites green, dropped sites grey, **P3 gold** with label “CMS match: we re-reference here”. Ears / A1A2 note: “original linked-ear reference”.
- **EPOC:** plot 14 Emotiv sites. Grey out AF3/AF4/FC5/FC6 as “not in the model”. CMS/DRL at P3/P4 marked.

**Degrade slider** (0 = clinical epoch as aligned, 1 = full fake EPOC):

| Slider | Effect |
|---|---|
| 0.00–0.25 | already 10 ch / 128 Hz (identity) |
| 0.25–0.50 | add Gaussian noise 0 → 8 µV |
| 0.50–0.75 | 14-bit quantize |
| 0.75–1.00 | extra noise + optional 4–8 Hz jitter |

Show the **same frozen model’s P(load)** as the slider moves. It should generally fall toward 0.5. If it does not, the sim is lying — fix `degrade.py`.

### Page 3 — Ablation & trust

- Bar chart of acc/F1/kappa for A/C/D/E
- Two confusion matrices (before / after)
- One sentence under the matrices: “Recovery must improve both classes.”
- Optional: per-subject after-minus-before accuracy (bar). Honest if some subjects get worse.

### Page 4 — India / access

Static is fine.

- Neurologist shortage line (< 2,500 for 1.4B)
- Price ladder: ₹8–40 lakh hospital · ₹1–5 lakh RMS · ~₹1 lakh EPOC · ₹20–40k Muse
- “Neurocom/Emotiv = open-data proxies for RMS-class vs wearable”
- **Not a diagnostic EEG. Not CDSCO-cleared. Cognitive-load screening research tool.**
- Next: paired RMS + EPOC in one Indian lab

### Dashboard rules

- Load artifacts at start; `st.cache_data`.
- If artifacts missing, show a red box: “Run `python -m src.run_all` first.”
- No training inside Streamlit.
- Dark theme is optional; readable beats pretty.
- Default to Page 1.

---

## 16. Scalp coordinates (copy)

Approximate 2D 10–20 for Plotly (nose up, x right):

```python
POS = {
    "Fp1": (-0.3, 0.90), "Fp2": (0.3, 0.90),
    "AF3": (-0.35, 0.75), "AF4": (0.35, 0.75),
    "F7": (-0.70, 0.55), "F3": (-0.35, 0.55), "Fz": (0.0, 0.55), "F4": (0.35, 0.55), "F8": (0.70, 0.55),
    "FC5": (-0.55, 0.35), "FC6": (0.55, 0.35),
    "T7": (-0.90, 0.00), "C3": (-0.40, 0.00), "Cz": (0.0, 0.00), "C4": (0.40, 0.00), "T8": (0.90, 0.00),
    "P7": (-0.70, -0.55), "P3": (-0.35, -0.55), "Pz": (0.0, -0.55), "P4": (0.35, -0.55), "P8": (0.70, -0.55),
    "O1": (-0.30, -0.90), "O2": (0.30, -0.90),
}
```

Draw a circle radius 1.0 as the head. Mark nose as a triangle at (0, 1.12).

---

## 17. `README.md` (short, for judges who open the repo)

Include:

- One-line pitch
- Direction: EEGMAT → STEW
- `pip install -r requirements.txt`
- How to get data (two URLs)
- `python -m src.run_all`
- `streamlit run app/streamlit_app.py`
- “No patient data. Public de-identified research sets only.”

---

## 18. 24-hour (and pre-hackathon) schedule

### Before the hackathon (mandatory)

| Task | Done? |
|---|---|
| venv + requirements | |
| EEGMAT downloaded, channel names printed | |
| STEW downloaded, 14 columns confirmed | |
| Both npz written, shape assert passed | |
| EEGMAT LOSO ≥ 0.70 | |
| Pipeline A STEW number recorded (the collapse) | |
| Pipeline C or D run once | |

If STEW is not downloaded before hour 0, the project is at risk. IEEE login is the bottleneck.

### Hours 0–8

- Re-run `run_all` from a clean shell
- Finish ablation A/C/D/E
- Save artifacts
- Record 15 s video of one successful CLI run

### Hours 8–16

- Streamlit Page 1 fully working from artifacts
- Page 2 scalps + degrade slider hooked to real model proba
- Page 3 tables from `ablation.json`

### Hours 16–22

- Page 4 India copy
- Polish: titles, units, “unlabeled” badge
- Demo rehearsal 90 seconds
- Backup video of the dashboard

### Hours 22–24

- Freeze code
- Do not retune on STEW test numbers
- If dashboard breaks: play the video, show `ablation.json` numbers, move on

### Cut order if late

1. Cut Tent / deep net  
2. Cut Hindi / PDF export  
3. Cut per-subject bars  
4. Cut 3D  
5. Cut Page 4 polish (keep the CDSCO disclaimer as text)  
6. **Never cut** ablation numbers or before/after toggle  

---

## 19. Verification checklist (run before you call it done)

```
[ ] Train data = EEGMAT only
[ ] Test data = STEW only
[ ] Classifier never saw STEW y
[ ] Adapter fit never used STEW y
[ ] 10 channels, same order, 128 Hz, 2 s, 1–40 Hz both sides
[ ] P3 re-reference on EEGMAT logged
[ ] Units are µV on both
[ ] EEGMAT LOSO acc and F1 written in train_meta.json
[ ] STEW before acc < after F1 is not a majority-class fake
[ ] Confusion matrices for before and after
[ ] Degrade slider changes P(load)
[ ] Dashboard runs offline from artifacts
[ ] Backup mp4 exists
[ ] Slide/README says proxies, not “we recorded RMS vs EPOC in India”
[ ] No seizure / diagnostic claim anywhere in the UI
```

---

## 20. Common failures and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| EEGMAT LOSO ~50% | labels swapped, or 30 Hz wipe, or volts vs µV mix | print y counts; plot PSD; check `* 1e6` |
| All STEW predictions = one class | scaler fitted on target, or z-score used labels, or class imbalance | fit scaler on EEGMAT only; check `predict_proba` |
| After adapter worse than before | EA estimated on too few epochs; over-noise in train | per-subject EA; reduce `NOISE_STD_UV` |
| MNE channel error | `EEG F3` not renamed | print `ch_names`; expand `canonicalize` |
| STEW 13 columns | header/index column | `skiprows`, `usecols` |
| Filter too slow | filtering every tiny epoch separately | filter the full recording, then window |
| Streamlit retrains every click | no `st.cache_data` / training in app | load joblib only |
| Beautiful sim, flat probabilities | degrade not using the same 10 ch / 30-D features | one `epoch_to_proba()` used by train, eval, and UI |

---

## 21. Exact mathematical definitions (so two people implement the same thing)

**Band-pass:** zero-phase FIR, Hamming, 1–40 Hz, duration 2 s at 128 Hz → 256 samples.

**Welch:** `nperseg=256`, no overlap inside the epoch (epoch is already 2 s), scaling `density`.

**Log band-power:** `log10( mean(psd[f0:f1]) + 1e-12 )`.

**Logistic regression:** `sklearn.linear_model.LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=2000, C=1.0, random_state=42)`.

**StandardScaler:** fit on EEGMAT feature matrix only; transform STEW features with that same scaler. Never refit on STEW.

**EA whitening:** \( R^{-1/2} X \), \( R = \frac{1}{n}\sum_i (X_i X_i^\top / T) + \varepsilon I \).

**Z-score:** subtract per-channel mean and divide by per-channel std, moments from unlabeled target epochs, broadcast over time.

---

## 22. Demo script (memorize)

1. “We train on a clinical Neurocom recording — stand-in for an Indian hospital EEG.”
2. Show left column: rest vs load, high confidence. Mention LOSO accuracy.
3. “Same frozen model on Emotiv EPOC, the wearable a campus can buy.”
4. Show right column **before**: collapse. Quote pipeline A accuracy.
5. Open sim: grey channels, gold P3, “different reference, fewer electrodes, more noise.” Move degrade slider.
6. Toggle **after** GAP-Align. Quote winner F1 and the ablation bar.
7. “No new labels. Not a diagnostic EEG. Next validation: RMS + EPOC in one Indian lab.”

Total: 90 seconds. Then stop talking.

---

## 23. Implementation order for the coding agent (checklist)

Work in this sequence. Commit after each numbered item if you use git.

1. Create folders, `requirements.txt`, `config.py`, `channels.py`.
2. Download EEGMAT; print channels and sfreq.
3. Download STEW; print shape and column names.
4. EEGMAT preprocess through P3-ref + 10 ch + resample + epochs → npz.
5. STEW preprocess → npz. Assert channel order.
6. Features + EEGMAT LOSO. **Stop until ≥ 0.70 or you have a written reason.**
7. Save frozen scaler + clf.
8. STEW pipeline A metrics (before).
9. Target z-score adapter (C). Compare to A.
10. Add EA (D). Compare.
11. Add train-time noise, retrain source, re-eval (E). Compare.
12. Dump ablation.json + confusion PNGs + demo windows.
13. Streamlit Page 1 from artifacts.
14. Page 2 scalps + degrade.
15. Page 3 ablation.
16. Page 4 India disclaimer.
17. Backup recording.
18. Freeze.

---

## 24. What “done” looks like

You can walk a judge to a laptop, run `streamlit run app/streamlit_app.py` with **no internet**, and in 90 seconds they see:

- a clinical window classified correctly  
- a wearable window failing  
- a hardware reason  
- an unlabeled recovery  
- a table proving which step bought the recovery  
- a sentence that this is an Indian access tool, not a diagnosis  

If any of those six is missing, it is not done.

---

## 25. File-level function list (implement all)

| File | Functions |
|---|---|
| `config.py` | constants only |
| `channels.py` | `canonicalize`, `pick_shared`, `stew_column_index` |
| `io_eegmat.py` | `list_subjects`, `load_raw`, `subject_id_from_path` |
| `io_stew.py` | `list_subjects`, `load_txt` |
| `preprocess.py` | `reref_p3`, `filter_resample`, `make_epochs`, `reject_ptp`, `process_eegmat_file`, `process_stew_file`, `build_npz` |
| `features.py` | `bandpower_vector`, `transform_epochs` |
| `alignment.py` | `ChannelZScore`, `EuclideanAlign`, `apply_gap` |
| `degrade.py` | `fake_emotiv`, `slider_degrade` |
| `train.py` | `loso_eegmat`, `fit_source`, `save_model` |
| `evaluate.py` | `metrics`, `run_ablation`, `save_demo_windows` |
| `run_all.py` | `main` |
| `app/streamlit_app.py` | four pages |
| `app/components.py` | `plot_traces`, `plot_scalp`, `proba_badge` |

---

End of guide. Follow Phase 0–5 before writing Streamlit. The numbers are the product; the UI is how you show them.
