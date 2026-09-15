# GAP-Align running audit

Append-only. One block per phase. Not a substitute for `channel_audit.txt`
(that file is written from real EDF/txt headers in Phase 5).

Standing rules (2026-09-15):
- Push to https://github.com/Bhavya-Shri/VMedithon after each meaningful phase.
- Append this log every time files change.
- STEW labels never used to fit classifier, scaler, z-score, or EA.

---

## Phase 1 -- repo scaffold -- 2026-09-15

What: folders, `.gitignore`, `README.md`, `src/__init__.py`.
Why: fixed disk layout so raw EEG, processed epochs, and frozen artifacts cannot mix.
Created: `data/raw/eegmat/`, `data/raw/stew/`, `data/processed/`, `src/`, `app/`, `artifacts/`, `recordings/`.
Verification: Guide §1 required gitignore lines present (`data/raw/`, `data/processed/`, `artifacts/*.joblib`, `__pycache__/`, `.venv/`).
Arrays: none.
Not tracked: TwinBite / NeuroShift leftovers, `.venv/`.

---

## Phase 2 -- environment -- 2026-09-15

What: `requirements.txt` (Guide §2.2 exact), `.venv`.
Why: isolate the allowed stack (mne, numpy, scipy, sklearn, no PyTorch).
Deviation: Guide wants Python 3.10/3.11. Machine had 3.13 default and 3.12.4. Venv is **3.12.4**.
Import check (venv):
```
python      3.12.4
mne         1.13.2
sklearn     1.9.1
numpy       2.5.3
scipy       1.18.1
joblib      1.6.0
pandas      3.0.5
matplotlib  3.11.2
plotly      7.0.0
streamlit   1.63.0
tqdm        4.70.1
```
`wfdb` not installed yet (Guide §5.1, download phase).

---

## Phase 3 -- config.py -- 2026-09-15

What: `config.py` as single source of truth.
Why: one place for SHARED_CH, 128 Hz, P3, bands, seed, noise LSB, paths.
Guide §3 constants: present and unchanged.
Extra constants (from later Guide sections, not invented): N_TIMES=256, STEW_CH_FILE, PTP 200/300, EA shrink 1e-6, LR params, LOSO gates.
Import check:
```
SHARED_CH   ['F3', 'F4', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2']
N_CH 10  N_TIMES 256  N_FEATURES 30
SFREQ_SRC 500  SFREQ_TGT 128
BANDPASS (1, 40)  NOTCH 50  EPOCH 2 s stride 1 s
CLINICAL_REF P3  SEED 42
NOISE_P 0.5  STD 5.0 uV  LSB 0.51 uV
ADAPTER_SECONDS 30
dirs_exist True
```
Shape contract locked: X (N, 10, 256) float32 uV @ 128 Hz -- not on disk yet.
Leakage: n/a (no data loaded).

---

## Phase 4 -- src/channels.py -- 2026-09-15

What: `canonicalize`, `pick_shared`, `stew_column_index`. `CH_ALIAS` added to `config.py`.
Why: Neurocom T3/T4/T5/T6 and `EEG F3` prefixes must become Emotiv T7/T8/P7/P8/F3 before any array math.
Verification (`python -m src.channels`): all string asserts passed.
```
EEG T3 -> T7, EEG T4 -> T8, EEG T5 -> P7, EEG T6 -> P8, EEG F3 -> F3
STEW col idx [2, 11, 1, 12, 4, 9, 5, 8, 6, 7]
STEW -> shared ['F3', 'F4', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2']
unknown 'STI 014' kept; missing channels raise ValueError (no impute)
```
Arrays: none. `channel_audit.txt` still PENDING real EDF/txt headers.
Not pushed: TwinBite / NeuroShift leftovers.

---

## Phase 5 -- data sample + audit -- 2026-09-15

What: stopped full 175 MB EEGMAT pull. Kept 2 subjects (00, 01), 4 EDFs.
Why: Review 1 is immediate; full 36 subjects are not needed to prove channels/units.
wfdb.dl_database failed: it requested Subject00_1.edf.hea (404). EEGMAT is EDF-only.
Fallback: direct PhysioNet file URLs. Incomplete 0-byte files from the failed wfdb run were deleted.

EEGMAT probe Subject00_1.edf (REAL header):
```
sfreq 500.0 Hz
n_ch 21  duration 182.0 s  n_times 91000
min/max V  -2.05e-4 / 6.55e-4   --> Volts, must *1e6
P3 present; all SHARED_CH present after canonicalize
EEG T3 -> T7, EEG T5 -> P7
```
STEW: not on disk. IEEE DataPort account required. Did not invent files.
Also wrote `src/io_eegmat.py`, `src/io_stew.py`, `REVIEW_1_Presentation.md`.
No preprocess (filter/epoch) yet -- Guide: audit before rest of preprocess.

---

## Git

Remote: https://github.com/Bhavya-Shri/VMedithon.git
Branch: main
Left untracked on purpose: TwinBite_Project_Design.md, NeuroShift_architecture_diagram.png, VMEDITHON_3.0_NeuroShift_Submission.pptx
