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

## Git

Remote: https://github.com/Bhavya-Shri/VMedithon.git
Branch: main
Left untracked on purpose: TwinBite_Project_Design.md, NeuroShift_architecture_diagram.png, VMEDITHON_3.0_NeuroShift_Submission.pptx
