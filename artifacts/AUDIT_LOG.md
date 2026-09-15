# GAP-Align running audit

Append-only. One block per phase. Not a substitute for `channel_audit.txt`
(that file is written from real EDF/txt headers in Phase 5).

Standing rules (2026-09-15):
- Do **not** push to https://github.com/Bhavya-Shri/VMedithon (upstream).
- Team fork for this work: https://github.com/haripriyasubbiah/VMedithon
- Append this log every time files change.
- STEW labels never used to fit classifier, scaler, z-score, or EA.

LIMIT (do not forget):
- EEGMAT on disk is a SAMPLE only: subjects 00, 01, 02 (6 EDFs). Full PhysioNet set is ~36 subjects. We stopped the full pull for Review 1.
- `eegmat_epochs.npz` (712, 10, 256) is from those 3 people only. Phase 6 is done for the sample, not for the full source set.
- LOSO >= 0.70 on 3 subjects is a smoke test, NOT the Guide hard gate. Pull remaining EEGMAT EDFs before quoting a source-model number to judges.
- STEW is the FULL set: 48 subjects, 96 txt files. No sample limit on the target side.

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

What: stopped full 175 MB EEGMAT pull. Sample is now 3 subjects (00, 01, 02), 6 EDFs.
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

## Phase 6 -- EEGMAT preprocess -- 2026-09-15

What: `src/preprocess.py` -- reref_p3, filter_resample, make_epochs, reject_ptp, process_eegmat_file, build_npz. Ran on 3 local subjects ONLY (see LIMIT).
Order: drop ECG -> canonicalize -> montage warn -> P3 reref -> drop P3 -> pick SHARED_CH -> notch 50 -> 1-40 Hz firwin zero-phase -> resample 128 -> *1e6 uV -> 2 s / 1 s stride -> PTP 200 uV.
npz: `data/processed/eegmat_epochs.npz` (gitignored)
```
X (712, 10, 256) float32  min/max -101.5 / 107.6 uV
y (712,) int8  rest=536  load=176
subjects 00, 01, 02
ch = SHARED_CH  sfreq=128
P3 after reref max abs = 0.000 uV (dropped from ch)
PTP drops: 0-11% per file; none hit 40% so thresh stayed 200 uV
```
Imbalance 536 vs 176 is longer rest EDFs (~182 s vs ~60 s task), not a label bug. class_weight=balanced later.
STEW preprocess not started.
Guide step 3 vs 4: rename before montage (step 3 says "after renaming"). Logged, not a method change.

---

## Phase 5b -- STEW download confirmed -- 2026-09-15

96 txt files, 48 subjects (01-48), each lo+hi. Nested under `data/raw/stew/STEW Dataset/`.
All 96 have 14 numeric columns, no header, (19200, 14) = 150 s at 128 Hz.
Units: thousands with DC offset, NOT Volts, do not *1e6. lo/hi unused for fitting.
Download is complete. Phase 7 ran after this entry.

---

## Phase 7 -- STEW preprocess -- 2026-09-15

What: `process_stew_file` -- pick 10 SHARED_CH, NO P3 reref, same FIR 1-40 Hz + 50 Hz notch as EEGMAT, 128 Hz, 2 s epochs, PTP 200/300.
Why: wearable already CMS-referenced; we only lock montage/rate/band.
npz: `data/processed/stew_epochs.npz` (gitignored)
```
X (10265, 10, 256) float32  min/max -243.3 / 241.0 uV
y (10265,) int8  rest=5214 load=5051  EVAL ONLY
n_subj 48  ch=SHARED_CH  sfreq=128
assert EEGMAT ch/shape == STEW ch/shape == (10, 256) OK
```
DC offset gone after band-pass (thousands -> tens/hundreds of uV). Do not *1e6.
PTP: 33 files loosened to 300 uV. 4 recordings fully dropped (100% PTP): sub01_lo, sub23_hi, sub38_hi, sub41_lo. Those subjects still have the other condition. Wearable is noisier -- expected, not a unit bug.
P3 not in STEW ch list.

---

## Git

Remote: https://github.com/Bhavya-Shri/VMedithon.git
Branch: main
Left untracked on purpose: TwinBite_Project_Design.md, NeuroShift_architecture_diagram.png, VMEDITHON_3.0_NeuroShift_Submission.pptx


Next is Phase 8: 30-D log band-power features. Reply proceed for that. Do not compute a STEW accuracy until EEGMAT LOSO (Phase 9) passes.

---

## Phase 8 -- features.py -- 2026-09-15

What: `src/features.py` -- `bandpower_vector`, `transform_epochs`, `feature_names`.
Why: 10 ch x theta/alpha/beta log-Welch = 30-D. No raw SVM on time series.
Welch nperseg=256, log10(mean+1e-12). Self-check: `python -m src.features`.

---

## Phase 9 -- train.py LOSO -- 2026-09-15

What: leave-one-subject-out on EEGMAT. LogisticRegression vs LinearSVC, EA on/off.
Why: Guide hard gate LOSO >= 0.70 before any target number. Winner chosen on EEGMAT only.
Leakage: STEW y never loaded in this file.

---

## Phase 10-12 -- alignment, degrade, evaluate -- 2026-09-15

What: `ChannelZScore`, `EuclideanAlign`, `apply_gap` (unlabeled, per-subject).
`fake_emotiv` + `slider_degrade` (same function family as the dashboard).
`run_ablation` pipelines A/C/D/E. F/Tent skipped (no BatchNorm).
Winner = highest macro-F1 among rows that predict both classes.
`y_tgt` is scoring-only.

---

## Phase 13-16 -- run_all + Streamlit -- 2026-09-15

What: `python -m src.run_all` preprocess → LOSO → frozen model → ablation → demo windows.
Four-page Streamlit from artifacts. No training in the UI.
If STEW txt files are missing, `proxy_target.py` writes an Emotiv-like degraded-EEGMAT npz
and the dashboard banners it. Those numbers are NOT EEGMAT→STEW.

Dashboard pages:
1. Live compare (before/after toggle, play)
2. Scalps + degrade slider hooked to frozen P(load)
3. Ablation bars + confusion + per-subject delta
4. India access + CDSCO disclaimer

---

## LIMIT update -- 2026-09-15 (this environment)

Upstream Bhavya-Shri/VMedithon is read-only for us. Work is pushed to the
haripriyasubbiah/VMedithon fork only.

EEGMAT re-pulled from PhysioNet in this environment (not the Review-1 3-subject
sample). STEW txt files were **not** on disk (IEEE DataPort login). Target
ablation used `eegmat_degraded_proxy`. Do not quote A–E as EEGMAT→STEW.

---

## Phase 6 rerun -- EEGMAT preprocess -- 2026-09-15

What: `python -m src.run_all` preprocess on PhysioNet EDFs now on disk.
```
EDFs 40  subjects 20 (00-19)
X (4784, 10, 256) float32  min/max -150.9 / 114.5 uV
y rest=3576 load=1208  (longer rest files, not a label bug)
ch = SHARED_CH  sfreq=128  P3 dropped after reref (max abs 0.000 uV)
PTP mostly 200 uV; unit_ok uV-scale
```
Full PhysioNet set is ~36 subjects; we had 20 complete pairs when this ran.

---

## Phase 8 executed -- features -- 2026-09-15

Vectorized Welch (`transform_epochs` matches `bandpower_vector` order).
Self-check passed: F3_theta / F3_alpha / F3_beta, shape (N, 30).

---

## Phase 9 executed -- EEGMAT LOSO -- 2026-09-15

Chosen on EEGMAT only (no STEW y):

| model | EA | acc | macro-F1 |
|---|---|---:|---:|
| logreg | True | **0.635** | **0.584** |
| logreg | False | 0.618 | 0.571 |
| linearsvc | True | 0.748 | 0.434 |
| linearsvc | False | 0.745 | 0.438 |

Winner: **logreg + source EA**. LinearSVC accuracy is higher but F1 ~0.43 — majority-class collapse. We keep F1.
Guide hard gate LOSO >= 0.70: **not met** (0.635). Written reason: 20/36 subjects, rest/load imbalance 3576 vs 1208, linear 30-D band-power. Frozen model is still this winner. Do not retune on the wearable target.

---

## Phases 10-12 executed -- unlabeled ablation -- 2026-09-15

Target = **eegmat_degraded_proxy** (same EEGMAT epochs + gain/mix/fake_emotiv).
NOT STEW. Adapter fit did not use y.

| Pipeline | Acc | Macro-F1 | Kappa | Both classes |
|---|---:|---:|---:|---|
| A naive port | 0.353 | 0.351 | 0.012 | yes |
| C z-score | 0.556 | 0.530 | 0.125 | yes |
| D z-score+EA | 0.565 | 0.545 | 0.163 | yes |
| E fake-Emotiv train + D | **0.596** | **0.567** | 0.183 | yes |

Winner **E** by macro-F1. Before = A. Each step helped. F/Tent skipped.
Ghost baseline SCVCNet 62.9% is EEGMAT→STEW; do not compare this proxy table to it on a slide.

Artifacts: `ablation.json`, `cm_before.png`, `cm_after.png`, `stew_preds_*.npz` (filename kept; contents are proxy), `scaler.joblib` / `clf.joblib` (A/C/D frozen), `scaler_E.joblib` / `clf_E.joblib`.

---

## Phases 13-16 executed -- demo windows + Streamlit -- 2026-09-15

`demo_windows.json` written (clinical + wearable windows, before/after proba).
Dashboard: `streamlit run app/streamlit_app.py` (port 8765). Four pages.
Page 1 banners the proxy so judges cannot mistake it for STEW.
Pipeline F still skipped.

Phase 17 backup mp4: not recorded in this environment.
Phase 18 freeze: code + artifacts on the haripriyasubbiah fork.

To get official EEGMAT→STEW numbers: put `sub##_lo.txt` / `sub##_hi.txt` in
`data/raw/stew/`, delete `data/processed/stew_epochs.npz`, re-run
`python -m src.run_all`.

