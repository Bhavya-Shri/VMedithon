# GAP-Align -- Review 1 talk sheet

Use this file in the review. Do not open the long design/guide docs unless a judge asks.
Timebox: **3 to 4 minutes**. Then stop and take questions.

Live repo: https://github.com/Bhavya-Shri/VMedithon

**Do not claim STEW accuracy numbers. We have not run STEW yet.**
**Do not say this is a diagnostic EEG or CDSCO device.**

---

## What to have on screen

1. This file, or GitHub README.
2. Optional 20-second code proof: `python -m src.channels` (shows T3 -> T7).
3. Optional: `config.py` SHARED_CH list.
4. If they want files: `artifacts/channel_audit.txt` (real EEGMAT header).

Do **not** open Streamlit. It is not built yet (step 16). Do not apologise -- say the numbers come first.

---

## Slide 1 -- One sentence (10 s)

Hospital EEG knowledge is locked in cities. GAP-Align is the calibration layer that lets a model trained on a clinical headset run on a wearable a campus or clinic can actually buy -- and we will show the failure and the unlabeled recovery.

---

## Slide 2 -- India problem (25 s)

India cannot put a neurologist and a 8 to 40 lakh EEG cart in every district. Fewer than 2,500 neurologists for 1.4 billion people.

Models get trained where the expensive machines and the labels live. The headset a college in Coimbatore or a counsellor in Nashik can buy is closer to an Emotiv EPOC, around 70 thousand to 1.5 lakh.

We are **not** claiming we recorded in an Indian hospital. Neurocom vs Emotiv is the open public-data stand-in for RMS-class hospital EEG vs a wearable India can deploy.

Price ladder if asked:
- Hospital cart: 8 to 40 lakh
- Indian clinical portable (RMS): 1 to 5 lakh
- Emotiv EPOC: ~1 lakh
- Muse-class: 20 to 40 thousand

---

## Slide 3 -- What we train and test (20 s)

Direction is non-negotiable:

- TRAIN / source = EEGMAT = Neurocom, wet clinical, 500 Hz
- TEST / target = STEW = Emotiv EPOC, 14-ch saline, 128 Hz
- Label = rest vs cognitive load
- Zero STEW labels used to fit the classifier or the adapter

Ghost baseline we must beat: SCVCNet, same pair, no target labels, about **62.9%** accuracy. We want to beat ~63% with zero STEW labels, plus an honest ablation table.

---

## Slide 4 -- Why a hospital model fails (40 s)

The cheap headset is not a smaller clinical headset. Each hardware gap is one operator:

| Gap | What it does | Our operator |
|---|---|---|
| 19-23 ch vs 14 ch | missing spatial info | keep 10 shared sites only |
| Ear ref vs CMS at P3 | different voltage zero | re-reference Neurocom to P3, then drop P3 |
| 500 Hz vs 128 Hz | extra high frequency | resample to 128 with anti-alias |
| different bandwidth | different passband | 1-40 Hz on both |
| gel vs saline gain | scale/offset | unlabeled per-channel z-score |
| dense vs sparse mixing | covariance changes | unlabeled Euclidean Alignment |
| clean vs 14-bit noisier | model addicted to SNR | train-time fake-Emotiv noise |

Shared 10 channels: F3 F4 F7 F8 T7 T8 P7 P8 O1 O2.

Clinical T3 is the same scalp site as Emotiv T7. We rename before any math.

---

## Slide 5 -- Method in one breath (25 s)

GAP-Align = Geometry And Physics Alignment.

Match the physics first (P3, 10 channels, 1-40 Hz, 128 Hz, 2-second windows). Then match unlabeled wearable statistics (z-score, Euclidean Alignment). Then, only while training, degrade some clinical windows into a fake Emotiv so the classifier is not addicted to hospital SNR.

The model is frozen logistic regression on 30 numbers: 10 channels times theta, alpha, beta log band-power. No neural net. No GPU. SCVCNet already showed deeper nets transfer worse on this pair.

Pipeline F / Tent is skipped: Tent needs BatchNorm; we use logistic regression.

---

## Slide 6 -- What we do NOT claim (15 s)

Not first cross-device EEG. Not a new learning theory. Not SOTA. Not that cheap headsets now match clinical EEG. Not epilepsy or dementia diagnosis. Wellness / screening research tool only.

---

## Slide 7 -- Implementation right now (40 s)  << SHOW CODE HERE

Shipped in the repo:

- Project layout, gitignore, README
- Python 3.12 venv, pinned stack: mne, sklearn, no PyTorch
- `config.py` -- every constant in one file (10 channels, 128 Hz, P3, seed 42)
- `src/channels.py` -- T3->T7, strip "EEG ", STEW column lock
- `src/io_eegmat.py` / `src/io_stew.py` -- loaders + audit, no filtering yet
- Real EEGMAT sample on disk: subjects 00 and 01 only (full 36 later)
- `artifacts/channel_audit.txt` from a real EDF header

Verified on Subject00_1.edf:

- 500 Hz from the header
- 21 channels, names like `EEG F3`, `EEG T3`, `EEG P3`
- P3 is present (we can do the CMS match)
- All 10 shared sites present after renaming
- Values are in **Volts** (min about -2e-4). We must convert to microvolts
- T3 becomes T7, T5 becomes P7 -- confirmed on the real file

Not done yet (by design, we are phase-gated):

- Full EEGMAT + STEW download (STEW needs your IEEE DataPort login)
- Filter / epoch / features
- LOSO on EEGMAT (hard gate: accuracy >= 0.70 before any STEW number)
- Frozen classifier, ablation A/C/D/E, Streamlit dashboard

If asked "where are the accuracy numbers?": we refuse to invent them. A fake 63% would kill credibility. Review 1 is problem + method + working I/O. Numbers are Review 2.

---

## Slide 8 -- Demo we will ship (20 s)

Four-page Streamlit, artifacts only, no training in the UI:

1. Live compare: clinical window vs wearable, before/after GAP-Align
2. Scalp simulation + degrade slider (same fake-Emotiv function as training)
3. Ablation table + confusion matrices
4. India access + CDSCO disclaimer

90-second final demo script is already written in the implementation guide.

---

## Slide 9 -- Ask / next 48 hours (15 s)

Need from us: IEEE DataPort download of STEW into `data/raw/stew/`. That is the bottleneck.

Then: preprocess both sides to (N, 10, 256) at 128 Hz, EEGMAT LOSO, then unlabeled STEW adaptation, then dashboard.

---

## If they ask these

**Why can't you use STEW labels to train?**
Those labels exist only to compute the final score. Fitting on them would be data leakage. We would no longer be allowed to compare against SCVCNet's 62.9% zero-label number.

**Why logistic regression not EEGNet?**
SCVCNet already used a deep net on this pair. Our product is the adapter, not a new architecture. A frozen linear model keeps the numbers honest.

**Did you record RMS vs Emotiv in India?**
No. Public datasets are the proxy. Next validation is RMS + EPOC in one Indian lab.

**Is this for epilepsy?**
No. Rest vs cognitive load only. Not a diagnostic EEG.

**Why P3?**
Emotiv's CMS (hardware reference) sits at P3. We subtract P3 from the clinical recording so both devices share the same voltage zero. Most teams skip this.

---

## 90-second version if they cut you off

India's EEG AI sits in metro hospitals. We train on Neurocom (EEGMAT), test on Emotiv EPOC (STEW), rest vs load. The wearable is a different physics: different reference, fewer electrodes, lower rate, worse SNR. GAP-Align inverts those gaps with unlabeled calibration -- P3 re-reference, 10-channel lock, z-score, Euclidean Alignment, train-time fake Emotiv. Ghost baseline 63%. No STEW labels in training. Not a diagnostic device. Code and channel audit are in the GitHub repo; accuracy comes after STEW is on disk.
