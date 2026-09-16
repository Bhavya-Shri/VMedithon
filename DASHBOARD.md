# GAP-Align dashboard — page-by-page guide

This document explains **every control, plot, and number** on the Streamlit app as it ships today. Use it to demo, to write slides, and to answer judges. It describes the **running UI**, not an older four-page sketch.

**Code:** `app/streamlit_app.py` (pages) · `app/components.py` (Plotly charts) · `app/hardware_viewer.html` (3D montage)  
**Talk sheet for the 3D + slider act:** `HARDWARE_SIMULATION.md`

---

## 0. What this app is (and is not)

GAP-Align is a **calibration layer**: a model trained on hospital-grade EEG (EEGMAT / Neurocom) is scored on a cheaper wearable (STEW / Emotiv EPOC) for **rest vs cognitive load**. The dashboard is the live proof of **failure then recovery**.

| It is | It is not |
|---|---|
| A viewer of **frozen artifacts** | A trainer. Streamlit never fits scaler, z-score, EA, or the classifier |
| A **2 s snapshot** player | A live Bluetooth / Emotiv stream |
| EEGMAT → STEW transfer | STEW → EEGMAT (that is `main.py`; do not quote it) |
| Screening / research UI | A diagnostic EEG, seizure detector, or CDSCO-cleared device |
| Neurocom vs Emotiv as **open-data proxies** | “We recorded RMS vs EPOC in an Indian hospital” |

**Hard leakage rule (say this if asked):** STEW labels are used only to **score** the dashboard and `ablation.json`. They were never used to fit the logistic regression, the scaler, z-score, Euclidean Alignment, or hyperparameters.

---

## 1. How to open it

From the repo root, with the project venv:

```bat
cd C:\Users\mail2_tdwzs1y\Desktop\bhavya_cursor\Bhav_VMedithon
.venv\Scripts\activate.bat
streamlit run app/streamlit_app.py --server.port 8765
```

Browser: **http://localhost:8765**  
Stop: **Ctrl+C** in that terminal (closing the tab does not stop the server).

Default page is **1 · Live compare**. Theme is dark (`#0b1220` background, gold `#e4b84a` accent) via `.streamlit/config.toml` plus extra CSS in the app.

If JSON artifacts are missing, a red box says: **Run `python -m src.run_all` first.** The app then stops. It will not invent numbers.

The **degrade slider** on Page 2 also needs `artifacts/clf.joblib` and `artifacts/scaler.joblib`. Those two files are gitignored. If they are absent locally, Pages 1 and 3 still show saved JSON/PNG metrics; the slider shows a warning and will not move `P(load)`.

---

## 2. What is loaded at start (the data contract)

Nothing on screen is computed from raw EDF/TXT at demo time. `st.cache_data` / `st.cache_resource` read:

| File | Used on | What it holds |
|---|---|---|
| `artifacts/demo_windows.json` | Pages 1–2 | 10 clinical + 10 wearable **2 s** windows (10 ch × 256 samples at 128 Hz), features, predictions, probabilities |
| `artifacts/train_meta.json` | Pages 1, 2 | EEGMAT LOSO, channel list, `target_source`, winner id |
| `artifacts/ablation.json` | Pages 1–3 | Official A/C/D/E scores, confusion matrices, winner **C** |
| `artifacts/clf.joblib` + `scaler.joblib` | Page 2 slider | Frozen logistic regression + standard scaler (EEGMAT only) |
| `artifacts/stew_preds_before.npz` / `_after.npz` | Page 3 | Per-window STEW predictions for the per-subject Δ bar |
| `artifacts/cm_before.png` / `cm_after.png` | (files on disk) | Same matrices as Page 3, also saved as images for slides |

### Labels

- `y = 0` → **Rest**
- `y = 1` → **Cognitive load**

On the wearable side the true `y` is an **eval label**. Caption on Page 1 says it was never used to fit.

### Shared channels (the 10 sites the model actually sees)

Order is a contract: **F3 F4 F7 F8 T7 T8 P7 P8 O1 O2**.

Each window is turned into a **30-D vector**: those 10 channels × three log band-powers

- **theta** 4–8 Hz  
- **alpha** 8–12 Hz  
- **beta** 13–30 Hz  

Welch PSD on the 2 s epoch, then `log10`. The classifier never sees raw voltage traces.

### Current frozen scores (quote these)

From `ablation.json`, **n = 10265** STEW epochs (48 subjects; 5214 rest / 5051 load). `target_source = stew`.

| Pipeline | Acc | Macro-F1 | Kappa | What it is |
|---|---:|---:|---:|---|
| **A** naive port | 50.8% | 0.368 | 0.030 | Channel/rate/filter lock only. Collapse |
| **C** unlabeled z-score | **65.0%** | **0.647** | 0.302 | **Winner.** Gain / impedance |
| **D** z-score + Euclidean Alignment | 63.8% | 0.638 | 0.276 | Spatial mixing; did not beat C |
| **E** fake-Emotiv train + D | 64.0% | 0.637 | 0.281 | SNR gap at train time; did not beat C |
| **F** Tent | skipped | — | — | Needs BatchNorm; we use linear logreg |

Ghost baseline to beat on slides: **SCVCNet EEGMAT→STEW, no target labels ≈ 62.9% acc**. Winner C is above that on accuracy.

EEGMAT LOSO (source, `train_meta.json`): acc **66.3%**, macro-F1 **0.614**. Guide gate was 70% — **not met**. The model was frozen anyway. Do not retune on STEW.

---

## 3. Chrome that appears on every page

### Title strip

- Kicker: **GAP-Align · Geometry And Physics Alignment**
- Title: **Hospital EEG knowledge, unlocked for a wearable**
- Subtitle: train EEGMAT (Neurocom), test STEW (Emotiv EPOC), rest vs load, **zero target labels for fitting**

### Sidebar

| Control | Meaning |
|---|---|
| **1 · Live compare** | Failure vs recovery on one window |
| **2 · Hardware simulation** | Why the headsets differ + degrade slider |
| **3 · Ablation & trust** | Official table and confusion matrices |
| Direction line | **EEGMAT → STEW**. Classifier never sees wearable labels |
| CDSCO line | Not a diagnostic EEG. Not CDSCO-cleared. Cognitive-load screening research tool |
| Data line | Public de-identified research sets only. No patient data |

There is **no Page 4**. The old India / price-ladder page was removed. The regulatory sentence stayed in the sidebar (Guide cut-order: cut Page 4 polish, keep the disclaimer).

### Proxy banner (Page 1 only, if it appears)

If `train_meta.json` has `target_source` other than `"stew"`, a yellow box says the wearable side is an Emotiv-**like proxy** (degraded EEGMAT), not STEW. **Do not quote those numbers as EEGMAT→STEW.** On this laptop the official run is real STEW, so this banner should stay hidden.

---

## 4. Page 1 — Live compare

**Job of this page:** in ~90 seconds, show a hospital window the frozen model can read, a wearable window it **fails** on (pipeline A), then the same wearable window **after unlabeled GAP-Align** (winner C).

Layout: **left clinical | center scores | right wearable**.

### 4.1 Controls

| Control | What it does | What it signifies |
|---|---|---|
| **Demo pair** | One dropdown. Slot `i` loads `clinical[i]` **and** `wearable[i]` | EEGMAT and STEW are **different people**. Slots 1–5 rest, 6–10 load |
| **Play** | Every 0.4 s advances the **pair** | Fake streaming. Both columns move |
| **Wearable view** radio | **Before GAP-Align** vs **After GAP-Align** | Same STEW traces; **different 30-D features and predictions**. Weights stay frozen |

Do not pick a STEW id and expect the left column to stay put. A previous bug always showed EEGMAT subject 27 (`clinical[0]`) because each STEW id had one window so the clinical index stayed 0.

**After vs before looking “more different” from clinical:** that is expected. (1) Score After against the **STEW eval label**, not against the EEGMAT `P(load)`. Before often matches clinical only because **both collapsed to load**. (2) After band-power bars are **z-scored**; they are not on the same vertical scale as clinical log-power. Traces (voltage) do not change when you toggle After.

### 4.2 Left column — Clinical · Neurocom / EEGMAT

| Widget | Input | Signifies |
|---|---|---|
| **Stacked traces** | `clinical[i]["x"]` shape (10, 256) | 2 seconds, 10 shared sites, already resampled to **128 Hz** and converted to µV. Offset so channels do not overlap |
| **θ / α / β grouped bars** | `clinical[i]["features"]` (30,) | What the classifier actually eats. One bar group per electrode |
| **Frozen model** label | `pred` 0/1 | Rest vs cognitive load |
| **P(cognitive load)** bar | `proba[1]` | Softmax-style probability of load from the **same** logreg used on STEW |
| Caption | — | “Hospital-grade stand-in · 19–23 ch wet · 500 Hz → aligned 10 ch 128 Hz” |
| True label line | `y` | EEGMAT rest (`_1` EDF) vs arithmetic (`_2` EDF), after epoching |

**What went into a clinical window (already done in `src/`):** EDF load → canonicalize T3→T7 etc. → **re-reference to P3** (Emotiv CMS lives there) → drop P3 → pick SHARED_CH → notch 50 Hz → band-pass 1–40 Hz → resample 128 Hz → 2 s epochs, 1 s stride → peak-to-peak reject → 30-D log band-power → **scaler + logreg fitted on EEGMAT only**.

The left column is the “this model is not random on its home device” shot. LOSO 66% is modest; that honesty lives in the center metrics.

### 4.3 Right column — Commercial · Emotiv / wearable

| Widget | Before GAP-Align | After GAP-Align |
|---|---|---|
| Traces | `wearable[j]["x"]` | **Same traces.** Adapter does not redraw the voltage |
| Band-power bars | `features_before` | `features_after` |
| Prediction | `pred_before` / `proba_before` | `pred_after` / `proba_after` |
| Caption | “Same frozen weights. No new labels.” | “Wearable stand-in · 14 ch saline · native 128 Hz” |

**What “before” means:** pipeline **A** — 10-ch lock, 128 Hz, 1–40 Hz, **no** unlabeled z-score. This is the naive port. On the full STEW set it is ~51% acc and F1 0.37 (almost chance, majority-ish collapse).

**What “after” means:** winner pipeline **C** — unlabeled **per-channel z-score** on the target (STEW), still the **same** EEGMAT logreg. No STEW `y` in that z-score fit.

Eval label caption: `Subject {id} · eval label: Rest|Cognitive load (never used to fit)`.

STEW files: `subXX_lo.txt` = rest, `subXX_hi.txt` = load. No P3 re-reference on STEW (CMS is already the hardware zero).

### 4.4 Center column — the claim in numbers

These numbers do **not** change when you move the window. They are the full-set scores from `ablation.json` / `train_meta.json`.

| Metric | Source | How to say it |
|---|---|---|
| Badge | hardcoded | “Unlabeled adaptation · no STEW labels used to train the classifier” |
| **EEGMAT LOSO accuracy** | `loso_acc` ≈ 66.3% | Source leave-one-subject-out. Not the transfer number |
| **Wearable accuracy before (A)** | `ablation["A"]["acc"]` ≈ 50.8% | Collapse on the cheap headset |
| **Wearable accuracy after** | winner C ≈ 65.0%, delta **+14.2 pp** | Recovery without new labels |
| **Kappa before / after** | 0.03 → 0.30 | Agreement beyond chance; A is almost none |
| **Macro-F1 after** | 0.647 | Must stay close to accuracy or we collapsed to one class |
| Winner caption | `winner: C` | Ghost 62.9%. C is the only adapter that won |

**Why both accuracy and F1:** pipeline A’s accuracy ~51% with F1 0.37 is the tell. It predicts almost everything as load (see Page 3 confusion matrix: 194 true-rest correct vs 5020 rest→load). Recovery has to lift **both** classes.

### 4.5 Are Page 1 graphs supposed to move?

- **Both columns’ traces:** yes, when you change **Demo pair** or hit **Play**. EEGMAT subject id in the left caption must change (27, 15, 23, …).
- **Center metrics:** **static.** Official n=10265 result.
- **After vs Before:** voltage traces stay the same; band bars and `P(load)` change.

---

## 5. Page 2 — Hardware simulation

**Job of this page:** the cheap headset is **not** a smaller clinical headset. Show geometry, then show that **SNR / bit-depth** moves the **same frozen model’s** `P(load)`.

Three stacked acts:

1. Teammate **3D montage** (visual)  
2. Optional **2D colour-key scalps** (Guide Page 2)  
3. **Degrade slider** (the scientific sim)

### 5.1 3D montage (`app/hardware_viewer.html`)

Origin: teammate PR #3 (`eeg_workload_dashboard.html` + `main.py`). We **embedded the visual**, not her scorer.

| Control | What it does | Signifies |
|---|---|---|
| **STEW · 14ch** | Emotiv EPOC montage | Test device |
| **EEGMAT · 19ch** | Neurocom-style 10–20 | Train device |
| **Rest / Task** | Recolours nodes + redraws PSD | Cognitive state overlay, not a new model |
| **Functional connectivity** | Shows/hides purple links | Correlation of the 10 shared traces in the demo pack (`\|r\| > 0.3`). Extra channels have no real edges from GAP-Align epochs |
| Drag on the canvas | Rotate the schematic head | Idle auto-rotate resumes after ~2 s |
| Hover a node | Tooltip: channel, role, region, activity %, state | Role is GAP-Align meaning, not her original “workload % only” |

**Node colours (GAP-Align skin, not her original cyan/amber-only map):**

| Colour | Sites | Meaning |
|---|---|---|
| Cyan → amber | F3 F4 F7 F8 T7 T8 P7 P8 O1 O2 | **Shared.** Model sees these. Colour = min-max **θ+β−α** from demo windows (clinical features for EEGMAT, `features_before` for STEW) |
| Gold | **P3** | Emotiv **CMS**. On Neurocom we re-reference here, then drop P3 |
| Grey | AF3 AF4 FC5 FC6, and clinical-only Fp/C/Pz/… | On a headset **or** on Neurocom, but **not in the 10-ch model** |

**PSD panel (right):** rest vs task Welch curves averaged over demo windows of that device, 1–45 Hz. This is **shared-channel average**, not a 19-ch clinical spectrum. Toggle device to switch EEGMAT vs STEW curves.

**What we deliberately hid**

Her HTML also had a CORAL scatter and an SVM “cross-device accuracy” bar. Those used **train STEW / test EEGMAT**. That is the **wrong direction** and the **wrong model**. They are CSS-hidden in `hardware_viewer.html`. Official numbers stay on Pages 1 and 3.

`main.py` is still in the repo as her standalone script. **Do not run it for judge quotes.**

**Honesty line under the 3D view:** positions are **schematic 10–20**, not a digitized head-shape. Three.js needs a network CDN load (`three.min.js`). If the iframe is blank, use the 2D expander.

### 5.2 Expander — 2D colour key (Guide Page 2)

Two Plotly scalps from `plot_scalp` in `app/components.py`. Nose up, approximate 10–20 (`POS` dict).

**Left — Neurocom 19-ch · linked-ear ref**

- Green: shared 10 (kept)  
- Gold P3: “CMS match: we re-reference here”  
- Grey: clinical-only (Fp1/2, Fz, C3/Cz/C4, Pz, P4, …) — wearable is blind here  
- Caption: original **linked-ear (A1+A2)** zero. We subtract P3 to imitate CMS, then drop P3  

**Right — Emotiv EPOC 14-ch · CMS at P3**

- Green: the 10 in the model  
- Grey: **AF3 AF4 FC5 FC6** — on the headset, **not in the model**  
- Gold P3: hardware CMS  
- Pink diamond **P4**: DRL (driven right leg / ground)  

This is the colour key you point at while talking montage and reference. The 3D head is the showpiece; these two discs are the **precise** claim.

### 5.3 Degrade slider — model-linked hardware sim

This is the only Page 2 widget that is allowed to change a **probability**.

| Control | Range | Effect |
|---|---|---|
| **Clinical window** | 0 … 9 | Which EEGMAT 2 s clip from `demo_windows.json` |
| **Degrade toward fake EPOC** | 0.00–1.00 | `src.degrade.slider_degrade` |

Slider schedule (same function family as train-time `fake_emotiv`):

| Amount | Physics being faked |
|---|---|
| 0.00–0.25 | Identity. Already 10 ch / 128 Hz |
| 0.25–0.50 | Gaussian noise 0 → **8 µV** (SNR) |
| 0.50–0.75 | **14-bit quantize**, LSB **0.51 µV** (Emotiv-class) |
| 0.75–1.00 | Extra noise + **4–8 Hz** sinusoidal jitter |

Then the **same** `scaler.joblib` + `clf.joblib` see 30-D features of the original vs degraded epoch.

| Metric | Meaning |
|---|---|
| **P(load) clinical** | Frozen model on the clean aligned hospital window |
| **P(load) after degrade** | Same weights after fake wearable physics |
| **Slider** | The mix amount |
| Degraded traces | What the model is now looking at |

**Expected behaviour:** as SNR and bit-depth fall, `P(load)` should drift **toward 0.5** (the model gets unsure). If it does not, the sim is lying — fix `degrade.py`, do not invent a second slider.

**What this slider is not:** it is not real STEW. Real STEW recovery is Page 1 After / Page 3 pipeline C. The slider is a **controlled** hardware-gap demo on a clinical epoch.

---

## 6. Page 3 — Ablation & trust

**Job of this page:** prove **which operator bought the points**, and that recovery is not majority-class cheating. Charts here are **supposed to stay static**. They are the frozen n=10265 result.

### 6.1 Grouped bar chart

X: pipelines **A, C, D, E**.  
Y: **acc**, **macro-F1**, **kappa**.  
Dotted line: **SCVCNet 62.9%**. Dashed line: **chance 0.5**.

**How to read it**

- A hugs chance on F1/kappa — naive port failed.  
- **C is the tallest F1 and the winner.**  
- D and E are close but **below** C. Extra geometry (EA) and train-time noise did not help this pair of datasets. Say that out loud; it is a trust feature, not a bug.

### 6.2 Confusion matrices

| Panel | Pipeline | Pattern |
|---|---|---|
| **Before** | A | Almost all mass on **pred load**. True rest: 194 correct vs 5020 dumped to load. True load: 5017/5051. Accuracy looks “50%” while the model is not doing two-class work |
| **After** | **C** | Rest 2891 / 2323; load 1270 / 3781. Both classes used. F1 ≈ acc |

Sentence under the plots: **“Recovery must improve both classes.”** If F1 is far below accuracy, the adapter collapsed.

If `target_source != stew`, a warning forbids putting the numbers on a slide as EEGMAT→STEW. Official run: real STEW, plus ghost-baseline caption 62.9%.

### 6.3 Per-subject Δ accuracy bar

From `stew_preds_before.npz` vs `stew_preds_after.npz`.

For each STEW subject: `(acc after − acc before)`. Green = helped, pink = **worse**.

**Signifies:** unlabeled z-score is not magic for every brain. Some subjects drop. That is the honest plot. Do not hide it.

### 6.4 Operator table

| ID | What it proves (hardware gap → operator) |
|---|---|
| **A** | Naive port after channel / rate / filter lock |
| **C** | Unlabeled per-channel z-score (**gain / impedance**) |
| **D** | Z-score + Euclidean Alignment (**spatial mixing**) |
| **E** | Train-time `fake_emotiv` noise (**SNR / 14-bit**) |
| **F** | Tent — skipped; needs BatchNorm; we use a linear model |

Winner **C** means: on EEGMAT→STEW, the gain/impedance fix was the one that recovered both classes. Montage lock was already paid in preprocess (A still failed). EA and fake-Emotiv were tried and did not win.

---

## 7. What was removed (so you do not look for it)

**Page 4 · India / access** (price ladder, neurologist-shortage copy) is gone from the sidebar. Design-doc India pitch still exists in `GAP_Align_Project_Design.md` and `REVIEW_1_Presentation.md` if you need it on a slide. The dashboard keeps only the **CDSCO / not a diagnosis** line.

---

## 8. Suggested 90-second click path

1. **Page 2** — STEW 14ch vs EEGMAT 19ch. Grey vs green. Gold P3.  
2. Slider 0 → ~0.8. Point at **P(load)** drifting.  
3. **Page 1** — wearable **Before** (collapse) → **After** (recovery). Center: 50.8% → 65.0%, F1 0.647.  
4. **Page 3** — A vs C matrices; “both classes”; C beat ghost 62.9%; D/E did not beat C.

One-sentence closer: *“No STEW labels in training. Not a diagnostic EEG. Next validation is RMS + EPOC in one lab.”*

---

## 9. File map (if a judge asks “where is that number?”)

```
app/streamlit_app.py          pages, artifact loader, 3D inject, slider
app/components.py             traces, band bars, 2D scalps, ablation, CMs
app/hardware_viewer.html      3D montage (teammate visual, GAP-Align colours)
src/degrade.py                slider_degrade + fake_emotiv
src/features.py               30-D log band-power
src/train.py                  predict_features (no fit in Streamlit)
artifacts/ablation.json       official A/C/D/E  ← quote this
artifacts/train_meta.json     LOSO, direction, channels
artifacts/demo_windows.json   10+10 snapshots for Pages 1–2
artifacts/clf.joblib          frozen logreg (gitignored)
main.py                       teammate CORAL/SVM, WRONG direction — do not quote
eeg_workload_dashboard.html   teammate original HTML (unskinned)
```

---

## 10. Conflicts worth knowing (do not mix stories)

| Source | Says | Dashboard reality |
|---|---|---|
| Implementation Guide §15 | Four pages, 2D-only Page 2 | **Three** pages; 3D montage **plus** 2D expander + slider |
| `main.py` | SVM + CORAL, STEW→EEGMAT | **Not** wired to official metrics |
| Page 1 traces | Look “live” | Frozen 2 s clips |
| Page 3 charts | Look like they might update | They **must not**; they are the paper numbers |

Code follows `GAP_Align_Implementation_Guide.md` for the **scorer** (logreg, EEGMAT→STEW, no STEW labels). The 3D viewer is a user-requested visual on top of that scorer.
