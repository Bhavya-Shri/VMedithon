# GAP-Align hardware simulation (demo sheet)

This is the simulation you show judges. It is **already in the Streamlit app**. Do not build a second simulator.

Run:

```powershell
cd C:\Users\mail2_tdwzs1y\Desktop\bhavya_cursor\Bhav_VMedithon
.\.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

Use **Page 2 (Hardware simulation)** then **Page 1 (Live compare)** then **Page 3 (Ablation)**.

Page 2 now embeds the teammate **3D montage** (`app/hardware_viewer.html`, from PR #3) plus the original **degrade slider**. The 3D head is schematic 10–20. Official accuracy is still Page 3 (pipeline C). Do **not** quote numbers from `main.py` (that file trains STEW→EEGMAT with CORAL/SVM — wrong direction).

---

## Is 3D possible?

**Technically yes.** Guide Page 2 said start in 2D. The teammate viewer is now embedded on Page 2 as a **montage cartoon**, not as the accuracy proof.

The claim is still: **each hardware gap changes the numbers the frozen model sees**. That is proven by (1) the degrade slider moving **P(load)** and (2) STEW **before vs after** GAP-Align. The rotating head does not replace those numbers.

---

## What you are simulating (one sentence)

The cheap headset is not a smaller clinical headset. Different reference, fewer electrodes, lower rate, worse SNR. Those differences make a hospital-trained model collapse. Unlabeled GAP-Align recovers part of that — without STEW labels.

---

## Colour key (say this while pointing at the scalps)

| Colour | Meaning |
|---|---|
| Green | Shared 10–20 site. The model **sees** this: F3 F4 F7 F8 T7 T8 P7 P8 O1 O2 |
| Gold **P3** | Emotiv CMS lives here. On Neurocom we **subtract P3**, then drop it |
| Grey (Neurocom) | Clinical-only (Fp, C, Pz, …). Wearable is blind here |
| Grey (EPOC) | On the headset but **not in the model**: AF3 AF4 FC5 FC6 |
| Pink diamond P4 | EPOC DRL (driven right leg / ground) |

Left scalp = Neurocom (train). Right scalp = Emotiv EPOC (test).

---

## Act 1 — Hardware differences, and how they hurt the model

Stay on **Page 2**.

### 1A. Montage (19–23 ch vs 14 ch)

Point at grey vs green.

**Say:** “The wearable cannot see Cz or Pz. We keep only the 10 sites both headsets share. That is a spatial information loss, not a software bug.”

**Effect on the model:** fewer channels → weaker 30-D band-power vector. The hospital model was trained after this lock, so this gap is already “paid” in preprocess. The leftover gaps are reference, gain, mixing, and SNR.

### 1B. Reference (ears vs CMS at P3)

Point at gold P3.

**Say:** “Every EEG voltage is electrode minus a zero. Neurocom’s zero is linked ears. Emotiv’s zero is CMS at P3. We re-reference the hospital recording to P3 so both devices share a zero. Most teams skip this.”

**Effect:** without P3 match, the whole voltage scale is measured against the wrong point. That is a domain shift in every sample.

### 1C. Rate and bandwidth (500 Hz vs 128 Hz, 1–40 Hz)

**Say:** “Emotiv never sees 128–500 Hz content. We anti-alias resample to 128 Hz and band-pass 1–40 Hz on both sides so we are not asking the model to use frequencies the wearable cannot record.”

### 1D. SNR / 14-bit — the live slider

This is the **interactive** hardware difference.

Pick one clinical window. Leave slider at 0. Read **P(load) clinical**.

Move the slider. The same frozen logistic regression runs on the degraded epoch (`slider_degrade` → 30-D features → `scaler` + `clf`).

| Slider | Hardware you are faking |
|---|---|
| 0.00–0.25 | Already 10 ch / 128 Hz (identity) |
| 0.25–0.50 | Gel vs saline: Gaussian noise up to 8 µV |
| 0.50–0.75 | 14-bit ADC, LSB 0.51 µV |
| 0.75–1.00 | Extra noise + 4–8 Hz jitter |

**Say:** “As I make the hospital clip look more like a cheap headset, the frozen model’s P(load) should fall toward 0.5 — it becomes unsure. That is the collapse we measure on real STEW as pipeline A.”

**Pass/fail:** if P(load) does not move, stop talking. The sim is disconnected from the model.

The waveform plot under the slider is the same 2 s, 10-channel clip getting noisier. Point at it.

---

## Act 2 — Apply adaptation, performance changes

Go to **Page 1 (Live compare)**.

- Left: clinical EEGMAT window, high-confidence rest vs load (source).
- Right: STEW wearable window.
- Toggle **Before GAP-Align** / **After GAP-Align**.

**Before** = pipeline A: same 10 channels / 128 Hz / 1–40 Hz only. Frozen EEGMAT classifier. **No** z-score, **no** Euclidean Alignment.

**After** = unlabeled adapter on STEW (no STEW labels in the fit): per-subject z-score, then (if used) Euclidean Alignment. Classifier stays frozen.

**Say:** “Same weights. We did not retrain on wearable labels. We only matched unlabeled statistics of this cheap headset — scale per channel, then spatial covariance.”

Then **Page 3** — quote the real ablation (n = 10265 STEW epochs, 48 subjects):

| Pipeline | What it is | Acc | Macro-F1 | Kappa | What it proves |
|---|---|---|---|---|---|
| A Before | naive port | 0.508 | 0.368 | 0.030 | collapse (near chance; majority-class risk — F1 much worse than acc) |
| C | unlabeled z-score | **0.650** | **0.647** | 0.302 | gain / impedance |
| D | z-score + EA | 0.638 | 0.638 | 0.276 | spatial mixing; did not beat C |
| E | + train-time fake Emotiv | 0.640 | 0.637 | 0.281 | SNR; did not beat C |

**Winner = C** (highest macro-F1, both classes present). If D or E is worse, we **keep C** and say so. Honesty is the point.

Ghost baseline: SCVCNet EEGMAT→STEW, no target labels, **62.9% acc**. Pipeline C is ~65% acc / 0.65 F1 with **zero STEW labels**. Do not claim SOTA. Do not claim cheap = clinical.

Source LOSO in `train_meta.json` is about **0.66 acc** — below the Guide’s 0.70 gate. If asked: “Source model is usable but not a clean pass; we still show the transfer drop and unlabeled recovery.”

---

## How the two acts connect (draw this verbally)

```
Neurocom physics  --P3, 10 ch, 128 Hz, 1-40 Hz-->  frozen logistic regression
                                                      |
Page 2 slider: add Emotiv-like noise/quantize         |  P(load) -> 0.5
                                                      |
Real STEW Page 1 BEFORE (pipeline A)                  |  ~51% acc, F1 0.37  COLLAPSE
                                                      |
Unlabeled z-score (pipeline C)                        |  ~65% acc, F1 0.65  RECOVERY
                                                      v
Page 3 table proves which hardware-gap operator bought the points
```

---

## What this simulation is not

- Not a live Emotiv Bluetooth stream
- Not a diagnostic EEG / seizure demo
- Not “we recorded RMS vs EPOC in India” (public-data proxy)
- Not 3D for its own sake

---

## 60-second version if they cut you off

“Two headsets, same 10 green electrodes. Gold P3 is why we re-reference. Grey is what the wearable cannot see. I degrade a hospital window toward 14-bit noisy Emotiv — the frozen model loses confidence. On real STEW, that is pipeline A: about 51%. After unlabeled z-score, about 65%, F1 0.65, no STEW labels used to train. Not a diagnosis. Next is RMS plus EPOC in one Indian lab.”
