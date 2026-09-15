# GAP-Align: Clinical-to-Consumer EEG Transfer

Hackathon design document — Neurocom (clinical) → Emotiv EPOC (commercial)  
**Market:** India (hospital EEG models → affordable wearables at colleges, clinics, and telehealth)

---

## 1. One-line pitch

India cannot put a neurologist and a ₹10-lakh EEG cart in every district. A model trained on hospital-grade EEG fails on the cheap headset a college or PHC can actually buy. We measure that drop, invert the hardware gap with unlabeled adaptation, and show how much reliability a wearable can inherit — without new labels or new hardware.

**India one-liner for judges**

"Hospital EEG knowledge is locked in cities. GAP-Align is the calibration layer that lets a model trained on a clinical headset run on an affordable Indian-deployable wearable — and we show the failure and the recovery live."

---

## 2. What we are (and are not) claiming

**We are claiming**

- Clinical → commercial is the product constraint: train on expensive hardware, deploy on an affordable headset.
- We measured the failure (accuracy drop) on a real device pair with public data.
- We built a small unlabeled adapter whose steps map 1:1 to measured hardware differences.
- We report which combination actually recovers accuracy — even if recovery is partial.

**We are not claiming**

- First people to do cross-device EEG.
- A new learning theory or unpublished SuperAlgo.
- Highest success rate in the EEG/BCI literature.
- Cheap headsets now match clinical reliability.
- Epilepsy / seizure / dementia diagnosis, or a CDSCO-cleared medical device (MVP is research/wellness cognitive-load screening only).

A combo of known methods is only defensible if it is **justified by the hardware gap** and we show **each step helps** (ablation table). Mixing z-score + CORAL + Tent because they sound good is not a contribution. Euclidean Alignment + AdaBN already exists as a 2021 EEG paper.

**Ghost baseline to beat:** SCVCNet already ran EEGMAT → STEW (Neurocom → Emotiv) at about **63%** with no target labels and **70%** with a bit of target tuning. Our bar is: beat ~63% with **zero STEW labels**, and show the ablation.

---

## 3. India market positioning

Yes — this idea is *more* valuable in India than in the US/EU, if we pitch **access**, not “we invented EEG AI.”

### 3.1 The real bottleneck (not just device cost)

| Fact | Why it matters |
|---|---|
| **&lt; 2,500 neurologists** for 1.4 billion people | EEG interpretation is urban and scarce |
| ~**52 epilepsy monitoring units**, many private | Advanced EEG is not a district-hospital default |
| EEG techs are informally trained; neurophysiology is thin | Wearables fail without a calibration layer |
| Mental-health treatment gap for common disorders is ~**85%** | Screening at college / PHC / tele-psych is the opening |
| India already has EEG AI (e.g. **MANAS-1**) | Do not claim “first Indian EEG model.” Claim **device transfer** so hospital-trained models can run on wearables |

Hardware is only half the gap. The other half: models are trained where the machines and labels live (AIIMS, NIMHANS, corporate hospitals) and never work on the headset a counsellor in Nashik or a college in Coimbatore can afford.

### 3.2 Indian price ladder (use on a slide)

Do not say “₹50 vs ₹50,000” vaguely. Use a real ladder:

| Tier | Typical India kit | Ballpark | Where it lives |
|---|---|---|---|
| Imported hospital EEG | Nihon Kohden / Natus / Compumedics, 32–64 ch, video-EEG | **₹8–40 lakh+** | Metro tertiary / private neuro |
| Indian clinical portable | **RMS Maximus** (Panchkula), Medicaid Neuromax, 24–32 ch wet | **₹1–5 lakh** | District hospital, medical college, diagnostic lab |
| Research wearable | **Emotiv EPOC X** 14-ch saline (our demo target) | **~₹70k–1.5 lakh** with software | Colleges, HCI labs, startups |
| Wellness band | Muse / NeuroSky-class, 4–5 ch | **~₹20–40k** | Consumer, coaching, early screening |

**Demo proxies (keep these — we have public data):**

- Clinical analog = **Neurocom / EEGMAT** ≈ RMS / Medicaid hospital EEG  
- Commercial analog = **Emotiv EPOC / STEW** ≈ the wearable India can actually deploy at scale  

Say this once: we did not record on RMS hardware; Neurocom vs Emotiv is the open-data stand-in for “Indian hospital cart vs Indian-deployable wearable.”

### 3.3 Who buys this in India

| Buyer | Job-to-be-done | Why GAP-Align |
|---|---|---|
| Medical colleges / NITs / IITs | teach BCI and psychophysiology without a ₹20-lakh cart for every student | one hospital-trained model, many EPOC kits |
| Tele-psych / campus counselling | rest vs cognitive-load / stress proxy, not diagnosis | wearable in hostel; model from lab EEG |
| IT / BPO occupational health | burnout / overload screening in Bengaluru, Hyderabad, Pune, NCR | cheap headset at clinic, not neurology OPD |
| District hospital / medical college EEG lab | reuse models when they later buy a cheaper ambulatory kit | unlabeled calibration, no new labelled study |
| Edtech / exam-prep (careful) | attention / load during study | wellness framing only |

**Do not sell to:** epilepsy clinics as a diagnostic EEG replacement. Wrong labels, wrong channels, CDSCO risk, and judges with an MBBS will destroy that claim.

### 3.4 India-safe problem statement

**In scope:** rest vs cognitive load / mental workload / stress-attention proxy on a wearable, after training on clinical-grade EEG.

**Out of scope for MVP:** seizure detection, stroke, dementia, “replace the neurologist,” Ayushman billing, CDSCO Class B/C SaMD.

Regulatory line for slides: public de-identified research data only; **wellness / screening research tool**; any clinical deployment needs CDSCO + Indian labelled data + a hospital partner (NIMHANS / AIIMS / state medical college).

### 3.5 India novelty (hackathon)

The **method** is still 3/10 in the global literature.  
The **India packaging** is ~**8/10 for an Indian Medithon** if you say:

- Hospital models should serve PHC / campus / MSME clinics.  
- Indian clinical EEG (RMS) and imported wearables (Emotiv) do not share a model today.  
- We show the drop and a zero-label adapter.  
- Roadmap: validate next on RMS + EPOC in one Indian lab.

That is localization + access, not a new algorithm. For VMedithon, that is the correct bet.

### 3.6 6–12 month India roadmap (slides only)

1. Partner one medical college or NIMHANS-affiliated lab: same task, **RMS 24-ch + Emotiv EPOC X**, 20–30 subjects.  
2. Keep GAP-Align frozen; report Indian paired-device numbers.  
3. Package as a Python module for college labs (₹0 licence for research).  
4. Only then talk telehealth / NHM. Not before paired Indian data.

---

## 4. The two headsets (primes for demo and simulation)

Use the pair we already have public data for. Do not pick BioSemi vs Muse for the live demo — we have no paired public rest/load recordings for that pair.

| | Clinical (source, train) | Commercial (target, test) |
|---|---|---|
| Device | **Neurocom EEG 23-ch** (XAI-MEDICA) | **Emotiv EPOC / EPOC X** |
| Dataset | EEGMAT (PhysioNet) | STEW (IEEE DataPort) |
| Price class | clinical cart, wet gel, technician | ~$800–$1000 wearable |
| Electrodes | wet **Ag/AgCl** | saline felt **Ag/AgCl** |
| Channels | 19–23, full 10–20 | **14** + CMS/DRL |
| Montage | Fp1, Fp2, Fz, F3, F4, F7, F8, C3, C4, Cz, P3, P4, Pz, O1, O2, T3, T4, T5, T6 | AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4 |
| Reference | linked **ear** (A1+A2) | **CMS at P3**, DRL at P4 |
| Rate | **500 Hz** | **128 Hz** (internal 2048 Hz, sequential ADC) |
| Bandwidth | typically ~0.5–30/40 Hz + 50 Hz notch | **0.2–45 Hz**, 50/60 Hz notch |
| ADC | clinical-grade, high bit depth | **14-bit** usable, LSB ≈ 0.51 µV |
| Subjects | 36 | 48 |
| Task in dataset | mental arithmetic (serial subtraction) | SIMKAP multitasking workload |
| Shared label | rest / baseline vs cognitive-load | rest / baseline vs cognitive-load |

This pair is the right choice because:

- They are real, named products, not imaginary “Headset A/B”.
- Public rest vs cognitive-load recordings exist on both.
- SCVCNet already used them, so we have a published ghost baseline.
- For slides we can show **EPOC X** photos. STEW was recorded on the older EPOC. Same 14 sites, same 128 Hz. Say that once.

### Shared 10–20 sites

Emotiv names T7/T8/P7/P8 are the same locations as clinical T3/T4/T5/T6.

**Keep:** `F3, F4, F7, F8, T7, T8, P7, P8, O1, O2`

**Clinical-only (drop on purpose):** Fp1, Fp2, Fz, C3, C4, Cz, P3, P4, Pz

**Emotiv-only (do not use at test if the model never saw them):** AF3, AF4, FC5, FC6

---

## 5. Hardware gap → one operator each

This is the method. Each difference becomes one transform. That is the “unique adaptation,” not a new neural net.

| Hardware gap | What it does to the signal | Operator |
|---|---|---|
| 19–23 ch vs 14 ch | spatial information missing | keep only the 10 shared sites |
| Ear ref vs CMS @ P3 | entire voltage is measured against a different point | **re-reference Neurocom to P3** before dropping P3 |
| 500 Hz vs 128 Hz | extra high-frequency content the Emotiv never sees | resample 500 → 128 with anti-alias |
| ~30/40 Hz vs 0.2–45 Hz | different passband | band-pass **1–40 Hz** on both |
| Gel vs saline, different gain | per-channel scale/offset | unlabeled **per-channel z-score** on Emotiv |
| Dense vs sparse mixing | covariance shape changes | unlabeled **Euclidean Alignment** on 10-ch covariances |
| 24-bit clean vs 14-bit noisier | model overfits to clinical SNR | while training, add light noise / 14-bit quantize on a copy of clinical trials |

---

## 6. GAP-Align (Geometry And Physics Alignment)

**Definition:** Match the physics of the headset (reference, rate, bandwidth, montage), then match the unlabeled statistics of the cheap device (z-score + Euclidean alignment).

**Borrowed pieces:** z-score, Euclidean Alignment, resampling.

**Ours:** the order, the P3 re-reference (Emotiv’s actual CMS), and training-time “fake Emotiv” degradation so the classifier is not addicted to clinical SNR.

Nobody has published that exact stack on Neurocom → EPOC for rest vs load. That is enough novelty for a hackathon. It is not enough to claim “highest success rate in EEG.”

### 5.1 Train (Neurocom / EEGMAT only)

1. Load 19-ch, 500 Hz.
2. Re-reference to **P3** (Emotiv CMS). This is the step most teams skip and the one to highlight on the scalp simulation.
3. Keep the 10 shared channels. Drop Fp / C / Pz. We lose midline and motor sites on purpose — that is the cheap headset.
4. Band-pass 1–40 Hz, resample to 128 Hz.
5. Epoch 2 s windows. Label rest vs load.
6. **Device randomization:** for each batch, with probability 0.5, quantize toward 14-bit and add small Gaussian noise (consumer SNR).
7. Features: log band-power θ (4–8 Hz), α (8–12 Hz), β (13–30 Hz) on 10 channels → 30-D vector.
8. Fit **logistic regression** (or linear SVM). Freeze it.

Do not lead with EEGNet / a deep net. SCVCNet already showed deeper models transfer worse on this pair. A frozen linear classifier keeps the “numbers are real” story intact.

### 5.2 Test (Emotiv / STEW, no labels for adaptation)

1. Rename T7/T8/P7/P8 if needed; keep the same 10 channels.
2. Already 128 Hz; band-pass 1–40 Hz; same 2 s epochs.
3. Unlabeled **z-score** per channel using a short STEW batch (20–30 s of mixed rest/task is enough).
4. Unlabeled **Euclidean Alignment:** whitening with the 10×10 covariance of that batch.
5. Same 30-D band-power → frozen classifier.

### 5.3 Ablation (this is the “each step helps” proof)

| Pipeline | STEW accuracy | What it tests |
|---|---|---|
| A. Channel + rate only | expect ~chance–55% | naive port |
| B. A + P3 re-reference | | reference gap |
| C. B + z-score | | gain / impedance |
| D. C + Euclidean Alignment | | spatial mixing |
| E. D + train-time noise | | SNR gap |
| F. E + Tent (optional) | often worse on EEG | do not keep if it drops |

The “novel combo” is whichever of B–E wins. If Tent loses, cut it and say so.

**Success bar**

| | Approx. ACC |
|---|---|
| Chance | 50% |
| Train EEGMAT, test STEW, no adaptation (SCVCNet) | ~63% |
| Same, with some target tuning (SCVCNet) | ~70% |
| Our goal | beat 63% with **zero STEW labels**, plus the ablation table |

If z-score alone goes 55% → 64%, that is a win. If Tent then drops to 51%, report that and keep z-score. Honesty is more impressive than a branded combo that was not ablated.

If we do not beat 63%, the demo still works if we show *which* hardware gap caused the remaining miss (usually missing Cz/Pz and the task mismatch: arithmetic vs SIMKAP).

---

## 7. Three deliverables for the final review

Make them one story, not three side projects.

### 6.1 AI model

Frozen clinical classifier: log band-power + logistic regression / linear SVM, trained only on Neurocom (EEGMAT). The product is the adapter, not a new network.

### 6.2 Dashboard — model I/O on both headsets

Same time window, two columns:

- **Left (clinical / Neurocom):** waveform, band-power bars, rest/load prediction, confidence.
- **Right (commercial / EPOC):** same, **before** GAP-Align and **after**.
- **Center:** accuracy drop in points, then recovered points.

If a judge only watches 90 seconds, this screen is the whole project.

### 6.3 Hardware simulation on the brain

This only adds value if it explains the drop. A rotating 3D headset with no link to the accuracy number is filler.

Show both montages on a scalp:

- **Neurocom:** 19 dots, ear reference highlighted, then an arrow: “we subtract P3 to imitate CMS.”
- **EPOC:** 14 dots, CMS/DRL at P3/P4, the clinical-only sites greyed out as “blind.”

Add a **degrade** slider: take a clinical epoch → drop channels → resample → add noise → the clinical model’s confidence falls. That is the mechanism. It ties the simulation to the metric.

---

## 8. What to say in the review

“India’s EEG knowledge sits in metro hospitals. We picked Neurocom vs Emotiv EPOC as an open-data stand-in for Indian hospital EEG vs a wearable a college or clinic can buy. The cheap headset is not a smaller clinical headset: different reference, fewer sites, lower rate, worse SNR. GAP-Align inverts those four gaps, then we measure which inversion actually recovers accuracy. This is a screening/research tool, not a diagnostic EEG.”

Do not say we invented a new learning theory. Do say the **P3 re-reference + montage lock + unlabeled EA/z-score + train-time EPOC noise** is the method, justified by these two devices.

---

## 9. Data and closest prior work

### Datasets

- **EEGMAT (clinical, Neurocom):** https://physionet.org/content/eegmat/1.0.0/
- **STEW (commercial, Emotiv EPOC):** IEEE DataPort doi: 10.21227/44r8-ya50

### Closest paper (same devices, same label)

**SCVCNet** — Wang et al., arXiv:2310.03749 / IEEE TIM 2025  
PDF: https://arxiv.org/pdf/2310.03749  
Code: https://github.com/7ohnKeats/SCVCNet

They used STEW + EEGMAT, binary rest vs load, 10 shared channels, EEGMAT 500 → 128 Hz. They solved it with a **new architecture**, not test-time adaptation.

Direct numbers (their direction EEGMAT → STEW):

- Train EEGMAT, test STEW, no target tuning: **62.9%** ACC / 59.1% F1
- Train EEGMAT, tune on STEW-VA, test STEW: **70.1%** ACC / 69.1% F1

Our gap vs this paper is the **adapter**, not the datasets.

### Other related work (do not over-cite)

- SDDA — cross-headset spatial distillation (different datasets, MI/P300): https://arxiv.org/pdf/2503.05349
- Euclidean Alignment (MOABB) — unlabeled covariance whitening
- Tent (ICLR 2021) — entropy-min TTA; often **hurts** EEG (NeuroAdapt-Bench 2026)
- Wu et al. 2017 — “Switching EEG Headsets Made Easy” (BioSemi / Emotiv / ABM)

---

## 10. Implementation notes (24-hour cut order)

**Must ship**

1. Alignment pipeline: P3 re-reference, 10-channel lock, resample, 1–40 Hz, 2 s epochs, shared rest/load label.
2. Train frozen classifier on EEGMAT only.
3. Zero-shot STEW number (pipeline A) — record the collapse.
4. Ablation B–E; freeze the winner.
5. Dashboard: two-column I/O + before/after + drop/recovery numbers.
6. Scalp sim: both montages + degrade slider.

**Cut first if short on time**

- Tent / any gradient TTA
- Deep model
- UI polish
- Extra headsets beyond these two

**Do not**

- Train on STEW and test on EEGMAT (that is cheap → expensive, the opposite of the pitch).
- Pretend we recorded both devices ourselves. Public datasets are the substitute; say that out loud.
- Live-retrain during judging. Replay saved, verified results.

**Stack (suggested)**

- Data: `mne`, `wfdb`, `numpy`, `scipy`
- Model: `scikit-learn` (band-power + logistic regression / SVM)
- Alignment: custom z-score + Euclidean Alignment
- Dashboard: Streamlit + Plotly
- Backup: saved `.npz` predictions + a short screen recording of one clean run

---

## 11. Review checklist

- [ ] Train only on Neurocom / EEGMAT
- [ ] Test only on Emotiv / STEW
- [ ] Shared 10 channels named on the slide
- [ ] P3 re-reference explained as CMS match
- [ ] Ablation table with real numbers (accuracy + F1 or kappa)
- [ ] Confusion matrices checked (recovery is not majority-class collapse)
- [ ] Dashboard before/after on the same window
- [ ] Scalp sim degrades clinical signal and the model confidence falls
- [ ] SCVCNet ~63% cited as ghost baseline, not ignored
- [ ] Honest remaining-error line (task mismatch + missing midline sites)
- [ ] India access pitch (neurologist shortage / hospital EEG vs wearable), not “first EEG AI”
- [ ] Price ladder in INR (hospital cart vs RMS vs Emotiv vs Muse)
- [ ] Explicit: not epilepsy diagnosis, not CDSCO device
- [ ] Neurocom/Emotiv named as public-data proxies for RMS-class vs wearable
