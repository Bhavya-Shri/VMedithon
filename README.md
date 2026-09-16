# GAP-Align

Hospital EEG knowledge is locked in cities. GAP-Align is the calibration layer that lets a model trained on a clinical headset run on an affordable Indian-deployable wearable — and we show the failure and the recovery live.

**Direction (non-negotiable):** train on EEGMAT (Neurocom, clinical) → test on STEW (Emotiv EPOC, commercial). Rest vs cognitive load. No STEW labels are used to fit the classifier or the adapter.

Team fork (push target): https://github.com/haripriyasubbiah/VMedithon  
Upstream (do not push): https://github.com/Bhavya-Shri/VMedithon

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Public de-identified research sets only. No patient data.

- EEGMAT (PhysioNet): https://physionet.org/content/eegmat/1.0.0/
- STEW (IEEE DataPort): https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset
  DOI: `10.21227/44r8-ya50`

Place EDFs in `data/raw/eegmat/` and STEW txt files in `data/raw/stew/`.

`python -m src.run_all` will try to download EEGMAT from PhysioNet if EDFs are missing. STEW needs an IEEE DataPort login — if those files are absent, the runner builds an **Emotiv-like proxy** from degraded EEGMAT so the dashboard still runs. That proxy is **not STEW**. Do not quote it as an EEGMAT→STEW number.

## Run

```bash
python -m src.run_all
streamlit run app/streamlit_app.py --server.port 8765
```

Three dashboard pages. Full page-by-page explanation: [`DASHBOARD.md`](DASHBOARD.md).

1. Live compare — clinical window vs wearable, before/after GAP-Align
2. Hardware simulation — teammate 3D montage + degrade slider on the frozen model
3. Ablation & trust — pipelines A/C/D/E, confusion matrices

Not a diagnostic EEG. Not CDSCO-cleared. Cognitive-load screening research tool. Neurocom/Emotiv are open-data proxies for RMS-class hospital EEG vs a wearable a campus can buy.
