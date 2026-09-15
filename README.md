# GAP-Align

Hospital EEG knowledge is locked in cities. GAP-Align is the calibration layer that lets a model trained on a clinical headset run on an affordable Indian-deployable wearable -- and we show the failure and the recovery live.

**Direction (non-negotiable):** train on EEGMAT (Neurocom, clinical) -> test on STEW (Emotiv EPOC, commercial). Rest vs cognitive load. No STEW labels are used to fit the classifier or the adapter.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Public de-identified research sets only. No patient data.

- EEGMAT (PhysioNet): https://physionet.org/content/eegmat/1.0.0/
- STEW (IEEE DataPort): https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset
  DOI: `10.21227/44r8-ya50`

Place EDFs in `data/raw/eegmat/` and STEW txt files in `data/raw/stew/`.

## Run

```powershell
python -m src.run_all
streamlit run app/streamlit_app.py
```

Not a diagnostic EEG. Not CDSCO-cleared. Cognitive-load screening research tool. Neurocom/Emotiv are open-data proxies for RMS-class hospital EEG vs a wearable a campus can buy.
