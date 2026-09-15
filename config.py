"""GAP-Align constants. Import from here; do not hard-code these values elsewhere.

Shapes we will produce later (not created in this file):
  epochs X: (N, 10, 256) float32, microvolts, 128 Hz
  labels y: (N,) int8
  features: (N, 30)  = 10 channels x 3 bands
"""
from pathlib import Path

# Path(__file__) is this file. .resolve().parent is the repo root even if
# we launch Python from another working directory.
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"

EEGMAT_DIR = DATA_RAW / "eegmat"
STEW_DIR = DATA_RAW / "stew"

# Guide §3 ---------------------------------------------------------------
SFREQ_SRC = 500.0          # expected EEGMAT; always overwrite from EDF header
SFREQ_TGT = 128.0          # Emotiv EPOC native rate
BANDPASS = (1.0, 40.0)     # Hz, both datasets
NOTCH = 50.0               # EEGMAT is Ukraine/EU line noise; STEW may be 50 or 60
EPOCH_SEC = 2.0
EPOCH_STRIDE_SEC = 1.0     # 50% overlap
RANDOM_SEED = 42

# Canonical 10 shared 10-20 sites (Emotiv names). Order is part of the contract.
SHARED_CH = ["F3", "F4", "F7", "F8", "T7", "T8", "P7", "P8", "O1", "O2"]

# Old 10-20 aliases (clinical EEGMAT) -> Emotiv / modern 10-20 (Guide §4).
CH_ALIAS = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Clinical-only site we re-reference TO, then DROP (Emotiv CMS lives at P3).
CLINICAL_REF = "P3"

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
}

# Train-time fake-Emotiv noise (pipeline E only)
NOISE_P = 0.5
NOISE_STD_UV = 5.0         # start here; tune only if source CV collapses
QUANTIZE_UV = 0.51         # Emotiv 14-bit LSB

# Unlabeled STEW used to estimate z-score / EA (no labels in that fit)
ADAPTER_SECONDS = 30.0

# Derived / named numbers from later Guide sections (still constants only) --
N_CH = len(SHARED_CH)                          # 10
N_TIMES = int(SFREQ_TGT * EPOCH_SEC)           # 256 samples in a 2 s window at 128 Hz
N_BANDS = len(BANDS)                           # 3
N_FEATURES = N_CH * N_BANDS                    # 30

# STEW txt column order as documented (Guide §4). Classifier keeps SHARED_CH only.
STEW_CH_FILE = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

# Peak-to-peak artifact reject (Guide §6.2 step 13), microvolts
PTP_REJECT_UV = 200.0
PTP_LOOSEN_UV = 300.0
PTP_DROP_FRAC = 0.40

# Euclidean Alignment (Guide §9.2)
EA_SHRINK = 1e-6
EA_MIN_EPOCHS = 5

# Welch + log band-power (Guide §8 / §21)
WELCH_NPERSEG = N_TIMES    # one window = the whole 2 s epoch
LOG_BP_EPS = 1e-12
ZSCORE_EPS = 1e-8

# Frozen logistic regression (Guide §21). Chosen on EEGMAT LOSO, never on STEW.
LR_SOLVER = "lbfgs"
LR_CLASS_WEIGHT = "balanced"
LR_MAX_ITER = 2000
LR_C = 1.0

# EEGMAT LOSO gates (Guide §11.1 and §14)
LOSO_PASS = 0.70
LOSO_WARN = 0.65
LOSO_EXIT = 0.55
