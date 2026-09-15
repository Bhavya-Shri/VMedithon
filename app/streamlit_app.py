"""GAP-Align dashboard. Reads artifacts only — never trains."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from config import ART, RANDOM_SEED, SHARED_CH
from components import (  # noqa: E402  (app/ is on path when Streamlit runs this file)
    plot_ablation,
    plot_bands,
    plot_cm,
    plot_scalp,
    plot_subject_delta,
    plot_traces,
)
from src.degrade import slider_degrade
from src.features import feature_names, transform_epochs
from src.train import predict_features

st.set_page_config(
    page_title="GAP-Align · clinical EEG → wearable",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #0b1220; color: #e8eef7; }
      [data-testid="stSidebar"] { background: #10192b; }
      .gap-kicker { letter-spacing: 0.14em; text-transform: uppercase; color: #e4b84a; font-size: 0.78rem; }
      .gap-pred { font-size: 1.6rem; font-weight: 700; margin: 0.2rem 0; }
      .gap-muted { color: #9bb0c9; font-size: 0.92rem; }
      .gap-badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px;
                   background: #14302c; color: #3dd6c6; font-size: 0.82rem; }
      .gap-warn { background: #3a2208; color: #fbbf24; padding: 0.7rem 1rem; border-radius: 8px; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _missing_box() -> None:
    st.error("Run `python -m src.run_all` first. The dashboard reads frozen artifacts; it does not train.")
    st.stop()


@st.cache_data
def load_json(name: str):
    path = ART / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_resource
def load_models():
    import joblib

    scaler_p, clf_p = ART / "scaler.joblib", ART / "clf.joblib"
    if not scaler_p.exists() or not clf_p.exists():
        return None, None
    return joblib.load(scaler_p), joblib.load(clf_p)


@st.cache_data
def load_npz(name: str):
    path = ART / name
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=False))


def _label(y: int) -> str:
    return "Rest" if int(y) == 0 else "Cognitive load"


def _pred_block(title: str, pred: int, proba, caption: str) -> None:
    p_load = float(proba[1])
    st.markdown(f"<div class='gap-kicker'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='gap-pred'>{_label(pred)}</div>", unsafe_allow_html=True)
    st.progress(min(max(p_load, 0.0), 1.0), text=f"P(cognitive load) = {p_load:.2f}")
    st.caption(caption)


def page_live(demo, meta, ablation) -> None:
    clinical = demo["clinical"]
    wearable = demo["wearable"]
    names = demo.get("feature_names") or feature_names()
    proxy = meta.get("target_source") != "stew"

    if proxy:
        st.markdown(
            "<div class='gap-warn'>Wearable side is an Emotiv-like <b>proxy</b> "
            "(degraded EEGMAT). Not STEW. Drop IEEE DataPort files into "
            "<code>data/raw/stew/</code> and re-run <code>python -m src.run_all</code> "
            "before quoting a STEW number.</div>",
            unsafe_allow_html=True,
        )

    subjects = sorted({w["subject"] for w in wearable})
    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([2, 2, 1])
    with c_ctrl1:
        sub = st.selectbox("Wearable subject", subjects, index=0)
    windows = [w for w in wearable if w["subject"] == sub] or wearable
    n_win = max(len(windows), 1)
    if "w_idx" not in st.session_state:
        st.session_state.w_idx = 0
    st.session_state.w_idx = min(st.session_state.w_idx, n_win - 1)
    with c_ctrl3:
        play = st.toggle("Play", value=st.session_state.get("playing", False))
        st.session_state.playing = play
    with c_ctrl2:
        if play:
            st.caption(f"Streaming window {st.session_state.w_idx + 1} / {n_win}")
            idx = st.session_state.w_idx
        else:
            idx = st.slider("Window", 0, n_win - 1, st.session_state.w_idx)
            st.session_state.w_idx = idx

    w = windows[idx]
    c = clinical[idx % len(clinical)]
    mode = st.radio("Wearable view", ["Before GAP-Align", "After GAP-Align"], horizontal=True)

    left, center, right = st.columns([1.15, 0.85, 1.15])
    with left:
        st.subheader("Clinical · Neurocom / EEGMAT")
        st.plotly_chart(plot_traces(c["x"], SHARED_CH, title="2 s · 10 shared sites · 128 Hz"), use_container_width=True)
        st.plotly_chart(plot_bands(c["features"], names), use_container_width=True)
        _pred_block("Frozen model", c["pred"], c["proba"], "Hospital-grade stand-in · 19–23 ch wet · 500 Hz → aligned 10 ch 128 Hz")
        st.caption(f"Subject {c['subject']} · true label: {_label(c['y'])}")

    with right:
        st.subheader("Commercial · Emotiv / wearable")
        st.plotly_chart(plot_traces(w["x"], SHARED_CH, title="2 s · same montage"), use_container_width=True)
        feats = w["features_after"] if mode.startswith("After") else w["features_before"]
        st.plotly_chart(plot_bands(feats, names, title="Band-power at this window"), use_container_width=True)
        if mode.startswith("After"):
            _pred_block("After GAP-Align", w["pred_after"], w["proba_after"], "Wearable stand-in · 14 ch saline · native 128 Hz")
        else:
            _pred_block("Before GAP-Align", w["pred_before"], w["proba_before"], "Same frozen weights. No new labels.")
        st.caption(f"Subject {w['subject']} · eval label: {_label(w['y'])} (never used to fit)")

    with center:
        st.markdown("<div class='gap-badge'>Unlabeled adaptation · no STEW labels used to train the classifier</div>", unsafe_allow_html=True)
        st.metric("EEGMAT LOSO accuracy", f"{meta.get('loso_acc', 0):.1%}")
        before = ablation["A"]
        after = ablation[ablation["winner"]]
        st.metric("Wearable accuracy before (A)", f"{before['acc']:.1%}")
        st.metric("Wearable accuracy after", f"{after['acc']:.1%}", delta=f"{(after['acc']-before['acc'])*100:.1f} pp")
        st.metric("Kappa before / after", f"{before['kappa']:.2f}  →  {after['kappa']:.2f}")
        st.metric("Macro-F1 after", f"{after['f1']:.3f}")
        st.caption(f"Winner pipeline: {ablation['winner']}. Ghost baseline SCVCNet EEGMAT→STEW = 62.9% (real STEW only).")
        st.caption("True labels on the wearable are for scoring the dashboard, not for fitting z-score, EA, or the classifier.")


def page_sim(demo, scaler, clf) -> None:
    st.subheader("Hardware simulation")
    st.write(
        "The cheap headset is not a smaller clinical headset. Green sites are the 10 channels the model sees. "
        "Gold is P3 — Emotiv’s CMS. Grey sites are blind to the wearable model."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_scalp("neurocom"), use_container_width=True)
        st.caption("Original linked-ear reference (A1+A2). We subtract P3 to imitate CMS, then drop P3.")
    with c2:
        st.plotly_chart(plot_scalp("epoc"), use_container_width=True)
        st.caption("AF3 / AF4 / FC5 / FC6 are on the headset but not in the model. DRL at P4.")

    if scaler is None or clf is None:
        st.warning("Frozen model artifacts missing — slider will not move P(load).")
        return

    clinical = demo["clinical"]
    idx = st.slider("Clinical window", 0, len(clinical) - 1, 0)
    amount = st.slider("Degrade toward fake EPOC", 0.0, 1.0, 0.0, 0.01)
    epoch = np.asarray(clinical[idx]["x"], dtype=np.float32)
    rng = np.random.default_rng(RANDOM_SEED + idx)
    degraded = slider_degrade(epoch, amount, rng)
    _, proba0 = predict_features(transform_epochs(epoch[None, ...]), scaler, clf)
    _, proba1 = predict_features(transform_epochs(degraded[None, ...]), scaler, clf)

    d1, d2, d3 = st.columns(3)
    d1.metric("P(load) clinical", f"{proba0[0, 1]:.2f}")
    d2.metric("P(load) after degrade", f"{proba1[0, 1]:.2f}", delta=f"{(proba1[0,1]-proba0[0,1]):+.2f}")
    d3.metric("Slider", f"{amount:.2f}")
    st.caption("0–0.25 identity · 0.25–0.50 noise · 0.50–0.75 14-bit quantize · 0.75–1 extra noise + 4–8 Hz jitter")
    st.plotly_chart(plot_traces(degraded, SHARED_CH, title="Degraded clinical epoch"), use_container_width=True)
    st.write(
        "As SNR and bit-depth fall, the frozen hospital model should drift toward 0.5. "
        "If it does not, the sim is lying — the slider uses the same `fake_emotiv` path as training."
    )


def page_ablation(ablation, meta) -> None:
    st.subheader("Ablation & trust")
    st.plotly_chart(plot_ablation(ablation), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_cm(ablation["A"]["cm"], "Before — pipeline A"), use_container_width=True)
    with c2:
        winner = ablation["winner"]
        st.plotly_chart(plot_cm(ablation[winner]["cm"], f"After — pipeline {winner}"), use_container_width=True)
    st.info("Recovery must improve both classes. If F1 is far below accuracy, the adapter collapsed to the majority class.")
    if ablation.get("target_source") != "stew":
        st.warning("Target is not STEW. Do not put these numbers on a slide as EEGMAT→STEW.")
    else:
        st.caption("Ghost baseline: SCVCNet train EEGMAT / test STEW, no target labels ≈ 62.9% accuracy.")

    before = load_npz("stew_preds_before.npz")
    after = load_npz("stew_preds_after.npz")
    if before and after:
        st.plotly_chart(plot_subject_delta(before, after), use_container_width=True)

    st.markdown(
        """
        | ID | What it proves |
        |---|---|
        | A | Naive port after channel / rate / filter lock |
        | C | Unlabeled per-channel z-score (gain / impedance) |
        | D | Z-score + Euclidean Alignment (spatial mixing) |
        | E | Train-time fake-Emotiv noise (SNR gap) |
        | F | Tent — skipped; needs BatchNorm, we use a linear model |
        """
    )


def page_india() -> None:
    st.subheader("Access, not a new EEG theory")
    st.write(
        "India cannot put a neurologist and an ₹8–40 lakh EEG cart in every district. "
        "Fewer than 2,500 neurologists serve 1.4 billion people. Models get trained where the "
        "machines and the labels live. GAP-Align is the calibration layer that lets a hospital-trained "
        "model run on a headset a campus or clinic can actually buy."
    )
    st.markdown(
        """
        | Tier | Typical India kit | Ballpark |
        |---|---|---|
        | Imported hospital EEG | Nihon Kohden / Natus / Compumedics | ₹8–40 lakh+ |
        | Indian clinical portable | RMS Maximus, Medicaid Neuromax | ₹1–5 lakh |
        | Research wearable | Emotiv EPOC X 14-ch saline | ~₹70k–1.5 lakh |
        | Wellness band | Muse / NeuroSky-class | ₹20–40k |
        """
    )
    st.write(
        "Neurocom vs Emotiv are **open-data proxies** for RMS-class hospital EEG vs a wearable India can deploy. "
        "We did not record on RMS hardware in an Indian hospital."
    )
    st.error("Not a diagnostic EEG. Not CDSCO-cleared. Cognitive-load screening research tool only.")
    st.write(
        "Out of scope: seizure detection, stroke, dementia, replacing a neurologist, Ayushman billing. "
        "Next validation is paired RMS + EPOC in one Indian lab, then a research Python module for college labs."
    )


def main() -> None:
    st.markdown("<div class='gap-kicker'>GAP-Align · Geometry And Physics Alignment</div>", unsafe_allow_html=True)
    st.title("Hospital EEG knowledge, unlocked for a wearable")
    st.caption("Train on EEGMAT (Neurocom). Test on STEW (Emotiv EPOC). Rest vs cognitive load. Zero target labels for fitting.")

    demo = load_json("demo_windows.json")
    meta = load_json("train_meta.json")
    ablation = load_json("ablation.json")
    scaler, clf = load_models()

    page = st.sidebar.radio(
        "Pages",
        [
            "1 · Live compare",
            "2 · Hardware simulation",
            "3 · Ablation & trust",
            "4 · India / access",
        ],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.write("Direction is non-negotiable: **EEGMAT → STEW**. Classifier never sees wearable labels.")
    st.sidebar.caption("Public de-identified research sets only. No patient data.")

    if page.startswith("4"):
        page_india()
        return
    if demo is None or meta is None or ablation is None:
        _missing_box()
    if page.startswith("1"):
        page_live(demo, meta, ablation)
    elif page.startswith("2"):
        page_sim(demo, scaler, clf)
    else:
        page_ablation(ablation, meta)

    if st.session_state.get("playing") and page.startswith("1"):
        time.sleep(0.4)
        n = max(len(demo["wearable"]), 1)
        st.session_state.w_idx = (st.session_state.get("w_idx", 0) + 1) % n
        st.rerun()


if __name__ == "__main__":
    main()
