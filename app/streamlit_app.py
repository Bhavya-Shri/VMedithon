"""GAP-Align dashboard. Reads artifacts only — never trains."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as st_components
from scipy.signal import welch

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


def _minmax(d: dict) -> dict:
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in d.items()}


def _windows_for_y(windows, y: int) -> list:
    return [w for w in windows if int(w.get("y", -1)) == int(y)]


def _channel_activity(windows, feat_key: str, y: int, names: list[str]) -> dict:
    rows = [np.asarray(w[feat_key], dtype=float) for w in _windows_for_y(windows, y) if feat_key in w]
    if not rows:
        return {}
    mean = np.mean(np.stack(rows, axis=0), axis=0)
    per_ch: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        ch, band = name.split("_", 1)
        per_ch.setdefault(ch, {})[band] = float(mean[i])
    raw = {}
    for ch, bands in per_ch.items():
        raw[ch] = bands.get("theta", 0.0) + bands.get("beta", 0.0) - bands.get("alpha", 0.0)
    return _minmax(raw)


def _psd_curve(windows, y: int, sfreq: float = 128.0) -> list:
    xs = [np.asarray(w["x"], dtype=float) for w in _windows_for_y(windows, y) if "x" in w]
    if not xs:
        return []
    X = np.stack(xs, axis=0)
    nperseg = min(X.shape[-1], 256)
    freqs, psd = welch(X, fs=sfreq, nperseg=nperseg, axis=-1)
    mean_p = psd.mean(axis=(0, 1))
    return [[float(f), float(p)] for f, p in zip(freqs, mean_p) if 1.0 <= f <= 45.0]


def _connectivity(windows, y: int, ch_names: list[str]) -> list:
    xs = [np.asarray(w["x"], dtype=float) for w in _windows_for_y(windows, y) if "x" in w]
    if not xs:
        return []
    avg = np.mean([np.corrcoef(x) for x in xs], axis=0)
    edges = []
    n = min(len(ch_names), avg.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            w = abs(float(avg[i, j]))
            if w > 0.3:
                edges.append([ch_names[i], ch_names[j], round(w, 3)])
    return edges


def hardware_viewer_payload(demo, meta, ablation) -> dict:
    """Feed the 3D montage from GAP-Align demo windows. No CORAL / no reverse-direction SVM."""
    names = list(demo.get("feature_names") or feature_names())
    clinical = demo["clinical"]
    wearable = demo["wearable"]
    winner = ablation.get("winner", "C")
    n_eegmat = int(meta.get("n_train_subjects") or 36)
    n_stew = 48 if meta.get("target_source") == "stew" else n_eegmat
    return {
        "n_subjects": {"eegmat": n_eegmat, "stew": n_stew},
        "overlap_channels": list(SHARED_CH),
        "activity": {
            "eegmat": {
                "rest": _channel_activity(clinical, "features", 0, names),
                "task": _channel_activity(clinical, "features", 1, names),
            },
            "stew": {
                "rest": _channel_activity(wearable, "features_before", 0, names),
                "task": _channel_activity(wearable, "features_before", 1, names),
            },
        },
        "psd": {
            "eegmat": {"rest": _psd_curve(clinical, 0), "task": _psd_curve(clinical, 1)},
            "stew": {"rest": _psd_curve(wearable, 0), "task": _psd_curve(wearable, 1)},
        },
        "connectivity": {
            "eegmat": {"rest": _connectivity(clinical, 0, SHARED_CH), "task": _connectivity(clinical, 1, SHARED_CH)},
            "stew": {"rest": _connectivity(wearable, 0, SHARED_CH), "task": _connectivity(wearable, 1, SHARED_CH)},
        },
        "metrics": {
            "accuracy_before": round(float(ablation["A"]["acc"]) * 100, 1),
            "accuracy_after": round(float(ablation[winner]["acc"]) * 100, 1),
        },
    }


def _pred_block(title: str, pred: int, proba, caption: str, true_y: int | None = None) -> None:
    p_load = float(proba[1])
    st.markdown(f"<div class='gap-kicker'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='gap-pred'>{_label(pred)}</div>", unsafe_allow_html=True)
    st.progress(min(max(p_load, 0.0), 1.0), text=f"P(cognitive load) = {p_load:.2f}")
    if true_y is not None:
        ok = int(pred) == int(true_y)
        st.caption(("Correct" if ok else "Wrong") + f" vs true {_label(true_y)}")
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

    n_pairs = min(len(wearable), len(clinical))
    if n_pairs < 1:
        st.error("demo_windows.json has no paired clinical/wearable snapshots.")
        return

    def _pair_label(i: int) -> str:
        ww = wearable[i]
        cc = clinical[i % len(clinical)]
        return (
            f"Slot {i + 1}/{n_pairs} · STEW {ww['subject']} {_label(ww['y'])}  ·  "
            f"EEGMAT {cc['subject']} {_label(cc['y'])}"
        )

    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([3, 2, 1])
    with c_ctrl3:
        play = st.toggle("Play", value=st.session_state.get("playing", False))
        st.session_state.playing = play
    if "w_idx" not in st.session_state:
        st.session_state.w_idx = 0
    st.session_state.w_idx = int(st.session_state.w_idx) % n_pairs

    if play:
        pack_i = st.session_state.w_idx % n_pairs
        with c_ctrl1:
            st.caption(f"Streaming {_pair_label(pack_i)}")
        with c_ctrl2:
            st.caption("EEGMAT and STEW are different people; slots are rest-with-rest, load-with-load.")
    else:
        with c_ctrl1:
            pack_i = st.selectbox(
                "Demo pair (updates BOTH columns)",
                list(range(n_pairs)),
                index=st.session_state.w_idx,
                format_func=_pair_label,
            )
            st.session_state.w_idx = pack_i
        with c_ctrl2:
            st.caption("Not the same skull. Pack slots 1–5 rest, 6–10 load on both sides.")

    w = wearable[pack_i]
    c = clinical[pack_i]
    mode = st.radio("Wearable view", ["Before GAP-Align", "After GAP-Align"], horizontal=True)
    show_after = mode.startswith("After")
    chart_key = f"p{pack_i}_{'a' if show_after else 'b'}"
    ok_b = int(w["pred_before"]) == int(w["y"])
    ok_a = int(w["pred_after"]) == int(w["y"])
    st.info(
        "GAP-Align does **not** copy the EEGMAT traces. After should match the **STEW eval label** "
        "(right column), not the hospital person on the left. Before often *looks* like clinical "
        "`P(load)` because both collapsed to load."
    )

    left, center, right = st.columns([1.15, 0.85, 1.15])
    with left:
        st.subheader("Clinical · Neurocom / EEGMAT")
        st.plotly_chart(
            plot_traces(c["x"], SHARED_CH, title=f"EEGMAT sub {c['subject']} · {_label(c['y'])} · 2 s"),
            use_container_width=True,
            key=f"clin_tr_{chart_key}",
        )
        st.plotly_chart(
            plot_bands(c["features"], names, title="Clinical log band-power (sensor µV · log10 PSD)"),
            use_container_width=True,
            key=f"clin_bp_{chart_key}",
        )
        _pred_block(
            "Frozen model",
            c["pred"],
            c["proba"],
            "Different person, hospital device. This is not the target After should copy.",
            true_y=c["y"],
        )
        st.caption(f"EEGMAT subject {c['subject']} · true label: {_label(c['y'])}")

    with right:
        st.subheader("Commercial · Emotiv / wearable")
        st.plotly_chart(
            plot_traces(w["x"], SHARED_CH, title=f"STEW sub {w['subject']} · {_label(w['y'])} · 2 s"),
            use_container_width=True,
            key=f"wear_tr_{chart_key}",
        )
        feats = w["features_after"] if show_after else w["features_before"]
        st.plotly_chart(
            plot_bands(
                feats,
                names,
                title="After: log-power of z-scored EEG (not the clinical scale)"
                if show_after
                else "Before: raw log-power (same units as clinical, often still wrong)",
            ),
            use_container_width=True,
            key=f"wear_bp_{chart_key}",
        )
        m1, m2 = st.columns(2)
        m1.metric(
            "P(load) before",
            f"{float(w['proba_before'][1]):.2f}",
            delta="correct" if ok_b else "wrong vs STEW",
            delta_color="normal" if ok_b else "inverse",
        )
        m2.metric(
            "P(load) after",
            f"{float(w['proba_after'][1]):.2f}",
            delta="correct" if ok_a else "wrong vs STEW",
            delta_color="normal" if ok_a else "inverse",
        )
        if show_after:
            _pred_block(
                "After GAP-Align",
                w["pred_after"],
                w["proba_after"],
                "Same frozen weights. Success = this matches STEW rest/load, not EEGMAT P(load).",
                true_y=w["y"],
            )
        else:
            _pred_block(
                "Before GAP-Align",
                w["pred_before"],
                w["proba_before"],
                "Naive port. Rest windows often look like load — that is the collapse.",
                true_y=w["y"],
            )
        st.caption(
            f"STEW subject {w['subject']} · eval {_label(w['y'])} (never used to fit) · "
            f"before {'correct' if ok_b else 'wrong'} → after {'correct' if ok_a else 'wrong'}"
        )

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
        st.caption("True labels on the wearable score this window, not the clinical traces. Different people, same rest/load slot.")
        st.caption(
            f"This STEW clip: before {'correct' if ok_b else 'wrong'} → after {'correct' if ok_a else 'wrong'} "
            f"vs {_label(w['y'])}. Official table above is n=10265, not this window."
        )


def page_sim(demo, scaler, clf, meta, ablation) -> None:
    st.subheader("Hardware simulation")
    st.write(
        "The cheap headset is not a smaller clinical headset. Toggle **EEGMAT · 19ch** vs **STEW · 14ch**. "
        "Shared 10–20 sites colour by band-power; grey nodes are dropped from the frozen model; gold is P3 (CMS)."
    )
    st.caption(
        "3D montage is the teammate hardware viewer (schematic 10–20, not a digitized head). "
        "It is **not** the official transfer score. Do not quote CORAL/SVM from `main.py` — "
        "that script trains STEW→EEGMAT, which is the wrong direction."
    )
    html_path = ROOT / "app" / "hardware_viewer.html"
    html = html_path.read_text(encoding="utf-8")
    payload = hardware_viewer_payload(demo, meta, ablation)
    html = html.replace("/*__GAP_ALIGN_REAL__*/", f"window.GAP_ALIGN_REAL = {json.dumps(payload)};")
    st_components.html(html, height=780, scrolling=True)

    with st.expander("2D colour key (Guide Page 2)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_scalp("neurocom"), use_container_width=True)
            st.caption("Original linked-ear reference (A1+A2). We subtract P3 to imitate CMS, then drop P3.")
        with c2:
            st.plotly_chart(plot_scalp("epoc"), use_container_width=True)
            st.caption("AF3 / AF4 / FC5 / FC6 are on the headset but not in the model. DRL at P4.")

    st.markdown("**Degrade slider — this is the model-linked hardware sim**")
    st.caption("Same frozen scaler/clf as Page 1. Slider uses `slider_degrade` / `fake_emotiv`.")

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
    st.caption("These charts are the frozen official scores. They do not move while you click around Page 1.")
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
        ],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.write("Direction is non-negotiable: **EEGMAT → STEW**. Classifier never sees wearable labels.")
    st.sidebar.caption("Not a diagnostic EEG. Not CDSCO-cleared. Cognitive-load screening research tool.")
    st.sidebar.caption("Public de-identified research sets only. No patient data.")

    if demo is None or meta is None or ablation is None:
        _missing_box()
    if page.startswith("1"):
        page_live(demo, meta, ablation)
    elif page.startswith("2"):
        page_sim(demo, scaler, clf, meta, ablation)
    else:
        page_ablation(ablation, meta)

    if st.session_state.get("playing") and page.startswith("1"):
        time.sleep(0.4)
        n = min(len(demo["wearable"]), len(demo["clinical"]))
        st.session_state.w_idx = (st.session_state.get("w_idx", 0) + 1) % max(n, 1)
        st.rerun()


if __name__ == "__main__":
    main()
