"""Plotly building blocks for the Streamlit dashboard. No training here."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from config import SHARED_CH

# Approximate 2D 10-20, nose up, x right (Guide §16).
POS = {
    "Fp1": (-0.30, 0.90), "Fp2": (0.30, 0.90),
    "AF3": (-0.35, 0.75), "AF4": (0.35, 0.75),
    "F7": (-0.70, 0.55), "F3": (-0.35, 0.55), "Fz": (0.00, 0.55), "F4": (0.35, 0.55), "F8": (0.70, 0.55),
    "FC5": (-0.55, 0.35), "FC6": (0.55, 0.35),
    "T7": (-0.90, 0.00), "C3": (-0.40, 0.00), "Cz": (0.00, 0.00), "C4": (0.40, 0.00), "T8": (0.90, 0.00),
    "P7": (-0.70, -0.55), "P3": (-0.35, -0.55), "Pz": (0.00, -0.55), "P4": (0.35, -0.55), "P8": (0.70, -0.55),
    "O1": (-0.30, -0.90), "O2": (0.30, -0.90),
}

NEUROCOM_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]
EPOC_14 = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
EPOC_UNUSED = {"AF3", "AF4", "FC5", "FC6"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,18,32,0.6)",
    font=dict(color="#d7e2f0", family="IBM Plex Sans, Source Sans 3, sans-serif", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def plot_traces(x, ch=None, sfreq: float = 128.0, title: str = "") -> go.Figure:
    ch = list(ch or SHARED_CH)
    x = np.asarray(x, dtype=float)
    t = np.arange(x.shape[-1]) / sfreq
    offset = max(np.ptp(x), 1.0) * 0.55
    fig = go.Figure()
    for i, name in enumerate(ch):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=x[i] + (len(ch) - 1 - i) * offset,
                mode="lines",
                name=name,
                line=dict(width=1.2),
                hovertemplate=f"{name}<br>t=%{{x:.2f}}s<extra></extra>",
            )
        )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        height=420,
        xaxis_title="Time (s)",
        yaxis=dict(showticklabels=False, title="Channels (offset µV)"),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")
    return fig


def plot_bands(features, names, title: str = "θ / α / β log band-power") -> go.Figure:
    features = np.asarray(features, dtype=float)
    bands = ["theta", "alpha", "beta"]
    colors = {"theta": "#7db7ff", "alpha": "#3dd6c6", "beta": "#e4b84a"}
    fig = go.Figure()
    for band in bands:
        idx = [i for i, n in enumerate(names) if n.endswith("_" + band)]
        fig.add_trace(
            go.Bar(
                x=[names[i].split("_")[0] for i in idx],
                y=[features[i] for i in idx],
                name=band,
                marker_color=colors[band],
            )
        )
    layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "legend"}
    fig.update_layout(
        **layout,
        title=title,
        barmode="group",
        height=260,
        yaxis_title="log10 PSD",
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12),
    )
    return fig


def _head_shape(fig: go.Figure) -> None:
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            line=dict(color="#8aa0b8", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0, -0.08, 0.08, 0.0],
            y=[1.12, 1.0, 1.0, 1.12],
            fill="toself",
            mode="lines",
            line=dict(color="#8aa0b8", width=1),
            fillcolor="#8aa0b8",
            hoverinfo="skip",
            showlegend=False,
        )
    )


def plot_scalp(kind: str = "neurocom") -> go.Figure:
    fig = go.Figure()
    _head_shape(fig)
    if kind == "neurocom":
        sites = NEUROCOM_19
        title = "Neurocom 19-ch · linked-ear ref"
    else:
        sites = EPOC_14
        title = "Emotiv EPOC 14-ch · CMS at P3"

    xs, ys, colors, sizes, texts, labels = [], [], [], [], [], []
    for name in sites:
        x, y = POS[name]
        xs.append(x)
        ys.append(y)
        labels.append(name)
        if name == "P3":
            colors.append("#e4b84a")
            sizes.append(18)
            texts.append("CMS match: we re-reference here" if kind == "neurocom" else "CMS (hardware ref)")
        elif name in SHARED_CH:
            colors.append("#3dd6c6")
            sizes.append(14)
            texts.append("shared site — kept")
        elif kind == "epoc" and name in EPOC_UNUSED:
            colors.append("#6b7280")
            sizes.append(11)
            texts.append("not in the model")
        else:
            colors.append("#4b5563")
            sizes.append(11)
            texts.append("clinical-only — dropped")

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=sizes, color=colors, line=dict(color="#0b1220", width=1)),
            customdata=texts,
            hovertemplate="%{text}<br>%{customdata}<extra></extra>",
            showlegend=False,
        )
    )
    if kind == "epoc":
        px, py = POS["P4"]
        fig.add_trace(
            go.Scatter(
                x=[px],
                y=[py],
                mode="markers+text",
                text=["P4"],
                textposition="top center",
                marker=dict(size=14, color="#fb7185", symbol="diamond"),
                hovertemplate="DRL at P4<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        height=420,
        xaxis=dict(visible=False, range=[-1.25, 1.25]),
        yaxis=dict(visible=False, range=[-1.25, 1.25], scaleanchor="x", scaleratio=1),
    )
    return fig


def plot_ablation(table: dict) -> go.Figure:
    keys = [k for k in ("A", "C", "D", "E") if k in table and "acc" in table[k]]
    fig = go.Figure()
    for metric, color in (("acc", "#7db7ff"), ("f1", "#3dd6c6"), ("kappa", "#e4b84a")):
        fig.add_trace(
            go.Bar(
                x=keys,
                y=[table[k][metric] for k in keys],
                name=metric,
                marker_color=color,
            )
        )
    fig.add_hline(y=0.629, line_dash="dot", line_color="#fb7185", annotation_text="SCVCNet 62.9%")
    fig.add_hline(y=0.5, line_dash="dash", line_color="#6b7280", annotation_text="chance")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Ablation on the wearable target (frozen EEGMAT model)",
        barmode="group",
        height=380,
        yaxis=dict(range=[0, 1], title="Score"),
    )
    return fig


def plot_cm(cm, title: str) -> go.Figure:
    z = np.asarray(cm)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Pred rest", "Pred load"],
            y=["True rest", "True load"],
            colorscale="Teal",
            text=z,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=320, yaxis=dict(autorange="reversed"))
    return fig


def plot_subject_delta(before_npz, after_npz) -> go.Figure | None:
    y = before_npz["y"]
    pred_b = before_npz["pred"]
    pred_a = after_npz["pred"]
    subs = before_npz["subject"]
    rows = []
    for sub in np.unique(subs):
        m = subs == sub
        acc_b = float((pred_b[m] == y[m]).mean())
        acc_a = float((pred_a[m] == y[m]).mean())
        rows.append((str(sub), acc_a - acc_b))
    rows.sort(key=lambda r: r[1])
    fig = go.Figure(
        go.Bar(
            x=[r[1] for r in rows],
            y=[r[0] for r in rows],
            orientation="h",
            marker_color=["#fb7185" if r[1] < 0 else "#3dd6c6" for r in rows],
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Per-subject accuracy change (after − before). Some people get worse — that is honest.",
        height=max(320, 18 * len(rows)),
        xaxis_title="Δ accuracy",
    )
    return fig
