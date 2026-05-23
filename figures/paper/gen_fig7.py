"""
fig7_dirichlet_overlay: overlay 14 empirical confusion-matrix configs on theoretical
Dirichlet-subspace violation curves (exposure + IV topologies).

X-axis: min diagonal dominance = min_k C[k,k]
Y-axis: rate (violation_rate / sign_flip_rate from Dirichlet subspace analysis)
Theory: violation_rate as gray-shaded area; sign_flip_rate as dashed line.
Empirical: 14 points placed at (min_diag, interpolated_violation_rate), markers
encode dataset/model. Demonstrates "theory (K=3) vs practice (K=2,3,10)" gap.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy.interpolate import PchipInterpolator

PROJ = Path("/home/ubuntu/.agent-ml-research-prod_0510/projects/causal_annot_error_attrib")
ART = PROJ / "artifacts"
OUT = PROJ / "figures/paper/fig7_dirichlet_overlay.pdf"

# ---------- global style ----------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.6,
})

# ---------- theoretical data (K=3 Dirichlet subspace) ----------
EXPOSURE_THEORY = {
    "diag_thresh":     [0.0,   0.5,   0.7,   0.9],
    "violation_rate":  [0.850, 0.325, 0.137, 0.119],
    "sign_flip_rate":  [0.834, 0.216, 0.021, 0.000],
}

with open(ART / "iv_dirichlet_subspace_results.json") as f:
    iv_raw = json.load(f)
IV_THEORY = {
    "diag_thresh":    [],
    "violation_rate": [],
    "sign_flip_rate": [],
}
for t in ("0.0", "0.5", "0.7", "0.9"):
    IV_THEORY["diag_thresh"].append(float(t))
    IV_THEORY["violation_rate"].append(iv_raw["thresholds"][t]["violation_rate"])
    IV_THEORY["sign_flip_rate"].append(iv_raw["thresholds"][t]["sign_flip_rate"])

# ---------- empirical 14 configs ----------
# (min_diag, dataset, model, prompt)  - 14 valid paper configs
# HumAID (K=10): full 10x10 CM
with open(ART / "plan006/asv_empirical_results.json") as f:
    plan006 = json.load(f)
HUMAID_CMS = plan006["confusion_matrices"]

# VAST and CivilComments diagonals extracted from plan006_full filter pipeline
# (vast: from plan006_full_filtered_stats.json["filtered_stats"]["vast_per_class_recall"])
VAST_DIAG = {
    "qwen2.5-7b_zero-shot":   [0.4613733905579399, 0.847457627118644,   0.10956175298804781],
    "qwen2.5-7b_few-shot-3":  [0.4132762312633833, 0.8210922787193974,  0.1254980079681275],
    "qwen2.5-7b_few-shot-5":  [0.4753747323340471, 0.775894538606403,   0.1254980079681275],
}
CIVIL_DIAG = {
    "qwen2.5-7b_zero-shot":   [0.7481536189069424, 0.684931506849315],
    "qwen2.5-7b_few-shot-3":  [0.6787296898079763, 0.7808219178082192],
    "qwen2.5-7b_few-shot-5":  [0.7370753323485968, 0.7602739726027398],
}

PAPER_CONFIGS = []
# humaid 8 (exclude qwen3_few-shot-5 which is all zeros)
HUMAID_KEYS = [
    "humaid_llama4_zero-shot",
    "humaid_llama4_few-shot-3",
    "humaid_llama4_few-shot-5",
    "humaid_qwen3_zero-shot",
    "humaid_qwen3_few-shot-3",
    "humaid_deepseek_zero-shot",
    "humaid_deepseek_few-shot-3",
    "humaid_deepseek_few-shot-5",
]
for k in HUMAID_KEYS:
    C = np.array(HUMAID_CMS[k])
    mind = float(np.min(np.diag(C)))
    parts = k.split("_", 2)
    PAPER_CONFIGS.append({
        "dataset": "humaid", "model": parts[1], "prompt": parts[2],
        "K": 10, "min_diag": mind,
    })
for k, diag in VAST_DIAG.items():
    model, prompt = k.split("_", 1)
    PAPER_CONFIGS.append({
        "dataset": "vast", "model": model, "prompt": prompt,
        "K": 3, "min_diag": float(min(diag)),
    })
for k, diag in CIVIL_DIAG.items():
    model, prompt = k.split("_", 1)
    PAPER_CONFIGS.append({
        "dataset": "civil_comments", "model": model, "prompt": prompt,
        "K": 2, "min_diag": float(min(diag)),
    })

assert len(PAPER_CONFIGS) == 14, f"expected 14, got {len(PAPER_CONFIGS)}"

# ---------- marker / color encoding ----------
# HumAID circles, blue shades by model
# VAST squares, orange
# CivilComments triangles, green
HUMAID_COLOR = {
    "llama4":   "#A6CEE3",   # light blue
    "qwen3":    "#1F78B4",   # mid blue
    "deepseek": "#08306B",   # dark blue
}
DATASET_STYLE = {
    "humaid":         {"marker": "o", "size": 55},
    "vast":           {"marker": "s", "color": "#E6550D", "size": 55},
    "civil_comments": {"marker": "^", "color": "#2CA25F", "size": 70},
}


def smooth_curve(xs, ys, n=200):
    """Monotonic interpolation on [0, 1]; extend constant beyond endpoints."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    interp = PchipInterpolator(xs, ys, extrapolate=False)
    xx = np.linspace(0.0, 1.0, n)
    yy = interp(xx)
    yy[xx < xs[0]] = ys[0]
    yy[xx > xs[-1]] = ys[-1]
    return xx, yy


def interp_at(x, xs, ys):
    """Evaluate monotone interp at a single x with constant tails."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    return float(PchipInterpolator(xs, ys, extrapolate=False)(x))


def draw_panel(ax, theory, panel_title, panel_letter, *, show_violation_fill=True):
    xs_t = theory["diag_thresh"]
    vr_t = theory["violation_rate"]
    sf_t = theory["sign_flip_rate"]

    xx, vr_curve = smooth_curve(xs_t, vr_t)
    _,  sf_curve = smooth_curve(xs_t, sf_t)

    # Theory: violation zone (gray fill above 0) + sign-flip dashed line
    if show_violation_fill:
        ax.fill_between(
            xx, 0, vr_curve,
            color="#888888", alpha=0.18, zorder=1,
            label="Violation rate (theory)",
        )
        ax.plot(xx, vr_curve, color="#666666", linewidth=1.2, zorder=2)
    else:
        # IV violation_rate is trivially 1.0; show as flat ceiling line w/o fill
        ax.axhline(1.0, color="#888888", linewidth=1.0, alpha=0.5, zorder=1,
                   label="Violation rate (theory) $=1$")
    ax.plot(xx, sf_curve, color="#B22222", linewidth=1.4,
            linestyle="--", zorder=3,
            label="Sign-flip rate (theory)")

    # Theory anchor points (small ticks at 0.0, 0.5, 0.7, 0.9)
    ax.scatter(xs_t, vr_t, s=18, color="#666666",
               marker="x", linewidth=1.0, zorder=4, label="_nolegend_")
    ax.scatter(xs_t, sf_t, s=18, color="#B22222",
               marker="x", linewidth=1.0, zorder=4, label="_nolegend_")

    # Safe / danger zone shading
    ax.axvspan(0.9, 1.0, color="#2CA25F", alpha=0.07, zorder=0)
    ax.axvspan(0.0, 0.5, color="#E6550D", alpha=0.07, zorder=0)

    # Empirical points: Y = interp(violation_rate at min_diag)
    handled = set()
    for cfg in PAPER_CONFIGS:
        x = cfg["min_diag"]
        y = interp_at(x, xs_t, vr_t)
        ds = cfg["dataset"]
        style = DATASET_STYLE[ds]
        marker = style["marker"]
        size = style["size"]
        color = HUMAID_COLOR[cfg["model"]] if ds == "humaid" else style["color"]

        legend_key = (ds, cfg["model"]) if ds == "humaid" else (ds,)
        if legend_key not in handled:
            handled.add(legend_key)
            if ds == "humaid":
                lbl = f"HumAID/{cfg['model']} (K=10)"
            elif ds == "vast":
                lbl = "VAST/qwen2.5-7b (K=3)"
            else:
                lbl = "CivilComments/qwen2.5-7b (K=2)"
        else:
            lbl = None

        ax.scatter(
            x, y, marker=marker, s=size,
            facecolor=color, edgecolor="white", linewidth=0.8,
            zorder=6, label=lbl,
        )

    # Zone annotations
    txt_safe = ax.text(
        0.95, 0.04, "safe\nzone", ha="center", va="bottom",
        fontsize=7.5, color="#1B7A45", fontstyle="italic",
        transform=ax.transAxes,
    )
    txt_safe.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
    txt_danger = ax.text(
        0.05, 0.45, "danger\nzone", ha="left", va="center",
        fontsize=7.5, color="#A53A0C", fontstyle="italic",
        transform=ax.transAxes,
    )
    txt_danger.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.12)
    ax.set_xlabel(r"min diagonal dominance  $\min_k C_{kk}$")
    if panel_letter == "a":
        ax.set_ylabel("Rate (theory) / dot Y = violation rate")
    ax.set_title(f"({panel_letter}) {panel_title}", loc="left", pad=4)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5, zorder=0)


# ---------- build figure ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)

draw_panel(axes[0], EXPOSURE_THEORY,
           "Exposure topology", "a",
           show_violation_fill=True)
draw_panel(axes[1], IV_THEORY,
           "IV topology", "b",
           show_violation_fill=False)

# Unified legend across both panels (de-duplicate)
handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    for hh, ll in zip(h, l):
        if ll and ll not in labels:
            handles.append(hh); labels.append(ll)
fig.legend(
    handles, labels,
    loc="lower center", bbox_to_anchor=(0.5, -0.04),
    ncol=4, frameon=False, columnspacing=1.2, handletextpad=0.5,
)

fig.suptitle(
    r"Theory (random $K{=}3$ $C$) vs. practice"
    r" (14 LLM-annotated configs, $K\in\{2,3,10\}$)",
    fontsize=10, y=1.01,
)

plt.tight_layout(rect=(0, 0.02, 1, 0.98))
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT)
plt.savefig(str(OUT).replace(".pdf", ".png"))
plt.close()
print(f"Saved: {OUT}")
print(f"Saved: {str(OUT).replace('.pdf', '.png')}")

# ---------- print empirical points for sanity ----------
print()
print(f"{'config':<48} {'K':>3} {'min_diag':>9}")
for cfg in sorted(PAPER_CONFIGS, key=lambda c: c["min_diag"]):
    key = f"{cfg['dataset']}_{cfg['model']}_{cfg['prompt']}"
    print(f"{key:<48} {cfg['K']:>3} {cfg['min_diag']:>9.4f}")
