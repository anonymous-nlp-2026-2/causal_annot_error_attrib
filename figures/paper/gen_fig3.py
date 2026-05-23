#!/usr/bin/env python3
"""Generate fig3_asv_ranking: Forest-plot comparison of ASV sensitivity rankings.

Three-panel layout: Magnitude ASV | Significance Power | Inline stats (+ sign-flip).
Narrative intent: different ASV types give DIFFERENT vulnerability rankings.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 1.8,
})

# ── Colors (colorblind-safe Okabe-Ito) ────────────────────────────────────
C_MAG  = '#0072B2'   # blue — magnitude ASV
C_SIG  = '#E69F00'   # orange — significance power
C_SIGN = '#D55E00'   # vermillion — sign-flip risk
C_FRAG = '#D55E00'   # fragile zone tint (low alpha)
C_IMMUNE = '#009E73'  # green — immune / structural guarantee

# ── Data ──────────────────────────────────────────────────────────────────
# Ordered by composite vulnerability (most fragile → most robust)
topologies = [
    'IV', 'Front-door', 'Exposure', 'Confounding',
    'Mediation', 'Collider', 'M-bias',
]
N = len(topologies)

# Magnitude ASV ranges (τ=0.5, 10% threshold). None=N/A, 'inf'=immune
mag_asv = {
    'IV':          (0.101, 0.607),
    'Front-door':  (0.064, 0.387),
    'Exposure':    None,
    'Confounding': (0.081, 0.487),
    'Mediation':   (0.076, 0.454),
    'Collider':    (0.338, 2.028),
    'M-bias':      'inf',
}

# Significance power (τ=0.05 filter). None=N/A
sig_power = {
    'IV':          0,
    'Front-door':  100,
    'Exposure':    None,
    'Confounding': 100,
    'Mediation':   0,
    'Collider':    0,
    'M-bias':      98,   # midpoint of 96–100%
}

# Sign-flip ASV
sign_flip = {
    'Exposure': '83.4%',
    'IV':       '1.1–6.7',
}

# Brief characterization for significance panel annotations
sig_note = {
    'Mediation': 'attenuating',
    'Collider':  'attenuating',
    'IV':        'Wald underpowered',
}


def fmt_mag(topo):
    v = mag_asv[topo]
    if v is None:
        return '—'
    if v == 'inf':
        return '∞'
    return f'{v[0]:.2f} – {v[1]:.2f}'


def fmt_sig(topo):
    v = sig_power[topo]
    if v is None:
        return '—'
    return f'{v}%'


def fmt_sf(topo):
    return sign_flip.get(topo, '—')


# ══════════════════════════════════════════════════════════════════════════
#  BUILD FIGURE
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(9.5, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.3, 2.0], wspace=0.08)

ax_mag  = fig.add_subplot(gs[0])
ax_sig  = fig.add_subplot(gs[1])
ax_stat = fig.add_subplot(gs[2])

# Shared y setup
for ax in (ax_mag, ax_sig, ax_stat):
    ax.set_ylim(N - 0.5, -0.5)
    # Alternating row backgrounds
    for i in range(N):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f5f5f5', zorder=0, lw=0)

# ── Panel 1: Magnitude ASV ────────────────────────────────────────────────
ax_mag.axvspan(0, 0.3, alpha=0.07, color=C_FRAG, zorder=0)
ax_mag.axvline(0.3, color=C_FRAG, lw=0.6, ls=':', alpha=0.35, zorder=0)

for i, t in enumerate(topologies):
    v = mag_asv[t]
    if v is None:
        ax_mag.text(0.8, i, 'N/A', fontsize=8.5, color='#bbbbbb',
                    ha='center', va='center', fontstyle='italic')
    elif v == 'inf':
        ax_mag.annotate(
            '', xy=(2.35, i), xytext=(1.5, i),
            arrowprops=dict(arrowstyle='->', color=C_IMMUNE, lw=2.2,
                            mutation_scale=14))
        ax_mag.text(2.4, i, '∞', fontsize=14, color=C_IMMUNE,
                    ha='left', va='center', fontweight='bold')
    else:
        lo, hi = v
        mid = (lo + hi) / 2
        ax_mag.plot([lo, hi], [i, i], color=C_MAG, lw=3,
                    solid_capstyle='round', zorder=2, alpha=0.7)
        ax_mag.plot(mid, i, 'o', color=C_MAG, ms=8, zorder=3,
                    mec='white', mew=0.9)
        ax_mag.plot(lo, i, '|', color=C_MAG, ms=6, mew=1.2, zorder=3)
        ax_mag.plot(hi, i, '|', color=C_MAG, ms=6, mew=1.2, zorder=3)

ax_mag.set_xlim(-0.02, 2.65)
ax_mag.set_xlabel('ASV value  (lower → more fragile)')
ax_mag.set_title('Magnitude ASV', fontweight='bold', color=C_MAG,
                 loc='left', pad=8)
ax_mag.set_yticks(range(N))
ax_mag.set_yticklabels(topologies)
ax_mag.grid(True, axis='x', alpha=0.12, zorder=0)
ax_mag.spines['left'].set_visible(False)
ax_mag.tick_params(axis='y', length=0)

# Italicize 'Exposure' (sign-flip-only topology)
for lbl in ax_mag.get_yticklabels():
    if lbl.get_text() == 'Exposure':
        lbl.set_fontstyle('italic')
        lbl.set_color('#888888')

# ── Panel 2: Significance Power ───────────────────────────────────────────
ax_sig.axvspan(0, 50, alpha=0.07, color=C_FRAG, zorder=0)
ax_sig.axvline(50, color=C_FRAG, lw=0.6, ls=':', alpha=0.35, zorder=0)

for i, t in enumerate(topologies):
    v = sig_power[t]
    if v is None:
        ax_sig.text(50, i, 'N/A', fontsize=8.5, color='#bbbbbb',
                    ha='center', va='center', fontstyle='italic')
    else:
        # Lollipop: thin line from 0 to value
        ax_sig.plot([0, v], [i, i], color=C_SIG, lw=1.5, alpha=0.4, zorder=2)
        ax_sig.plot(v, i, '^', color=C_SIG, ms=10, zorder=3,
                    mec='white', mew=0.9)
        # Brief annotation for 0% power topologies
        if t in sig_note:
            ax_sig.text(
                4, i + 0.28, sig_note[t], fontsize=6.5,
                color='#999999', fontstyle='italic', va='top')

ax_sig.set_xlim(-5, 115)
ax_sig.set_xlabel('Power (%)')
ax_sig.set_title('Significance Power', fontweight='bold', color=C_SIG,
                 loc='left', pad=8)
ax_sig.set_yticks(range(N))
ax_sig.set_yticklabels([])
ax_sig.grid(True, axis='x', alpha=0.12, zorder=0)
ax_sig.spines['left'].set_visible(False)
ax_sig.tick_params(axis='y', length=0)

# ── Panel 3: Inline stats table ───────────────────────────────────────────
ax_stat.set_xlim(0, 1)
ax_stat.axis('off')

cx = [0.01, 0.40, 0.66]
headers = ['Mag ASV', 'Power', 'Sign-flip']
h_colors = [C_MAG, C_SIG, C_SIGN]

# Column headers
for j, (hdr, hc) in enumerate(zip(headers, h_colors)):
    ax_stat.text(cx[j], -0.35, hdr, fontsize=9, fontweight='bold',
                 color=hc, va='center', clip_on=False)

# Header underline
ax_stat.plot([0.0, 0.98], [-0.12, -0.12], color='#dddddd', lw=0.6,
             clip_on=False, transform=ax_stat.transData)

for i, t in enumerate(topologies):
    # Magnitude value
    mc = C_IMMUNE if mag_asv[t] == 'inf' else '#444444'
    ax_stat.text(cx[0], i, fmt_mag(t), fontsize=8.5, color=mc,
                 va='center', family='monospace')
    # Significance power
    ax_stat.text(cx[1], i, fmt_sig(t), fontsize=8.5, color='#444444',
                 va='center', family='monospace')
    # Sign-flip
    sf = fmt_sf(t)
    if t in sign_flip:
        ax_stat.plot(cx[2] - 0.04, i, 'D', color=C_SIGN, ms=5,
                     mec='white', mew=0.5, zorder=3, clip_on=False)
        ax_stat.text(cx[2], i, sf, fontsize=8.5, color=C_SIGN,
                     va='center', fontweight='bold', family='monospace')
    else:
        ax_stat.text(cx[2], i, sf, fontsize=8.5, color='#cccccc',
                     va='center', family='monospace')

# ── Suptitle (insight, not description) ───────────────────────────────────
fig.suptitle(
    'Different ASV types yield different vulnerability rankings',
    fontsize=11.5, fontstyle='italic', color='#444444', y=1.01)

# ── Legend ─────────────────────────────────────────────────────────────────
handles = [
    plt.Line2D([0], [0], marker='o', color=C_MAG, ls='-', lw=2.5,
               ms=7, mec='white', mew=0.9, alpha=0.7,
               label='Magnitude ASV (range)'),
    plt.Line2D([0], [0], marker='^', color=C_SIG, ls='-', lw=1.5,
               ms=9, mec='white', mew=0.9, alpha=0.6,
               label='Significance power'),
    plt.Line2D([0], [0], marker='D', color=C_SIGN, ls='None',
               ms=6, mec='white', mew=0.5,
               label='Sign-flip risk'),
    mpatches.Patch(fc=C_FRAG, alpha=0.07, ec='none',
                   label='Fragile zone'),
]
fig.legend(handles=handles, loc='lower center', ncol=4,
           bbox_to_anchor=(0.42, -0.04), frameon=False, fontsize=8.5)

fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.13, wspace=0.08)

# ── Save ──────────────────────────────────────────────────────────────────
outdir = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(outdir, 'fig3_asv_ranking')
fig.savefig(f'{base}.pdf')
fig.savefig(f'{base}.png', dpi=300)
plt.close()
print(f'Saved: {base}.pdf')
print(f'Saved: {base}.png')
