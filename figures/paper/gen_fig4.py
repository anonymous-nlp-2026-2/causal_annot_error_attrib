#!/usr/bin/env python3
"""Generate fig4_binarization: Binarization collapse Δ across 6 topologies.

Narrative intent: Collapsing K=3 to binary is a dangerous shortcut —
Δ varies 150x from M-bias (0.006) to IV (0.926), and sign flips
emerge in exposure + IV.

Data source: registry plan008_binarization_collapse (100k Dirichlet samples).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data from registry (plan008_binarization_collapse) ─────────────────
# Sorted by Δ_mean ascending for visual impact

data = {
    'M-bias':      {'D_mean': 0.0063, 'D_med': 0.0054, 'D_P95': 0.0147, 'D_max': 0.0328,
                    'frac_gt_001': 0.181, 'frac_gt_005': 0.0, 'sign_flip': 0.0},
    'Collider':    {'D_mean': 0.0169, 'D_med': 0.0125, 'D_P95': 0.0472, 'D_max': 0.1069,
                    'frac_gt_001': 0.572, 'frac_gt_005': 0.039, 'sign_flip': 0.0},
    'Confounding': {'D_mean': 0.0253, 'D_med': 0.0182, 'D_P95': 0.0733, 'D_max': 0.2213,
                    'frac_gt_001': 0.686, 'frac_gt_005': 0.138, 'sign_flip': 0.0},
    'Mediation':   {'D_mean': 0.0446, 'D_med': 0.0355, 'D_P95': 0.1179, 'D_max': 0.252,
                    'frac_gt_001': 0.835, 'frac_gt_005': 0.36, 'sign_flip': 0.0},
    'Exposure':    {'D_mean': 0.5164, 'D_med': 0.506, 'D_P95': 0.8453, 'D_max': 1.3876,
                    'frac_gt_001': 1.0, 'frac_gt_005': 0.999, 'sign_flip': 0.14},
    'IV':          {'D_mean': 0.926, 'D_med': 0.9394, 'D_P95': 1.3426, 'D_max': 2.0809,
                    'frac_gt_001': 1.0, 'frac_gt_005': 1.0, 'sign_flip': 0.175},
}

topos = list(data.keys())
n = len(topos)
y_pos = np.arange(n)

# Colors
C_LOW  = '#0072B2'   # blue — low collapse
C_HIGH = '#D55E00'   # vermillion — high collapse
C_MED  = '#E69F00'   # orange — medium
C_DANGER = '#FFF0E0'
C_SAFE   = '#E8F5E8'

def get_color(d_mean):
    if d_mean < 0.02:
        return C_LOW
    elif d_mean < 0.1:
        return C_MED
    else:
        return C_HIGH

fig, ax = plt.subplots(figsize=(8, 4.5))

# Background danger band
ax.axvspan(0.05, 2.5, color=C_DANGER, alpha=0.3, zorder=0)
ax.axvspan(0, 0.05, color=C_SAFE, alpha=0.3, zorder=0)
ax.axvline(x=0.05, color='gray', linestyle=':', linewidth=0.8, alpha=0.7, zorder=1,
           label='Δ = 0.05 (practical threshold)')

for i, topo in enumerate(topos):
    d = data[topo]
    color = get_color(d['D_mean'])

    # Range bar: median to P95
    ax.plot([d['D_med'], d['D_P95']], [y_pos[i], y_pos[i]],
            color=color, linewidth=2.5, zorder=3, alpha=0.6)

    # Extend thin line to max
    ax.plot([d['D_P95'], d['D_max']], [y_pos[i], y_pos[i]],
            color=color, linewidth=1, zorder=3, alpha=0.4)

    # Mean marker
    ax.plot(d['D_mean'], y_pos[i], 'o', color=color, markersize=9, zorder=4,
            markeredgecolor='white', markeredgewidth=1)

    # Median marker (smaller)
    ax.plot(d['D_med'], y_pos[i], '|', color=color, markersize=12, zorder=4,
            markeredgewidth=2)

    # Numeric label
    label = f"Δ̄={d['D_mean']:.3f}"
    if d['sign_flip'] > 0:
        label += f"  ⚠ {d['sign_flip']*100:.0f}% sign flip"
    ax.text(d['D_max'] + 0.03, y_pos[i], label,
            va='center', ha='left', fontsize=8.5, color=color, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(topos, fontsize=10)
ax.set_xlabel('Binarization collapse Δ = |bias$_{K=3}$ − bias$_{K→2}$|', fontsize=10)
ax.set_xlim(-0.02, 2.5)
ax.invert_yaxis()

# 150x annotation
ax.annotate('150×', xy=(0.926, 5), xytext=(0.926, 0),
            fontsize=14, fontweight='bold', color=C_HIGH, alpha=0.3,
            ha='center', va='center',
            arrowprops=dict(arrowstyle='<->', color=C_HIGH, alpha=0.3, lw=1.5))

ax.set_title('Binarization Collapse Across DAG Topologies\n'
             'Collapsing K=3 → binary introduces topology-dependent bias',
             fontsize=11, fontweight='bold', loc='left')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
leg_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C_LOW, markersize=8,
               label='Low collapse (Δ̄ < 0.02)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C_MED, markersize=8,
               label='Medium (0.02 ≤ Δ̄ < 0.1)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C_HIGH, markersize=8,
               label='High collapse (Δ̄ ≥ 0.1)'),
    plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=1,
               label='Practical threshold (Δ = 0.05)'),
]
ax.legend(handles=leg_elements, loc='lower right', fontsize=8, framealpha=0.9)

plt.tight_layout()

# ── Save ────────────────────────────────────────────────────────────────
out_base = '/home/ubuntu/.agent-ml-research-prod_0510/projects/causal_annot_error_attrib/figures/paper/fig4_binarization'
fig.savefig(f'{out_base}.pdf', bbox_inches='tight', dpi=300, pad_inches=0.1)
fig.savefig(f'{out_base}.png', bbox_inches='tight', dpi=300, pad_inches=0.1)
print(f'Saved: {out_base}.pdf and .png')
plt.close()
