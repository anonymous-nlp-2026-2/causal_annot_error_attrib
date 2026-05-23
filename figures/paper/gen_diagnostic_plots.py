"""Generate diagnostic plots for §4.6 empirical validation.

Data source: HumAID (K=10) ONLY — real LLM confusion matrices from
plan_006_asv_empirical.py, loaded directly from asv_empirical_results.json.

Outputs:
  fig5_f1_vs_bias.pdf / .png  — F1 vs downstream bias scatter (HumAID only)
  fig6_asv_heatmap.pdf / .png — ASV magnitude by topology (HumAID only)
"""

import json
import numpy as np
import matplotlib.pyplot as plt

JSON_PATH = '../../artifacts/plan006/asv_empirical_results.json'

with open(JSON_PATH) as f:
    data = json.load(f)

TOPOS = ['confounding', 'mediation', 'collider', 'exposure', 'M-bias', 'IV', 'front-door']
TOPO_KEYS = ['confounding', 'mediation', 'collider', 'exposure', 'mbias', 'iv', 'frontdoor']

# Identify zero-matrix configs (macro_f1 == 0) to exclude from all stats.
zero_configs = {(e['model'], e['prompt'])
                for e in data['f1_vs_bias'] if e['macro_f1'] == 0}


def is_valid(e):
    return (e['model'], e['prompt']) not in zero_configs

topo_colors = {
    'confounding': '#1f77b4', 'mediation': '#ff7f0e', 'collider': '#2ca02c',
    'exposure': '#d62728', 'mbias': '#9467bd', 'iv': '#8c564b',
    'frontdoor': '#e377c2',
}
topo_markers = {
    'confounding': 'o', 'mediation': 's', 'collider': '^',
    'exposure': 'D', 'mbias': 'v', 'iv': 'X',
    'frontdoor': 'P',
}

# ── Figure 5: F1 vs Downstream Bias (exact values from JSON) ──

entries = [e for e in data['f1_vs_bias'] if is_valid(e) and e['macro_f1'] > 0]

fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))

for ti, tk in enumerate(TOPO_KEYS):
    pts = [e for e in entries if e['topology'] == tk]
    if not pts:
        continue
    f1s = np.array([e['macro_f1'] for e in pts])
    biases = np.array([abs(e['bias_pct']) * 100 for e in pts])
    ax.scatter(f1s, biases, c=topo_colors[tk],
               marker=topo_markers[tk], s=40, alpha=0.8, linewidths=0.4,
               edgecolors='k', label=TOPOS[ti], zorder=3)

ax.axhline(10, color='#d62728', ls='--', lw=0.8, alpha=0.6)
ax.text(0.692, 10.5, '10% bias threshold', ha='right', va='bottom',
        fontsize=7, color='#d62728', alpha=0.8)

iv_pts = [e for e in entries if e['topology'] == 'iv']
max_iv = max(iv_pts, key=lambda e: abs(e['bias_pct']))
ax.annotate(f'IV: {abs(max_iv["bias_pct"])*100:.1f}% bias at F1={max_iv["macro_f1"]:.2f}',
            xy=(max_iv['macro_f1'], abs(max_iv['bias_pct'])*100),
            xytext=(max_iv['macro_f1'] + 0.025, abs(max_iv['bias_pct'])*100 - 4),
            fontsize=7, ha='left', color='#8c564b',
            arrowprops=dict(arrowstyle='->', color='#8c564b', lw=0.8))

ax.set_xlabel('Macro F1', fontsize=10)
ax.set_ylabel('|Relative Bias| (%)', fontsize=10)
f1_all = [e['macro_f1'] for e in entries]
ax.set_xlim(min(f1_all) - 0.01, max(f1_all) + 0.01)
ax.set_ylim(0, 48)
ax.legend(fontsize=6.5, ncol=2, loc='upper right', framealpha=0.9,
          handletextpad=0.3, columnspacing=0.8)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.15)
ax.set_title('HumAID (K=10): Topology Determines Bias, Not F1', fontsize=10, pad=8)

plt.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(f'fig5_f1_vs_bias.{ext}', dpi=300, bbox_inches='tight')
plt.close()
print("fig5: plotted", len(entries), "exact data points from JSON")


# ── Figure 6: ASV Magnitude by Topology (HumAID K=10 only) ──

# Compute ASV ranges and unsafe counts from JSON type1_type2 data
t1t2 = [e for e in data['type1_type2'] if is_valid(e)]

asv_ranges = {}
asv_unsafe = {}
for tk in TOPO_KEYS:
    topo_entries = [e for e in t1t2 if e['topology'] == tk]
    magnitudes = [e['magnitude_asv'] for e in topo_entries
                  if e['magnitude_asv'] is not None]
    unsafe_count = sum(1 for e in topo_entries if not e['asv_safe'])
    total_count = len(topo_entries)
    asv_unsafe[tk] = (unsafe_count, total_count)
    if magnitudes and unsafe_count > 0:
        asv_ranges[tk] = (min(magnitudes), max(magnitudes))
    else:
        asv_ranges[tk] = None

fig, (ax_asv, ax_bar) = plt.subplots(1, 2, figsize=(6.0, 2.8),
                                      gridspec_kw={'width_ratios': [3, 1.3], 'wspace': 0.08})

y_pos = np.arange(len(TOPO_KEYS))

for i, tk in enumerate(TOPO_KEYS):
    rng = asv_ranges[tk]
    if rng is None:
        ax_asv.barh(i, 2.0, left=0, color='#d4edda', edgecolor='#2ca02c',
                    linewidth=0.8, height=0.6)
        ax_asv.text(1.0, i, 'safe (bias < 10%)', ha='center', va='center',
                    fontsize=7, color='#2ca02c', fontweight='bold')
    else:
        lo, hi = rng
        mid = (lo + hi) / 2
        width = hi - lo
        color = '#d62728' if hi < 0.5 else '#ff7f0e' if hi < 1.0 else '#ffc107'
        ax_asv.barh(i, width, left=lo, color=color, edgecolor='k',
                    linewidth=0.5, height=0.6, alpha=0.8)
        ax_asv.plot([lo, hi], [i, i], 'k-', lw=1.5)
        ax_asv.plot(lo, i, 'k|', markersize=8)
        ax_asv.plot(hi, i, 'k|', markersize=8)
        ax_asv.text(mid, i + 0.35, f'{lo:.2f}--{hi:.2f}', ha='center',
                    va='bottom', fontsize=6.5)

ax_asv.set_xlim(0, 2.0)
ax_asv.set_yticks(y_pos)
ax_asv.set_yticklabels(TOPOS, fontsize=8)
ax_asv.set_xlabel('Magnitude ASV ($\\delta^*$)', fontsize=9)
ax_asv.set_title('ASV Range (HumAID, K=10)', fontsize=9, pad=6)
ax_asv.axvline(1.0, color='grey', ls=':', lw=0.6, alpha=0.5)
ax_asv.text(1.02, 6.5, '$\\delta^*{=}1$\n(reference)', fontsize=6, color='grey', va='top')
ax_asv.invert_yaxis()

unsafe_pcts = [asv_unsafe[tk][0] / asv_unsafe[tk][1] * 100 for tk in TOPO_KEYS]
colors_bar = ['#d62728' if p > 50 else '#ff7f0e' if p > 0 else '#2ca02c' for p in unsafe_pcts]
ax_bar.barh(y_pos, unsafe_pcts, color=colors_bar, edgecolor='white', linewidth=0.5, height=0.6)
ax_bar.set_xlim(0, 115)
ax_bar.set_yticks([])
ax_bar.set_xlabel('% Unsafe', fontsize=8)
ax_bar.set_title('ASV Filter', fontsize=9, pad=6)
ax_bar.tick_params(labelsize=7)

for i, pct in enumerate(unsafe_pcts):
    label = f'{pct:.0f}%' if pct == 0 or pct == 100 else f'{pct:.1f}%'
    ax_bar.text(pct + 2, i, label, va='center', fontsize=6.5)

ax_bar.invert_yaxis()

plt.subplots_adjust(left=0.14, right=0.97, top=0.86, bottom=0.16)
for ext in ['pdf', 'png']:
    fig.savefig(f'fig6_asv_heatmap.{ext}', dpi=300, bbox_inches='tight')
plt.close()

print("fig6: ASV ranges and unsafe counts computed from JSON")
for tk in TOPO_KEYS:
    u, t = asv_unsafe[tk]
    rng = asv_ranges[tk]
    rng_str = f'{rng[0]:.4f}--{rng[1]:.4f}' if rng else 'safe'
    print(f"  {tk}: {u}/{t} unsafe ({u/t*100:.1f}%), ASV range: {rng_str}")

print("\nGenerated: fig5_f1_vs_bias.pdf/.png, fig6_asv_heatmap.pdf/.png")
print("Data source: HumAID (K=10) ONLY — all values from asv_empirical_results.json")
