#!/usr/bin/env python3
"""Figure 1: DAG Taxonomy — annotation error positions across 7 causal topologies.
v3: 4+3 layout (7 panels), fixed collider label, added conditioned legend."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

R = 0.058
BLACK = '#1a1a1a'
BLUE = '#0072B2'
GRAY = '#555555'
LGRAY = '#888888'
VERM = '#D55E00'


def draw_node(ax, x, y, label, style='observed'):
    if style == 'conditioned':
        box = FancyBboxPatch(
            (x - R * 1.3, y - R * 1.1), R * 2.6, R * 2.2,
            boxstyle='round,pad=0.008',
            fc='#E8F0FE', ec=BLUE, lw=1.6, zorder=10)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10,
                color=BLUE, fontweight='bold', zorder=11)
        return
    kw = dict(fc='white', zorder=10)
    if style == 'latent':
        kw.update(ec=BLACK, lw=1.0, ls=(0, (4, 3)))
    elif style == 'star':
        kw.update(ec=BLUE, lw=1.6, fc='#E8F0FE')
    else:
        kw.update(ec=BLACK, lw=1.0)
    ax.add_patch(Circle((x, y), R, **kw))
    c = BLUE if style == 'star' else BLACK
    w = 'bold' if style == 'star' else 'normal'
    ax.text(x, y, label, ha='center', va='center', fontsize=10,
            color=c, fontweight=w, zorder=11)


def draw_edge(ax, p1, p2, style='causal', rad=0):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    d = np.hypot(dx, dy)
    if d < 1e-6:
        return
    g = R * 1.22
    s = (p1[0] + dx / d * g, p1[1] + dy / d * g)
    e = (p2[0] - dx / d * g, p2[1] - dy / d * g)
    kw = dict(arrowstyle='->', mutation_scale=12, zorder=5,
              connectionstyle=f'arc3,rad={rad}')
    if style == 'meas':
        kw.update(lw=1.0, ls='--', color=LGRAY)
    else:
        kw.update(lw=1.3, color=BLACK)
    ax.add_patch(FancyArrowPatch(s, e, **kw))


# ── Figure: 4+3 layout ──
fig = plt.figure(figsize=(12, 5.5))
gs = GridSpec(2, 12, figure=fig,
              hspace=0.32, wspace=0.05,
              left=0.01, right=0.99, top=0.92, bottom=0.07)

panels = [
    ('(a) Confounding',      'Attenuation (toward OVB)'),
    ('(b) Exposure',         'Sign reversal (83.4%)'),
    ('(c) Mediation',        'Amplification'),
    ('(d) Collider',         'Attenuation (no sign flip)'),
    ('(e) Front-door',       'Moderate sensitivity'),
    ('(f) M-bias',           'Immune'),
    ('(g) IV (Instrument)',  'Fragile (Wald amplification)'),
]

axs = []
# Row 1: 4 panels, each spanning 3 of 12 columns
for i in range(4):
    ax = fig.add_subplot(gs[0, i*3:(i+1)*3])
    ax.set_xlim(0, 1.35)
    ax.set_ylim(0, 0.85)
    ax.set_aspect('equal')
    ax.axis('off')
    axs.append(ax)

# Row 2: 3 panels, each spanning 4 of 12 columns
for i in range(3):
    ax = fig.add_subplot(gs[1, i*4:(i+1)*4])
    ax.set_xlim(0, 1.35)
    ax.set_ylim(0, 0.85)
    ax.set_aspect('equal')
    ax.axis('off')
    axs.append(ax)

for i in range(7):
    ax = axs[i]
    title, subtitle = panels[i]
    ax.text(0.5, 1.10, title, ha='center', va='top', fontsize=10.5,
            fontweight='bold', transform=ax.transAxes)
    sub_color = VERM if i == 1 else GRAY
    ax.text(0.5, 0.99, subtitle, ha='center', va='top', fontsize=9,
            fontstyle='italic', color=sub_color, transform=ax.transAxes)

# Exposure panel: subtle red background
rect = Rectangle((0, 0), 1.35, 0.85, fc='#FFF5F3', ec=VERM, lw=1.5, zorder=0)
axs[1].add_patch(rect)


# ═══════════════════════════════════════════════════
# (a) Confounding: A→X, A→Y, X→Y, A⇢A*
# ═══════════════════════════════════════════════════
ax = axs[0]
A  = (0.50, 0.68); As = (0.98, 0.68)
X  = (0.25, 0.20); Y  = (0.85, 0.20)
draw_node(ax, *A,  'A',  'latent')
draw_node(ax, *As, 'A*', 'star')
draw_node(ax, *X,  'X')
draw_node(ax, *Y,  'Y')
draw_edge(ax, A, X)
draw_edge(ax, A, Y)
draw_edge(ax, X, Y)
draw_edge(ax, A, As, 'meas')


# ═══════════════════════════════════════════════════
# (b) Exposure: A→Y, A⇢A*
# ═══════════════════════════════════════════════════
ax = axs[1]
A  = (0.32, 0.55); As = (1.02, 0.55); Y = (0.67, 0.17)
draw_node(ax, *A,  'A',  'latent')
draw_node(ax, *As, 'A*', 'star')
draw_node(ax, *Y,  'Y')
draw_edge(ax, A, Y)
draw_edge(ax, A, As, 'meas')


# ═══════════════════════════════════════════════════
# (c) Mediation: X→A, A→Y, X→Y (direct), A⇢A*
# ═══════════════════════════════════════════════════
ax = axs[2]
X  = (0.08, 0.35); A  = (0.55, 0.67)
As = (1.02, 0.67); Y  = (1.24, 0.35)
draw_node(ax, *X,  'X')
draw_node(ax, *A,  'A',  'latent')
draw_node(ax, *As, 'A*', 'star')
draw_node(ax, *Y,  'Y')
draw_edge(ax, X, A)
draw_edge(ax, A, Y)
draw_edge(ax, X, Y, rad=-0.2)
draw_edge(ax, A, As, 'meas')


# ═══════════════════════════════════════════════════
# (d) Collider: X→A, Y→A, A⇢A*, condition on A*
# ═══════════════════════════════════════════════════
ax = axs[3]
X  = (0.22, 0.70); Y  = (1.12, 0.70)
A  = (0.67, 0.42); As = (0.67, 0.10)
draw_node(ax, *X,  'X')
draw_node(ax, *Y,  'Y')
draw_node(ax, *A,  'A',  'latent')
draw_node(ax, *As, 'A*', 'conditioned')
draw_edge(ax, X, A)
draw_edge(ax, Y, A)
draw_edge(ax, A, As, 'meas')


# ═══════════════════════════════════════════════════
# (e) Front-door: X→A→Y, U→X, U→Y, A⇢A*
# ═══════════════════════════════════════════════════
ax = axs[4]
X  = (0.15, 0.35); A  = (0.55, 0.67)
As = (1.02, 0.67); Y  = (1.20, 0.35)
U  = (0.67, 0.10)
draw_node(ax, *X,  'X')
draw_node(ax, *A,  'A',  'latent')
draw_node(ax, *As, 'A*', 'star')
draw_node(ax, *Y,  'Y')
draw_node(ax, *U,  'U',  'latent')
draw_edge(ax, X, A)
draw_edge(ax, A, Y)
draw_edge(ax, U, X)
draw_edge(ax, U, Y)
draw_edge(ax, A, As, 'meas')


# ═══════════════════════════════════════════════════
# (f) M-bias: U₁→X, U₁→A, U₂→A, U₂→Y, X→Y, A⇢A*
# ═══════════════════════════════════════════════════
ax = axs[5]
U1 = (0.12, 0.73); U2 = (1.22, 0.73)
X  = (0.12, 0.27); Y  = (1.22, 0.27)
A  = (0.67, 0.52); As = (0.67, 0.10)
draw_node(ax, *U1, 'U₁', 'latent')
draw_node(ax, *U2, 'U₂', 'latent')
draw_node(ax, *X,  'X')
draw_node(ax, *Y,  'Y')
draw_node(ax, *A,  'A',  'latent')
draw_node(ax, *As, 'A*', 'star')
draw_edge(ax, U1, X)
draw_edge(ax, U1, A)
draw_edge(ax, U2, A)
draw_edge(ax, U2, Y)
draw_edge(ax, X, Y)
draw_edge(ax, A, As, 'meas')


# ═══════════════════════════════════════════════════
# (g) IV: A→X, X→Y, U→X, U→Y, A⇢A*
# ═══════════════════════════════════════════════════
ax = axs[6]
A  = (0.15, 0.68); As = (0.65, 0.68)
X  = (0.40, 0.34); Y  = (0.98, 0.34)
U  = (0.69, 0.07)
draw_node(ax, *A,  'A',  'latent')
draw_node(ax, *As, 'A*', 'star')
draw_node(ax, *X,  'X')
draw_node(ax, *Y,  'Y')
draw_node(ax, *U,  'U',  'latent')
draw_edge(ax, A, X)
draw_edge(ax, X, Y)
draw_edge(ax, U, X)
draw_edge(ax, U, Y)
draw_edge(ax, A, As, 'meas')


# ── Legend ──
handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
           markeredgecolor=BLACK, markeredgewidth=1.0, markersize=9,
           linestyle='none', label='Observed'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
           markeredgecolor=BLACK, markeredgewidth=1.0, markersize=9,
           linestyle='none', label='Latent (dashed ○)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E8F0FE',
           markeredgecolor=BLUE, markeredgewidth=1.6, markersize=9,
           linestyle='none', label='Misclassified (A*)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#E8F0FE',
           markeredgecolor=BLUE, markeredgewidth=1.6, markersize=9,
           linestyle='none', label='Conditioned on (A*)'),
    Line2D([0], [0], color=BLACK, lw=1.3, linestyle='-',
           marker='>', markersize=4.5, markeredgewidth=0,
           label='Causal'),
    Line2D([0], [0], color=LGRAY, lw=1.0, linestyle='--',
           marker='>', markersize=4.5, markeredgewidth=0,
           label='Measurement'),
]
fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8.5,
           frameon=False, handletextpad=0.4, columnspacing=1.5,
           bbox_to_anchor=(0.5, 0.005))


# ── Save ──
out_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, 'fig1_dag_taxonomy.pdf')
png_path = os.path.join(out_dir, 'fig1_dag_taxonomy.png')

fig.savefig(pdf_path, format='pdf')
fig.savefig(png_path, format='png')
plt.close()

print(f'PDF: {pdf_path}')
print(f'PNG: {png_path}')
for p in [pdf_path, png_path]:
    print(f'  {os.path.basename(p)}: {os.path.getsize(p):,} bytes')
