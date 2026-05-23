#!/usr/bin/env python3
"""Figure 2: 2×4 Bias Distribution Across 7 DAG Topologies under K=3 Misclassification."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
import numpy as np
from scipy.stats import gaussian_kde
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 1.8,
})

SAFE = '#0072B2'    # blue (colorblind-friendly)
DANGER = '#E69F00'  # orange (colorblind-friendly)
DIST = '#0072B2'
REF = '#555555'
N_MC = 100_000
N_POP = 2_000_000

rng_main = np.random.default_rng(42)
Cs = np.zeros((N_MC, 3, 3))
for j in range(3):
    Cs[:, :, j] = rng_main.dirichlet(np.ones(3), size=N_MC)


def softmax_vec(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def softmax_sample_batch(logits, rng):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    cu = np.cumsum(p, axis=1)
    u = rng.uniform(size=logits.shape[0])
    return (u >= cu[:, 0]).astype(int) + (u >= cu[:, 1]).astype(int)


def safe_batch_solve(A, b):
    try:
        return np.linalg.solve(A, b[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        n = A.shape[0]
        results = np.full((n, A.shape[1]), np.nan)
        for i in range(n):
            try:
                results[i] = np.linalg.solve(A[i], b[i])
            except np.linalg.LinAlgError:
                pass
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFOUNDING: A→T, A→Y, T→Y.  OLS: Y ~ 1+T+D*₁+D*₂.  Coeff on T.
# ═══════════════════════════════════════════════════════════════════════════════
def compute_confounding(Cs):
    P_A = np.array([0.4, 0.35, 0.25])
    a1, a2 = 1.5, -1.0
    beta = np.array([0.0, 1.0, 1.0, -0.5])
    p0, p1, p2 = P_A
    n = len(Cs)

    ET = a1 * p1 + a2 * p2
    ET2 = a1**2 * p1 + a2**2 * p2 + 1.0
    q = Cs @ P_A
    ET_Ds = a1 * Cs[:, :, 1] * p1 + a2 * Cs[:, :, 2] * p2

    EZZ = np.zeros((n, 4, 4))
    EZZ[:, 0, 0] = 1
    EZZ[:, 0, 1] = EZZ[:, 1, 0] = ET
    EZZ[:, 0, 2] = EZZ[:, 2, 0] = q[:, 1]
    EZZ[:, 0, 3] = EZZ[:, 3, 0] = q[:, 2]
    EZZ[:, 1, 1] = ET2
    EZZ[:, 1, 2] = EZZ[:, 2, 1] = ET_Ds[:, 1]
    EZZ[:, 1, 3] = EZZ[:, 3, 1] = ET_Ds[:, 2]
    EZZ[:, 2, 2] = q[:, 1]
    EZZ[:, 3, 3] = q[:, 2]

    EZX = np.zeros((n, 4, 4))
    EZX[:, 0, :] = [1, ET, p1, p2]
    EZX[:, 1, :] = [ET, ET2, a1 * p1, a2 * p2]
    EZX[:, 2, 0] = q[:, 1]
    EZX[:, 2, 1] = ET_Ds[:, 1]
    EZX[:, 2, 2] = Cs[:, 1, 1] * p1
    EZX[:, 2, 3] = Cs[:, 1, 2] * p2
    EZX[:, 3, 0] = q[:, 2]
    EZX[:, 3, 1] = ET_Ds[:, 2]
    EZX[:, 3, 2] = Cs[:, 2, 1] * p1
    EZX[:, 3, 3] = Cs[:, 2, 2] * p2

    rhs = np.einsum('nij,j->ni', EZX, beta)
    return safe_batch_solve(EZZ, rhs)[:, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXPOSURE: A→Y.  OLS: Y ~ 1+D*₁+D*₂.  Coeff on D*₁.
# ═══════════════════════════════════════════════════════════════════════════════
def compute_exposure(Cs):
    P_A = np.array([0.4, 0.35, 0.25])
    BETA = np.array([0.0, 1.0, -0.5])
    p0, p1, p2 = P_A
    n = len(Cs)

    q = Cs @ P_A
    EZZ = np.zeros((n, 3, 3))
    EZZ[:, 0, 0] = 1
    EZZ[:, 0, 1] = EZZ[:, 1, 0] = q[:, 1]
    EZZ[:, 0, 2] = EZZ[:, 2, 0] = q[:, 2]
    EZZ[:, 1, 1] = q[:, 1]
    EZZ[:, 2, 2] = q[:, 2]

    EZX = np.zeros((n, 3, 3))
    EZX[:, 0, :] = [1, p1, p2]
    EZX[:, 1, 0] = q[:, 1]
    EZX[:, 1, 1] = Cs[:, 1, 1] * p1
    EZX[:, 1, 2] = Cs[:, 1, 2] * p2
    EZX[:, 2, 0] = q[:, 2]
    EZX[:, 2, 1] = Cs[:, 2, 1] * p1
    EZX[:, 2, 2] = Cs[:, 2, 2] * p2

    rhs = np.einsum('nij,j->ni', EZX, BETA)
    plim = safe_batch_solve(EZZ, rhs)
    return plim[:, 1], plim[:, 2]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MEDIATION: T→A→Y, T→Y(direct).  OLS: Y ~ 1+T+D*₁+D*₂.  Coeff on T.
# ═══════════════════════════════════════════════════════════════════════════════
def compute_mediation(Cs):
    BETA = np.array([0.0, 1.0, -0.5])
    TAU = 1.0
    GAMMA = np.array([0.0, 0.5, -0.3])
    DELTA = np.array([0.0, 0.8, -0.6])
    n = len(Cs)

    p0 = softmax_vec(GAMMA)
    p1 = softmax_vec(GAMMA + DELTA)
    pm = 0.5 * (p0 + p1)
    q0 = Cs @ p0
    q1 = Cs @ p1
    qm = 0.5 * (q0 + q1)

    EZZ = np.zeros((n, 4, 4))
    EZZ[:, 0, 0] = 1
    EZZ[:, 0, 1] = EZZ[:, 1, 0] = 0.5
    EZZ[:, 0, 2] = EZZ[:, 2, 0] = qm[:, 1]
    EZZ[:, 0, 3] = EZZ[:, 3, 0] = qm[:, 2]
    EZZ[:, 1, 1] = 0.5
    EZZ[:, 1, 2] = EZZ[:, 2, 1] = 0.5 * q1[:, 1]
    EZZ[:, 1, 3] = EZZ[:, 3, 1] = 0.5 * q1[:, 2]
    EZZ[:, 2, 2] = qm[:, 1]
    EZZ[:, 3, 3] = qm[:, 2]

    EZX = np.zeros((n, 4, 4))
    EZX[:, 0, :] = [1, 0.5, pm[1], pm[2]]
    EZX[:, 1, :] = [0.5, 0.5, 0.5 * p1[1], 0.5 * p1[2]]
    EZX[:, 2, 0] = qm[:, 1]
    EZX[:, 2, 1] = 0.5 * q1[:, 1]
    EZX[:, 2, 2] = Cs[:, 1, 1] * pm[1]
    EZX[:, 2, 3] = Cs[:, 1, 2] * pm[2]
    EZX[:, 3, 0] = qm[:, 2]
    EZX[:, 3, 1] = 0.5 * q1[:, 2]
    EZX[:, 3, 2] = Cs[:, 2, 1] * pm[1]
    EZX[:, 3, 3] = Cs[:, 2, 2] * pm[2]

    b = np.array([BETA[0], TAU, BETA[1], BETA[2]])
    rhs = np.einsum('nij,j->ni', EZX, b)
    return safe_batch_solve(EZZ, rhs)[:, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# 4/5. COLLIDER & M-BIAS: precompute (T,Y,A) stats, vectorise per-C
#   OLS: Y ~ 1+T+D*₁+D*₂.  Coeff on T.
# ═══════════════════════════════════════════════════════════════════════════════
def _precompute_stats(T, Y, A):
    return dict(
        ET=T.mean(), ET2=(T**2).mean(), EY=Y.mean(), ETY=(T * Y).mean(),
        p=np.array([np.mean(A == j) for j in range(3)]),
        ET_A=np.array([np.mean(T * (A == j)) for j in range(3)]),
        EY_A=np.array([np.mean(Y * (A == j)) for j in range(3)]),
    )


def _plim_from_stats(Cs, st):
    n = len(Cs)
    q = Cs @ st['p']
    ETDs = Cs @ st['ET_A']
    EYDs = Cs @ st['EY_A']

    EZZ = np.zeros((n, 4, 4))
    EZZ[:, 0, 0] = 1
    EZZ[:, 0, 1] = EZZ[:, 1, 0] = st['ET']
    EZZ[:, 0, 2] = EZZ[:, 2, 0] = q[:, 1]
    EZZ[:, 0, 3] = EZZ[:, 3, 0] = q[:, 2]
    EZZ[:, 1, 1] = st['ET2']
    EZZ[:, 1, 2] = EZZ[:, 2, 1] = ETDs[:, 1]
    EZZ[:, 1, 3] = EZZ[:, 3, 1] = ETDs[:, 2]
    EZZ[:, 2, 2] = q[:, 1]
    EZZ[:, 3, 3] = q[:, 2]

    EZY = np.zeros((n, 4))
    EZY[:, 0] = st['EY']
    EZY[:, 1] = st['ETY']
    EZY[:, 2] = EYDs[:, 1]
    EZY[:, 3] = EYDs[:, 2]
    return safe_batch_solve(EZZ, EZY)[:, 1]


def compute_collider(Cs):
    print("  Precomputing collider (N=2M)...")
    rng = np.random.default_rng(999)
    N = N_POP
    T = rng.normal(0, 1, N)
    Y = 1.0 * T + rng.normal(0, 1, N)
    logits = np.column_stack([np.zeros(N), 0.5 * T + 0.8 * Y, -0.3 * T + 0.6 * Y])
    A = softmax_sample_batch(logits, rng)
    return _plim_from_stats(Cs, _precompute_stats(T, Y, A))


def compute_mbias(Cs):
    print("  Precomputing M-bias (N=2M)...")
    rng = np.random.default_rng(999)
    N = N_POP
    U1 = rng.normal(0, 1, N)
    U2 = rng.normal(0, 1, N)
    T = 1.0 * U1 + rng.normal(0, 1, N)
    Y = 1.0 * T + 1.0 * U2 + rng.normal(0, 1, N)
    logits = np.column_stack([np.zeros(N), 0.8 * U1 + 0.6 * U2, -0.5 * U1 + 0.9 * U2])
    A = softmax_sample_batch(logits, rng)
    return _plim_from_stats(Cs, _precompute_stats(T, Y, A))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. IV (naive OLS): Z(binary)→A(3-class)→Y, U→{A,Y}.
#   Naive OLS: Y ~ 1+D*₁+D*₂.  Coeff on D*₁.  True β₁=1.0.
# ═══════════════════════════════════════════════════════════════════════════════
def compute_iv(Cs):
    print("  Precomputing IV naive OLS (N=2M)...")
    rng = np.random.default_rng(999)
    N = N_POP
    Z_iv = rng.binomial(1, 0.5, N).astype(float)
    U = rng.normal(0, 1, N)
    logits = np.column_stack([
        np.zeros(N),
        1.0 * Z_iv + 0.8 * U,
        -0.5 * Z_iv + 0.6 * U,
    ])
    A = softmax_sample_batch(logits, rng)
    Y = (1.0 * (A == 1).astype(float)
         - 0.5 * (A == 2).astype(float)
         + 1.0 * U + rng.normal(0, 1, N))

    p = np.array([np.mean(A == j) for j in range(3)])
    EY_A = np.array([np.mean(Y * (A == j)) for j in range(3)])

    n = len(Cs)
    q = Cs @ p
    EYDs = Cs @ EY_A
    EY_cond = EYDs / np.clip(q, 1e-10, None)
    return EY_cond[:, 1] - EY_cond[:, 0]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. IV (Wald estimator): Z(binary)→A→Y, U→{A,Y}.
#   Wald = Cov(A*,Y)/Cov(A*,Z).  52.6% sign reversal.
# ═══════════════════════════════════════════════════════════════════════════════
def compute_iv_wald(Cs):
    print("  Precomputing IV Wald (N=2M)...")
    rng = np.random.default_rng(999)
    N = N_POP
    Z_iv = rng.binomial(1, 0.5, N).astype(float)
    U = rng.normal(0, 1, N)
    logits = np.column_stack([
        np.zeros(N),
        1.0 * Z_iv + 0.8 * U,
        -0.5 * Z_iv + 0.6 * U,
    ])
    A = softmax_sample_batch(logits, rng)
    Y = (1.0 * (A == 1).astype(float)
         - 0.5 * (A == 2).astype(float)
         + 1.0 * U + rng.normal(0, 1, N))

    D1_true = (A == 1).astype(float)
    cov_D1Z = np.mean(D1_true * Z_iv) - np.mean(D1_true) * np.mean(Z_iv)
    cov_YZ = np.mean(Y * Z_iv) - np.mean(Y) * np.mean(Z_iv)
    tau_true_wald = cov_YZ / cov_D1Z

    p_z0 = np.array([np.mean((A == j) & (Z_iv == 0)) / np.mean(Z_iv == 0) for j in range(3)])
    p_z1 = np.array([np.mean((A == j) & (Z_iv == 1)) / np.mean(Z_iv == 1) for j in range(3)])
    EY_z0 = np.mean(Y[Z_iv == 0])
    EY_z1 = np.mean(Y[Z_iv == 1])

    n = len(Cs)
    q_z0 = Cs @ p_z0
    q_z1 = Cs @ p_z1
    first_stage = q_z1[:, 1] - q_z0[:, 1]
    reduced_form = EY_z1 - EY_z0
    wald = reduced_form / np.where(np.abs(first_stage) > 1e-10, first_stage, np.nan)
    return wald, tau_true_wald


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE ALL
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing distributions...")
plim_conf = compute_confounding(Cs)
plim_exp1, plim_exp2 = compute_exposure(Cs)
plim_med = compute_mediation(Cs)
plim_col = compute_collider(Cs)
plim_mb = compute_mbias(Cs)
plim_iv = compute_iv(Cs)
plim_iv_wald, tau_wald = compute_iv_wald(Cs)

BETA_EXP = np.array([0.0, 1.0, -0.5])
sf_exp_combined = np.nanmean(
    (plim_exp1 * BETA_EXP[1] < 0) | (plim_exp2 * BETA_EXP[2] < 0)
)
d_wald = plim_iv_wald[np.isfinite(plim_iv_wald)]
sf_iv_wald = np.mean(d_wald * tau_wald < 0)

print(f"\nExposure combined sign-flip rate: {sf_exp_combined:.1%}")
for name, data, true in [
    ("Confounding", plim_conf, 1.0),
    ("Exposure(γ1)", plim_exp1, 1.0),
    ("Mediation", plim_med, 1.0),
    ("Collider", plim_col, 1.0),
    ("M-bias", plim_mb, 1.0),
    ("IV(OLS)", plim_iv, 1.0),
    ("IV(Wald)", plim_iv_wald, tau_wald),
]:
    d = data[np.isfinite(data)]
    sf = np.mean(d * true < 0)
    print(f"  {name:15s}: median={np.median(d):.4f}, mean={np.mean(d):.4f}, "
          f"sf={sf:.1%}, range=[{np.percentile(d, 1):.3f}, {np.percentile(d, 99):.3f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT — 2×4 grid, per-panel x-axis, regime bands, KDE + histogram
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPlotting...")
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

panels = [
    dict(label='(a) Confounding', sub='Residual confounding',
         data=plim_conf, tau=1.0, sub_color=DANGER),
    dict(label='(b) Exposure', sub=f'{sf_exp_combined:.1%} β₁ sign reversal',
         data=plim_exp1, tau=1.0, sub_color=DANGER, include_zero=True),
    dict(label='(c) Mediation', sub='Direct-effect amplification',
         data=plim_med, tau=1.0, sub_color=DANGER),
    dict(label='(d) Collider', sub='Collider bias attenuated',
         data=plim_col, tau=1.0, sub_color=SAFE),
    dict(label='(e) M-bias', sub='Near-immune',
         data=plim_mb, tau=1.0, sub_color=SAFE),
    dict(label='(f) IV (naive OLS)', sub='Endogeneity compounded',
         data=plim_iv, tau=1.0, sub_color=DANGER),
    dict(label='(g) IV (Wald)', sub='52.6% sign reversal',
         data=plim_iv_wald, tau=tau_wald, sub_color=DANGER, include_zero=True),
]

axes.flat[7].set_visible(False)

for ax, cfg in zip(axes.flat, panels):
    data = cfg['data']
    tau = cfg['tau']
    d = data[np.isfinite(data)]

    # ── Per-panel x-range ────────────────────────────────────────────────────
    p_lo, p_hi = np.percentile(d, [0.5, 99.5])
    x_lo = min(p_lo, tau)
    x_hi = max(p_hi, tau)
    if cfg.get('include_zero', False):
        x_lo = min(x_lo, -0.2)
    span = x_hi - x_lo
    pad = max(0.1 * span, 0.02)
    x_range = (x_lo - pad, x_hi + pad)

    # ── Regime bands ─────────────────────────────────────────────────────────
    if x_range[0] < 0:
        ax.axvspan(x_range[0], 0, alpha=0.07, color=DANGER, zorder=0)
    safe_w = 0.05 * abs(tau)
    ax.axvspan(max(tau - safe_w, x_range[0]), min(tau + safe_w, x_range[1]),
               alpha=0.12, color=SAFE, zorder=0)

    # ── Histogram ────────────────────────────────────────────────────────────
    d_plot = d[(d >= x_range[0]) & (d <= x_range[1])]
    ax.hist(d_plot, bins=60, range=x_range, density=True,
            alpha=0.25, color=DIST, edgecolor='none', zorder=2)

    # ── KDE ──────────────────────────────────────────────────────────────────
    if len(d_plot) > 50:
        try:
            kde = gaussian_kde(d_plot, bw_method='scott')
            xk = np.linspace(x_range[0], x_range[1], 400)
            yk = kde(xk)
            ax.plot(xk, yk, color=DIST, lw=1.5, zorder=3)
            ax.fill_between(xk, yk, alpha=0.12, color=DIST, zorder=2)
        except Exception:
            pass

    # ── Reference line ───────────────────────────────────────────────────────
    ax.axvline(tau, color=REF, ls='--', lw=1.2, alpha=0.7, zorder=4)

    # ── Title + subtitle ─────────────────────────────────────────────────────
    ax.set_title(cfg['label'], fontsize=12, fontweight='bold', pad=16)
    ax.text(0.5, 1.01, cfg['sub'], transform=ax.transAxes,
            fontsize=9, ha='center', va='bottom', color=cfg['sub_color'],
            fontstyle='italic')

    # ── Stats box ────────────────────────────────────────────────────────────
    sf_rate = np.nanmean(d * tau < 0)
    viol_rate = np.nanmean(np.abs(d - tau) / max(abs(tau), 1e-8) > 0.10)
    box_txt = (f'sign_flip={sf_rate:.1%}\n'
               f'viol>10%={viol_rate:.1%}\n'
               f'n={len(d):,}')
    ax.text(0.97, 0.95, box_txt, transform=ax.transAxes,
            fontsize=7.5, ha='right', va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85,
                      ec='#cccccc', lw=0.5))

    # ── Axis limits ──────────────────────────────────────────────────────────
    ax.set_xlim(x_range)
    ax.set_ylim(bottom=0)

    # ── τ_true label (after ylim is set) ─────────────────────────────────────
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    offset = 0.03 * (x_range[1] - x_range[0])
    t = ax.text(tau + offset, 0.75, r'$\tau_{true}$', fontsize=8,
                color=REF, alpha=0.6, va='center', transform=trans)
    t.set_path_effects([pe.withStroke(linewidth=2, foreground='white')])

for ax in axes[1, :]:
    if ax.get_visible():
        ax.set_xlabel(r'plim $\hat{\tau}$', fontsize=11)
for ax in axes[:, 0]:
    ax.set_ylabel('Density', fontsize=13)

fig.suptitle(
    'Bias Distribution Under K=3 Non-Differential Misclassification\n'
    '100,000 Random Confusion Matrices (Dirichlet(1,1,1) per column)',
    fontsize=13, y=1.04)

plt.tight_layout(h_pad=2.5, w_pad=1.2)

out_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(out_dir, 'fig2_bias_distribution.pdf')
png_path = os.path.join(out_dir, 'fig2_bias_distribution.png')
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=300)
plt.close()

print(f"\nSaved: {pdf_path}")
print(f"Saved: {png_path}")
print(f"PDF size: {os.path.getsize(pdf_path) / 1024:.0f} KB")
print(f"PNG size: {os.path.getsize(png_path) / 1024:.0f} KB")
