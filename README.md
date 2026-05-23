# Multi-Class Annotation Errors in Causal Inference

Code and data for "Multi-Class Annotation Errors in Causal Inference: Topology-Dependent Bias, Sensitivity, and Correction" (EMNLP 2026 submission).

## Structure

- `annot_sensitivity/` — Core Python package (bias computation, ASV framework, correction methods)
- `artifacts/` — Experiment scripts and results
- `figures/paper/` — Figure generation scripts
- `docs/paper/` — LaTeX source

## Requirements

Python 3.10+

```bash
pip install numpy scipy pandas matplotlib seaborn statsmodels
```

## Quick Start

```bash
# Run core tests
python -m pytest annot_sensitivity/tests/

# Reproduce Table 1 (bias taxonomy)
python artifacts/compute_all_tasks.py

# Reproduce Figure 2 (bias distributions)
python figures/paper/gen_fig2.py
```

## Reproducing All Results

Each `artifacts/exp_*.py` and `artifacts/plan_*.py` script generates the corresponding experiment results. JSON files contain cached results.
