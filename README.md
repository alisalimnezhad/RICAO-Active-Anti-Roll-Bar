# Worst-case transient bounds for constrained electrohydraulic active anti-roll-bar control of heavy vehicles

## Overview

This repository contains the Python implementation and frozen inputs used to reproduce the numerical studies reported in the associated article. It includes the ten-state electrohydraulic vehicle model, reference-response data, controller synthesis and support-function bound calculations, operating-envelope and delay audits, nonlinear screening utilities, and the final plant/controller stress records. The article source and its tables and figures are included for claim-to-output traceability.

The principal guarantees are pointwise numerical bounds over the declared steering family and evaluated linear plant bank. They are not interval enclosures between plant parameters, formally verified arithmetic proofs, nonlinear rollover guarantees, or experimental validation.

## Repository structure

```text
Github_RICAO/
├── README.md
├── .gitignore
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── validation/                         # Model, frozen parameters, reference data, checks
├── certified_aarb_v2/                  # Nominal transient-bound reconstruction
├── certified_aarb_v3/                  # Axle fairness, current matching, multi-model study
├── certified_aarb_v4/                  # 170-plant operating-envelope study
├── certified_aarb_v5_rear_audit/       # Rear reserve, error budget, delay audits
├── certified_aarb_v8_sensitivity/      # FIR, steering, hydraulic, and delay sensitivity
├── certified_aarb_v9_delay_aware/      # 5 ms redesign and nonlinear critical-case screen
├── certified_aarb_v10_presentation/    # Complexity and claim-scope asset generation
├── certified_aarb_v14_combined_robustness/  # 1,536-record combined stress audit
├── certified_aarb_v14_release/        # Final audit gate
├── data/publication_tables/            # Machine-readable tables from the article
└── paper/                               # Main/supplement LaTeX sources and required assets
    ├── Main_Manuscript.tex
    ├── Supplementary_Material.tex
    ├── references.bib
    ├── figures/
    └── tables/
```

The model and fixed parameter set are in `validation/vehicle_model.py` and `validation/parameters.py`. Reference trajectories and steering inputs are frozen CSVs under `validation/data/`, with source notes in `validation/data/provenance.json`. Stage-specific final JSON/CSV records and controller coefficients are kept under each study's `outputs/` directory. The merged v14 stress records and their sampling inputs are included in the release; the v14 run below can reproduce them from the frozen study code and inputs.

This is a Python research-code release; the final analysis does not require MATLAB or Simulink, and no Simulink model is part of the release. SciPy's `linprog(method="highs")` interface supplies the LP solver; no separate HiGHS installation is required.

## Requirements

- Python 3.10 or newer.
- NumPy 2.x, SciPy 1.13 or newer (below 2.0), and Matplotlib 3.8 or newer (below 4.0).
- Pytest 8.x or 9.x to run the regression tests.
- A SciPy build with its HiGHS-backed `linprog` method available.

Install the supported dependency ranges from the repository root:

```bash
python -m pip install -r requirements.txt
```

The archived verification environment was CPython 3.13.15, NumPy 2.4.1, SciPy 1.17.0, Matplotlib 3.10.8, and pytest 9.1.1. Versions are recorded in the relevant study outputs; other supported versions may produce small floating-point differences.

To compile the article and supplement, install a TeX Live distribution with `elsarticle`, `latexmk`, and the packages imported by the sources, then run from `paper/`:

```bash
latexmk -pdf -interaction=nonstopmode Main_Manuscript.tex
latexmk -pdf -interaction=nonstopmode Supplementary_Material.tex
```

The generated PDFs and TeX build intermediates are ignored and are not part of the source release.

## Reproduce the studies

Run commands from the repository root, in the listed order. Some synthesis and full-bank steps are computationally intensive and may take substantially longer than the regression tests.

### Baseline and nominal bound

```bash
python validation/run_validation.py
python certified_aarb_v2/run_reproduction.py
python certified_aarb_v2/verify_results.py
```

### Axle-wise comparisons and operating envelope

```bash
python certified_aarb_v3/run_stage3.py
python certified_aarb_v3/verify_stage3.py
python certified_aarb_v3/make_artifacts.py
python certified_aarb_v4/run_design.py
python certified_aarb_v4/run_envelope.py
python certified_aarb_v4/verify_envelope.py
python certified_aarb_v4/make_artifacts.py
```

### Rear-reserve audit, sensitivity, and delay-aware redesign

```bash
python certified_aarb_v5_rear_audit/run_error_budget.py
python certified_aarb_v5_rear_audit/run_reserve_design.py --target 0.01 --guard 0.00025
python certified_aarb_v5_rear_audit/run_reserve_design.py --target 0.02 --guard 0.00042
python certified_aarb_v5_rear_audit/run_reserve_design.py --target 0.05 --guard 0.00095
python certified_aarb_v5_rear_audit/make_reserve_comparison.py
python certified_aarb_v5_rear_audit/verify_reserve_designs.py
python certified_aarb_v8_sensitivity/run_sensitivity.py
python certified_aarb_v8_sensitivity/make_artifacts.py
python certified_aarb_v8_sensitivity/verify_sensitivity.py
python certified_aarb_v9_delay_aware/run_delay_aware.py
python certified_aarb_v9_delay_aware/run_nonlinear.py
python certified_aarb_v9_delay_aware/make_artifacts.py
python certified_aarb_v9_delay_aware/verify_delay_aware.py
python certified_aarb_v10_presentation/build_presentation.py
python certified_aarb_v10_presentation/verify_presentation.py
```

The frozen study outputs let readers inspect coefficients and results without rerunning synthesis. The `run_*` commands recompute those studies from the included model and parameter files; the audit scope remains the one documented in the paper.

### Full combined plant/controller stress audit

```bash
python certified_aarb_v14_combined_robustness/run_all.py --force
```

This runs two resumable shards for the 512-point ±5% and 1,024-point ±10% Sobol banks, refines limiting cases, independently verifies critical records, regenerates the combined-audit table and figure, and runs the release checks. The bundled full record set is `certified_aarb_v14_combined_robustness/outputs/results.json`. Each record retains its Sobol input coordinates together with the evaluated plant and controller diagnostics. Shard JSONL files are temporary checkpoints and are excluded from the release. Summary, protocol, refinement, independent-verification, and hash records are included alongside the final results. Omit `--force` to resume an interrupted run.

### Regression tests

Run test files in separate pytest processes because several historical study modules use the same local module name (`study.py`):

```bash
for test in \
  validation/test_validation.py \
  certified_aarb_v2/test_reproduction.py \
  certified_aarb_v3/test_stage3.py \
  certified_aarb_v4/test_stage4.py \
  certified_aarb_v5_rear_audit/test_error_budget.py \
  certified_aarb_v5_rear_audit/test_reserve_designs.py \
  certified_aarb_v8_sensitivity/test_sensitivity.py \
  certified_aarb_v9_delay_aware/test_delay_aware.py \
  certified_aarb_v10_presentation/test_presentation.py \
  certified_aarb_v14_combined_robustness/test_robustness.py \
  certified_aarb_v14_release/test_release.py; do
  python -m pytest -q "$test" || exit 1
done
```

The complete frozen outputs needed by these regression tests are included; the suite does not require committed PDF builds or temporary artifact manifests.

## Inspect the data

- `data/publication_tables/` contains the 19 CSV tables and manifest prepared from the final article tables.
- `validation/data/` contains the fixed steering input, reference response traces, expected regression metrics, and their provenance.
- Each study's `outputs/` contains its frozen final result JSON/CSV records and controller coefficient CSVs. Files with `partial` in the name are transient checkpoints and are excluded from the clean release; rerunning a study may recreate them locally.
- The v14 `results.json` contains all 1,536 full records, including sampled input coordinates; shard files are disposable resumable checkpoints and are not committed.

## Citation and license

Please cite the associated article and the exact repository release used:

> Ali Salimnezhad and Masoud Masih-Tehrani. “Worst-case transient bounds for constrained electrohydraulic active anti-roll-bar control of heavy vehicles.” *Results in Control and Optimization*. Publication year and DOI to be added after publication.

`CITATION.cff` provides machine-readable citation metadata. Replace the DOI placeholder in this guidance with the article and repository DOI after they are assigned.

The software is released under the MIT License in `LICENSE`. Before public distribution, the authors should confirm they hold the rights to license all source code and data. In particular, the reference-response CSVs are digitized from previously published figures; see `validation/data/provenance.json` and verify that their redistribution is permitted. The LaTeX source still carries the submission-stage data-availability wording (“available ... upon reasonable request”); update it with the permanent repository URL/DOI when the public deposit is made.