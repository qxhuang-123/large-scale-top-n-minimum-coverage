# Reproducibility code

The paper uses the current `main` branch. The implementation is organized by method:

- `methods/TPCAR/`: exact TPCAR runners for OP, Yelp, VG, TG, and SO.
- `src/bgcr/`: current compiled C++ BGCR core, build entry point, and five-dataset runner.
- `methods/MCF/`: minimum-cost-flow runners.
- `methods/Gurobi/`: binary Gurobi models.
- `methods/Gurobi-LP/`: the same totally-unimodular model solved as an LP relaxation.

All methods use `alpha=0.40`, `N=10`, candidate universe `I^alpha`, and
`D_target=ceil(|I^alpha|*p/100)`. Candidate K is `ceil(alpha*valid_items)`
(with the Top-N lower bound where applicable). Ties are score descending,
then item id ascending.

The authoritative score matrices and CSR candidate caches are declared once in
`src/common/shared_experiment_inputs.py`. MCF, TPCAR initialisation, and C++
BGCR load this same input contract; MCF intentionally fails if a shared cache
is missing instead of creating a method-specific replacement.

## Data

The repository contains code, configuration templates, checksums, and result
schemas. Dense LETTER score matrices are not committed to ordinary GitHub
storage: several exceed GitHub's 100 MB per-file limit. Download public source
data under its original terms and obtain the derived score matrices from the
corresponding author or a data release. Place them under `data/raw/<dataset>/`
using the filenames in `config/datasets.example.yaml`.

Set `TOPN_DATA_ROOT` to a directory containing `raw/` and `processed/` (the
default is this repository's `data/`). Run `python tools/prepare_shared_inputs.py`
once. It creates the only score/candidate cache consumed by MCF, TPCAR and C++
BGCR, so every solver uses identical inputs without machine-specific paths.

Historical BGCR runners and superseded result artifacts have been removed; use
the current source directories above.
