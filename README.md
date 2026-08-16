# Reproducibility code

The paper uses the current `main` branch. The implementation is organized by method:

- `methods/TPCAR/`: exact TPCAR runners for OP, Yelp, VG, TG, and SO.
- `methods/BGCR/`: greedy BGCR runners for the five datasets.
- `methods/MCF/`: minimum-cost-flow runners.
- `methods/Gurobi/`: binary Gurobi models.
- `methods/Gurobi-LP/`: the same totally-unimodular model solved as an LP relaxation.

All methods use `alpha=0.40`, `N=10`, candidate universe `I^alpha`, and
`D_target=ceil(|I^alpha|*p/100)`. Candidate K is `ceil(alpha*valid_items)`
(with the Top-N lower bound where applicable). Ties are score descending,
then item id ascending.

## Data

The repository contains code, configuration templates, checksums, and result
schemas. Raw review data and predicted score matrices are not committed because
of size and redistribution restrictions. Put the five datasets under
`data/raw/<dataset>/` and configure paths in `config/datasets.yaml`.

The old `legacy/` duplicate runner directory has been removed; use only the
method directories above.