# Processed experiment results

- `bgcr/` contains machine-readable JSON outputs for all five datasets.
- `exact/` contains the retained TPCAR workbooks for OP, Yelp, VG, and TG.
- `gurobi/` contains the OP binary and LP benchmark workbooks.
- `vg_tg_alpha_sensitivity.csv` contains the available VG/TG retention-rate
  sensitivity results.

These files are derived experiment outputs, not raw ratings or predicted score
matrices. Workbooks are retained without value or formatting changes for audit
traceability. Future releases should add CSV/JSON exports for every manuscript
table and figure while keeping these originals as provenance records.
