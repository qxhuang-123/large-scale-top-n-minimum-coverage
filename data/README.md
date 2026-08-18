# Data availability and layout

This repository does not redistribute raw Amazon/Yelp records or the
LETTER-generated dense score matrices because those files exceed ordinary
GitHub limits. Download source records under their original terms and obtain
the derived score matrices from the corresponding author or a data release.
Place them under `data/raw/<dataset>/` as configured in
`config/datasets.example.yaml`.

Known ratings are excluded before candidate filtering. Per user, retain the
first `ceil(alpha * eligible_items)` items in descending score order, with at
least `N` candidates.

```bash
python tools/make_data_manifest.py data/raw data/data_manifest.local.csv
python tools/prepare_shared_inputs.py
```

This creates `data/processed/<dataset>/scores_float32.npy` and one CSR cache.
MCF, TPCAR, and BGCR use only those same files. Keep large matrices outside
Git. Archive their relative names, sizes, and SHA-256 checksums with a release
so other researchers can verify inputs.

