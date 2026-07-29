# Data availability and layout

This repository does not redistribute raw Amazon/Yelp records or the
LETTER-generated dense score matrices. Download source datasets under their
respective terms, run LETTER, and place scores under `data/raw/<dataset>/` as
configured in `config/datasets.yaml`.

Known ratings are excluded before candidate filtering. Per user, retain the
first `ceil(alpha * eligible_items)` items in descending score order, with at
least `N` candidates.

```bash
python tools/make_data_manifest.py data/raw data/data_manifest.local.csv
python src/preprocessing/build_candidates.py --config config/datasets.yaml
```

Keep large matrices outside Git. Archive their relative names, sizes, and
SHA-256 checksums with a release so other researchers can verify inputs.

