# Minimum-coverage Top-N recommendation: TPCA and DGCA

This repository accompanies *Large-Scale Top-N Recommendation with Minimum
Item Coverage: A Network-Flow Reformulation and Specialized Algorithms*.

**Current revised algorithms (22 October 2026):**

- **TPCA**: exact coverage-gap augmentation with local shortest-path repair,
  boundary queries, donor aggregation, and terminal aggregation.
- **DGCA**: current-minimum-loss direct replacement with a verified user heap.
- **Baselines**: explicit MCF, Gurobi MIP/LP, and R/A/B/C/D direct-reallocation controls.

The maintained revised experiment package is in [`current/`](current/).
The older `methods/` and `src/` directories are retained for the COR submission;
their runners and numerical results are not the revised TPCA/DGCA benchmarks.
The previous repository README is preserved in
[`paper/LEGACY_COR_README.md`](paper/LEGACY_COR_README.md).

## Setup and native builds

The measured benchmark workflow uses Windows x64 and Python 3.9. Install a C++17
compiler (MSVC Build Tools on Windows) before building the extensions.

```powershell
python -m pip install -r current/requirements.txt
python current/build_revised.py
python current/validate_revised.py
```

The validation command uses only small synthetic graphs, compares exact
objectives with independent LP solutions, and checks DGCA decisions and
R/A/B/C/D equivalence. It does not run the five large paper instances.
Gurobi is needed only for its own baseline; validation skips it by default.

## Shared inputs and five-dataset entries

All methods consume the **same immutable candidate graph and integer scores**.
Set the directory containing `OP/ready.json`, `YELP/ready.json`, etc.:

```powershell
$env:TOPN_SHARED_GRAPH_ROOT = "D:/topn_shared_graphs"
python TPCA-OP.py --repeats 1
python DGCA-OP.py --repeats 1
python MCF-OP.py --repeats 1
```

Replace `OP` with `YELP`, `VG`, `TG`, or `SO`. Default settings are N=10,
the 100% nominal coverage target, and one independent run. For target sweeps:

```powershell
python TPCA-TG.py --percentages 0 20 40 60 80 100 --repeats 1
python DGCA-Strategy-VG.py --help
python TPCA-Ablation.py --help
python TPCA-Gaps.py --help
```

PyCharm users can open the repository and create Python run configurations
for the identically named root entry points. No absolute interpreter path is
stored in the public package.

The four required arrays per shared graph are `user_indptr.npy`,
`user_items.npy`, `scores_integer.npy`, and `scores_original.npy`, together with
`graph.json` and dataset-level `ready.json`. `current/shared_graphs.json` records
the five tested graph IDs and sizes. Candidate arrays total about 9.90 GB and
are not ordinary Git files. Derived score matrices are linked in the paper's
[Zenodo data record](https://doi.org/10.5281/zenodo.21996444); these score exports
are distinct from the prepared immutable graph caches. Small recorded exchange-witness arrays are included for verification. No raw
Amazon/Yelp data are redistributed. Do not substitute a different cache without recording a new
graph ID and rerunning the corresponding comparisons.

The public raw-input adapters use `TOPN_DATA_ROOT` and
`TOPN_CANDIDATE_CACHE_ROOT`; see `current/PORTABILITY_NOTES.md`. Reusing the
tested graph caches avoids reinterpreting dataset-specific eligibility.

## Figures, tables, and recorded evidence

`current/Figures/` contains Python scripts, frozen summaries and trajectories,
and table sources. Plotting does not call optimization algorithms:

```powershell
python current/Figures/plot_F02_independent_runtime_r1.py
python current/Figures/plot_F03_peak_memory_r1.py
python current/Figures/plot_F04_coverage_quality.py
python current/Figures/rebuild_supplementary_tables.py
```

The manuscript layout helper `compact_manuscript_supplements.py --paper PATH`
splits the recorded S3 table at a dataset boundary without changing its cells.
The manuscript itself is not published by this code update.

Main timings use the first independently started run, including method-specific
construction and solution time. Repeats/medians, cumulative native trajectories,
and resource stops remain separately identified. An unattained DGCA target is
direct-neighborhood exhaustion, not a proof of global infeasibility. V0 remains
deferred and is not used for a controlled speedup claim.

The revised input provenance distinguishes LETTER-based OP/Yelp/VG/TG scores
from SO biased-matrix-factorization scores. OP is not described as uniformly
unobserved-only. The exact TG exporter and VG checkpoint-to-export link remain
documented provenance limitations; this repository update does not claim to
have recovered them.
