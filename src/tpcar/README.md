# Build the TPCAR extension

The production experiments import `tpcar_core_fast`, the included pybind11
C++ extension that builds the implicit exchange graph and runs Dijkstra with
Johnson potentials.

```bash
python -m pip install -r requirements-build.txt
cd src/tpcar
python setup.py build_ext --inplace
```

Use a C++17 compiler matching the active Python architecture. On Windows, use
the corresponding MSVC Build Tools. Compiled `.pyd` files are excluded so the
repository does not distribute opaque, platform-specific binaries.

## Run the SO exact experiment

Place the SO cache files at:

- `cache/so/SO_scores_float32.npy`
- `cache/so/SO_cand_pos_top40_seed20260704_v2.npz`

Then run:

```bash
python src/tpcar/run_tpcar_so.py --d-percentages 0 20 40
```

Results are written under `outputs/tpcar/so/`. The runner uses the shared
`tpcar_core_fast` extension above and preserves the iteration-history,
spreadsheet, JSON, and figure outputs used for the SO exact experiment.

