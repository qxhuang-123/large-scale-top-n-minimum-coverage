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

