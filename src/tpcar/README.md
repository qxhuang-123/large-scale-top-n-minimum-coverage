# TPCAR implementation note

Production experiments import `tpcar_core_fast`, a pybind11 C++ extension that
builds the implicit exchange graph and runs Dijkstra with Johnson potentials.
The historical Python-only OP prototype is included for algorithm inspection,
but it materializes a dense selection matrix and is unsuitable for the largest
datasets.

The extension source was referenced by experiment runners but was not present
in the manuscript directory or located dataset folders. It must be added
before claiming a fully self-contained artifact. Until then, the runner files
and reported outputs are traceable, but the optimized TPCAR binary cannot be
rebuilt from this repository alone.

