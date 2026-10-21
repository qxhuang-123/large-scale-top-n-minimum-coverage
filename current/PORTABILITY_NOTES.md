# Compatibility and provenance

The C++ sources listed in CORE_SOURCE_SHA256.json are byte-identical to the
locally tested TPCA/DGCA implementations. Internal TPCAR, BGCR-D, state_source,
and dynamic_user_heap names remain compatibility identifiers in saved records.
They do not select the original COR algorithms.

Public-only adapter changes: optional TOPN_SHARED_GRAPH_ROOT for immutable graph
caches, relative/environment-based raw-input locations, and --skip-gurobi for
the small exact-adapter check. These do not change candidates, rounding, ties,
objectives, augmentations, or selection rules. Original benchmark numbers were
not recomputed with these adapter changes.

The benchmark supervisors measure Windows process working sets and require
Windows. The native setup scripts include other compiler flags, but the full
benchmark workflow is validated on Windows, not asserted portable to all OSes.
No opaque .pyd binaries or large score/candidate arrays are committed.
