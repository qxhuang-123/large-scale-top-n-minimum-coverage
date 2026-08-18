# Reproducibility results

The JSON files in this directory are the result artifacts used to reproduce
the reported BGCR and TPCAR summaries. All current BGCR runs use the same
candidate cache, candidate fraction `alpha=0.40`, seed `20260704`, and the
deterministic tie rule: score descending, then item id ascending.

## C++ BGCR reruns

`bgcr_cpp/{OP,Yelp,VG,TG,SO}/` contains the full C++-core BGCR reruns for
every `N` in `{10,15,20,25,30}`.  Each artifact records deterministic Top-N
initialization (score descending, item id ascending) followed by the C++
batched direct-reallocation heuristic at all six nominal coverage targets.
The source and build entry points are in `src/bgcr/`.

## MCF at N=10

`mcf/n10/latest_run_status.md` records the latest shared-input explicit-network
MCF rerun. The OP/Yelp and VG JSON output artifacts are retained alongside the
run record; the SO transposed-matrix run is deliberately excluded.

## TPCAR at N=10

`tpcar/n10/shared_initial/` contains the authoritative shared Top-N initial
solution for every dataset. These files are the exact initialization records
that match BGCR: they use the same score matrix, candidate cache, and tie rule.

`tpcar/n10/final_shared/` contains the final complete N=10 TPCAR result JSON
artifacts for OP, Yelp, VG, and TG. They use the shared candidate sets,
deterministic score-descending/item-id-ascending tie rule, shared Top-N
initialization, and the target definition
`D = ceil(q * |I^alpha|)`. Yelp's 100% nominal target is infeasible; its JSON
therefore records the exact attained boundary coverage of 7,008.
