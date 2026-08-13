# Experiment-to-code crosswalk

| Manuscript component | Datasets | Code |
| --- | --- | --- |
| Candidate filtering, `alpha=40%` | OP, Yelp, VG, TG, SO | `src/preprocessing/` and candidate builders in each runner |
| Binary Gurobi model | OP (scalability attempts on larger data) | `src/gurobi/run_gurobi.py` with LP mode disabled |
| Gurobi-LP exact benchmark | OP | `src/gurobi/run_gurobi.py` |
| Explicit-network MCF | OP, Yelp, VG, TG; SO memory-infeasible | `src/mcf/run_mcf_op_so.py` and `src/mcf/run_mcf_op_yelp.py` |
| TPCAR exact algorithm | OP, Yelp, VG, TG, SO | `src/tpcar/run_tpcar_op.py`, `src/tpcar/run_tpcar_so.py`, and the shared C++ extension |
| BGCR heuristic | OP, Yelp, VG, TG, SO | `src/bgcr/run_bgcr_five_datasets.py` |
| List-length comparison | OP, Yelp, VG, TG, SO | BGCR runner with `N={10,15,20,25,30}` |
| Sensitivity to coverage ratio | OP, Yelp, VG, TG | exact runners at six nominal ratios |
| Sensitivity to `alpha` | OP, Yelp, VG, TG | `src/sensitivity/` |

The manuscript reports raw dataset statistics before candidate filtering and feasibility preprocessing, while optimization instances may be slightly smaller after removing users or items with no eligible edge. Always report both raw and post-filter candidate-graph shapes.

