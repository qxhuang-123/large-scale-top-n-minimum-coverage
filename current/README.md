# TPCA and DGCA

Open this directory in PyCharm. The maintained entries are TPCA-OP.py and DGCA-OP.py; replace OP with YELP, VG, TG or SO. Use Python 3.9 on Windows x64 with the bundled modules, or rebuild the C++17 extensions for your Python version. Install requirements.txt first.

Both algorithms accept --n 10 --repeats 3 --percentages 0 20 40 60 80 100. DGCA calls dgca_stage2 in dynamic_user_grouped_cached mode. The reference core is a dependency of the current implementation and its validation.

Configure 数据路径.json with the immutable candidate graph folders. Each graph contains graph.json, user_indptr.npy, user_items.npy, scores_integer.npy and scores_original.npy. Large derived score and candidate arrays are provided separately from the code supplement. The synthetic example under examples/tiny_graph is included.

## Reported tables and figure

records/Table4_TableS1_paper.csv and records/Table6_paper.csv contain the manuscript values. records/main_paper_runs.csv, DGCA_paper_runs.csv and strategy_paper_runs.csv provide the measurements used in the means; each completed reported configuration uses three process runs. The supplied sample standard deviations use ddof=1. Separately scheduled warmups are not included. Record IDs are in reported_run_manifest.json.

records/Figure2_memory.csv and records/figures/Figure2_memory.* contain the updated memory plot. Each value is the peak worker-process working set, including resident mapped pages, from the first reported repetition identified in memory_run_manifest.json. It is not a mean across repetitions. Table5_tpca_component.csv is the single-run component experiment. Tables 2 and 3 are also included as CSV.

Total is method-specific Build plus the complete Solve call. Build includes input assembly and validation of the method-specific representation. Solve includes Top-N initialization and algorithm-specific index construction. Shared score prediction/candidate preprocessing and loading, final solution auditing and export are excluded. Details are in records/measurement_protocol.json.

Verify the reported CSV means, standard deviations, repetition counts and DGCA memory values without running the solvers:

    python records/verify_paper_data.py

## Build and validation

Run setup_compact.py from 公共代码/TPCA, setup.py from 公共代码/DGCA/latest, and setup.py from 公共代码/TPCA/component_comparison to build the corresponding native extensions. MCF/Gurobi and component-control sources are retained for the comparisons. Gurobi requires its own installation and license.

Run 公共代码/DGCA/latest/validate_stage2.py for the small-graph exhaustive greedy comparison. Its archived result covers 600 graphs and 4236 calls. examples/tiny_graph supplies a small runnable input; full-scale data are separate.
