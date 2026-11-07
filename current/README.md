# TPCA and DGCA submission code

Open this directory in PyCharm. Use Python 3.9 on Windows x64 with the bundled native modules, or rebuild the C++17 extensions for your Python version. Install requirements.txt first.

TPCA-OP.py and DGCA-OP.py are the maintained production entries; replace OP with YELP, VG, TG or SO. Both accept --n 10 --repeats 3 --percentages 0 20 40 60 80 100. DGCA calls dgca_stage2 in dynamic_user_grouped_cached mode. C++ sources are included. The reference core is required by the current implementation and validation, and is not an alternative production entry.

Configure 数据路径.json with immutable candidate graph folders. Each folder contains graph.json, user_indptr.npy, user_items.npy, scores_integer.npy and scores_original.npy. Large score/candidate arrays are external; obtain the derived data from the study data deposit (the accompanying derived-data deposit). Public dataset sources alone do not reconstruct the archived prediction matrices. Use the graph metadata to verify identity.

Total = method-specific Build (including input assembly and validation) + complete native Solve (including Top-N and all algorithm-specific index construction). Shared loading/preprocessing and final auditing/export are excluded. Samples are independent processes; reported standard deviations are sample standard deviations. Historical memory is the worker peak working set including resident mapped pages, not private bytes.

records/paper_protocol contains the original complete batch; records/supplement_after_warmup contains later measurements and separately archived warmups. The historical adjusted summary selected original and supplementary observations retrospectively. It is preserved for traceability, not relabeled as an independent three-run batch. DGCA_all_measured_summary.csv uses all measured observations: four per target on OP/YELP/VG/TG and five on SO; the separately scheduled warmups are excluded. No new large experiments were run during packaging. The first original repetition supplies the memory column. Manuscript timing ratios must use the same chosen summary throughout.

Current TPCA component controls and MCF/Gurobi benchmark sources are retained under 公共代码 for evaluating the paper; they are not historical production variants. Build TPCA with 公共代码/TPCA/setup_compact.py, DGCA with 公共代码/DGCA/latest/setup.py, and component controls with 公共代码/TPCA/component_comparison/setup.py, each from its own directory.

DGCA validation: run 公共代码/DGCA/latest/validate_stage2.py. This compares the production implementation with exhaustive greedy choices and the reference core on small graphs. Archived validation is supplied. No claim of reproducing full-scale timings is made by an import test.

Unified replacement tables: records/Table4_TableS1_unified.csv and records/Table6_unified.csv. records/DGCA_memory_original_r1.csv fixes the memory repetition convention. These use all measured DGCA observations; three-run historical adjusted values remain traceable in records.
