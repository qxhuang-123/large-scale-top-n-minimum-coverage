# Reported experiment data

Table4_TableS1_paper.csv and Table6_paper.csv reproduce the submitted manuscript tables. Times are means of three independently started process measurements per completed setting. main_paper_runs.csv, DGCA_paper_runs.csv and strategy_paper_runs.csv provide the underlying measurements; total_sd_s is the sample standard deviation. Resource-guard stops are status records, not completed three-run means. MG and Infeas. are preserved as printed in the paper.

reported_run_manifest.json identifies the three DGCA records used in each mean. All have measured Build and complete Solve times. The separately scheduled warmup jobs are excluded. Original recorded run identifiers are preserved. Other historical observations are archived locally; exclusion from this report does not relabel them as warmups.

Figure2_memory.csv and figures/Figure2_memory.* are the updated manuscript memory data and figure. They use the first reported repetition, identified by memory_run_manifest.json. The metric is worker-process peak working set including resident mapped pages, not private bytes or a three-run mean. The single-run TPCA ablation is not averaged into these tables.

Predictor-source evidence: current SO scores closely match an archived biased matrix-factorization model in the input audit. The original exporter and the reason for choosing that predictor have not been recovered. The available evidence does not verify a LETTER origin for SO. OP eligibility evidence is included in input_audit_20261107.json.
