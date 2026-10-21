# VG、TG补跑完成：五数据集策略对照表5

新增加VG、TG各R/A/B/C/D一次，共10次，全部正常结束。原OP、Yelp、SO仍使用各自第一次运行；随机扫描统一种子2026101901。表5共25行。

## 文件

- Lean_Main_Preview.pdf：更新后的8页正文图表预览，表5为五数据集，表6为五数据集模块消融；消融仅保留表格，正文不放图5—7。
- tables/T05_strategy_full.png：600 dpi完整表5；同名PDF/SVG为矢量版，CSV为数据，TeX为booktabs双栏表。
- tables/T06_module_ablation_r1：五数据集模块消融核心时间，数据保持原批次首次记录。
- tables/TS1_module_ablation_detailed_r1：模块消融补充明细，保留核心时间、总时间、峰值内存。
- data/strategy_runs.csv：保留原66条和新增10条，共76条；绘图只选择repeat=1共25条。
- data/strategy_original_66_runs.csv：原批次冻结副本；data/strategy_extension_10_runs.csv：补跑10条。
- data/raw_strategy_extension：每次的结果、任务、交换事件、推荐边、时间轨迹、内存轨迹和日志。
- data/strategy_five_dataset_audit.json：新旧主实验输入一致性、精确前沿核对及A/B、C/D逐事件一致性检查。

## 统一设置

N=10，D为候选物品并集。候选图graph_id、初始覆盖和整数目标均逐项与原主比较核对。评分仍为rint(score×1e6)，Top-N仍按整数评分降序、物品ID升序破同分。原生二进制SHA256为f47819f2f464073ce9f2c5edd616c9fb5999b833a9ca1b5f742d73f5cf4caf1d，与原策略对照相同。

R随机排列初始未覆盖物品一次；每次访问该物品仍选择当前最小损失可行替换。A/B保留初始损失固定排序并单次扫描，B采用供给端聚合。C/D每轮选择当前全局最小损失直接替换；C使用物品堆，D使用用户堆与终端聚合。A/B与C/D分别得到完全相同交换序列。

不同选择策略允许达到不同覆盖和得分。VG、TG各版本终止前直接替换邻域均已耗尽；未达到请求目标不能据此推断全局不可行。随机扫描的终止覆盖高于另两类策略，是本次实际结果，不作事后更换种子。表中Loss相对于初始Top-N得分，不能在不同K下仅凭损失数值断言质量优劣；原始数据另保留同K精确参考差距。

Core含原生初始化及求解；Total还包含实际方法构建及输入验证，公共图载入和最终导出在外。各数据集来自各自正式批次，未用主比较或模块消融较快时间替换。首轮计时是描述性结果，不报告虚构误差棒或显著性。

## 复现与PyCharm

绘图使用E:/PythonProject/.venv/Scripts/python.exe（requirements.txt固定版本），执行make_all.py可从冻结数据重建8页预览；执行table_T05_strategy_full.py只重建表5。绘图不调用求解器、不需要大图数组。verify_reproduction.py在隔离目录验证导出文件完全一致。

补跑求解入口位于E:/PythonProject/BGCR_D/run_strategy_extension_20261021.py，PyCharm已有BGCR-STRATEGY-VG、BGCR-STRATEGY-TG、BGCR-STRATEGY-VG-TG三个配置。BGCR-STRATEGY-TABLE5-5DATASETS仅从冻结数据重画五数据集表5。已完成求解记录会复用。experiment_code为本批代码快照，原图数组及共享依赖仍使用现有PythonProject接口。

## 正文可用简短叙述

We compare five selection-strategy implementations on all five datasets using the same candidate graph, Top-N initialization, rounded scores, target, and feasibility checks. The random scan uses a fixed seed and changes only the order in which initially uncovered items are visited. Table 5 reports each strategy's terminal coverage, score loss, and first-run construction and solution times.

The paired implementations A/B and C/D generate identical exchange sequences on every dataset, allowing their implementation costs to be compared without changing the selection rule. Dynamic minimum-loss selection does not uniformly maximize terminal coverage: on VG and TG, the random scan covers more items. We therefore report the attained coverage alongside score loss and runtime.

附表S2/S3分组排版：运行 rebuild_supplementary_tables.py --compile；原始CSV不变。单表PDF与600 dpi PNG位于 tables/TS2_progressive_grouped.* 与 TS3_repeated_times_grouped.*。
