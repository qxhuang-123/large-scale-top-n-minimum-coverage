# OP/Yelp逐层消融与覆盖缺口实验

本批在现有五图单模块消融结束后串行运行，BGCR随后接续。V0（每轮重跑Dijkstra）不新增实验；已有结果和正在运行的实验保留。

| 版本 | 局部树修复 | 边界堆 | 供给端聚合 | 终端聚合 | 原生模式 |
| --- | --- | --- | --- | --- | --- |
| V1 | 是 | 否，完整列扫描 | 否 | 否 | state_progress_v1 |
| V2 | 是 | 是 | 否 | 否 | state_progress_v2 |
| V3 | 是 | 是 | 是 | 否 | state_source_only |
| V4 | 是 | 是 | 是 | 是 | state_source |

四版均保留精确增广与最短路树复用。V1不等于逐轮重跑Dijkstra。新增组合使用独立native_progressive编译产物，未覆盖正在使用的native_ablation源码及二进制。关闭终端聚合时将未覆盖物品显式纳入搜索，新增组合已通过LP评分及边界证书核查。

## 目标与预算

同一不可变候选图、整数评分q=rint(raw*1e6)、Top-N规则，N=10。Kmax从已核验的精确TPCAR结果取得，参考边界计算不计入本批算法时间。D=K0+ceil(rho*(Kmax-K0))，rho为1%、10%、50%、100%。

| 数据集 | K0 | Kmax | 1%目标D | 10%目标D | 50%目标D | 100%目标D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OP | 780 | 2367 | 796 | 939 | 1574 | 2367 |
| Yelp | 352 | 7008 | 419 | 1018 | 3680 | 7008 |

每设置先独立运行1次，共32次，属于控制时间成本的首轮测量。不绘重复实验误差棒，不声称统计显著。V3/V4也使用本批相同独立编译文件重新测量，不能把旧高目标轨迹截点当成独立小目标耗时。

每目标单独启动进程、重新构建方法数据结构，记录构建、核心、合计、OS峰值工作集、采样私有内存、覆盖、整数目标和轨迹。保持原预算：构建加求解1800秒、私有内存12GiB软守卫、系统可用内存1GiB守卫。大缺口的V1可能较慢，超时或内存受限如实报告，不外推为实测完成时间。

评分与原精确轨迹在相同D下的整数目标核对，再核对版本之间一致性。逐层收益取决于加入顺序；现有Full减单模块结果保留为补充。允许任何版本比前一版本更慢。

## 验证及入口

- validate_progressive.py：100图、1044次调用，820可行、224不可行边界核查；完整V4与原完整后端261次等价核查，整数评分无误差。
- validation/progressive_worker_pipeline/validation.json：四版本独立进程、审计和导出检查通过，不计入正式结果。
- run_progressive_gap.py：运行及断点复用；--plan-only仅检查并导出计划，不启动求解。
- progressive_gap_report.py --wait：完成后生成核心时间、构建加求解时间、峰值内存图和三线表，自动检查后仍须查看最终图表。
- PyCharm入口：TRMNL-PROGRESSIVE-GAP、TRMNL-PLOT-PROGRESSIVE-GAP。

## 已有轨迹图

输出paper_report/existing_trajectories/T01_existing_high_target_trajectories，OP和Yelp分别采用原消融V3/V4第一次运行的真实轨迹。横轴为已增加的覆盖K-K0，纵轴为累计核心时间，保留原生初始化时间，不含Python方法构建；剔除最终无路验证端点。Yelp原请求D=7017，绘至实际边界7008。

每条曲线只有1次运行，沿途点相互依赖，不是上千次独立计时样本。该图与新批次各目标独立运行的图分开命名。

## 输出

outputs/progressive_gap_20261018中的progress.json/current_job.json记录进度，plan.json/protocol.json保存计划，progressive_results.csv/all_results.json/checks.json保存结果及评分核查。paper_report/independent_targets与tables为新批次完成后的图表；existing_trajectories为已有数据图。
