# 新版 TPCAR / MCF / Gurobi 统一比较

本机配置：2026-10-18。使用现有 `E:/PythonProject/.venv/Scripts/python.exe`。没有安装或升级包，没有改原始候选缓存、原始评分或原 `TRMNL-*-F` 工程。比较程序另编译七数组接口，去掉冗余反向评分副本；原 F 的搜索、压缩、堆与局部修复逻辑保持一致。BGCR 不在本轮范围。

## PyCharm 运行

打开 `E:/PythonProject`，选择：

- `TRMNL-OP-TPCAR`、`TRMNL-YELP-TPCAR`、`TRMNL-TG-TPCAR`、`TRMNL-VG-TPCAR`、`TRMNL-SO-TPCAR`
- 相应的 `TRMNL-OP-MCF` 等五个 MCF 配置
- 相应的 `TRMNL-OP-GUROBI-LP` / `TRMNL-OP-GUROBI-MIP` 等十个配置
- `TRMNL-ALL-COMPARE`：整批串行比较，TPCAR/MCF 五图，LP/MIP 先运行 OP，沿用原文基准范围
- `TRMNL-STOP-COMPARE`：请求停止当前比较，保留完成结果；可用相同 `--session` 续跑
- `TRMNL-RESUME-COMPARE`：直接续跑本次 `first_batch_final_20261018`，跳过已完成任务

共 23 个运行配置（20 个数据集/方法入口、整批入口、停止入口、续跑入口）。TPCAR 指新版 `state_source`；不使用旧 TPCAR。原 `TRMNL-*-F` 是原评分的历史入口，与这里统一取整的正式比较分开保存。

每个单独配置**默认跑六档目标，各独立重复三次**，不是只跑一次最高目标。配置未刷新时重新打开父项目。不要同时启动多个配置；进程锁保证串行，以免速度与内存互相干扰。

常用参数：

```text
--repeats 3
--percentages 100 --repeats 1
--target 7009 --repeats 1
--timeout 1800 --private-limit-gib 12 --reserve-gib 1
```

`--target` 是绝对请求值；优先使用 `--percentages` 跨数据集运行。N 默认 10，也可 `--n` 指定。

## 完全共用的问题输入

现有 F 已准备的候选图只读，保留所有候选成员；没有重做 40% 筛选。为比较另建不可变快照：

    q_ui = numpy.rint(float64(original_score_ui) * 1,000,000)

取整采用 ties-to-even；每行统一按 q 降序、物品 ID 升序排列。保存原评分供重算，保存指针/候选/评分的内容 SHA-256 和统一 `graph_id`。所有方法读取同一套 q、E、N、D。

TPCAR 也直接使用 q 的整数数值（以 float64 无损承载），而非 q/scale；目标与中间规模检查确保本批整数目标可安全表示。MCF 使用 -q 整数费用；Gurobi 使用 q 系数。因此比较主指标是**整数目标完全相等**，不是打印小数碰巧相同。

目标 D=ceil(p C/100)，p=0/20/40/60/80/100，使用整数算式处理上取整。K0 由统一 q 和同分规则的 Top-N 得到；所有方法报告同一个请求缺口 max(D-K0,0)。MCF、LP/MIP 按标准模型从头求解，不强迫它们采用 TPCAR 的 Top-N 热启动。

同分最优解仍可能有不同的选边、最终 K≥D，甚至略不同的原评分总和；这是共同取整目标下的替代最优解，不应强行修改输出。`comparison_checks.json` 只对原请求的 OPTIMAL 结果比较整数目标与可行性。F 对不可行请求给出的边界方案单独记录，不与另一个问题的最优值混比。

## 方法与公平成本

- 新版 TPCAR：从当前 F 源码派生的独立七数组接口 `state_variants_bench`；从标准 CSR 在内存重建其专用双向索引、评分视图和边号，计入 build 与峰值内存。反向权重通过 `user_scores[item_edges[p]]` 取得，不额外复制 E 个 float64；F 的边界堆分支本来就采用此查询。反向边号先构建，再顺序填充反向用户数据；额外 q 输入映射在复制后释放。原生调用内的 Top-N、首次搜索、源池、边界堆、增广和修复计入 solve。接口来源与精确变更见 `compact_backend_provenance.json`，原 F 工程未改。
- MCF：OR-Tools 9.15.6755 `SimpleMinCostFlow`；分块建立显式网络，计入 build。每用户 N 个推荐；覆盖汇点恰好接收 D 个来自不同物品的首单位，其他流进入普通汇点，等价于覆盖至少 D。读回所有已选边后统一审计。
- Gurobi 12.0.3 LP：最低覆盖连续模型，simplex，一线程。
- Gurobi MIP：同一目标和约束，x/z 为二进制，MIPGap/MIPGapAbs=0，一线程。

LP/MIP 不是用缩小的候选图代替全图；独立单图入口会尝试该图全规模。整批默认只在 OP 跑它们，避免把本轮扩成所有大图的通用 IP 研究。

所有方法报告数据加载、专用构造、优化、抽取审计、输出以及完整进程时间。主要优化合计是 build+solve。共享候选快照取整/排序单独记录，不计入任何一方。每次方法专用结构重新构造，但没有清空 Windows 磁盘缓存；不把它称为磁盘冷启动实验。

## 时间与内存监测

每个方法/目标/重复是独立进程，串行运行。默认每 0.2 秒采样内存和阶段。Windows venv 启动器会创建真正的 Python 进程：这里依据 worker 自报 PID 监测算法进程，不能仅监测约数 MB 的启动器。

- `peak_working_set_bytes`：Windows 生存期峰值工作集，包含文件映射页。
- `sampled_peak_private_bytes`：采样得到的私有字节峰值，可能漏掉短峰。
- `peak_commit_os_bytes`：Windows 生存期峰值进程提交计数。

默认构造＋优化预算 1800 秒；私有字节软守卫 12 GiB；系统可用内存连续 5 次低于 1 GiB 会触发系统保护。守卫按采样检查，不是声称设置了 OS 硬内存配额。阈值对所有方法一致，实际触发条件保存在结果中。

`PRIVATE_MEMORY_LIMIT`、`SYSTEM_MEMORY_GUARD`、`TIME_LIMIT` 与 `INFEASIBLE` 分开。系统保护触发只说明本次环境和守卫下没有结果，不能据此宣称模型不可行或算法绝对需要超过整台机器 RAM。受限设置只尝试一次后跳过其剩余重复，不编造三次中位数。

## 结果与轨迹

结果位于 `TRMNL_BENCH/outputs/<会话>/`：

- `protocol.json`：图、精度、重复、资源、源代码版本约定。
- `current_job.json` / `progress.json`：当前任务与会话是否完成。
- `comparison.csv` / `comparison_report.md`：逐次数据与汇总。
- `comparison_checks.json`：跨方法整数目标一致性，发现冲突立即停止整批。
- 每次的 `result.json`、日志、推荐列表、已选边、`memory_trace.csv`、`trajectory.csv/json`。

TPCAR 保存每次覆盖增广的评分和时间；MCF Python API 没有暴露同样的内部增广回调，因此只保存真实最终端点和阶段/内存轨迹，不伪造其逐次增广曲线。LP 的 simplex 曲线包含未可行的松弛状态，不能当整数推荐方案的 APS；MIP 保存其 incumbent 记录。所有方法都有经过共同审计的最终端点。

正式版专用索引在 RAM 构造，进程退出后自然释放，不为每次重复保存 GB 级副本。初期文件映射构造的校准会话单独保留并排除出正式计时；其停止后的临时数组清理仅限该次 scratch 内的五个明确生成文件，不删除原始候选或共享输入。TPCAR 还分别记录索引组装与完整双向一致性检查时间；二者都包含在当前报告的 build 中，没有隐藏校验开销。

## 续跑与合并

整批支持 `--session 已有会话路径`，协议相同的已完成 job 会复用。改变时间或资源设置应选新会话，不能混入旧协议。一个 PyCharm 单图配置运行后产生独立会话；用整批入口最便于自动跨方法核对。

也可合并指定会话：

```text
collect_results.py --sessions 会话A 会话B --output 新汇总目录
```

选择相同 N 和资源协议的会话；检查 `graph_id` 和源版本一致后用于正式计时统计。先前试运行用于验证接口，不应悄悄混作正式独立重复。

## 已做验证

四种适配器已通过 12 个小图、116 次独立 LP 核对：96 可行、20 不可行调用，整数目标误差 0；F 的不可行边界评分和证书也核对。七数组接口另与原 F 在 100 个图、300 个目标上逐项比较：评分、覆盖、选边与工作计数完全一致。进程隔离、实际 PID 内存监测、导出、资源停止及停止后续跑已验证。实际大图结果以 outputs 内的日志为准，不将配置完成等同整批运行完成。

本次自动执行的正式会话为 `outputs/first_batch_final_20261018`，最高覆盖目标优先。`launch_first_batch.py` 在 SO 全规模接口验证成功后自动继续整批；该次已完成、协议及源码完全相同的独立运行可作为 SO 最高目标的第 1 次重复复用，来源保存在 `validation_reuse.json`。早期文件映射和八数组内存压力校准不混入正式结果。整批默认最多 216 个计划运行，资源受限的重复会按规则减少，因此可能运行较长时间；随时查看 progress 或使用停止配置。

官方 API 依据：

- https://developers.google.com/optimization/flow/mincostflow
- https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex
