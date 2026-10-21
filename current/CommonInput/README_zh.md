# F 算法：局部修复＋精确边界堆＋供给端聚合＋终点聚合

本包整理了此前 Yelp 实验中 `state_source` 模式的实际实现。三个 C++ 核心文件
与该实验包逐字节一致；新增的是输入校验、通用 Python 接口、命令行入口和示例。
实验核心保留了其他消融分支，方便追溯；本包对外入口固定调用 F。

## 1. 模型与输入约定

每个用户从其候选物品中选择恰好 N 个不同物品，在总覆盖数至少 D 的条件下最大化总得分。
如果 D 不可行，返回最大可实现覆盖下的得分最优解，并输出可独立检查的覆盖不可行性证书。
如果初始 Top-N 已满足 D，立即返回；此时不代表已经计算了最大覆盖量。

- 用户、物品编号均从 0 开始。每个用户至少有 N 个不同候选。
- 传入的候选图就是优化允许的边集合；本接口不隐式实施 alpha 筛选。
- 候选得分必须有限，可以为零或负数。矩阵接口默认所有有限值均可推荐。
- 若自己的数据以 0 表示已知评分或禁用边，务必显式传入 `eligible`，或先删掉那些边。
- C++ 使用 int32 索引、float64 得分。用户数+物品数+2、边数、U*N 均须小于 2^31-1。
- 当前实现是带浮点容差的精确优化算法实现，不是有理数符号运算程序。

## 2. 环境与编译

要求 Python 3.10+、GCC 或 Clang 的 C++17 编译器。Linux 已实际编译验证。
Windows 建议使用 WSL；保留的实验源码含 GCC/Clang 内建函数，未适配原生 MSVC。
macOS 可使用 Clang，但本次未在 macOS 上执行验证。

进入解压后的本目录：

```bash
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
python run_f.py --demo --n 1 --target 3 --output output_demo
```

示例预期：初始覆盖 2，最终覆盖 3，1 次增广、2 次用户推荐替换，总得分约 14.8。
编译得到的动态库依赖本机 Python 和平台；因此压缩包不附预编译二进制。

## 3. 使用自己的数据

小型或中型候选边表：CSV 表头必须为 `user_id,item_id,score`，不要含重复用户-物品对。

```bash
python run_f.py --edges-csv examples/candidates.csv --n 1 --target 3 --output output_csv
```

可通过 `--users`、`--items` 指定完整编号范围，否则从最大编号推断。
较大实例优先复用 CSR 缓存，避免把大型矩阵转成 Python 对象：

```bash
python run_f.py --csr /path/to/data_directory --n 10 --target 7017 --output output_yelp
```

这个目录须含以下八个数组（与此前 Yelp 数据缓存格式一致）：
`user_indptr.npy`、`user_items.npy`、`user_scores.npy`、`edge_users.npy`、
`item_indptr.npy`、`item_users.npy`、`item_edges.npy`、`item_scores.npy`。
也可将这八个键保存在同一个 NPZ 中，传给 `--csr`。
行中按得分降序、物品编号升序排列；物品列中的边号严格递增。
不需要额外 metadata 文件。用户只需提供已经按论文候选规则预处理的数组。

密集 NPY 矩阵也可运行：

```bash
python run_f.py --scores-npy scores.npy --eligible-npy eligible.npy --n 10 --target 100
```

Python 接口：

```python
import numpy as np
from f_algorithm import solve, CandidateGraph, solve_graph

scores = np.array([[5.,4.9,0.], [5.2,0.,0.], [0.,4.8,4.7]])
result = solve(scores, N=1, D=3)
print(result['recommendations'])
print(result['coverage'], result['score'], result['target_feasible'])

# 已有大规模 CSR 缓存时：
# graph = CandidateGraph.load('data_directory')
# result = solve_graph(graph, N=10, D=7017)
```

CSV 和密集矩阵适合方便接入；大数据建议 CSR。输入构造、排序、完整校验和结果审计
均在核心计时之外，并会实际消耗时间和内存，不能忽略它们来报告端到端时间。

## 4. 输出与计时

- `result.json`：可行性、覆盖、总分、增广轨迹、操作计数和分项时间。
- `recommendations.csv`：每个用户选中的 N 个物品，按得分排序。
- `selected_edges.npy`：选中边的 CSR 行序号。
- `reachable_nodes.npy`：若运行进入最短路阶段，保存核心返回的可达标记。
  只在目标不可行并耗尽增广时，将它作为最大覆盖证书使用；提前到达目标时不作此解释。

`call_wall_seconds` 是 C++ 完整调用耗时，包含初始化、首次最短路、建堆、所有增广、
局部修复及返回值转换。`input_validation_seconds`、`audit_seconds` 单独列出。
`history[*].time_sec` 是本次运行的累计核心时间；不等于对应目标独立重跑的耗时。
`maximum_coverage_certified` 为 true 表示不可行目标的闭集证书通过，null 表示未作该断言。

## 5. 核心代码导航

|算法步骤|位置|
|---|---|
|Top-N、初始势、增广交换与主循环|`src/state_variants.cpp`|
|供给池与指针|`StateRepair` 构造函数、`refresh_source`|
|终点指针及全局堆|`terminal_edge`、`build_targets`、`choose_terminal`|
|首次压缩图最短路|`initial_search`|
|恢复路径、维护供给入口与失效子树|`augment_state`|
|边界接入和局部最短路修复|`repair_state`|
|每个物品的精确边界堆|`src/local_repair_impl.inc` 中 `boundary_min`|
|通用输入、调用、可行性与闭集证书检查|`f_algorithm.py`|

F 的开关：供给聚合 true、终点聚合 true、边界堆 true、候选激活索引 false、交集索引 false。
供给端采用有序池与指针；两个精确查询堆分别是终点堆和物品边界堆。
局部 Dijkstra 另有距离堆。当前 F 展开修复用户时仍扫描其原始候选行。

## 6. 验证与历史结果

```bash
python -m pip install -r requirements-validation.txt
python validate_f.py --instances 40
```

验证使用 SciPy/HiGHS 独立 LP 对照，覆盖整数同分、连续分数、负分、稀疏候选、
单覆盖转化、提前结束及不可行目标。它核对返回解的整数可行性与 LP 上界是否一致。
本次实际验证结果保存在 `validation/current_validation.json`。
源文件身份保存在 `provenance.json`；完整包文件哈希保存在 `MANIFEST_SHA256.json`。

`reference/` 保留的是此前 Yelp 实测记录，本次整理没有重跑大型 Yelp 数据：
B/E/F 的三次独立运行中位数分别为 17.280010 / 14.767682 / 10.422996 秒。
数据 U=9415、I=9373、E=35,243,883、N=10、alpha=0.4；初始覆盖 352。
请求 D=7017 不可行，最终覆盖 7008，实际增广 6656 次。
F 相对 B 减少约 39.68% 耗时；这只是该实验环境下的结果，不是普适提速保证。

大型原始数据和八个缓存数组不包含在本代码包中。数据来源可追溯到原实验记录：
https://zenodo.org/records/21996444
https://github.com/qxhuang-123/large-scale-top-n-minimum-coverage
这些链接是历史包中的来源记录，本次未重新访问或核验可下载状态。

## 7. 理论边界

源指针累计前进不超过 U*N，终点指针累计前进不超过 E；边界堆首次建堆的总入堆条目
不超过 E。边界键刷新、临时排除/恢复和修复区候选行扫描仍可能多次发生。
因此不能将 F 的整个过程写成 O(E)，也不能将实测加速直接写成已证明的最坏情况降阶。
附带 PDF 给出完整流程、正确性依据和与上述函数对应的伪代码。
