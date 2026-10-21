# TPCAR与BGCR-D消融设计（五数据集）

适用数据：OP、Yelp、SO、VG、TG。每图N=10，D=C（当前候选集并集物品数），完整配置和三个单模块关闭配置，各独立进程重复3次。候选图、整数评分、取整规则、初始Top-N、目标、资源预算保持一致。预计每种算法60个模块比较运行点；资源受限设置按协议跳过剩余重复。

| 配置 | 新TPCAR | BGCR-D |
| --- | --- | --- |
| Full | state_source | dynamic_user_heap |
| -Src | state_terminal：关闭源端聚合 | dynamic_user_heap_nosource：每次扫描当前N个槽位取最便宜可移除项 |
| -Term | state_source_only：关闭终端聚合 | dynamic_user_heap_noterminal：每次从候选行首重新查找未覆盖目标 |
| -Heap | state_source_noheap：精确列扫描替代修复边界堆 | dynamic_user_scan：每步扫描全部用户替代全局用户堆 |

两者采用同一控制思路：每次仅关闭一种加速机制，保留原算法的决策规则。因此需要核查同算法不同配置的整数目标、实际覆盖与结果状态一致；BGCR小图验证还逐步核对交换事件。TPCAR保留跨用户交换链、精确增广和局部树修复；BGCR保留当前最小损失的单用户直接替换。两种算法的Heap不是同一种堆，不能据名称作同一机制的解释。

TPCAR消融不包含关闭局部树修复的版本，因此不能单独归因局部修复的加速贡献。BGCR每步精确选择当前最小直接替换仍是对全局问题的启发式，最终评分和覆盖允许不同于TPCAR。跨算法评分差距须在同一实际覆盖K下比较。

记录：方法构造时间、核心求解时间、构造+求解时间、OS峰值工作集、采样私有内存、整数评分、实际覆盖、轨迹和工作量计数。完整设置报告3次中位数及范围，同时保留原始记录。TIME_LIMIT、SYSTEM_MEMORY_GUARD等标为资源受限；不视为问题不可行。Yelp最高请求若不可达，TPCAR各模块核对相同可达边界，BGCR只判断直接替换邻域终止。

BGCR另做同邻域策略对照：OP、Yelp、SO，fixed_source（初始损失固定排序单扫描）、id_source（物品ID顺序单扫描），与完整动态BGCR-D比较。该对照改变决策规则，允许覆盖和评分不同；用于回应COR审稿人关于排序与动态最小损失选择的疑问，不能混作不改变决策的模块消融。

TPCAR扩展：原OP/Yelp/SO队列继续，extend_ablation_VG_TG.py等待其结束后复用已有result.json，补VG/TG最多24次。共同完成条件由ablation_completion.py检查。五图合计最多60次，不重跑已有完成设置。PyCharm入口TRMNL-ABLATION-ADD-VG-TG。

BGCR：五图完整配置15次复用其主比较，三个关闭模块配置新增45次；主比较90次、不同N20次、策略对照18次，总计划最多173次。BGCR-ALL-COMPARE等待TPCAR五图消融完成后启动。

## 用户确认的范围
本轮不新增每次从头运行Dijkstra的Base/SSP实验，不新增Repair-only实验；基础算法沿用原有数据。当前五图队列仍仅包含Full及三个单模块关闭配置，BGCR按既定队列等待。


Progressive OP/Yelp follow-up: V1--V4 at 1/10/50/100 percent of the attainable coverage gap, one independent run each (32 runs). V0 remains deferred. See PROGRESSIVE_GAP_PROTOCOL.md. Runs after the current five-dataset ablation, before BGCR.
