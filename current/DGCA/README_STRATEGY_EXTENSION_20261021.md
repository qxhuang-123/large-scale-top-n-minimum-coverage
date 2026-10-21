# VG、TG选择策略补充实验

本批次仅增加VG与TG的R/A/B/C/D各一次，共10次。N=10、D为候选物品并集，R随机种子2026101901；评分、初始解、候选图和可行性检查全部复用原实验接口。

运行入口为run_strategy_extension_20261021.py；PyCharm配置为BGCR-STRATEGY-VG、BGCR-STRATEGY-TG、BGCR-STRATEGY-VG-TG。全部入口串行使用原有执行锁；已完成任务复用，禁止为了重画表格重跑计时。

新增代码仅负责十项任务的排程与汇总，原生二进制、原生源码、strategy_worker.py和strategy_supervisor.py均保持原版本，启动前逐项校验哈希。

输出目录：outputs/BGCR_strategy_extension_VG_TG_20261021。每项保留result.json、selected_edges.npy、events.npy、trajectory.csv、memory_trace.csv等；extension_audit.json检查同输入、与精确轨迹同覆盖评分对齐，以及A/B和C/D交换事件一致。

同一算法的等价实现A/B、C/D应得到相同事件；R、固定扫描和动态策略之间允许终止覆盖及目标值不同。目标未达到时只说明直接替换邻域耗尽，不能据此宣称全局不可行。

原有三数据集66条记录保留；论文表5从旧批次选第一次15条，再加入本批10条，共25条。所有时间为单次描述结果，不据此作显著性推断。
