# TPCA / DGCA 实验代码

PyCharm 项目 E:/PythonProject 中选择 TPCA-OP、DGCA-OP 等配置。

- 五数据集：OP、YELP、VG、TG、SO；TPCA、DGCA、MCF 各有五个入口。
- 默认 N=10、100%目标、一次运行。可加 --n 15 或 --percentages 0 20 40 60 80 100。
- Gurobi-OP / Gurobi-LP-OP 需有效许可证。
- TPCA/ 为精确、MCF、消融及渐进缺口代码；DGCA/ 为直接替换、策略与消融；CommonInput/ 为输入准备与审核。
- TPCA/data 为同一批已测的统一图，数组以硬链接保留，不额外复制数GB。删除旧路径不影响新路径，搬电脑须完整复制数组，数组只读使用。
- 历史 .pyd 二进制名称不改，适配 Windows x64 / Python 3.9，同时保留 C++源码与构建脚本。内部 TPCAR / BGCR-D 是现有核心的兼容标识。
- Figures/ 是图表代码小文件夹，含冻结CSV、每图独立入口、高清输出；运行绘图不启动求解器。
- 新计算写入 TPCA/outputs 或 DGCA/outputs，不覆盖历史结果。

## 批处理入口

- DGCA-Strategy-OP.py 等五个入口：R/A/B/C/D 各一次，随机种子固定；DGCA-Strategies.py 可指定 --datasets，一次覆盖五图。
- TPCA-Exact-N.py：已有 N=15/20/25/30 精确批处理。
- TPCA-Ablation.py：保留原匹配消融协议（最多3次/配置，资源停止则跳过后续重复）。
- TPCA-Gaps.py：OP/Yelp 的 V1--V4 与 1/10/50/100% 缺口；--plan-only 只检查计划。
- 旧按日期命名的扩展/报告脚本保留用于来源追溯；当前策略重跑请用新的五图入口。
