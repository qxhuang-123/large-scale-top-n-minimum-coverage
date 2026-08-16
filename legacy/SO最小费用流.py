from __future__ import annotations

import sys
from pathlib import Path

# 共享网络流求解器脚本所在目录。这个绝对路径避免复制到 E 盘运行时 import 失败。
SHARED_SOLVER_DIR = Path(r"C:\Users\24qxh\Documents\Codex\2026-07-11\new-chat-2\outputs")
if str(SHARED_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_SOLVER_DIR))

from run_network_flow_N10_D_percent_5datasets import DATASETS, run_dataset, write_outputs


def main() -> None:
    # SO only. 固定 N=10，D=0/20/40/60/80/100%。
    # 第一次运行会从 E:\Users\24qxh\Desktop\SO 下的 3 个 CSV 构建缓存，时间和内存开销会比较大。
    rows, summary, meta = run_dataset(
        "SO",
        DATASETS["SO"],
        dry_run=False,
        d_percentages=[0, 20, 40, 60, 80, 100],
        skip_cache_build=False,
    )
    write_outputs(rows, [summary], [meta])


if __name__ == "__main__":
    main()
