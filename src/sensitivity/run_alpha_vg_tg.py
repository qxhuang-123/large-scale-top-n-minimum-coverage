from __future__ import annotations

import argparse
import ctypes
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.graph.python import min_cost_flow


N = 10
ALPHAS = [round(i / 10, 1) for i in range(1, 11)]
SEED = 20260704
SCORE_SCALE = 1 << 23
ARC_BATCH_SIZE = 1_000_000

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(r"E:\Users\24qxh\Desktop\VG_TG_最小费用流灵敏度分析")

DATASETS = {
    "VG": {
        "score_file": SCRIPT_DIR / "VG贪心算法_cache" / "VG_scores_float32.npy",
        "expected_shape": (24442, 10350),
    },
    "TG": {
        "score_file": SCRIPT_DIR / "TG贪心算法_cache" / "TG_scores_float32.npy",
        "expected_shape": (19412, 11924),
    },
}


def load_scores(dataset: str, config: dict) -> np.ndarray:
    path = Path(config["score_file"])
    if not path.exists():
        raise FileNotFoundError(f"{dataset} 评分缓存不存在：{path}")
    scores = np.load(path, mmap_mode="r")
    expected_shape = tuple(config["expected_shape"])
    if scores.shape != expected_shape:
        raise ValueError(
            f"{dataset} 评分矩阵维度错误：实际 {scores.shape}，预期 {expected_shape}"
        )
    print(f"{dataset} 使用评分缓存：{path}，shape={scores.shape}", flush=True)
    return scores


def flush_arc_batch(
    solver: min_cost_flow.SimpleMinCostFlow,
    tails_parts: list[np.ndarray],
    heads_parts: list[np.ndarray],
    costs_parts: list[np.ndarray],
) -> None:
    if not tails_parts:
        return
    tails = np.concatenate(tails_parts).astype(np.int32, copy=False)
    heads = np.concatenate(heads_parts).astype(np.int32, copy=False)
    costs = np.concatenate(costs_parts).astype(np.int64, copy=False)
    capacities = np.ones(tails.size, dtype=np.int64)
    solver.add_arcs_with_capacity_and_unit_cost(tails, heads, capacities, costs)
    tails_parts.clear()
    heads_parts.clear()
    costs_parts.clear()


def build_exact_flow_model(scores: np.ndarray, alpha: float):
    user_count, item_count = map(int, scores.shape)
    solver = min_cost_flow.SimpleMinCostFlow()
    rng = np.random.default_rng(SEED)

    user_supplies = np.zeros(user_count, dtype=np.int64)
    item_seen = np.zeros(item_count, dtype=bool)
    naive_item_seen = np.zeros(item_count, dtype=bool)

    tails_parts: list[np.ndarray] = []
    heads_parts: list[np.ndarray] = []
    costs_parts: list[np.ndarray] = []
    pending_edges = 0
    candidate_edges = 0
    naive_score_sum = 0
    minimum_score = np.iinfo(np.int64).max
    maximum_score = np.iinfo(np.int64).min

    started = time.perf_counter()
    for user in range(user_count):
        row = np.asarray(scores[user], dtype=np.float32)
        valid_items = np.flatnonzero(np.isfinite(row) & (row > 0.0))
        if valid_items.size == 0:
            continue

        candidate_count = min(
            valid_items.size,
            max(N, int(np.ceil(alpha * valid_items.size))),
        )
        valid_scores = row[valid_items]
        if candidate_count < valid_items.size:
            jitter = rng.random(valid_items.size, dtype=np.float32) * np.float32(1e-7)
            selected_local = np.argpartition(
                valid_scores + jitter, -candidate_count
            )[-candidate_count:]
            selected_items = valid_items[selected_local]
            selected_scores = valid_scores[selected_local]
        else:
            selected_items = valid_items
            selected_scores = valid_scores

        order = np.argsort(-selected_scores, kind="mergesort")
        selected_items = selected_items[order].astype(np.int32, copy=False)
        selected_scores = selected_scores[order].astype(np.float64, copy=False)
        integer_scores = np.rint(selected_scores * SCORE_SCALE).astype(np.int64)

        if not np.array_equal(
            integer_scores.astype(np.float64) / SCORE_SCALE,
            selected_scores,
        ):
            raise ValueError("评分无法无损转换为整数费用，请检查评分范围")

        recommendation_count = min(N, selected_items.size)
        user_supplies[user] = recommendation_count
        naive_score_sum += int(integer_scores[:recommendation_count].sum())
        naive_item_seen[selected_items[:recommendation_count]] = True
        item_seen[selected_items] = True

        minimum_score = min(minimum_score, int(integer_scores.min()))
        maximum_score = max(maximum_score, int(integer_scores.max()))

        count = int(selected_items.size)
        tails_parts.append(np.full(count, user, dtype=np.int32))
        heads_parts.append(user_count + selected_items)
        costs_parts.append(-integer_scores)
        pending_edges += count
        candidate_edges += count

        if pending_edges >= ARC_BATCH_SIZE:
            flush_arc_batch(solver, tails_parts, heads_parts, costs_parts)
            pending_edges = 0

        if (user + 1) % 500 == 0:
            print(
                f"  建模用户 {user + 1:,}/{user_count:,}，候选边 {candidate_edges:,}",
                flush=True,
            )

    flush_arc_batch(solver, tails_parts, heads_parts, costs_parts)
    total_recommendations = int(user_supplies.sum())
    if total_recommendations == 0:
        raise ValueError("没有可用的候选推荐边")

    score_range = int(maximum_score - minimum_score)
    coverage_bonus = total_recommendations * score_range + 1
    if coverage_bonus * item_count >= np.iinfo(np.int64).max:
        raise OverflowError("最小费用流整数目标可能溢出")

    item_nodes = user_count + np.arange(item_count, dtype=np.int32)
    sink = user_count + item_count
    sink_nodes = np.full(item_count, sink, dtype=np.int32)

    coverage_arc_start = solver.num_arcs()
    solver.add_arcs_with_capacity_and_unit_cost(
        item_nodes,
        sink_nodes,
        np.ones(item_count, dtype=np.int64),
        np.full(item_count, -coverage_bonus, dtype=np.int64),
    )
    solver.add_arcs_with_capacity_and_unit_cost(
        item_nodes,
        sink_nodes,
        np.full(item_count, total_recommendations, dtype=np.int64),
        np.zeros(item_count, dtype=np.int64),
    )

    supply_nodes = np.concatenate(
        (np.arange(user_count, dtype=np.int32), np.array([sink], dtype=np.int32))
    )
    supplies = np.concatenate(
        (user_supplies, np.array([-total_recommendations], dtype=np.int64))
    )
    solver.set_nodes_supplies(supply_nodes, supplies)

    return {
        "solver": solver,
        "candidate_edges": candidate_edges,
        "items_with_candidates": int(item_seen.sum()),
        "naive_score_sum": naive_score_sum,
        "naive_diversity": int(naive_item_seen.sum()),
        "total_recommendations": total_recommendations,
        "coverage_bonus": coverage_bonus,
        "coverage_arc_start": coverage_arc_start,
        "build_seconds": time.perf_counter() - started,
    }


def run_one(dataset: str, scores: np.ndarray, alpha: float) -> dict:
    user_count, item_count = map(int, scores.shape)
    print(
        f"\n--- OR-Tools 精确最小费用流 {dataset}: "
        f"N={N}, alpha={alpha:.1f}, D=I={item_count} ---",
        flush=True,
    )

    total_started = time.perf_counter()
    model = build_exact_flow_model(scores, alpha)
    solver = model["solver"]
    candidate_edges = int(model["candidate_edges"])
    build_seconds = float(model["build_seconds"])
    print(
        f"建模完成：候选边={candidate_edges:,}，"
        f"建模时间={build_seconds:.1f}s，开始求解",
        flush=True,
    )

    solve_started = time.perf_counter()
    status = solver.solve()
    solve_seconds = time.perf_counter() - solve_started
    if status != solver.OPTIMAL:
        raise RuntimeError(f"OR-Tools 最小费用流求解失败，状态={status}")

    diversity = sum(
        1
        for arc in range(
            model["coverage_arc_start"],
            model["coverage_arc_start"] + item_count,
        )
        if solver.flow(arc) > 0
    )
    score_sum = (
        -int(solver.optimal_cost())
        - diversity * int(model["coverage_bonus"])
    )
    total_recommendations = int(model["total_recommendations"])
    accuracy = score_sum / SCORE_SCALE / total_recommendations
    objective = score_sum / SCORE_SCALE
    naive_accuracy = (
        int(model["naive_score_sum"])
        / SCORE_SCALE
        / total_recommendations
    )

    result = {
        "数据集": dataset,
        "N": N,
        "风险系数alpha": alpha,
        "风险系数百分比": int(round(alpha * 100)),
        "D_target": item_count,
        "用户数": user_count,
        "项目数I": item_count,
        "候选边数": int(model["candidate_edges"]),
        "有候选用户的项目数": int(model["items_with_candidates"]),
        "推荐总数": total_recommendations,
        "推荐准确性": round(accuracy, 6),
        "总体多样性": diversity,
        "目标函数值": round(objective, 6),
        "Naive推荐准确性": round(naive_accuracy, 6),
        "Naive总体多样性": int(model["naive_diversity"]),
        "迭代次数": np.nan,
        "交换次数": np.nan,
        "target_feasible": diversity >= item_count,
        "max_reached": diversity,
        "建模时间s": round(float(model["build_seconds"]), 3),
        "求解时间s": round(solve_seconds, 3),
        "总时间s": round(time.perf_counter() - total_started, 3),
        "求解后端": "OR-Tools SimpleMinCostFlow C++",
    }
    print(
        f"{dataset} alpha={alpha:.1f} 完成：准确性={accuracy:.6f}，"
        f"总体多样性={diversity}，求解={solve_seconds:.1f}s",
        flush=True,
    )

    del solver
    del model
    gc.collect()
    return result


def load_partial(partial_path: Path) -> list[dict]:
    if not partial_path.exists():
        return []
    return pd.read_csv(partial_path, encoding="utf-8-sig").to_dict("records")


def result_frame(rows: list[dict]) -> pd.DataFrame:
    return (
        pd.DataFrame(rows)
        .sort_values(["数据集", "风险系数alpha"])
        .reset_index(drop=True)
    )


def save_partial(rows: list[dict], partial_path: Path) -> None:
    result_frame(rows).to_csv(partial_path, index=False, encoding="utf-8-sig")


def save_final_tables(rows: list[dict]) -> None:
    frame = result_frame(rows)
    csv_path = OUT_DIR / "vg_tg_mcf_sensitivity_results_ortools.csv"
    xlsx_path = OUT_DIR / "vg_tg_mcf_sensitivity_results_ortools.xlsx"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="灵敏度分析", index=False)
        for dataset in DATASETS:
            frame.loc[frame["数据集"] == dataset].to_excel(
                writer, sheet_name=dataset, index=False
            )
    print(f"最终 CSV：{csv_path}", flush=True)
    print(f"最终 Excel：{xlsx_path}", flush=True)


def single_result_path(dataset: str, alpha: float) -> Path:
    alpha_tag = f"{alpha:.1f}".replace(".", "p")
    return OUT_DIR / f"single_{dataset.lower()}_alpha_{alpha_tag}.json"


def run_single(dataset: str, alpha: float) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"未知数据集：{dataset}")
    scores = load_scores(dataset, DATASETS[dataset])
    row = run_one(dataset, scores, alpha)
    path = single_result_path(dataset, alpha)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)
    print(f"单轮结果已保存：{path}", flush=True)


def available_memory_gb() -> float:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return float("inf")
    return status.available_physical / (1024 ** 3)


def wait_for_memory(minimum_gb: float = 14.0, timeout_seconds: int = 300) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        available = available_memory_gb()
        if available >= minimum_gb:
            print(f"可用内存 {available:.1f} GB，开始下一轮", flush=True)
            return
        print(f"等待内存释放：当前 {available:.1f} GB", flush=True)
        gc.collect()
        time.sleep(10)
    print("内存等待超时，继续尝试运行", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", nargs=2, metavar=("DATASET", "ALPHA"))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.single:
        run_single(args.single[0], float(args.single[1]))
        return

    partial_path = OUT_DIR / "vg_tg_mcf_sensitivity_partial_ortools.csv"
    rows = load_partial(partial_path)
    done = {
        (str(row["数据集"]), round(float(row["风险系数alpha"]), 1))
        for row in rows
    }

    for dataset in DATASETS:
        for alpha in ALPHAS:
            key = (dataset, round(alpha, 1))
            if key in done:
                print(f"{dataset} alpha={alpha:.1f} 已完成，跳过", flush=True)
                continue

            result_path = single_result_path(dataset, alpha)
            result_path.unlink(missing_ok=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single",
                dataset,
                f"{alpha:.1f}",
            ]
            print(f"启动独立子进程：{dataset} alpha={alpha:.1f}", flush=True)
            for attempt in range(2):
                wait_for_memory()
                try:
                    subprocess.run(command, check=True)
                    break
                except subprocess.CalledProcessError:
                    if attempt == 1:
                        raise
                    print("子进程失败，等待内存恢复后重试一次", flush=True)
                    time.sleep(15)
            if not result_path.exists():
                raise RuntimeError(f"子进程没有生成结果：{result_path}")

            row = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(row)
            done.add(key)
            save_partial(rows, partial_path)
            result_path.unlink(missing_ok=True)
            print(f"部分结果已保存：{partial_path}", flush=True)

    save_final_tables(rows)
    print("VG、TG 灵敏度分析全部完成。", flush=True)


if __name__ == "__main__":
    main()
