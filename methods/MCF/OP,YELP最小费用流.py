from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

try:
    from ortools.graph.python import min_cost_flow
except ImportError as exc:
    raise SystemExit("This script requires OR-Tools: python -m pip install ortools") from exc


ROOT = Path(r"C:\Users\24qxh\Documents\Codex\2026-07-11\new-chat-2")
OUT_DIR = ROOT / "outputs" / "network_flow_OP_Yelp_exact_D_base"
OUT_XLSX = OUT_DIR / "network_flow_OP_Yelp_exact_D_base.xlsx"
OUT_JSON = OUT_DIR / "network_flow_OP_Yelp_exact_D_base.json"

N = 10
D_PERCENTAGES = [0, 20, 40, 60, 80, 100]
SCORE_SCALE = 1_000_000


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    scores_path: Path
    cand_path: Path


DATASETS = {
    "OP": DatasetConfig(
        name="OP",
        scores_path=Path(
            r"C:\Users\24qxh\Documents\Codex\2026-07-05\import-json-import-sys-import-time-4"
            r"\work\op_exact_cache\op_full_scores_float32.npy"
        ),
        cand_path=Path(
            r"C:\Users\24qxh\Documents\Codex\2026-07-05\import-json-import-sys-import-time-4"
            r"\work\op_exact_cache\op_full_cand_frac_0p4_seed20260704.npz"
        ),
    ),
    "Yelp": DatasetConfig(
        name="Yelp",
        scores_path=Path(
            r"C:\Users\24qxh\Documents\Codex\2026-07-04\new-chat"
            r"\work\yelp_cache\yelp_unknown_scores_float32.npy"
        ),
        cand_path=Path(
            r"C:\Users\24qxh\Documents\Codex\2026-07-04\new-chat"
            r"\work\yelp_cache\yelp_cand_frac_0p4_seed20260704.npz"
        ),
    ),
}


def load_candidate_arrays(scores: np.ndarray, cand_path: Path):
    z = np.load(cand_path)
    user_indptr = z["user_indptr"].astype(np.int64, copy=False)
    item_key = "user_items" if "user_items" in z.files else "user_indices"
    user_items = z[item_key].astype(np.int32, copy=False)

    num_items = scores.shape[1]
    user_quota = np.minimum(N, np.diff(user_indptr)).astype(np.int32, copy=False)
    item_counts = np.bincount(user_items, minlength=num_items)
    items_with_candidates = int(np.count_nonzero(item_counts))

    return user_indptr, user_items, user_quota, items_with_candidates


def canonical_top_n(
    scores: np.ndarray,
    user: int,
    items: np.ndarray,
    k: int,
) -> np.ndarray:
    """Select by score descending, breaking score ties by item ID ascending."""
    if k <= 0:
        return items[:0]
    if k >= len(items):
        return items

    user_scores = np.asarray(scores[user, items], dtype=np.float64)
    order = np.lexsort((items, -user_scores))
    return items[order[:k]]


def compute_naive(scores: np.ndarray, user_indptr: np.ndarray, user_items: np.ndarray, user_quota: np.ndarray):
    num_items = scores.shape[1]
    counts = np.zeros(num_items, dtype=np.int32)
    total_score = 0.0
    total_slots = int(user_quota.sum())

    for u in range(len(user_quota)):
        s, e = int(user_indptr[u]), int(user_indptr[u + 1])
        k = int(user_quota[u])
        if k <= 0:
            continue
        candidates = user_items[s:e]
        selected = canonical_top_n(scores, u, candidates, k)
        counts[selected] += 1
        total_score += float(np.asarray(scores[u, selected], dtype=np.float64).sum())

    coverage = int(np.count_nonzero(counts))
    return {
        "objective": total_score,
        "accuracy": total_score / total_slots if total_slots else 0.0,
        "coverage": coverage,
    }


def solve_network_flow_for_d(
    scores: np.ndarray,
    user_indptr: np.ndarray,
    user_items: np.ndarray,
    user_quota: np.ndarray,
    d_target: int,
) -> dict:
    num_users, num_items = scores.shape
    total_slots = int(user_quota.sum())

    source = 0
    user0 = 1
    item0 = user0 + num_users
    cover_sink = item0 + num_items
    sink = cover_sink + 1
    num_nodes = sink + 1

    solver = min_cost_flow.SimpleMinCostFlow()

    active_users = np.flatnonzero(user_quota > 0).astype(np.int64, copy=False)
    solver.add_arcs_with_capacity_and_unit_cost(
        np.full(active_users.size, source, dtype=np.int64),
        user0 + active_users,
        user_quota[active_users].astype(np.int64, copy=False),
        np.zeros(active_users.size, dtype=np.int64),
    )

    edge_count = int(len(user_items))
    edge_starts = np.empty(edge_count, dtype=np.int64)

    for u in range(num_users):
        s, e = int(user_indptr[u]), int(user_indptr[u + 1])
        if e > s:
            edge_starts[s:e] = user0 + u

    edge_ends = item0 + user_items.astype(np.int64, copy=False)
    edge_costs = np.empty(edge_count, dtype=np.int64)

    for u in range(num_users):
        s, e = int(user_indptr[u]), int(user_indptr[u + 1])
        if e > s:
            items = user_items[s:e]
            edge_costs[s:e] = -np.rint(
                np.asarray(scores[u, items], dtype=np.float64) * SCORE_SCALE
            ).astype(np.int64)

    edge_arc_ids = solver.add_arcs_with_capacity_and_unit_cost(
        edge_starts,
        edge_ends,
        np.ones(edge_count, dtype=np.int64),
        edge_costs,
    )

    item_nodes = item0 + np.arange(num_items, dtype=np.int64)

    solver.add_arcs_with_capacity_and_unit_cost(
        item_nodes,
        np.full(num_items, cover_sink, dtype=np.int64),
        np.ones(num_items, dtype=np.int64),
        np.zeros(num_items, dtype=np.int64),
    )

    solver.add_arcs_with_capacity_and_unit_cost(
        item_nodes,
        np.full(num_items, sink, dtype=np.int64),
        np.full(num_items, total_slots, dtype=np.int64),
        np.zeros(num_items, dtype=np.int64),
    )

    supplies = [0] * num_nodes
    supplies[source] = total_slots
    supplies[sink] = -(total_slots - d_target)
    supplies[cover_sink] = -d_target

    for node, supply in enumerate(supplies):
        solver.set_node_supply(node, int(supply))

    t0 = time.time()
    status = solver.solve()
    solve_time = time.time() - t0

    if status != solver.OPTIMAL:
        return {
            "status": int(status),
            "solve_time": solve_time,
            "objective": None,
            "accuracy": None,
            "coverage": None,
            "selected_edges": None,
        }

    objective_scaled = 0
    item_flow = np.zeros(num_items, dtype=np.int32)
    selected_edges = 0

    for arc in edge_arc_ids:
        flow = solver.flow(int(arc))
        if flow <= 0:
            continue
        end = solver.head(int(arc))
        cost = solver.unit_cost(int(arc))
        selected_edges += flow
        item_flow[end - item0] += flow
        objective_scaled += -cost * flow

    objective = objective_scaled / SCORE_SCALE

    return {
        "status": "OPTIMAL",
        "solve_time": solve_time,
        "objective": objective,
        "accuracy": objective / total_slots if total_slots else 0.0,
        "coverage": int(np.count_nonzero(item_flow)),
        "selected_edges": int(selected_edges),
    }


def run_dataset(cfg: DatasetConfig):
    print(f"\n=== {cfg.name} ===", flush=True)

    if not cfg.scores_path.exists():
        raise FileNotFoundError(cfg.scores_path)
    if not cfg.cand_path.exists():
        raise FileNotFoundError(cfg.cand_path)

    scores = np.load(cfg.scores_path, mmap_mode="r")
    user_indptr, user_items, user_quota, items_with_candidates = load_candidate_arrays(scores, cfg.cand_path)

    num_users, num_items = scores.shape
    total_slots = int(user_quota.sum())
    candidate_edges = int(len(user_items))

    start_all = time.time()
    naive = compute_naive(scores, user_indptr, user_items, user_quota)
    naive_time = time.time() - start_all

    meta = {
        "dataset": cfg.name,
        "N": N,
        "num_users": int(num_users),
        "num_items": int(num_items),
        "candidate_edges": candidate_edges,
        "items_with_candidates": items_with_candidates,
        "total_slots": total_slots,
        "D_target_base": "items_with_candidates",
        "D_target_rounding": "ceil",
        "top_n_tie_break": "score_desc_item_id_asc",
        "engine": "ortools_simple_min_cost_flow_exact_algorithm_D_base",
        "scores_path": str(cfg.scores_path),
        "cand_path": str(cfg.cand_path),
    }

    print(
        f"shape={num_users:,}x{num_items:,}, slots={total_slots:,}, "
        f"candidate_edges={candidate_edges:,}, items_with_candidates={items_with_candidates:,}",
        flush=True,
    )

    rows = []
    last_good = None

    for iteration, pct in enumerate(D_PERCENTAGES):
        raw_target = int(math.ceil(items_with_candidates * pct / 100.0))
        model_d_target = min(raw_target, items_with_candidates, total_slots)
        target_feasible_by_candidates = raw_target <= min(items_with_candidates, total_slots)

        print(
            f"Solving {cfg.name} D-{pct}%: "
            f"D_target={raw_target:,}, model_D_target={model_d_target:,}",
            flush=True,
        )

        if raw_target == 0:
            # The unconstrained problem decomposes by user. Returning the
            # canonical Top-N optimum prevents solver-dependent tie choices.
            result = {
                "status": "CANONICAL_TOP_N",
                "solve_time": naive_time,
                "objective": naive["objective"],
                "accuracy": naive["accuracy"],
                "coverage": naive["coverage"],
                "selected_edges": total_slots,
            }
        else:
            result = solve_network_flow_for_d(
                scores=scores,
                user_indptr=user_indptr,
                user_items=user_items,
                user_quota=user_quota,
                d_target=model_d_target,
            )

        elapsed = time.time() - start_all
        target_feasible = bool(
            target_feasible_by_candidates
            and result["coverage"] is not None
            and result["coverage"] >= raw_target
        )

        row = {
            "数据集": cfg.name,
            "D比例": f"D-{pct}%",
            "D_target": raw_target,
            "D_target_base": "items_with_candidates",
            "D_target_rounding": "ceil",
            "top_n_tie_break": "score_desc_item_id_asc",
            "model_D_target": model_d_target,
            "迭代次数": iteration,
            "总体多样性": result["coverage"],
            "目标函数值": result["objective"],
            "准确多样性": result["accuracy"],
            "时间s": elapsed,
            "单次求解时间s": result["solve_time"],
            "target_feasible": target_feasible,
            "status": result["status"],
        }

        rows.append(row)

        if result["objective"] is not None:
            last_good = row

        print(
            f"  status={result['status']}, coverage={result['coverage']}, "
            f"objective={result['objective']}, elapsed={elapsed:.1f}s",
            flush=True,
        )

    summary = {
        "数据集": cfg.name,
        "N": N,
        "总时间s": time.time() - start_all,
        "目标函数值": None if last_good is None else last_good["目标函数值"],
        "准确多样性": None if last_good is None else last_good["准确多样性"],
        "总体多样性": None if last_good is None else last_good["总体多样性"],
        "Naive目标函数值": naive["objective"],
        "Naive准确多样性": naive["accuracy"],
        "Naive总体多样性": naive["coverage"],
        "target_feasible": None if last_good is None else last_good["target_feasible"],
        "max_reached": None if last_good is None else last_good["总体多样性"],
        "total_recs": total_slots,
    }

    return rows, summary, meta


def write_outputs(rows: list[dict], summaries: list[dict], metas: list[dict]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="D比例结果", index=False)
        pd.DataFrame(summaries).to_excel(writer, sheet_name="总结果", index=False)
        pd.DataFrame(metas).to_excel(writer, sheet_name="meta", index=False)

    wb = openpyxl.load_workbook(OUT_XLSX)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.font = openpyxl.styles.Font(bold=True)
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 12), 42)
    wb.save(OUT_XLSX)

    OUT_JSON.write_text(
        json.dumps(
            {
                "percent_results": rows,
                "summary": summaries,
                "meta": metas,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {OUT_XLSX}", flush=True)
    print(f"Saved: {OUT_JSON}", flush=True)


def main():
    all_rows = []
    summaries = []
    metas = []

    for name in ["OP", "Yelp"]:
        rows, summary, meta = run_dataset(DATASETS[name])
        all_rows.extend(rows)
        summaries.append(summary)
        metas.append(meta)
        write_outputs(all_rows, summaries, metas)


if __name__ == "__main__":
    main()