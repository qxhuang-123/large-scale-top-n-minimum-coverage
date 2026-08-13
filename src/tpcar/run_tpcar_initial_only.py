"""Report the shared TPCAR/BGCR coverage-relaxed Top-N solution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.common.initial_topn import initial_topn


def load_candidate_arrays(scores: np.ndarray, path: Path):
    with np.load(path) as cache:
        user_indptr = cache["user_indptr"].astype(np.int64, copy=False)
        item_key = "user_items" if "user_items" in cache.files else "user_indices"
        user_items = cache[item_key].astype(np.int32, copy=False)
        if "user_scores" in cache.files:
            user_scores = cache["user_scores"].astype(np.float32, copy=False)
        else:
            user_scores = np.empty(len(user_items), dtype=np.float32)
            for user in range(scores.shape[0]):
                start, end = int(user_indptr[user]), int(user_indptr[user + 1])
                user_scores[start:end] = scores[user, user_items[start:end]]
        item_indptr = cache["item_indptr"].astype(np.int64, copy=False)
        item_users = cache["item_users"].astype(np.int32, copy=False)
    return user_indptr, user_items, user_scores, item_indptr, item_users


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scores = np.load(args.scores, mmap_mode="r")
    user_indptr, user_items, user_scores, item_indptr, _ = load_candidate_arrays(scores, args.candidates)
    _, _, _, objective, total_recs, diversity = initial_topn(
        scores.shape[0], scores.shape[1], user_indptr, user_items, user_scores, args.n
    )
    result = {
        "dataset": args.dataset, "N": args.n, "mode": "TPCAR_INITIAL_ONLY",
        "objective": float(objective),
        "average_score": float(objective) / total_recs if total_recs else 0.0,
        "initial_D": int(diversity), "total_recs": int(total_recs),
        "items_with_candidates": int(np.count_nonzero(np.diff(item_indptr) > 0)),
        "initialization": "score_desc_item_id_asc",
    }
    output = args.output or Path("outputs") / "tpcar" / f"{args.dataset.lower()}_initial_N{args.n}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved_json={output}")


if __name__ == "__main__":
    main()