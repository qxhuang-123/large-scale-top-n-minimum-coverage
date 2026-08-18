"""Create the one portable score/candidate cache consumed by all solvers."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.common.shared_experiment_inputs import ALPHA, DATA_ROOT, N, shared_input


def load_scores(paths: list[Path]) -> np.ndarray:
    blocks = []
    for path in paths:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(path, index_col=0)
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, index_col=0)
        else:
            raise ValueError(f"Unsupported score file: {path}")
        blocks.append(frame.to_numpy(dtype=np.float32, copy=True))
    return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)


def build_cache(scores: np.ndarray) -> dict[str, np.ndarray]:
    users, items = scores.shape
    user_indptr = np.zeros(users + 1, dtype=np.int64)
    rows, values, reverse = [], [], [[] for _ in range(items)]
    total = 0
    for user, row in enumerate(scores):
        eligible = np.flatnonzero(np.isfinite(row))
        keep = min(eligible.size, max(N, math.ceil(ALPHA * eligible.size)))
        selected = eligible[np.lexsort((eligible, -row[eligible]))[:keep]].astype(np.int32, copy=False)
        rows.append(selected); values.append(row[selected].astype(np.float32, copy=False))
        for item in selected: reverse[int(item)].append(user)
        total += selected.size; user_indptr[user + 1] = total
    user_items = np.concatenate(rows)
    item_indptr = np.zeros(items + 1, dtype=np.int64)
    item_users = []; cursor = 0
    for item, user_list in enumerate(reverse):
        arr = np.asarray(user_list, dtype=np.int32); item_users.append(arr)
        cursor += arr.size; item_indptr[item + 1] = cursor
    item_users = np.concatenate(item_users)
    item_ids = np.repeat(np.arange(items, dtype=np.int32), np.diff(item_indptr))
    return {"user_indptr": user_indptr, "user_items": user_items,
            "user_scores": np.concatenate(values), "item_indptr": item_indptr,
            "item_users": item_users, "item_scores": scores[item_users, item_ids].astype(np.float32, copy=False)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/datasets.example.yaml"))
    parser.add_argument("--datasets", nargs="*", choices=("OP", "Yelp", "VG", "TG", "SO"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for name in args.datasets or list(cfg["datasets"]):
        entry = cfg["datasets"][name]; source = entry["scores"]
        paths = [DATA_ROOT / x for x in (source if isinstance(source, list) else [source])]
        scores = load_scores(paths)
        if "expected_shape" in entry and tuple(scores.shape) != tuple(entry["expected_shape"]):
            raise ValueError(f"{name}: got {scores.shape}, expected {tuple(entry['expected_shape'])}")
        target = shared_input(name); target.scores.parent.mkdir(parents=True, exist_ok=True)
        np.save(target.scores, scores); np.savez_compressed(target.candidates, **build_cache(scores))
        print(f"{name}: {target.scores} | {target.candidates}")


if __name__ == "__main__":
    main()
