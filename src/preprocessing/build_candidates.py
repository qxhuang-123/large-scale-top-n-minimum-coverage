"""Build per-user candidate CSR caches from configured score matrices."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_scores(value: str | list[str]) -> np.ndarray:
    paths = [Path(value)] if isinstance(value, str) else [Path(item) for item in value]
    parts = []
    for path in paths:
        if path.suffix.lower() == ".npy":
            part = np.load(path, mmap_mode="r")
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            part = pd.read_excel(path, index_col=0).to_numpy(dtype=np.float32)
        elif path.suffix.lower() == ".csv":
            part = pd.read_csv(path, index_col=0).to_numpy(dtype=np.float32)
        else:
            raise ValueError(f"Unsupported score format: {path}")
        parts.append(np.asarray(part, dtype=np.float32))
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


def build_csr(scores: np.ndarray, alpha: float, n: int) -> dict[str, np.ndarray]:
    indptr = [0]
    items = []
    values = []
    quotas = []
    for row in scores:
        eligible = np.flatnonzero(np.isfinite(row))
        quota = min(n, eligible.size)
        keep = min(eligible.size, max(n, math.ceil(alpha * eligible.size)))
        order = np.argsort(-row[eligible], kind="stable")[:keep]
        selected = eligible[order]
        items.append(selected.astype(np.int32, copy=False))
        values.append(row[selected].astype(np.float32, copy=False))
        indptr.append(indptr[-1] + keep)
        quotas.append(quota)
    return {
        "user_indptr": np.asarray(indptr, dtype=np.int64),
        "user_items": np.concatenate(items) if items else np.empty(0, dtype=np.int32),
        "user_scores": np.concatenate(values) if values else np.empty(0, dtype=np.float32),
        "user_quota": np.asarray(quotas, dtype=np.int32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/datasets.yaml"))
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    defaults = config["defaults"]
    names = args.datasets or list(config["datasets"])
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        entry = config["datasets"][name]
        scores = load_scores(entry["scores"])
        expected = entry.get("expected_shape")
        if expected and list(scores.shape) != expected:
            raise ValueError(f"{name}: got {scores.shape}, expected {tuple(expected)}")
        arrays = build_csr(scores, float(defaults["alpha"]), int(defaults["list_length"]))
        target = output_dir / f"{name.lower()}_candidates_alpha_0p4.npz"
        np.savez_compressed(target, **arrays)
        print(f"{name}: shape={scores.shape}, edges={arrays['user_items'].size:,} -> {target}")


if __name__ == "__main__":
    main()
