"""Portable input contract shared by TPCAR, BGCR, and MCF."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ALPHA = 0.40
N = 10
SEED = 20260704
D_PERCENTAGES = (0, 20, 40, 60, 80, 100)


@dataclass(frozen=True)
class SharedDatasetInput:
    scores: Path
    candidates: Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TOPN_DATA_ROOT", REPOSITORY_ROOT / "data")).resolve()
PROCESSED_ROOT = DATA_ROOT / "processed"

def _processed(dataset: str) -> SharedDatasetInput:
    folder = PROCESSED_ROOT / dataset.lower()
    return SharedDatasetInput(
        folder / "scores_float32.npy",
        folder / f"candidates_alpha_0p4_seed{SEED}.npz",
    )


SHARED_INPUTS: dict[str, SharedDatasetInput] = {name: _processed(name) for name in ("OP", "Yelp", "VG", "TG", "SO")}


def shared_input(dataset: str) -> SharedDatasetInput:
    try:
        return SHARED_INPUTS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {', '.join(SHARED_INPUTS)}") from exc
