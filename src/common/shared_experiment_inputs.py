"""Authoritative input contract shared by TPCAR, BGCR, and MCF.

Every method must load these *same immutable score and candidate-cache files*.
The cache already encodes the alpha=0.40 screened graph and its deterministic
candidate ordering; no solver is permitted to rebuild it implicitly.
"""
from __future__ import annotations

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


_BGCR_DIR = Path(r"E:\PythonProject\.venv\NIPT_预测结果")

SHARED_INPUTS: dict[str, SharedDatasetInput] = {
    "OP": SharedDatasetInput(
        Path(r"C:\Users\24qxh\Documents\Codex\2026-07-05\import-json-import-sys-import-time-4\work\op_exact_cache\op_full_scores_float32.npy"),
        Path(r"C:\Users\24qxh\Documents\Codex\2026-07-05\import-json-import-sys-import-time-4\work\op_exact_cache\op_full_cand_frac_0p4_seed20260704.npz"),
    ),
    "Yelp": SharedDatasetInput(
        Path(r"C:\Users\24qxh\Documents\Codex\2026-07-04\new-chat\work\yelp_cache\yelp_unknown_scores_float32.npy"),
        Path(r"C:\Users\24qxh\Documents\Codex\2026-07-04\new-chat\work\yelp_cache\yelp_cand_frac_0p4_seed20260704.npz"),
    ),
    "VG": SharedDatasetInput(
        _BGCR_DIR / "VG贪心算法_cache" / "VG_scores_float32.npy",
        _BGCR_DIR / "VG贪心算法_cache" / "VG_cand_pos_top40_seed20260704_v2.npz",
    ),
    "TG": SharedDatasetInput(
        _BGCR_DIR / "TG贪心算法_cache" / "TG_scores_float32.npy",
        _BGCR_DIR / "TG贪心算法_cache" / "TG_cand_pos_top40_seed20260704_v2.npz",
    ),
    "SO": SharedDatasetInput(
        _BGCR_DIR / "SO贪心算法_cache" / "SO_scores_float32.npy",
        _BGCR_DIR / "SO贪心算法_cache" / "SO_cand_pos_top40_seed20260704_v2.npz",
    ),
}


def shared_input(dataset: str) -> SharedDatasetInput:
    try:
        return SHARED_INPUTS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {', '.join(SHARED_INPUTS)}") from exc
