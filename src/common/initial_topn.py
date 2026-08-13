"""Deterministic Top-N initialization shared by BGCR and TPCAR."""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:
    njit = None


if njit is not None:

    @njit(cache=True)
    def initial_topn_numba(num_users, num_items, user_indptr, user_items, user_scores, n):
        selected_items = np.full((num_users, n), -1, dtype=np.int32)
        selected_scores = np.zeros((num_users, n), dtype=np.float32)
        counts = np.zeros(num_items, dtype=np.int32)
        total_score = 0.0
        total_recs = 0
        for u in range(num_users):
            s = user_indptr[u]
            e = user_indptr[u + 1]
            length = e - s
            if length <= 0:
                continue
            k = n if n < length else length
            order = np.argsort(user_scores[s:e])[::-1]
            # Sort item ids only inside equal-score groups; avoids O(length^2).
            group_start = 0
            while group_start < length:
                group_end = group_start + 1
                base_score = user_scores[s + order[group_start]]
                while group_end < length and user_scores[s + order[group_end]] == base_score:
                    group_end += 1
                for i in range(group_start + 1, group_end):
                    key = order[i]
                    key_item = user_items[s + key]
                    j = i - 1
                    while j >= group_start and user_items[s + order[j]] > key_item:
                        order[j + 1] = order[j]
                        j -= 1
                    order[j + 1] = key
                group_start = group_end
            for pos in range(k):
                edge = s + order[pos]
                item = user_items[edge]
                selected_items[u, pos] = item
                selected_scores[u, pos] = user_scores[edge]
                counts[item] += 1
                total_score += user_scores[edge]
                total_recs += 1
        diversity = 0
        for item in range(num_items):
            if counts[item] > 0:
                diversity += 1
        return selected_items, selected_scores, counts, total_score, total_recs, diversity


if njit is None:
    initial_topn_numba = _initial_topn_impl

def initial_topn_from_arrays(user_indptr, user_items, user_scores, num_items, n):
    if njit is None:
        raise RuntimeError("numba is required for deterministic Top-N initialization")
    return initial_topn_numba(
        int(len(user_indptr) - 1), int(num_items),
        np.asarray(user_indptr, dtype=np.int64),
        np.asarray(user_items, dtype=np.int32),
        np.asarray(user_scores, dtype=np.float32), int(n),
    )


def initial_topn(num_users, num_items, user_indptr, user_items, user_scores, n):
    return initial_topn_numba(int(num_users), int(num_items), np.asarray(user_indptr, dtype=np.int64), np.asarray(user_items, dtype=np.int32), np.asarray(user_scores, dtype=np.float32), int(n))
