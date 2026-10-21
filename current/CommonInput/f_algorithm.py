"""F: exact local repair + boundary heaps + source and terminal aggregation.

The C++ sources are byte-identical to the previous Yelp experiment. This file
adds input validation and a general interface; it never changes the algorithm.
Scores may be zero/negative. Eligibility must be specified explicitly when
zero denotes a previously observed rating in a particular dataset.
"""
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np

KEYS = ("user_indptr", "user_items", "user_scores", "edge_users",
        "item_indptr", "item_users", "item_edges", "item_scores")
INT_MAX = np.iinfo(np.int32).max


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _ids(values, name):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a 1-D integer array")
    if a.size and (a.min() < 0 or a.max() >= INT_MAX):
        raise ValueError(f"{name} exceeds the supported nonnegative int32 range")
    return np.ascontiguousarray(a, dtype=np.int32)


@dataclass
class CandidateGraph:
    n_users: int
    n_items: int
    arrays: tuple

    @classmethod
    def from_edges(cls, n_users, n_items, users, items, scores):
        """Create both CSR orientations; IDs start at zero; no duplicate pairs."""
        U, I = _integer(n_users, "n_users"), _integer(n_items, "n_items")
        if min(U, I) <= 0 or U + I + 2 >= INT_MAX:
            raise ValueError("Invalid graph dimensions")
        u, j = _ids(users, "users"), _ids(items, "items")
        r = np.asarray(scores, dtype=np.float64)
        if r.ndim != 1 or len(u) != len(j) or len(u) != len(r):
            raise ValueError("users, items and scores must have equal 1-D shapes")
        E = len(u)
        if E >= INT_MAX or not np.all(np.isfinite(r)):
            raise ValueError("Too many edges or nonfinite candidate scores")
        if E and (u.max() >= U or j.max() >= I):
            raise ValueError("Candidate ID outside the declared graph")
        order = np.lexsort((j, -r, u))
        u, j, r = u[order], j[order], r[order]
        up = np.r_[0, np.cumsum(np.bincount(u, minlength=U))].astype(np.int32)
        ie = np.argsort(j, kind="stable").astype(np.int32)
        ip = np.r_[0, np.cumsum(np.bincount(j, minlength=I))].astype(np.int32)
        graph = cls(U, I, (up, j, r, u, ip, u[ie], ie, r[ie]))
        graph.validate()
        return graph

    @classmethod
    def from_matrix(cls, scores, eligible=None):
        """Finite matrix entries are eligible by default, including zeros."""
        r = np.asarray(scores, dtype=np.float64)
        if r.ndim != 2:
            raise ValueError("scores must be a users-by-items matrix")
        mask = np.isfinite(r) if eligible is None else np.asarray(eligible, dtype=bool)
        if mask.shape != r.shape or not np.all(np.isfinite(r[mask])):
            raise ValueError("Bad eligibility mask or nonfinite eligible score")
        u, j = np.nonzero(mask)
        return cls.from_edges(*r.shape, u, j, r[u, j])

    @classmethod
    def load(cls, path):
        """Load an NPZ or the eight NPY arrays used by the Yelp experiment."""
        path = Path(path)
        if path.is_dir():
            a = tuple(np.load(path / f"{k}.npy", allow_pickle=False) for k in KEYS)
        else:
            with np.load(path, allow_pickle=False) as z:
                a = tuple(z[k].copy() for k in KEYS)
        fixed = tuple(np.ascontiguousarray(x, dtype=np.float64)
                      if k.endswith("scores") else _ids(x, k)
                      for k, x in zip(KEYS, a))
        graph = cls(len(fixed[0]) - 1, len(fixed[4]) - 1, fixed)
        graph.validate()
        return graph

    def save(self, path):
        np.savez(path, **dict(zip(KEYS, self.arrays)))

    def validate(self):
        """Validate both sparse views before passing pointers to native code."""
        U, I = self.n_users, self.n_items
        if min(U, I) <= 0 or U + I + 2 >= INT_MAX or len(self.arrays) != 8:
            raise ValueError("Invalid graph dimensions")
        up, j, r, u, ip, iu, ie, ir = self.arrays
        E = len(j)
        if E >= INT_MAX or any(a.ndim != 1 or not a.flags.c_contiguous for a in self.arrays):
            raise ValueError("CSR arrays must be 1-D, contiguous and int32-sized")
        if any(a.dtype != (np.float64 if k.endswith("scores") else np.int32)
               for k, a in zip(KEYS, self.arrays)):
            raise ValueError("CSR indices require int32; scores require float64")
        if len(up) != U+1 or len(ip) != I+1 or any(len(a) != E for a in (r,u,iu,ie,ir)):
            raise ValueError("Inconsistent CSR lengths")
        for ptr in (up, ip):
            if ptr[0] != 0 or ptr[-1] != E or np.any(np.diff(ptr) < 0):
                raise ValueError("Invalid CSR pointer array")
        if E and (j.min() < 0 or j.max() >= I or u.min() < 0 or u.max() >= U
                  or ie.min() < 0 or ie.max() >= E):
            raise ValueError("CSR index out of bounds")
        if not np.all(np.isfinite(r)):
            raise ValueError("Nonfinite scores")
        for v in range(U):
            lo, hi = int(up[v]), int(up[v+1]); row, score = j[lo:hi], r[lo:hi]
            if np.any(u[lo:hi] != v) or len(np.unique(row)) != len(row):
                raise ValueError("Duplicate user-item pair or inconsistent edge_users")
            if np.any(score[1:] > score[:-1]) or np.any((score[1:] == score[:-1]) & (row[1:] < row[:-1])):
                raise ValueError("Each row must be sorted by score DESC, item ID ASC")
        for item in range(I):
            lo, hi = int(ip[item]), int(ip[item+1]); edges = ie[lo:hi]
            if np.any(edges[1:] <= edges[:-1]) or np.any(j[edges] != item):
                raise ValueError("Invalid item-column edge ordering or membership")
            if not np.array_equal(iu[lo:hi], u[edges]) or not np.array_equal(ir[lo:hi], r[edges]):
                raise ValueError("The two CSR orientations disagree")


def audit_solution(result, graph, N):
    """Check recommendation counts, candidate membership, score and coverage."""
    U, I = graph.n_users, graph.n_items
    a = graph.arrays; e = np.asarray(result["selected_edges"], dtype=np.int32)
    assert len(e) == U*N and len(np.unique(e)) == len(e)
    assert np.all((e >= 0) & (e < len(a[1])))
    assert np.all(np.bincount(a[3][e], minlength=U) == N)
    counts = np.bincount(a[1][e], minlength=I)
    assert np.count_nonzero(counts) == result["tpcar_D"]
    error = abs(float(a[2][e].sum()) - result["tpcar_obj"])
    assert error <= 1e-7 + 1e-12*abs(result["tpcar_obj"])
    return e, counts, error


def audit_maximum_coverage(result, graph, selected, counts):
    """Check a closed reachable set in the original, uncompressed residual graph.

    This certifies maximum attainable coverage, not score optimality by itself.
    It is only used after an infeasible target exhausts augmentation.
    """
    U, I = graph.n_users, graph.n_items
    a = graph.arrays
    reachable = np.asarray(result["reachable_nodes"], dtype=bool)
    assert reachable.shape == (U+I+2,) and reachable[I+U]
    ri, ru = reachable[:I], reachable[I:I+U]
    assert not np.any((counts >= 2) & ~ri)
    assert not np.any(ri[a[1][selected]] & ~ru[a[3][selected]])
    mask = np.zeros(len(a[1]), dtype=bool); mask[selected] = True
    for lo in range(0, len(mask), 1_000_000):
        sl = slice(lo, lo+1_000_000)
        assert not np.any(ru[a[3][sl]] & ~ri[a[1][sl]] & ~mask[sl])
    assert not np.any(ri & (counts == 0))
    return True


def solve_graph(graph, N, D, progress_every=0):
    """Run F only. Graph construction/validation and auditing are timed separately.

    Returns the original diagnostics plus readable aliases and recommendations.
    Only target_feasible=False certifies coverage is globally maximal; an early
    successful target does not compute the maximum possible coverage.
    """
    N, D = _integer(N, "N"), _integer(D, "D")
    progress_every = _integer(progress_every, "progress_every")
    U, I = graph.n_users, graph.n_items
    if N <= 0 or U*N >= INT_MAX or not 0 <= D <= I or progress_every < 0:
        raise ValueError("Require N>0, 0<=D<=n_items and nonnegative progress interval")
    t = time.perf_counter(); graph.validate()
    if np.any(np.diff(graph.arrays[0]) < N):
        raise ValueError("Every user must have at least N distinct eligible candidates")
    validation_seconds = time.perf_counter()-t
    try:
        import state_variants as core
    except ImportError as ex:
        raise RuntimeError("First run: python setup.py build_ext --inplace") from ex
    start = time.perf_counter()
    result = dict(core.run(U, I, *graph.arrays, N, D, progress_every, "state_source"))
    call_seconds = time.perf_counter()-start
    t = time.perf_counter()
    selected, counts, error = audit_solution(result, graph, N)
    certificate = None
    if not result["target_feasible"]:
        certificate = audit_maximum_coverage(result, graph, selected, counts)
    order = np.lexsort((graph.arrays[1][selected], -graph.arrays[2][selected], graph.arrays[3][selected]))
    recs = graph.arrays[1][selected[order]].reshape(U, N).tolist()
    result.update(algorithm="F", mode="state_source", requested_coverage=D,
                  coverage=int(result["tpcar_D"]), score=float(result["tpcar_obj"]),
                  initial_coverage=int(result["naive_D"]), recommendations=recs,
                  call_wall_seconds=call_seconds, input_validation_seconds=validation_seconds,
                  audit_seconds=time.perf_counter()-t, score_recompute_abs_error=error,
                  maximum_coverage_certified=certificate)
    return result


def solve(scores, N, D, eligible=None, progress_every=0):
    """Convenience entry for a dense matrix; prefer CandidateGraph for large data."""
    return solve_graph(CandidateGraph.from_matrix(scores, eligible), N, D, progress_every)
