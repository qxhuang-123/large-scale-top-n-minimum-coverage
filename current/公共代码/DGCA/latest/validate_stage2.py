"""Validate stage-two-only optimization; first-stage original code is reused."""
import hashlib
import json
from pathlib import Path
import numpy as np
import bgcr_d_bench as baseline
import dgca_stage2 as stage2

HERE = Path(__file__).resolve().parent

def arrays(rows):
    ptr, items, scores = [0], [], []
    for row in rows:
        row = sorted(row, key=lambda pair: (-int(pair[1]), int(pair[0])))
        items.extend(j for j, _ in row)
        scores.extend(q for _, q in row)
        ptr.append(len(items))
    return np.array(ptr, np.int32), np.array(items, np.int32), np.array(scores, np.int64)


def oracle(a, I, N, D):
    row, items, q = a
    selected = (row[:-1, None] + np.arange(N)).astype(np.int32)
    counts = np.bincount(items[selected.ravel()], minlength=I)
    K = np.count_nonzero(counts)
    events = []
    while K < D:
        best = None
        for u, chosen in enumerate(selected):
            for pos, old in enumerate(chosen):
                if counts[items[old]] < 2:
                    continue
                for edge in range(row[u], row[u + 1]):
                    j = int(items[edge])
                    if counts[j]:
                        continue
                    key = (int(q[old]) - int(q[edge]), j, u, pos, edge)
                    if best is None or key < best:
                        best = key
        if best is None:
            break
        _, j, u, pos, edge = best
        old = int(selected[u, pos])
        counts[items[old]] -= 1
        counts[j] += 1
        selected[u, pos] = edge
        events.append((u, pos, old, edge))
        K += 1
    return np.asarray(events, np.int32).reshape(-1, 4), selected



def main():
    calls = 0
    modes = ('dynamic_user_heap_cached', 'dynamic_user_grouped_cached')
    for seed in range(600):
        rng = np.random.default_rng(20261107000 + seed)
        U, I = int(rng.integers(1, 10)), int(rng.integers(2, 18))
        N = int(rng.integers(1, min(I, 5) + 1))
        if seed % 40 == 0:
            U, N = 3, [15, 20, 25, 30][(seed // 40) % 4]
            I = N + 5
        rows = []
        for _ in range(U):
            ids = rng.choice(I, int(rng.integers(N, I + 1)), replace=False)
            q = rng.integers(-5, 6, len(ids), dtype=np.int64) * 1000000
            if seed % 7 == 0:
                q[:] = 5000000
            if seed % 5 == 0:
                q = rng.integers(-2000000, 6000000, len(ids), dtype=np.int64)
            rows.append(list(zip(ids, q)))
        a = arrays(rows)
        floating = a[2].astype(np.float64)
        K0 = len(np.unique(a[1][a[0][:-1, None] + np.arange(N)]))
        for D in sorted({0, K0, min(I, K0 + 1), min(I, K0 + 3), I}):
            events, selected = oracle(a, I, N, D)
            ref = baseline.solve_user(a[0], a[1], floating, I, N, D)
            assert np.array_equal(events, ref['events'])
            for mode in modes:
                r = stage2.solve_user(a[0], a[1], floating, I, N, D, mode)
                assert np.array_equal(events, r['events']), (seed, D, mode, 'events')
                assert np.array_equal(selected, r['selected']), (seed, D, mode, 'selection')
                assert np.array_equal(ref['history'][:, :2], r['history'][:, :2])
                for key in ('initial_coverage','initial_objective','coverage','objective','target_reached','swaps'):
                    assert r[key] == ref[key], (seed, D, mode, key)
                q_total = sum(map(int, a[2][selected.ravel()]))
                assert q_total == r['objective']
                counts=np.bincount(a[1][selected.ravel()],minlength=I)
                assert np.count_nonzero(counts)==r['coverage']
                if r['coverage'] < D:
                    assert all(not np.any(counts[a[1][selected[u]]] >= 2) or
                               not np.any(counts[a[1][a[0][u]:a[0][u+1]]] == 0)
                               for u in range(U))
                calls += 1
        if (seed+1)%100==0: print('validated',seed+1,'graphs;',calls,'calls',flush=True)
    source = HERE/'vendor/reference_core.cpp'
    original = HERE.parent/'native/bgcr_d_bench.cpp'
    assert source.read_bytes()==original.read_bytes()
    result=dict(passed=True,graphs=600,stage2_calls=calls,
                original_constructor_and_swap_reused_verbatim=True,
                checks=['independent exhaustive greedy oracle','complete events','selected edges',
                        'initial Top-N plan via zero target','initial objective and coverage',
                        'full coverage/objective trajectories','terminal direct-neighborhood exhaustion'],
                source_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in
                    ('stage2.cpp','vendor/reference_core.cpp','validate_stage2.py')})
    (HERE/'validation_stage2.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
