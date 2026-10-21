"""Independent small LP checks of all four standardized benchmark adapters."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.optimize import linprog

from bench_common import ROOT, SCALE, SharedGraph, write_json
from bench_solvers import solve_tpcar, solve_mcf, solve_gurobi


def make_graph(folder, U, I, rows, values):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    up = [0]; items = []; integer = []; raw = []
    for j, r in zip(rows, values):
        j = np.asarray(j, dtype=np.int32); r = np.asarray(r, dtype=np.float64)
        q = np.rint(r * SCALE).astype(np.int64); order = np.lexsort((j, -q))
        items.extend(j[order]); integer.extend(q[order]); raw.extend(r[order]); up.append(len(items))
    for name, a in [('user_indptr', np.array(up, dtype=np.int32)), ('user_items', np.array(items, dtype=np.int32)),
                    ('scores_integer', np.array(integer, dtype=np.int64)), ('scores_original', np.array(raw, dtype=np.float64))]:
        np.save(folder / (name + '.npy'), a)
    write_json(folder / 'graph.json', dict(n_users=U, n_items=I, edges=len(items),
        candidate_items=len(set(items)), max_abs_integer_score=max(abs(x) for x in integer),
        graph_id=hashlib.sha256(np.array(integer).tobytes() + np.array(items).tobytes()).hexdigest()))
    return SharedGraph(folder)


def oracle(g, N, D):
    E = g.E; up = np.array(g.up); users = np.repeat(np.arange(g.U), np.diff(up))
    eq = np.zeros((g.U, E + g.I)); eq[users, np.arange(E)] = 1
    ub = np.zeros((g.I + 1, E + g.I)); ub[g.items, np.arange(E)] = -1
    ub[np.arange(g.I), E + np.arange(g.I)] = 1; ub[-1, E:] = -1
    rhs = np.zeros(g.I + 1); rhs[-1] = -D
    return linprog(np.r_[-np.asarray(g.q, dtype=np.float64), np.zeros(g.I)],
        A_eq=eq, b_eq=np.full(g.U, N), A_ub=ub, b_ub=rhs, bounds=(0, 1), method='highs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-gurobi', action='store_true', help='Validate TPCA/MCF without a Gurobi license')
    args = parser.parse_args()
    methods = ('TPCAR', 'MCF') if args.skip_gurobi else ('TPCAR', 'MCF', 'GUROBI-LP', 'GUROBI-MIP')
    (ROOT / 'validation').mkdir(exist_ok=True)
    stats = dict(instances=12, adapter_calls=0, feasible_calls=0, infeasible_calls=0, max_integer_objective_error=0)
    with tempfile.TemporaryDirectory(prefix='adapters_', dir=str(ROOT / 'validation')) as tmp:
        task = Path(tmp).resolve()
        assert task.is_relative_to((ROOT / 'validation').resolve())
        for seed in range(stats['instances']):
            rng = np.random.default_rng(20261018 + seed)
            U, I, N = int(rng.integers(3, 7)), int(rng.integers(5, 10)), 2
            rows = []; values = []
            for u in range(U):
                size = int(rng.integers(N, I + 1)); j = rng.choice(I, size, replace=False)
                if seed % 3 == 0: j = j[j != I - 1]
                if len(j) < N: j = np.arange(N)
                r = rng.integers(-2, 6, len(j)).astype(float) + rng.choice([0, 1e-7, 5e-7, 8e-7], len(j))
                rows.append(j); values.append(r)
            g = make_graph(task / ('g%d' % seed), U, I, rows, values)
            K0, initial = g.initial(N)
            for D in sorted(set([0, min(K0 + 1, I), I])):
                ref = oracle(g, N, D)
                if ref.success: expected = int(round(-ref.fun))
                else:
                    assert ref.status == 2
                    expected = None
                for method in methods:
                    folder = task / ('%d_%d_%s' % (seed, D, method)); folder.mkdir()
                    args = dict(g=g, N=N, D=D, folder=folder, phase=lambda *a, **k: None)
                    if method == 'TPCAR': result = solve_tpcar(**args)
                    elif method == 'MCF': result = solve_mcf(**args)
                    else: result = solve_gurobi(**args, mip=method == 'GUROBI-MIP', budget_seconds=30)
                    stats['adapter_calls'] += 1
                    if expected is not None:
                        assert result['status'] == 'OPTIMAL', (seed, D, method, result['status'])
                        error = abs(result['audit']['integer_objective'] - expected)
                        assert error == 0, (seed, D, method, error)
                        stats['feasible_calls'] += 1
                    else:
                        assert result['status'] == 'INFEASIBLE', (seed, D, method, result['status'])
                        if method == 'TPCAR':
                            assert result['maximum_coverage_certified']
                            boundary = oracle(g, N, result['audit']['coverage'])
                            assert boundary.success and abs(result['audit']['integer_objective'] + boundary.fun) < .1
                        stats['infeasible_calls'] += 1
            g.close()
    stats['passed'] = True
    write_json(ROOT / 'validation' / 'adapter_validation.json', stats)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__': main()
