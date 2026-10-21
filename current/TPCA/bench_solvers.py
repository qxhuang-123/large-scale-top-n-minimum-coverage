"""Exact methods for the same rounded minimum-coverage problem."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from bench_common import F_ROOT, SCALE, audit_selected, close_array


def validate_compact_graph(graph):
    U, I = graph.n_users, graph.n_items
    up, j, r, u, ip, iu, ie = graph.arrays
    E = len(j)
    if any(a.ndim != 1 or not a.flags.c_contiguous for a in graph.arrays):
        raise ValueError('Noncontiguous compact CSR input')
    if r.dtype != np.float64 or any(a.dtype != np.int32 for a in (up, j, u, ip, iu, ie)):
        raise ValueError('Compact CSR dtype mismatch')
    if len(up) != U + 1 or len(ip) != I + 1 or any(len(a) != E for a in (r, u, iu, ie)):
        raise ValueError('Compact CSR shape mismatch')
    for p in (up, ip):
        if p[0] != 0 or p[-1] != E or np.any(np.diff(p) < 0): raise ValueError('Bad CSR pointers')
    if not np.all(np.isfinite(r)) or np.any(j < 0) or np.any(j >= I): raise ValueError('Bad candidates/scores')
    for v in range(U):
        lo, hi = int(up[v]), int(up[v + 1]); row, score = j[lo:hi], r[lo:hi]
        if np.any(u[lo:hi] != v) or len(np.unique(row)) != len(row): raise ValueError('Bad user row')
        if np.any(score[1:] > score[:-1]) or np.any((score[1:] == score[:-1]) & (row[1:] < row[:-1])):
            raise ValueError('Bad rounded-score ordering')
    for item in range(I):
        lo, hi = int(ip[item]), int(ip[item + 1]); edges = ie[lo:hi]
        if np.any(edges < 0) or np.any(edges >= E) or np.any(edges[1:] <= edges[:-1]): raise ValueError('Bad reverse edges')
        if np.any(j[edges] != item) or not np.array_equal(iu[lo:hi], u[edges]): raise ValueError('Reverse CSR disagreement')


def solve_tpcar(g, N, D, folder, phase):
    started = time.perf_counter(); phase('build')
    from f_algorithm import audit_maximum_coverage
    import state_variants_bench as state_variants
    with ExitStack() as stack:
        arrays = {}
        for key in ('user_scores', 'edge_users', 'item_users', 'item_edges'):
            arrays[key] = np.empty(g.E, dtype=np.float64 if key.endswith('scores') else np.int32)
        counts = np.zeros(g.I, dtype=np.int64)
        for lo in range(0, g.E, 1_000_000):
            counts += np.bincount(g.items[lo:lo + 1_000_000], minlength=g.I)
        ip = np.r_[0, np.cumsum(counts)].astype(np.int32)
        cursor = ip[:-1].astype(np.int64)
        for u in range(g.U):
            lo, hi = int(g.up[u]), int(g.up[u + 1])
            j = g.items[lo:hi]; q = g.q[lo:hi]
            arrays['user_scores'][lo:hi] = q
            arrays['edge_users'][lo:hi] = u
            positions = cursor[j]
            arrays['item_edges'][positions] = np.arange(lo, hi, dtype=np.int32)
            cursor[j] += 1
        # Build only the reverse edge IDs with scattered writes, then fill
        # the larger reverse payloads sequentially. Avoid simultaneous random
        # writes across several GB-scale mapped files.
        del q
        close_array(g.q)
        for lo in range(0, g.E, 500_000):
            hi = min(g.E, lo + 500_000)
            edges = arrays['item_edges'][lo:hi]
            arrays['item_users'][lo:hi] = arrays['edge_users'][edges]
        graph = SimpleNamespace(n_users=g.U, n_items=g.I, arrays=(g.up, g.items, arrays['user_scores'],
            arrays['edge_users'], ip, arrays['item_users'], arrays['item_edges']))
        index_assembly_seconds = time.perf_counter() - started
        phase('build', index_assembly_seconds=index_assembly_seconds, construction_stage='dedicated_index_validation')
        validation_started = time.perf_counter()
        validate_compact_graph(graph)
        index_validation_seconds = time.perf_counter() - validation_started
        build_seconds = time.perf_counter() - started
        phase('solve', build_seconds=build_seconds)
        start = time.perf_counter()
        result = dict(state_variants.run(g.U, g.I, *graph.arrays, N, D, 0, 'state_source'))
        solve_seconds = time.perf_counter() - start
        phase('audit', build_seconds=build_seconds, solve_seconds=solve_seconds)
        g.q = np.load(g.folder / 'scores_integer.npy', mmap_mode='r')
        selected = np.asarray(result.pop('selected_edges'), dtype=np.int32)
        audit = audit_selected(g, selected, N, D)
        if int(round(result['tpcar_obj'])) != audit['integer_objective'] or result['tpcar_obj'] != audit['integer_objective']:
            raise AssertionError('TPCAR native objective disagrees with exact integer recomputation')
        cert = None
        if not result['target_feasible']:
            item_counts = np.bincount(g.items[selected], minlength=g.I)
            cert = audit_maximum_coverage(result, graph, selected, item_counts)
        result.pop('reachable_nodes', None)
        history = result.pop('history', [])
        for row in history:
            row['trajectory_kind'] = 'exact_coverage_augmentation'
            if 'diversity' in row: row['coverage'] = int(row['diversity'])
            if 'objective' in row:
                row['integer_objective'] = int(round(row['objective']))
                row['rounded_score'] = row['integer_objective'] / SCALE
                row['rounded_APS'] = row['rounded_score'] / (g.U * N)
        return dict(status='OPTIMAL' if audit['target_feasible'] else 'INFEASIBLE',
                    build_seconds=build_seconds, solve_seconds=solve_seconds,
                    index_assembly_seconds=index_assembly_seconds,
                    index_validation_seconds=index_validation_seconds,
                    selected=selected, audit=audit, native_diagnostics=result,
                    maximum_coverage_certified=cert, trajectory=history,
                    native_score_units='integer_scores_q_before_division_by_scale',
                    trajectory_kind='exact_coverage_augmentation',
                    backend_file=str(state_variants.__file__))


def solve_mcf(g, N, D, folder, phase):
    phase('build'); started = time.perf_counter()
    from ortools.graph.python import min_cost_flow
    import ortools
    solver = min_cost_flow.SimpleMinCostFlow()
    source = 0; user0 = 1; item0 = g.U + 1
    cover = item0 + g.I; sink = cover + 1
    ids = np.arange(g.U, dtype=np.int32)
    solver.add_arcs_with_capacity_and_unit_cost(np.zeros(g.U, dtype=np.int32),
        user0 + ids, np.full(g.U, N, dtype=np.int64), np.zeros(g.U, dtype=np.int64))
    edge_offset = solver.num_arcs()
    for lo in range(0, g.E, 500_000):
        hi = min(g.E, lo + 500_000)
        edge_ids = np.arange(lo, hi, dtype=np.int64)
        users = np.searchsorted(g.up[1:], edge_ids, side='right').astype(np.int32)
        solver.add_arcs_with_capacity_and_unit_cost(user0 + users, item0 + g.items[lo:hi],
            np.ones(hi - lo, dtype=np.int64), -g.q[lo:hi])
    j = item0 + np.arange(g.I, dtype=np.int32)
    solver.add_arcs_with_capacity_and_unit_cost(j, np.full(g.I, cover, dtype=np.int32),
        np.ones(g.I, dtype=np.int64), np.zeros(g.I, dtype=np.int64))
    solver.add_arcs_with_capacity_and_unit_cost(j, np.full(g.I, sink, dtype=np.int32),
        np.full(g.I, N * g.U, dtype=np.int64), np.zeros(g.I, dtype=np.int64))
    nodes = np.array([source, cover, sink], dtype=np.int32)
    solver.set_nodes_supplies(nodes, np.array([N * g.U, -D, -(N * g.U - D)], dtype=np.int64))
    close_array(g.q)
    build_seconds = time.perf_counter() - started
    phase('solve', build_seconds=build_seconds)
    t = time.perf_counter(); status = solver.solve(); solve_seconds = time.perf_counter() - t
    phase('audit', build_seconds=build_seconds, solve_seconds=solve_seconds)
    g.q = np.load(g.folder / 'scores_integer.npy', mmap_mode='r')
    if status != solver.OPTIMAL:
        return dict(status='INFEASIBLE' if status == solver.INFEASIBLE else 'SOLVER_ERROR',
                    native_status=str(status), build_seconds=build_seconds, solve_seconds=solve_seconds,
                    selected=None, trajectory=[], trajectory_kind='endpoint_only_no_native_callback',
                    backend_version=ortools.__version__)
    selected = []
    for lo in range(0, g.E, 500_000):
        hi = min(g.E, lo + 500_000)
        flows = solver.flows(np.arange(edge_offset + lo, edge_offset + hi, dtype=np.int32))
        selected.append((np.flatnonzero(flows) + lo).astype(np.int32))
    selected = np.concatenate(selected)
    audit = audit_selected(g, selected, N, D)
    if not audit['target_feasible'] or -int(solver.optimal_cost()) != audit['integer_objective']:
        raise AssertionError('MCF solution failed the common exact audit')
    return dict(status='OPTIMAL', build_seconds=build_seconds, solve_seconds=solve_seconds,
                selected=selected, audit=audit, trajectory=[],
                trajectory_kind='endpoint_only_no_native_callback', backend_version=ortools.__version__)


def solve_gurobi(g, N, D, folder, phase, mip=False, budget_seconds=1800):
    phase('build'); started = time.perf_counter()
    import gurobipy as gp
    from scipy import sparse
    env = gp.Env(empty=True); env.setParam('OutputFlag', 0); env.start()
    model = gp.Model(env=env)
    trajectory = []; last_callback = [0.0]
    try:
        model.Params.Threads = 1; model.Params.Seed = 20261018
        model.Params.LogToConsole = 0; model.Params.OutputFlag = 1
        model.Params.LogFile = str(Path(folder) / 'gurobi.log')
        model.Params.FeasibilityTol = 1e-8; model.Params.OptimalityTol = 1e-9
        if mip:
            model.Params.MIPGap = 0; model.Params.MIPGapAbs = 0; model.Params.IntFeasTol = 1e-9
        else:
            model.Params.Method = 1  # simplex returns a basic LP solution
        x = model.addMVar(g.E + g.I, lb=0, ub=1,
                vtype=gp.GRB.BINARY if mip else gp.GRB.CONTINUOUS, name='xz')
        objective = np.r_[np.asarray(g.q, dtype=np.float64), np.zeros(g.I)]
        model.setObjective(objective @ x, gp.GRB.MAXIMIZE)
        del objective
        user_matrix = sparse.csr_matrix((np.ones(g.E), np.arange(g.E, dtype=np.int32), g.up),
                                      shape=(g.U, g.E + g.I))
        model.addMConstr(user_matrix, x, '=', np.full(g.U, N, dtype=np.float64), name='user_quota')
        del user_matrix
        row = np.r_[g.items, np.arange(g.I, dtype=np.int32)]
        col = np.arange(g.E + g.I, dtype=np.int32)
        data = np.r_[np.ones(g.E), -np.ones(g.I)]
        item_matrix = sparse.coo_matrix((data, (row, col)), shape=(g.I, g.E + g.I)).tocsr()
        del row, col, data
        model.addMConstr(item_matrix, x, '>', np.zeros(g.I), name='item_indicator')
        del item_matrix
        floor = sparse.csr_matrix((np.ones(g.I), np.arange(g.E, g.E + g.I, dtype=np.int32),
                                   np.array([0, g.I], dtype=np.int32)), shape=(1, g.E + g.I))
        model.addMConstr(floor, x, '>', np.array([D], dtype=np.float64), name='coverage_floor')
        del floor
        model.update()
        close_array(g.q)
        build_seconds = time.perf_counter() - started
        model.Params.TimeLimit = max(.01, budget_seconds - build_seconds)
        phase('solve', build_seconds=build_seconds)
        t = time.perf_counter()

        def callback(m, where):
            now = time.perf_counter()
            if where == gp.GRB.Callback.MIPSOL:
                trajectory.append(dict(time_sec=now - t,
                    integer_objective=float(m.cbGet(gp.GRB.Callback.MIPSOL_OBJ)),
                    trajectory_kind='mip_incumbent_no_coverage_trace'))
            elif not mip and where == gp.GRB.Callback.SIMPLEX and now - last_callback[0] >= .5:
                last_callback[0] = now
                trajectory.append(dict(time_sec=now - t,
                    relaxation_objective=float(m.cbGet(gp.GRB.Callback.SPX_OBJVAL)),
                    primal_infeasibility=float(m.cbGet(gp.GRB.Callback.SPX_PRIMINF)),
                    dual_infeasibility=float(m.cbGet(gp.GRB.Callback.SPX_DUALINF)),
                    trajectory_kind='simplex_relaxation_not_an_integer_plan'))

        model.optimize(callback)
        solve_seconds = time.perf_counter() - t
        phase('audit', build_seconds=build_seconds, solve_seconds=solve_seconds)
        g.q = np.load(g.folder / 'scores_integer.npy', mmap_mode='r')
        statuses = {gp.GRB.OPTIMAL: 'OPTIMAL', gp.GRB.INFEASIBLE: 'INFEASIBLE',
                    gp.GRB.TIME_LIMIT: 'TIME_LIMIT', gp.GRB.INF_OR_UNBD: 'INF_OR_UNBD'}
        status = statuses.get(model.Status, 'SOLVER_STATUS_%s' % model.Status)
        result = dict(status=status, native_status=int(model.Status), build_seconds=build_seconds,
                      solve_seconds=solve_seconds, selected=None, trajectory=trajectory,
                      trajectory_kind='mip_incumbent' if mip else 'simplex_relaxation',
                      backend_version=list(gp.gurobi.version()))
        if model.SolCount:
            values = x.X[:g.E]
            if np.max(np.abs(values - np.rint(values)), initial=0) > 1e-6:
                result.update(status='NONINTEGRAL_LP_SOLUTION', max_fractionality=float(np.max(np.abs(values - np.rint(values)))))
                return result
            selected = np.flatnonzero(values > .5).astype(np.int32)
            audit = audit_selected(g, selected, N, D)
            if not audit['target_feasible'] or abs(model.ObjVal - audit['integer_objective']) > .25:
                raise AssertionError('Gurobi solution disagrees with exact integer audit')
            result.update(selected=selected, audit=audit, native_objective=float(model.ObjVal))
        return result
    finally:
        model.dispose(); env.dispose()
