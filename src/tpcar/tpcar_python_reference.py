"""
TPCAR exact algorithm (Algorithm 1) for OP dataset.
Optimized v2: vectorized graph construction, fixed covered-set bug, incremental item_counts.
"""

import json, os, time, numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

BASE = r'e:\Users\24qxh\Desktop\LETTER-main\LETTER-main\dataset\OP'
NPY_PATH = r'e:\Users\24qxh\Desktop\LETTER-main\LETTER-main\predictions\OP_R_ui_full_user_item.npy'
N_VALUES = [10, 15, 20, 25, 30]
OUTPUT_JSON = r'e:\Users\24qxh\Desktop\op_results_exact.json'
RISK_PERCENTILE = 0.40

print("Loading prediction matrix...")
R = np.fromfile(NPY_PATH, dtype=np.float32)
num_users, num_items = 4903, 2419
R = R.reshape(num_users, num_items)
print(f"  R shape: {R.shape}, range: [{R.min():.4f}, {R.max():.4f}]")

print("Building known rating mask...")
known_mask = np.zeros((num_users, num_items), dtype=bool)
for split in ['train', 'val', 'test']:
    with open(os.path.join(BASE, f'{split}.json')) as f:
        data = json.load(f)
    for uid_str, items in data.items():
        uid = int(uid_str)
        for pair in items:
            iid = int(pair[1])
            if 0 <= uid < num_users and 0 <= iid < num_items:
                known_mask[uid, iid] = True
print(f"  Known ratings: {known_mask.sum()}")

R_masked = R.copy()
R_masked[known_mask] = 0.0

SIGMA = num_items + num_users
TAU = num_items + num_users + 1
N_NODES = num_items + num_users + 2

def user_node(u):
    return num_items + u

def user_from_node(n):
    return n - num_items


def build_candidate_sets(R_masked, percentile):
    candidates_set = []
    item_to_cand_users = {}
    for u in range(num_users):
        row = R_masked[u]
        nonzero_idx = np.where(row > 0)[0]
        if len(nonzero_idx) == 0:
            candidates_set.append(set())
            continue
        threshold = np.percentile(row[nonzero_idx], (1 - percentile) * 100)
        cand = set(int(i) for i in nonzero_idx if row[i] >= threshold)
        candidates_set.append(cand)
        for i in cand:
            if i not in item_to_cand_users:
                item_to_cand_users[i] = set()
            item_to_cand_users[i].add(u)
    return candidates_set, item_to_cand_users


def tpcar_exact(R_masked, candidates_set, item_to_cand_users, N,
                add_j_arr, add_u_arr, add_cost_arr):
    # Phase 1: Top-N per user
    in_list = np.zeros((num_users, num_items), dtype=bool)
    item_counts = np.zeros(num_items, dtype=np.int32)

    for u in range(num_users):
        cand = candidates_set[u]
        if len(cand) == 0:
            continue
        cand_arr = np.array(list(cand))
        scores = R_masked[u, cand_arr]
        top_idx = np.argsort(-scores)[:min(N, len(cand_arr))]
        for ti in top_idx:
            ii = int(cand_arr[ti])
            in_list[u, ii] = True
            item_counts[ii] += 1

    K0 = int((item_counts > 0).sum())
    uncovered_arr = item_counts == 0  # True for uncovered items

    naive_pred_sum = 0.0
    total_recs = 0
    for u in range(num_users):
        for i in np.where(in_list[u])[0]:
            naive_pred_sum += R_masked[u, i]
            total_recs += 1
    naive_pred = naive_pred_sum / total_recs if total_recs > 0 else 0
    naive_D = K0

    Delta = int(uncovered_arr.sum())
    if Delta <= 0:
        return naive_pred, naive_pred, naive_D, naive_D

    print(f"    Phase 1: K0={K0}, Delta={Delta}")

    potential = np.zeros(N_NODES, dtype=np.float64)
    swaps_done = 0
    t0 = time.time()

    for iteration in range(Delta):
        if iteration % 200 == 0 and iteration > 0:
            elapsed = time.time() - t0
            eta = elapsed / iteration * (Delta - iteration)
            cur_D = int((item_counts > 0).sum())
            print(f"    Iteration {iteration}/{Delta}, D={cur_D}, swaps={swaps_done}, {elapsed:.1f}s, ETA {eta:.0f}s")

        # ── Build edges (vectorized) ──

        # 1. Source arcs: (SIGMA, j) for j in uncovered
        uncov_list = np.where(uncovered_arr)[0]
        src_rows = np.full(len(uncov_list), SIGMA, dtype=np.int32)
        src_cols = uncov_list.astype(np.int32)
        src_cost = np.maximum(0.0, potential[SIGMA] - potential[uncov_list])

        # 2. Addition arcs: (j, u_node) for j not in user_lists[u]
        active_add = ~in_list[add_u_arr, add_j_arr]
        a_j = add_j_arr[active_add]
        a_u = add_u_arr[active_add]
        a_c = add_cost_arr[active_add]
        a_u_node = a_u + num_items
        add_reduced = np.maximum(0.0, a_c + potential[a_j] - potential[a_u_node])

        # 3. Removal arcs: (u_node, i) for i in user_lists[u] — vectorized
        rem_u, rem_i = np.where(in_list)
        rem_u_node = (rem_u + num_items).astype(np.int32)
        rem_i = rem_i.astype(np.int32)
        rem_raw = R_masked[rem_u, rem_i].astype(np.float64)
        rem_reduced = np.maximum(0.0, rem_raw + potential[rem_u_node] - potential[rem_i])

        # 4. Termination arcs: (i, TAU) for i with item_counts >= 2
        term_items = np.where(item_counts >= 2)[0].astype(np.int32)
        term_cols = np.full(len(term_items), TAU, dtype=np.int32)
        term_cost = np.maximum(0.0, potential[term_items] - potential[TAU])

        # Combine all edges
        all_rows = np.concatenate([src_rows, a_j, rem_u_node, term_items])
        all_cols = np.concatenate([src_cols, a_u_node, rem_i, term_cols])
        all_data = np.concatenate([src_cost, add_reduced, rem_reduced, term_cost])

        graph = csr_matrix((all_data, (all_rows, all_cols)), shape=(N_NODES, N_NODES))

        # Dijkstra from SIGMA
        dists, preds = dijkstra(graph, indices=SIGMA, directed=True, return_predecessors=True)

        if dists[TAU] == np.inf:
            print(f"    No augmenting path found at iteration {iteration}")
            break

        # Reconstruct path: TAU -> ... -> SIGMA
        path = []
        node = TAU
        while node != SIGMA and node >= 0:
            path.append(node)
            node = preds[node]
        if node != SIGMA:
            break
        path.append(SIGMA)
        path.reverse()
        # path = [SIGMA, j1, u1_node, i1, u2_node, i2, ..., ik, TAU]

        # Execute exchange chain
        for pi in range(1, len(path) - 1, 2):
            j_add = path[pi]
            if pi + 1 >= len(path) - 1:
                break
            u_n = path[pi + 1]
            u = user_from_node(u_n)

            if pi + 2 <= len(path) - 2:
                i_remove = path[pi + 2]
            else:
                break

            # User u: add j_add, remove i_remove
            in_list[u, i_remove] = False
            in_list[u, j_add] = True
            item_counts[i_remove] -= 1
            item_counts[j_add] += 1

            # Update uncovered_arr
            if item_counts[i_remove] == 0:
                uncovered_arr[i_remove] = True
            if item_counts[j_add] > 0:
                uncovered_arr[j_add] = False

            swaps_done += 1

        # Update potentials (Johnson)
        finite = dists < np.inf
        potential[finite] += dists[finite]

    tpcar_D = int((item_counts > 0).sum())
    tpcar_pred_sum = 0.0
    total_recs = 0
    rem_u2, rem_i2 = np.where(in_list)
    for u_idx in range(len(rem_u2)):
        tpcar_pred_sum += R_masked[rem_u2[u_idx], rem_i2[u_idx]]
        total_recs += 1
    tpcar_pred = tpcar_pred_sum / total_recs if total_recs > 0 else 0

    return naive_pred, tpcar_pred, naive_D, tpcar_D


# ── Main ──
print("\nBuilding candidate sets (risk filter: top 40%)...")
candidates_set, item_to_cand_users = build_candidate_sets(R_masked, RISK_PERCENTILE)
avg_cand = np.mean([len(c) for c in candidates_set])
print(f"  Avg candidates per user: {avg_cand:.1f}")
print(f"  Items with candidate users: {len(item_to_cand_users)}")

# Precompute addition arcs
print("Precomputing addition arcs...")
add_j_list = []
add_u_list = []
add_cost_list = []
for j, users in item_to_cand_users.items():
    for u in users:
        add_j_list.append(j)
        add_u_list.append(u)
        add_cost_list.append(-float(R_masked[u, j]))
add_j_arr = np.array(add_j_list, dtype=np.int32)
add_u_arr = np.array(add_u_list, dtype=np.int32)
add_cost_arr = np.array(add_cost_list, dtype=np.float64)
print(f"  Total addition arcs: {len(add_j_arr)}")

results = []
for N in N_VALUES:
    print(f"\n--- N={N} ---")
    t0 = time.time()
    naive_pred, tpcar_pred, naive_D, tpcar_D = tpcar_exact(
        R_masked, candidates_set, item_to_cand_users, N,
        add_j_arr, add_u_arr, add_cost_arr)
    elapsed = time.time() - t0
    print(f"   N={N}  |  Naive: D={naive_D}, pred={naive_pred:.4f}  |  TPCAR: D={tpcar_D}, pred={tpcar_pred:.4f}  |  {elapsed:.1f}s")
    results.append({
        'N': N,
        'naive_pred': round(naive_pred, 4),
        'tpcar_pred': round(tpcar_pred, 4),
        'naive_D': naive_D,
        'tpcar_D': tpcar_D,
    })

print("\n" + "=" * 60)
print("Final OP Results (Exact Algorithm):")
print(f"{'N':>6} {'前人D':>8} {'前人Pred':>10} {'本文D':>8} {'本文Pred':>10}")
print("-" * 60)
for r in results:
    print(f"{r['N']:>6} {r['naive_D']:>8,} {r['naive_pred']:>10.4f} {r['tpcar_D']:>8,} {r['tpcar_pred']:>10.4f}")

with open(OUTPUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUTPUT_JSON}")
