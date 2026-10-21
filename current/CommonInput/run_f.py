"""Command-line entry: examples, candidate CSV, dense NPY or cached CSR."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from f_algorithm import CandidateGraph, solve_graph


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true")
    group.add_argument("--edges-csv", type=Path)
    group.add_argument("--scores-npy", type=Path)
    group.add_argument("--csr", type=Path, help="NPZ, or directory with the eight NPY arrays")
    p.add_argument("--users", type=int); p.add_argument("--items", type=int)
    p.add_argument("--eligible-npy", type=Path)
    p.add_argument("--n", type=int, default=1); p.add_argument("--target", type=int, default=3)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("output_f"))
    args = p.parse_args()
    if args.demo:
        # Optimal initial A,A,B becomes A,B,C via u0:A->B and u2:B->C.
        graph = CandidateGraph.from_matrix(np.array([[5.,4.9,0.],[5.2,0.,0.],[0.,4.8,4.7]]))
    elif args.edges_csv:
        with args.edges_csv.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        users = np.array([int(r["user_id"]) for r in rows], dtype=np.int64)
        items = np.array([int(r["item_id"]) for r in rows], dtype=np.int64)
        scores = np.array([float(r["score"]) for r in rows])
        U = args.users if args.users is not None else (int(users.max())+1 if len(users) else 0)
        I = args.items if args.items is not None else (int(items.max())+1 if len(items) else 0)
        graph = CandidateGraph.from_edges(U,I,users,items,scores)
    elif args.scores_npy:
        scores = np.load(args.scores_npy, allow_pickle=False)
        eligible = np.load(args.eligible_npy, allow_pickle=False) if args.eligible_npy else None
        graph = CandidateGraph.from_matrix(scores, eligible)
    else:
        graph = CandidateGraph.load(args.csr)
    result = solve_graph(graph, args.n, args.target, args.progress_every)
    args.output.mkdir(parents=True, exist_ok=True)
    chosen = np.asarray(result.pop("selected_edges"), dtype=np.int32)
    np.save(args.output/"selected_edges.npy", chosen)
    if "reachable_nodes" in result:
        np.save(args.output/"reachable_nodes.npy", np.asarray(result.pop("reachable_nodes"), dtype=np.uint8))
    recs = result.pop("recommendations")
    with (args.output/"recommendations.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer=csv.writer(f); writer.writerow(["user_id","rank","item_id"])
        for u, row in enumerate(recs):
            writer.writerows((u,rank,j) for rank,j in enumerate(row,1))
    (args.output/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["algorithm","requested_coverage","initial_coverage",
        "coverage","score","target_feasible","augmentations","call_wall_seconds",
        "maximum_coverage_certified"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
