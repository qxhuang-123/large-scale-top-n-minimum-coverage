"""Dataset entry points for the delivered F (state_source) algorithm."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
from f_algorithm import CandidateGraph, solve_graph
from prepare_datasets import prepare

ROOT = Path(__file__).resolve().parent
REFERENCE = {
    "OP": {"coverage": 2367, "score": 242777.856372118, "augmentations": 1587},
    "YELP": {"coverage": 7008, "score": 451026.2084059715, "augmentations": 6656},
}


def export_result(result, folder):
    folder.mkdir(parents=True, exist_ok=True)
    data = dict(result)
    np.save(folder / "selected_edges.npy", np.asarray(data.pop("selected_edges"), dtype=np.int32))
    if "reachable_nodes" in data:
        np.save(folder / "reachable_nodes.npy", np.asarray(data.pop("reachable_nodes"), dtype=np.uint8))
    recs = data.pop("recommendations")
    with (folder / "recommendations.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "rank", "item_id"])
        for u, row in enumerate(recs):
            writer.writerows((u, rank, item) for rank, item in enumerate(row, 1))
    (folder / "result.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def main(dataset):
    parser = argparse.ArgumentParser(description="Run TRMNL-%s-F with N=10 and the original top-40%% graph." % dataset)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--target", type=int)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--check-inputs", action="store_true", help="Check source cache headers without conversion or solving (SO/VG/TG)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.check_inputs:
        if dataset not in ("SO", "VG", "TG"):
            parser.error("--check-inputs is available for SO/VG/TG")
        from prepare_cached_datasets import check_inputs
        info = check_inputs(dataset)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return info
    started = time.perf_counter()
    if dataset in ("SO", "VG", "TG"):
        from prepare_cached_datasets import prepare as prepare_cached
        meta = prepare_cached(dataset)
    else:
        meta = prepare(dataset)
    target = meta["candidate_items"] if args.target is None else args.target
    name = "TRMNL-%s-F" % dataset
    output = ROOT / "outputs" / name
    stamp = time.strftime("%Y%m%d_%H%M%S")
    session = output / stamp
    session.mkdir(parents=True, exist_ok=False)
    t = time.perf_counter()
    graph = CandidateGraph.load(Path(meta.get("array_directory", ROOT / "data" / dataset)))
    graph_load_seconds = time.perf_counter() - t
    print("%s | users=%d items=%d edges=%d N=10 D=%d" % (
        name, meta["n_users"], meta["n_items"], meta["candidate_edges"], target), flush=True)
    print("Graph loading + initial validation: %.3f s" % graph_load_seconds, flush=True)
    rows = []
    first_ready_seconds = None
    for iteration in range(1, args.repeats + 1):
        t = time.perf_counter()
        result = solve_graph(graph, N=10, D=target, progress_every=args.progress_every)
        solve_seconds = time.perf_counter() - t
        t = time.perf_counter()
        saved = export_result(result, session / ("run_%02d" % iteration))
        export_seconds = time.perf_counter() - t
        row = dict(run=iteration, initial_coverage=result["initial_coverage"],
                   requested_coverage=target, coverage=result["coverage"],
                   score=result["score"], average_score=result["score"]/(meta["n_users"]*10),
                   target_feasible=result["target_feasible"],
                   maximum_coverage_certified=result["maximum_coverage_certified"],
                   augmentations=result["augmentations"], swaps=result["swaps"],
                   full_searches=result.get("full_searches"), fallbacks=result.get("fallbacks"),
                   call_wall_seconds=result["call_wall_seconds"],
                   input_validation_seconds=result["input_validation_seconds"],
                   audit_seconds=result["audit_seconds"], solve_wall_seconds=solve_seconds,
                   export_seconds=export_seconds, score_recompute_abs_error=result["score_recompute_abs_error"])
        if args.target is None and dataset in REFERENCE:
            ref = REFERENCE[dataset]
            row["reference_score_difference"] = result["score"] - ref["score"]
            if result["coverage"] != ref["coverage"] or abs(row["reference_score_difference"]) > 1e-4:
                raise RuntimeError("Result does not match archived exact coverage/score; review saved output")
            if result["augmentations"] != ref["augmentations"]:
                raise RuntimeError("Unexpected coverage augmentation count")
        rows.append(row)
        if first_ready_seconds is None:
            first_ready_seconds = graph_load_seconds + solve_seconds + export_seconds
        print("Run %d: core=%.6f s, solve+checks=%.6f s, K=%d, score=%.9f, feasible=%s, certificate=%s" % (
            iteration, result["call_wall_seconds"], solve_seconds, result["coverage"], result["score"],
            result["target_feasible"], result["maximum_coverage_certified"]), flush=True)
        del result, saved
    import state_variants
    pyd = Path(state_variants.__file__)
    summary = dict(name=name, dataset=dataset, algorithm="F", engine_mode="state_source",
                   python=sys.version, interpreter=sys.executable, platform=platform.platform(),
                   numpy=np.__version__, extension=str(pyd), extension_sha256=hashlib.sha256(pyd.read_bytes()).hexdigest(),
                   candidate_metadata=meta, graph_load_and_validation_seconds=graph_load_seconds,
                   first_run_with_load_checks_and_export_seconds=first_ready_seconds,
                   repeats=args.repeats, median_core_seconds=statistics.median(r["call_wall_seconds"] for r in rows),
                   min_core_seconds=min(r["call_wall_seconds"] for r in rows),
                   max_core_seconds=max(r["call_wall_seconds"] for r in rows),
                   total_session_seconds=time.perf_counter()-started, runs=rows,
                   timing_note="Core call includes native initialization and return conversion. Graph load, validation, auditing, and export are separately reported. First-time XLSX conversion is a separate preparation stage.")
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    (session / "summary.json").write_text(payload, encoding="utf-8")
    (output / "latest_summary.json").write_text(payload, encoding="utf-8")
    with (session / "timings.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print("Median core: %.6f s" % summary["median_core_seconds"], flush=True)
    print("Results:", session, flush=True)
    return summary
