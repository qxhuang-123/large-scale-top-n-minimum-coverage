"""Prepare the same OP-full / Yelp-unobserved top-40% candidate graphs.

The original exports store a complete numeric row, with the user ID first.
Rows are streamed from XLSX; files and source datasets are never modified.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from zipfile import ZipFile

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('TOPN_DATA_ROOT', str(ROOT / 'data'))).expanduser().resolve()
SPECS = {
    "OP": {
        "source": str(DATA_ROOT / "raw/OP/IUIC1_Rui_pred_user_item_FULL_display1.xlsx"),
        "users": 4903, "items": 2419, "edges": 4746104,
        "candidate_items": 2367, "initial_coverage": 780,
        "reference_initial_score": 242825.25429558754,
        "eligibility": "finite_positive_full_prediction_export",
    },
    "YELP": {
        "source": str(DATA_ROOT / "raw/YELP/Yelp_R_ui_UNKNOWN_ONLY_known_ratings_0_display1.xlsx"),
        "users": 9415, "items": 9373, "edges": 35243883,
        "candidate_items": 7017, "initial_coverage": 352,
        "reference_initial_score": 453219.4965727329,
        "eligibility": "finite_positive_unobserved_export_known_pairs_are_zero",
    },
}
VALUE = re.compile(rb"<v>([^<]*)</v>")


def stream_rows(path):
    with ZipFile(path) as book:
        sheets = [n for n in book.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
        if len(sheets) != 1:
            raise ValueError("Expected a single worksheet, found %s" % sheets)
        with book.open(sheets[0]) as xml:
            buf = b""
            while True:
                block = xml.read(4 * 1024 * 1024)
                if not block:
                    break
                buf += block
                while True:
                    start = buf.find(b"<row ")
                    if start < 0:
                        break
                    end = buf.find(b"</row>", start)
                    if end < 0:
                        buf = buf[start:]
                        break
                    yield buf[start:end + 6]
                    buf = buf[end + 6:]


def prepare(dataset):
    spec = SPECS[dataset]
    folder = ROOT / "data" / dataset
    ready = folder / "metadata.json"
    if ready.exists():
        meta = json.loads(ready.read_text(encoding="utf-8"))
        if all((folder / (k + ".npy")).exists() for k in [
            "user_indptr", "user_items", "user_scores", "edge_users",
            "item_indptr", "item_users", "item_edges", "item_scores",
        ]):
            print("Using prepared dataset", dataset, flush=True)
            return meta
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    path = Path(spec["source"])
    U, I = spec["users"], spec["items"]
    dense_path = folder / "scores_float32.npy"
    if dense_path.exists():
        scores = np.load(dense_path, mmap_mode="r")
        if scores.shape != (U, I):
            raise ValueError("Unexpected existing dense cache dimensions")
    else:
        temporary = folder / "scores_float32.preparing.npy"
        scores = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(U, I))
        rows = stream_rows(path)
        header = next(rows)
        if len(re.findall(rb"<c(?:\s|>)", header)) != I + 1:
            raise ValueError("Unexpected number of worksheet columns")
        count = 0
        for row in rows:
            values = VALUE.findall(row)
            if len(values) != I + 1:
                raise ValueError("Incomplete numeric row %d: %d cells" % (count, len(values)))
            if count >= U or int(values[0]) != count + 1:
                raise ValueError("Unexpected user row ID")
            scores[count] = np.fromstring(b" ".join(values[1:]), sep=" ", dtype=np.float32)
            count += 1
            if count % 1000 == 0:
                print(dataset, "decoded rows", count, "/", U, "elapsed", round(time.perf_counter() - started, 2), flush=True)
        if count != U:
            raise ValueError("Wrong user count: %d" % count)
        scores.flush()
        del scores
        temporary.replace(dense_path)
        scores = np.load(dense_path, mmap_mode="r")
    print(dataset, "dense input ready", flush=True)
    degrees = np.empty(U, dtype=np.int32)
    for u in range(U):
        count = np.count_nonzero(np.isfinite(scores[u]) & (scores[u] > 0))
        degrees[u] = max(10, int(math.ceil(0.4 * count)))
        if count < 10:
            raise ValueError("User has fewer than N eligible items")
    E = int(degrees.sum())
    if E != spec["edges"]:
        raise ValueError("Candidate edge mismatch: %d != %d" % (E, spec["edges"]))
    up = np.r_[0, np.cumsum(degrees, dtype=np.int64)].astype(np.int32)
    items = np.lib.format.open_memmap(folder / "user_items.npy", mode="w+", dtype=np.int32, shape=(E,))
    values = np.lib.format.open_memmap(folder / "user_scores.npy", mode="w+", dtype=np.float64, shape=(E,))
    users = np.lib.format.open_memmap(folder / "edge_users.npy", mode="w+", dtype=np.int32, shape=(E,))
    initial_score = 0.0
    initial_items = []
    counts = np.zeros(I, dtype=np.int64)
    # Reproduce the archived candidate-cache builder, including boundary ties.
    # After choosing eligibility, sort the retained row as required by F.
    rng = np.random.default_rng(20260704)
    for u in range(U):
        row = scores[u]
        valid = np.flatnonzero(np.isfinite(row) & (row > 0))
        k = int(degrees[u])
        jitter = rng.random(valid.size, dtype=np.float32) * np.float32(1e-7)
        retained = valid[np.argpartition(row[valid] + jitter, -k)[-k:]]
        chosen = retained[np.lexsort((retained, -row[retained]))].astype(np.int32)
        lo, hi = int(up[u]), int(up[u + 1])
        items[lo:hi] = chosen
        values[lo:hi] = row[chosen]
        users[lo:hi] = u
        counts += np.bincount(chosen, minlength=I)
        initial_items.extend(chosen[:10].tolist())
        initial_score += float(np.asarray(row[chosen[:10]], dtype=np.float64).sum())
    candidate_items = int(np.count_nonzero(counts))
    initial_coverage = len(set(initial_items))
    if candidate_items != spec["candidate_items"] or initial_coverage != spec["initial_coverage"]:
        raise ValueError("Graph does not match expected candidate/initial coverage: %s" % ((candidate_items, initial_coverage),))
    print(dataset, "building transpose", E, "edges", flush=True)
    # CSC conversion sorts row-edge IDs inside each item column. E fits int32.
    index = csr_matrix((np.arange(E, dtype=np.int32), items, up), shape=(U, I)).tocsc()
    ie = np.asarray(index.data, dtype=np.int32)
    np.save(folder / "item_indptr.npy", np.asarray(index.indptr, dtype=np.int32))
    np.save(folder / "item_edges.npy", ie)
    np.save(folder / "item_users.npy", np.asarray(users[ie], dtype=np.int32))
    np.save(folder / "item_scores.npy", np.asarray(values[ie], dtype=np.float64))
    np.save(folder / "user_indptr.npy", up)
    items.flush(); values.flush(); users.flush()
    meta = dict(dataset=dataset, source=str(path), source_bytes=path.stat().st_size,
                source_mtime_ns=path.stat().st_mtime_ns, eligibility=spec["eligibility"],
                n_users=U, n_items=I, candidate_edges=E, candidate_items=candidate_items,
                N=10, alpha=0.4, ordering="score_desc_item_id_asc",
                candidate_selection="archived_float32_argpartition_jitter_seed20260704",
                initial_coverage=initial_coverage, initial_score=initial_score,
                reference_initial_score=spec["reference_initial_score"],
                initial_score_reference_difference=initial_score-spec["reference_initial_score"],
                score_dtype_source="float32", score_dtype_core="float64",
                preparation_seconds=time.perf_counter()-started)
    ready.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=list(SPECS))
    prepare(p.parse_args().dataset)
