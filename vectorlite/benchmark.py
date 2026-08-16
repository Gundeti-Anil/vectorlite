"""Benchmark ANN vs. flat search across increasing dataset scales.

Measures build time, mean query latency (ms) and recall@k of the ANN index,
using exhaustive flat search as ground truth.

Usage:
    python benchmark.py                       # 1k / 10k / 100k, dim=128
    python benchmark.py --scales 1000 10000 50000 --dim 64 --queries 50
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np

from vectorlite.index import ANNIndex, FlatIndex


def synthetic(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


def recall_at_k(truth: List[List[str]], approx: List[List[str]]) -> float:
    """Fraction of true top-k ids recovered by the approximate search."""
    hits = total = 0
    for t, a in zip(truth, approx):
        hits += len(set(t) & set(a))
        total += len(t)
    return hits / max(1, total)


def run_scale(n: int, dim: int, k: int, n_queries: int, metric: str, ef: int) -> Dict[str, float]:
    data = synthetic(n, dim, seed=n)
    queries = synthetic(n_queries, dim, seed=99_999)

    flat = FlatIndex(dim=dim, metric=metric)
    t0 = time.perf_counter()
    for i, vec in enumerate(data):
        flat.insert(f"v{i}", vec)
    flat_build = time.perf_counter() - t0

    ann = ANNIndex(dim=dim, metric=metric, M=16, ef_construction=100, ef_search=ef)
    t0 = time.perf_counter()
    for i, vec in enumerate(data):
        ann.insert(f"v{i}", vec)
    ann_build = time.perf_counter() - t0

    # Warm the BLAS thread pool so the first timed query is not an outlier.
    flat.query(queries[0], k)
    ann.query(queries[0], k)

    truth, flat_times = [], []
    for q in queries:
        t0 = time.perf_counter()
        res = flat.query(q, k)
        flat_times.append((time.perf_counter() - t0) * 1000)
        truth.append([r.id for r in res])

    approx, ann_times = [], []
    for q in queries:
        t0 = time.perf_counter()
        res = ann.query(q, k)
        ann_times.append((time.perf_counter() - t0) * 1000)
        approx.append([r.id for r in res])

    return {
        "n": n,
        "flat_build_s": flat_build,
        "ann_build_s": ann_build,
        "flat_ms": float(np.mean(flat_times)),
        "ann_ms": float(np.mean(ann_times)),
        "recall": recall_at_k(truth, approx),
        "speedup": float(np.mean(flat_times) / max(1e-9, np.mean(ann_times))),
    }


def print_table(rows: List[Dict[str, float]]) -> None:
    header = (
        f"{'N':>8} {'flat build(s)':>14} {'ann build(s)':>13} "
        f"{'flat q(ms)':>11} {'ann q(ms)':>10} {'speedup':>8} {'recall':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{int(r['n']):>8} {r['flat_build_s']:>14.2f} {r['ann_build_s']:>13.2f} "
            f"{r['flat_ms']:>11.3f} {r['ann_ms']:>10.3f} {r['speedup']:>7.1f}x {r['recall']:>7.3f}"
        )
    print()


def plot(rows: List[Dict[str, float]], out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(ns, [r["flat_ms"] for r in rows], "o-", label="flat (exact)")
    ax1.plot(ns, [r["ann_ms"] for r in rows], "s-", label="ANN (graph)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("dataset size")
    ax1.set_ylabel("mean query latency (ms)")
    ax1.set_title("Query latency")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(ns, [r["recall"] for r in rows], "o-", color="seagreen")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("dataset size")
    ax2.set_ylabel("recall@k vs. exact search")
    ax2.set_title("ANN recall")
    ax2.grid(alpha=0.3)

    fig.suptitle("VectorLite: ANN vs. flat search")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"chart saved to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="VectorLite benchmark")
    p.add_argument("--scales", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    p.add_argument("--ef", type=int, default=100)
    p.add_argument("--out", default="benchmark_results.png")
    args = p.parse_args()

    rows = []
    for n in args.scales:
        print(f"running scale n={n} ...", flush=True)
        rows.append(run_scale(n, args.dim, args.k, args.queries, args.metric, args.ef))
    print_table(rows)
    plot(rows, args.out)


if __name__ == "__main__":
    main()
