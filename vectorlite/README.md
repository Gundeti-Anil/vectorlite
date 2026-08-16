# VectorLite

A lightweight, embeddable vector search engine written from scratch in Python.
No faiss / hnswlib / annoy / chroma — only numpy for the math.

```
vectorlite/
├── vectorlite/
│   ├── index/
│   │   ├── base.py      # shared interface, metrics, vectorized scoring
│   │   ├── flat.py      # exact brute-force index (numpy matrix product)
│   │   └── ann.py       # simplified HNSW-inspired graph index
│   ├── persistence.py   # atomic npz + JSON save/load
│   ├── config.py        # env-driven config, index factory
│   └── api.py           # FastAPI service
├── benchmark.py         # latency + recall benchmark, saves a chart
├── tests/               # pytest suite
└── requirements.txt
```

## Architecture

**`BaseIndex`** defines `insert / delete / query / stats / __len__`, so the flat
and ANN indexes are fully interchangeable. All scoring goes through
`score_matrix()`, which returns a *higher-is-better* score for both metrics
(cosine similarity, or negated euclidean distance) — ranking code never needs to
branch on the metric.

**FlatIndex** keeps all vectors in one contiguous `(n, dim)` float32 array.
A query is a single matrix-vector product plus `argpartition` for top-k — no
python loop over vectors. Deletes are soft (alive mask) and compacted when dead
rows exceed 50%, keeping deletes amortized O(1).

**ANNIndex** builds a single-layer navigable small world graph. Each new node
beam-searches the existing graph (`ef_construction`) and links to its `M` best
neighbors bidirectionally; early nodes end up with long-range edges, which is
what makes greedy traversal converge quickly. Queries run a beam search of width
`ef_search` from the entry point. Neighbor lists are capped at `2M` by dropping
the worst edges.

**Persistence** writes one `.npz`: the vector matrix natively as binary, and
ids / metadata / graph adjacency / hyper-parameters as a JSON blob. Saves are
atomic (temp file + `os.replace`). See the module docstring for why this beats
both pure pickle (unsafe) and pure JSON (slow, bloated).

## Running

```bash
pip install -r requirements.txt

# flat index
VECTORLITE_INDEX=flat VECTORLITE_DIM=128 uvicorn vectorlite.api:app --reload

# ANN index
VECTORLITE_INDEX=ann VECTORLITE_DIM=128 VECTORLITE_EF_SEARCH=100 uvicorn vectorlite.api:app
```

Config env vars: `VECTORLITE_INDEX` (flat|ann), `VECTORLITE_DIM`,
`VECTORLITE_METRIC` (cosine|euclidean), `VECTORLITE_PERSIST_PATH`,
`VECTORLITE_AUTOSAVE`, `VECTORLITE_M`, `VECTORLITE_EF_CONSTRUCTION`,
`VECTORLITE_EF_SEARCH`.

### API

| Method | Path             | Body / notes                                        |
| ------ | ---------------- | --------------------------------------------------- |
| POST   | `/insert`        | `{id, vector, metadata?, text?}` — upserts by id     |
| POST   | `/query`         | `{vector, k}` → `{results: [{id, score, metadata}]}` |
| DELETE | `/vectors/{id}`  | 404 if unknown                                       |
| GET    | `/stats`         | size, dim, metric, memory usage, graph stats         |
| POST   | `/save`          | force a snapshot to disk                             |

```bash
curl -X POST localhost:8000/insert -H 'content-type: application/json' \
  -d '{"id":"doc1","vector":[0.1,0.2,...],"metadata":{"src":"wiki"},"text":"hello"}'
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"vector":[0.1,0.2,...],"k":5}'
```

Embeddings are **not** generated internally — callers supply raw vectors.

### Embedded use

```python
from vectorlite import ANNIndex, save_index, load_index

idx = ANNIndex(dim=128, metric="cosine")
idx.insert("a", vec, {"src": "docs"})
hits = idx.query(query_vec, k=10)
save_index(idx, "index.npz")
idx = load_index("index.npz")
```

## Benchmarks

```bash
python benchmark.py                                  # 1k / 10k / 100k
python benchmark.py --scales 1000 10000 --dim 64
```

Prints a table and writes `benchmark_results.png` (query latency vs. scale, and
ANN recall@10 vs. exact search).

## Tests

```bash
pytest
```

Covers flat correctness against a numpy reference, persistence round-trip
identity, ANN recall > 85% on synthetic data, and the HTTP API.

## Limitations

- Single-layer graph, no probabilistic level assignment — ANN build is slower
  and recall degrades faster at very large N than real HNSW.
- No heuristic neighbor diversification (HNSW's `selectNeighborsHeuristic`), so
  clustered data can create redundant edges.
- Deletes are soft in the ANN graph: nodes remain as routing hops, so memory is
  only reclaimed by rebuilding.
- Single-process, single in-memory index guarded by one lock; no sharding,
  replication, or concurrent writers.
- Full-snapshot persistence (no WAL / incremental commits); autosave on every
  write is fine for modest sizes, turn it off and call `/save` for bulk loads.
- No metadata filtering, quantization, or authentication.

### Note on the crossover point

Below roughly 50k vectors the flat index is often *faster* than the ANN index:
brute force is one BLAS call over a contiguous array, while the graph walk pays
python interpreter overhead per hop. ANN wins as N grows (flat is O(N·d) per
query, the graph walk is ~O(log N) hops), which is exactly what the benchmark
chart shows when you include the 100k scale.
