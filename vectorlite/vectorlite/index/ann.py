"""Navigable small-world ANN index with a coarse hub layer and beam search.

Build: every inserted node beam-searches the base graph (`ef_construction`),
links to `M` diversified neighbors, and adds reciprocal edges. A random subset
of nodes is also linked on a coarser hub graph so queries can jump long range
before refining on the base layer.

Search: optional hub-layer beam search to pick an entry point, then base-layer
beam search of width `ef`. Return the best `k`.

Scoring uses cached unit vectors (cosine) or squared norms (euclidean) so each
hop is one small matrix-vector product, not a re-normalize of the whole index.
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import BaseIndex, SearchResult, VALID_METRICS, normalize, unit_vector


class ANNIndex(BaseIndex):
    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 100,
        ef_search: int = 100,
        seed: int = 42,
    ) -> None:
        if metric not in VALID_METRICS:
            raise ValueError(f"metric must be one of {VALID_METRICS}")
        self.dim = int(dim)
        self.metric = metric
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self._rng = np.random.default_rng(seed)

        self._data: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self._unit: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self._sq_norm: np.ndarray = np.zeros(0, dtype=np.float32)
        self._alive_mask: np.ndarray = np.zeros(0, dtype=bool)
        self._hub_mask: np.ndarray = np.zeros(0, dtype=bool)
        self._stamp: np.ndarray = np.zeros(0, dtype=np.uint32)
        self._epoch: int = 1
        self._count: int = 0
        self._ids: List[Optional[str]] = []
        self._graph: List[List[int]] = []
        self._upper: List[List[int]] = []
        self._row_of: Dict[str, int] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._texts: Dict[str, Optional[str]] = {}
        self._entry: Optional[int] = None
        self._upper_entry: Optional[int] = None

    # ------------------------------------------------------------- internals

    def _resize(self, capacity: int) -> None:
        data = np.zeros((capacity, self.dim), dtype=np.float32)
        unit = np.zeros((capacity, self.dim), dtype=np.float32)
        sq_norm = np.zeros(capacity, dtype=np.float32)
        alive = np.zeros(capacity, dtype=bool)
        hubs = np.zeros(capacity, dtype=bool)
        stamp = np.zeros(capacity, dtype=np.uint32)
        n = self._count
        if n:
            data[:n] = self._data[:n]
            unit[:n] = self._unit[:n]
            sq_norm[:n] = self._sq_norm[:n]
            alive[:n] = self._alive_mask[:n]
            hubs[:n] = self._hub_mask[:n]
            stamp[:n] = self._stamp[:n]
        self._data, self._unit, self._sq_norm = data, unit, sq_norm
        self._alive_mask, self._hub_mask, self._stamp = alive, hubs, stamp

    def _append(self, vec: np.ndarray) -> int:
        if self._count == self._data.shape[0]:
            self._resize(max(16, self._data.shape[0] * 2))
        i = self._count
        self._data[i] = vec
        self._unit[i] = unit_vector(vec)
        self._sq_norm[i] = float(vec @ vec)
        self._alive_mask[i] = True
        self._count += 1
        return i

    def _prepare(self, q: np.ndarray) -> np.ndarray:
        return unit_vector(q) if self.metric == "cosine" else q

    def _score_prepared(self, qp: np.ndarray, rows: Sequence[int]) -> np.ndarray:
        idx = np.asarray(rows, dtype=np.intp)
        if idx.size == 0:
            return np.empty(0, dtype=np.float32)
        if self.metric == "cosine":
            return self._unit[idx] @ qp
        dots = self._data[idx] @ qp
        diff_sq = self._sq_norm[idx] - 2.0 * dots + float(qp @ qp)
        return -np.sqrt(np.maximum(diff_sq, 0.0))

    def _score_to(self, q: np.ndarray, rows) -> np.ndarray:
        """Batch-score a query against a list of nodes (vectorized per hop)."""
        if len(rows) == 0:
            return np.empty(0, dtype=np.float32)
        return self._score_prepared(self._prepare(q), rows)

    def _begin_search(self) -> int:
        epoch = self._epoch + 1
        if epoch == 0:
            self._stamp.fill(0)
            epoch = 1
        self._epoch = epoch
        return epoch

    def _pick_live(self, entry: Optional[int], hubs_only: bool = False) -> Optional[int]:
        if entry is not None and self._alive_mask[entry] and (
            not hubs_only or self._hub_mask[entry]
        ):
            return entry
        mask = self._alive_mask[: self._count]
        if hubs_only:
            mask = mask & self._hub_mask[: self._count]
        live = np.flatnonzero(mask)
        if live.size == 0:
            return None
        return int(live[0])

    def _select_neighbors(self, node: int, candidates: Sequence[int], M: int) -> List[int]:
        """Keep up to M neighbors that are close to `node` but not clustered."""
        seen = set()
        uniq: List[int] = []
        for c in candidates:
            c = int(c)
            if c == node or c in seen:
                continue
            seen.add(c)
            uniq.append(c)
        if len(uniq) <= M:
            return uniq

        qp = self._prepare(self._data[node])
        scores = self._score_prepared(qp, uniq)
        order = np.argsort(-scores)
        selected: List[int] = []
        for i in order:
            c = uniq[int(i)]
            if not selected:
                selected.append(c)
                continue
            to_sel = self._score_prepared(self._prepare(self._data[c]), selected)
            if float(np.max(to_sel)) < float(scores[int(i)]):
                selected.append(c)
            if len(selected) >= M:
                break
        if len(selected) < M:
            for i in order:
                c = uniq[int(i)]
                if c not in selected:
                    selected.append(c)
                if len(selected) >= M:
                    break
        return selected

    def _prune(self, graph: List[List[int]], node: int, cap: int) -> None:
        rows = graph[node]
        if len(rows) <= cap:
            return
        scores = self._score_to(self._data[node], rows)
        keep = np.asarray(rows, dtype=np.intp)[np.argsort(-scores)[:cap]]
        graph[node] = keep.tolist()

    def _link(self, graph: List[List[int]], a: int, b: int, cap: int) -> None:
        ga = graph[a]
        if b not in ga:
            ga.append(b)
        gb = graph[b]
        if a not in gb:
            gb.append(a)
        self._prune(graph, b, cap)

    def _search_on(
        self,
        q: np.ndarray,
        ef: int,
        graph: List[List[int]],
        entries: Sequence[Optional[int]],
    ) -> List[tuple]:
        """Beam search over `graph`. Returns [(score, node)] sorted best-first."""
        seeds = [int(e) for e in entries if e is not None]
        if not seeds:
            return []

        qp = self._prepare(q)
        epoch = self._begin_search()
        candidates: List[tuple] = []
        results: List[tuple] = []
        for entry in seeds:
            if self._stamp[entry] == epoch:
                continue
            self._stamp[entry] = epoch
            s = float(self._score_prepared(qp, [entry])[0])
            candidates.append((-s, entry))
            results.append((s, entry))
        heapq.heapify(candidates)
        heapq.heapify(results)

        while candidates:
            neg_score, node = heapq.heappop(candidates)
            if -neg_score < results[0][0] and len(results) >= ef:
                break
            unseen: List[int] = []
            for n in graph[node]:
                if self._stamp[n] != epoch:
                    self._stamp[n] = epoch
                    unseen.append(n)
            if not unseen:
                continue
            scores = self._score_prepared(qp, unseen)
            worst = results[0][0]
            for n, s in zip(unseen, scores):
                s = float(s)
                if len(results) < ef or s > worst:
                    heapq.heappush(candidates, (-s, n))
                    if self._alive_mask[n]:
                        heapq.heappush(results, (s, n))
                        if len(results) > ef:
                            heapq.heappop(results)
                        worst = results[0][0]
        return sorted(results, key=lambda t: -t[0])

    # ------------------------------------------------------------------ write

    def insert(
        self,
        id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ) -> None:
        vec = self._validate(vector)
        if id in self._row_of:
            self.delete(id)

        node = self._append(vec)
        self._ids.append(id)
        self._graph.append([])
        self._upper.append([])
        self._row_of[id] = node
        self._metadata[id] = dict(metadata or {})
        self._texts[id] = text

        is_hub = self._upper_entry is None or bool(
            self._rng.random() < (1.0 / max(self.M, 2))
        )
        self._hub_mask[node] = is_hub

        if self._entry is None:
            self._entry = node
            self._upper_entry = node
            return

        hits = self._search_on(vec, self.ef_construction, self._graph, [self._entry])
        for neighbor in self._select_neighbors(node, [n for _, n in hits], self.M):
            self._link(self._graph, node, neighbor, self.M * 2)

        if is_hub:
            uhits = self._search_on(
                vec, self.ef_construction, self._upper, [self._upper_entry]
            )
            for neighbor in self._select_neighbors(node, [n for _, n in uhits], self.M):
                self._link(self._upper, node, neighbor, self.M * 2)
            if self._upper_entry is None:
                self._upper_entry = node

    def delete(self, id: str) -> bool:
        node = self._row_of.pop(id, None)
        if node is None:
            return False
        self._alive_mask[node] = False
        self._ids[node] = None
        self._metadata.pop(id, None)
        self._texts.pop(id, None)
        return True

    # ------------------------------------------------------------------- read

    def query(self, vector: np.ndarray, k: int = 10, ef: Optional[int] = None) -> List[SearchResult]:
        q = self._validate(vector)
        if not self._row_of or k <= 0:
            return []
        ef = max(ef or self.ef_search, k)
        seeds: List[int] = []
        hub = self._pick_live(self._upper_entry, hubs_only=True)
        if hub is not None:
            uhits = self._search_on(q, max(16, ef // 2), self._upper, [hub])
            seeds.extend(n for _, n in uhits[:8])
        entry = self._pick_live(self._entry)
        if entry is not None:
            seeds.append(entry)
        hits = self._search_on(q, ef, self._graph, seeds)

        out: List[SearchResult] = []
        for score, node in hits:
            vid = self._ids[node]
            if vid is None or not self._alive_mask[node]:
                continue
            out.append(
                SearchResult(vid, float(score), self._metadata.get(vid, {}), self._texts.get(vid))
            )
            if len(out) == k:
                break
        return out

    def __len__(self) -> int:
        return len(self._row_of)

    def stats(self) -> Dict[str, Any]:
        vec_bytes = self._count * self.dim * 4
        edges = sum(len(g) for g in self._graph)
        hubs = int(self._hub_mask[: self._count].sum()) if self._count else 0
        return {
            "type": "ann",
            "size": len(self),
            "dim": self.dim,
            "metric": self.metric,
            "nodes": self._count,
            "hubs": hubs,
            "edges": int(edges),
            "avg_degree": round(edges / max(1, self._count), 2),
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "vector_bytes": vec_bytes,
            "memory_mb": round((vec_bytes + edges * 8) / (1024 * 1024), 4),
        }

    # ------------------------------------------------------- (de)serialization

    def state(self) -> Dict[str, Any]:
        return {
            "kind": "ann",
            "dim": self.dim,
            "metric": self.metric,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "vectors": self._data[: self._count].copy(),
            "ids": list(self._ids),
            "alive": self._alive_mask[: self._count].tolist(),
            "graph": [sorted(g) for g in self._graph],
            "upper": [sorted(g) for g in self._upper],
            "hubs": self._hub_mask[: self._count].tolist(),
            "metadata": self._metadata,
            "texts": self._texts,
            "entry": self._entry,
            "upper_entry": self._upper_entry,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "ANNIndex":
        idx = cls(
            dim=state["dim"],
            metric=state["metric"],
            M=state["M"],
            ef_construction=state["ef_construction"],
            ef_search=state["ef_search"],
        )
        vectors = np.asarray(state["vectors"], dtype=np.float32).reshape(-1, idx.dim)
        n = int(vectors.shape[0])
        idx._resize(max(16, n))
        if n:
            idx._data[:n] = vectors
            idx._unit[:n] = normalize(vectors)
            idx._sq_norm[:n] = np.einsum("ij,ij->i", vectors, vectors)
        idx._count = n
        idx._ids = list(state["ids"])
        alive = np.asarray(state["alive"], dtype=bool)
        idx._alive_mask[:n] = alive
        idx._graph = [list(g) for g in state["graph"]]
        idx._upper = [list(g) for g in state.get("upper", [[] for _ in range(n)])]
        if n:
            hubs = np.asarray(state.get("hubs", [False] * n), dtype=bool)
            idx._hub_mask[:n] = hubs
        idx._metadata = dict(state["metadata"])
        idx._texts = dict(state["texts"])
        idx._entry = state["entry"]
        idx._upper_entry = state.get("upper_entry")
        idx._row_of = {
            vid: i for i, vid in enumerate(idx._ids) if vid is not None and idx._alive_mask[i]
        }
        return idx
