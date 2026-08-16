"""Flat (brute-force) index: exact search, fully vectorized with numpy.

Storage model
-------------
Vectors live in one contiguous (capacity, dim) float32 array so every query is
a single BLAS-backed matrix-vector product instead of a python loop. Capacity
grows by doubling so inserts stay amortized O(1).

Cosine queries use a cached unit-length copy of each row; Euclidean queries use
cached squared norms. That way a query never re-normalizes the whole matrix.

Deletes are *soft*: the row is marked dead in `alive` and its id mapping is
dropped. Dead rows are compacted away once they exceed 50% of the array, which
keeps deletes O(1) amortized instead of O(n) each.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseIndex, SearchResult, VALID_METRICS, normalize, unit_vector


class FlatIndex(BaseIndex):
    def __init__(self, dim: int, metric: str = "cosine") -> None:
        if metric not in VALID_METRICS:
            raise ValueError(f"metric must be one of {VALID_METRICS}")
        self.dim = int(dim)
        self.metric = metric

        self._vectors: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self._unit: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self._sq_norm: np.ndarray = np.zeros(0, dtype=np.float32)
        self._ids: List[Optional[str]] = []          # row -> id (None when dead)
        self._alive: np.ndarray = np.zeros(0, dtype=bool)
        self._row_of: Dict[str, int] = {}            # id -> row
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._texts: Dict[str, Optional[str]] = {}
        self._dead = 0

    def _nrows(self) -> int:
        return len(self._ids)

    def _grow(self, extra: int = 1) -> None:
        need = self._nrows() + extra
        cap = int(self._vectors.shape[0])
        if need <= cap:
            return
        new_cap = max(16, cap * 2, need)
        vectors = np.zeros((new_cap, self.dim), dtype=np.float32)
        unit = np.zeros((new_cap, self.dim), dtype=np.float32)
        sq_norm = np.zeros(new_cap, dtype=np.float32)
        alive = np.zeros(new_cap, dtype=bool)
        n = self._nrows()
        if n:
            vectors[:n] = self._vectors[:n]
            unit[:n] = self._unit[:n]
            sq_norm[:n] = self._sq_norm[:n]
            alive[:n] = self._alive[:n]
        self._vectors, self._unit, self._sq_norm, self._alive = vectors, unit, sq_norm, alive

    def _store_row(self, row: int, vec: np.ndarray) -> None:
        self._vectors[row] = vec
        self._unit[row] = unit_vector(vec)
        self._sq_norm[row] = float(vec @ vec)
        self._alive[row] = True

    def _score_all(self, q: np.ndarray) -> np.ndarray:
        n = self._nrows()
        if self.metric == "cosine":
            return self._unit[:n] @ unit_vector(q)
        dots = self._vectors[:n] @ q
        diff_sq = self._sq_norm[:n] - 2.0 * dots + float(q @ q)
        return -np.sqrt(np.maximum(diff_sq, 0.0))

    # ------------------------------------------------------------------ write

    def insert(
        self,
        id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ) -> None:
        vec = self._validate(vector)
        if id in self._row_of:  # upsert: drop the old row first
            self.delete(id)

        self._grow(1)
        row = self._nrows()
        self._store_row(row, vec)
        self._ids.append(id)
        self._row_of[id] = row
        self._metadata[id] = dict(metadata or {})
        self._texts[id] = text

    def delete(self, id: str) -> bool:
        row = self._row_of.pop(id, None)
        if row is None:
            return False
        self._alive[row] = False
        self._ids[row] = None
        self._metadata.pop(id, None)
        self._texts.pop(id, None)
        self._dead += 1
        if self._dead > max(16, self._nrows() // 2):
            self._compact()
        return True

    def _compact(self) -> None:
        n = self._nrows()
        keep = np.flatnonzero(self._alive[:n])
        self._vectors = self._vectors[keep].copy()
        self._unit = self._unit[keep].copy()
        self._sq_norm = self._sq_norm[keep].copy()
        self._ids = [self._ids[i] for i in keep]
        self._alive = np.ones(len(self._ids), dtype=bool)
        self._row_of = {vid: i for i, vid in enumerate(self._ids) if vid is not None}
        self._dead = 0

    # ------------------------------------------------------------------- read

    def query(self, vector: np.ndarray, k: int = 10) -> List[SearchResult]:
        q = self._validate(vector)
        if not self._row_of or k <= 0:
            return []

        scores = self._score_all(q)
        n = self._nrows()
        scores = np.where(self._alive[:n], scores, -np.inf)

        k = min(k, int(self._alive[:n].sum()))
        # argpartition gives the top-k in O(n); we only sort those k.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]

        out: List[SearchResult] = []
        for row in top:
            vid = self._ids[int(row)]
            if vid is None:
                continue
            out.append(
                SearchResult(
                    id=vid,
                    score=float(scores[int(row)]),
                    metadata=self._metadata.get(vid, {}),
                    text=self._texts.get(vid),
                )
            )
        return out

    def get(self, id: str) -> Optional[np.ndarray]:
        row = self._row_of.get(id)
        return None if row is None else self._vectors[row].copy()

    def __len__(self) -> int:
        return len(self._row_of)

    def stats(self) -> Dict[str, Any]:
        n = self._nrows()
        vec_bytes = int(self._vectors[:n].nbytes)
        return {
            "type": "flat",
            "size": len(self),
            "dim": self.dim,
            "metric": self.metric,
            "rows_allocated": n,
            "deleted_rows": int(self._dead),
            "vector_bytes": vec_bytes,
            "memory_mb": round(vec_bytes / (1024 * 1024), 4),
        }

    # ------------------------------------------------------- (de)serialization

    def state(self) -> Dict[str, Any]:
        self._compact()
        n = self._nrows()
        return {
            "kind": "flat",
            "dim": self.dim,
            "metric": self.metric,
            "vectors": self._vectors[:n].copy(),
            "ids": list(self._ids),
            "metadata": self._metadata,
            "texts": self._texts,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "FlatIndex":
        idx = cls(dim=state["dim"], metric=state["metric"])
        vectors = np.asarray(state["vectors"], dtype=np.float32).reshape(-1, idx.dim)
        n = int(vectors.shape[0])
        idx._vectors = vectors.copy()
        idx._unit = normalize(vectors) if n else np.zeros((0, idx.dim), dtype=np.float32)
        idx._sq_norm = (
            np.einsum("ij,ij->i", vectors, vectors).astype(np.float32)
            if n
            else np.zeros(0, dtype=np.float32)
        )
        idx._ids = list(state["ids"])
        idx._alive = np.ones(n, dtype=bool)
        idx._row_of = {vid: i for i, vid in enumerate(idx._ids) if vid is not None}
        idx._metadata = dict(state["metadata"])
        idx._texts = dict(state["texts"])
        idx._dead = 0
        return idx
