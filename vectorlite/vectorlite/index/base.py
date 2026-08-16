"""Common interface + shared math for all VectorLite indexes.

Both the flat and the ANN index implement `BaseIndex`, so they are drop-in
interchangeable everywhere (API, benchmarks, tests).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

Metric = str  # "cosine" | "euclidean"
VALID_METRICS = ("cosine", "euclidean")


@dataclass
class SearchResult:
    """One hit returned by `query`."""

    id: str
    score: float  # higher == better, always (see `_scores` below)
    metadata: Dict[str, Any]
    text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": float(self.score),
            "metadata": self.metadata,
            "text": self.text,
        }


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zero rows stay zero (guarded division)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def unit_vector(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D vector; a zero vector is returned unchanged."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    nrm = float(np.linalg.norm(vec))
    if nrm == 0.0:
        return vec.copy()
    return vec / nrm


def score_matrix(query: np.ndarray, matrix: np.ndarray, metric: Metric) -> np.ndarray:
    """Vectorized similarity of one query vector against a (n, d) matrix.

    Returns a 1-D array where **higher is always better**, so all ranking code
    can just use argmax/argpartition regardless of metric:
      - cosine    -> cosine similarity in [-1, 1]
      - euclidean -> negated L2 distance in (-inf, 0]
    """
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.float32)

    if metric == "cosine":
        q = query / (np.linalg.norm(query) or 1.0)
        m = normalize(matrix)
        return m @ q
    if metric == "euclidean":
        # ||a - b||^2 = ||a||^2 - 2 a·b + ||b||^2, fully vectorized.
        diff_sq = (
            np.einsum("ij,ij->i", matrix, matrix)
            - 2.0 * (matrix @ query)
            + float(query @ query)
        )
        return -np.sqrt(np.maximum(diff_sq, 0.0))
    raise ValueError(f"unknown metric {metric!r}, expected one of {VALID_METRICS}")


class BaseIndex(ABC):
    """Insert / delete / query contract shared by FlatIndex and ANNIndex."""

    dim: int
    metric: Metric

    @abstractmethod
    def insert(
        self,
        id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ) -> None: ...

    @abstractmethod
    def delete(self, id: str) -> bool: ...

    @abstractmethod
    def query(self, vector: np.ndarray, k: int = 10) -> List[SearchResult]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]: ...

    def _validate(self, vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"expected dimension {self.dim}, got {vec.shape[0]}")
        return vec
