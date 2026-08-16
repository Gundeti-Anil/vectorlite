"""VectorLite — a lightweight, embeddable vector search engine."""

from .index import ANNIndex, BaseIndex, FlatIndex, SearchResult
from .persistence import load_index, save_index

__version__ = "0.1.0"
__all__ = ["ANNIndex", "BaseIndex", "FlatIndex", "SearchResult", "load_index", "save_index"]
