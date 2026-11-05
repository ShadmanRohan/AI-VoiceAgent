"""FAISS-backed local vector index persisted under data/vectors."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from loguru import logger

from app.config import get_settings


class VectorStore:
    def __init__(self, user_id: str, dimension: int = 384) -> None:
        settings = get_settings()
        self.user_id = user_id
        self.dimension = dimension
        self.dir = settings.data_dir / "vectors" / user_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.faiss"
        self.meta_path = self.dir / "meta.json"
        self.vectors_path = self.dir / "vectors.npy"
        self._index = None
        self._id_map: list[str] = []
        self._vectors: np.ndarray = np.zeros((0, dimension), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        import faiss

        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self._id_map = list(meta.get("ids", []))
            self.dimension = int(meta.get("dimension", self.dimension))
        if self.vectors_path.exists():
            self._vectors = np.load(self.vectors_path).astype(np.float32)
        else:
            self._vectors = np.zeros((0, self.dimension), dtype=np.float32)
        self._index = faiss.IndexFlatIP(self.dimension)
        if len(self._vectors):
            self._index.add(np.ascontiguousarray(self._vectors))

    def _save(self) -> None:
        import faiss

        faiss.write_index(self._index, str(self.index_path))
        np.save(self.vectors_path, self._vectors)
        self.meta_path.write_text(
            json.dumps({"ids": self._id_map, "dimension": self.dimension}),
            encoding="utf-8",
        )

    def add(self, vector_id: str, vector: np.ndarray) -> None:
        import faiss

        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self.dimension:
            raise ValueError(f"Expected dim {self.dimension}, got {vec.shape[1]}")
        if vector_id in self._id_map:
            self.remove(vector_id)
        self._vectors = (
            np.vstack([self._vectors, vec]) if len(self._vectors) else vec.copy()
        )
        self._index.add(vec)
        self._id_map.append(vector_id)
        self._save()

    def remove(self, vector_id: str) -> bool:
        import faiss

        if vector_id not in self._id_map:
            return False
        keep = [i for i, vid in enumerate(self._id_map) if vid != vector_id]
        self._id_map = [self._id_map[i] for i in keep]
        self._vectors = (
            self._vectors[keep]
            if keep
            else np.zeros((0, self.dimension), dtype=np.float32)
        )
        self._index = faiss.IndexFlatIP(self.dimension)
        if len(self._vectors):
            self._index.add(np.ascontiguousarray(self._vectors))
        self._save()
        return True

    def search(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if self._index.ntotal == 0:
            return []
        vec = np.asarray(query, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._id_map):
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    def clear(self) -> None:
        import faiss

        self._index = faiss.IndexFlatIP(self.dimension)
        self._id_map = []
        self._vectors = np.zeros((0, self.dimension), dtype=np.float32)
        self._save()
        logger.info("Cleared vector store for user {}", self.user_id)


_stores: dict[str, VectorStore] = {}


def get_vector_store(user_id: str, dimension: int = 384) -> VectorStore:
    key = f"{user_id}:{dimension}"
    if key not in _stores:
        _stores[key] = VectorStore(user_id, dimension=dimension)
    return _stores[key]


def reset_vector_stores() -> None:
    _stores.clear()
