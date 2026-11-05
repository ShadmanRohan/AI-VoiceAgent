"""Local embedding generation with versioned model support (E5 / BGE)."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Sequence

import numpy as np
from loguru import logger

from app.config import ACTIVE_EMBEDDING_MODEL, get_settings


class EmbeddingService:
    """Lazy-loaded sentence-transformers embedder with E5 prefixes."""

    def __init__(self, model_name: str | None = None, *, version: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.version = version or settings.embedding_index_version
        self._model = None
        self._lock = asyncio.Lock()
        self.dimension = 384
        self.is_e5 = "e5" in self.model_name.lower()

    async def _ensure_model(self):
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model

            def _load():
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model {} (version={})", self.model_name, self.version)
                return SentenceTransformer(self.model_name)

            self._model = await asyncio.to_thread(_load)
            dim_fn = getattr(self._model, "get_embedding_dimension", None) or getattr(
                self._model, "get_sentence_embedding_dimension"
            )
            self.dimension = int(dim_fn())
            return self._model

    def _prepare(self, texts: Sequence[str], *, kind: str) -> list[str]:
        prepared: list[str] = []
        for t in texts:
            text = (t or "").strip()
            if self.is_e5:
                prefix = "query: " if kind == "query" else "passage: "
                if not text.lower().startswith(("query:", "passage:")):
                    text = prefix + text
            prepared.append(text)
        return prepared

    async def embed(self, texts: Sequence[str], *, kind: str = "passage") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = await self._ensure_model()
        prepared = self._prepare(texts, kind=kind)

        def _encode():
            vectors = model.encode(
                prepared,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype=np.float32)

        return await asyncio.to_thread(_encode)

    async def embed_one(self, text: str, *, kind: str = "passage") -> np.ndarray:
        arr = await self.embed([text], kind=kind)
        return arr[0]

    async def embed_query(self, text: str) -> np.ndarray:
        return await self.embed_one(text, kind="query")

    async def embed_passage(self, text: str) -> np.ndarray:
        return await self.embed_one(text, kind="passage")


@lru_cache
def get_embedding_service(model_name: str | None = None, version: str | None = None) -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(
        model_name or settings.embedding_model,
        version=version or settings.embedding_index_version,
    )


def reset_embedding_services() -> None:
    get_embedding_service.cache_clear()
