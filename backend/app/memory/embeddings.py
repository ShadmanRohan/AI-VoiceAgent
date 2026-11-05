"""Local embedding generation using BGE-small."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Sequence

import numpy as np
from loguru import logger

from app.config import get_settings


class EmbeddingService:
    """Lazy-loaded sentence-transformers embedder."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._lock = asyncio.Lock()
        self.dimension = 384

    async def _ensure_model(self):
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model

            def _load():
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model {}", self.model_name)
                model = SentenceTransformer(self.model_name)
                return model

            self._model = await asyncio.to_thread(_load)
            dim_fn = getattr(
                self._model, "get_embedding_dimension", None
            ) or getattr(self._model, "get_sentence_embedding_dimension")
            self.dimension = int(dim_fn())
            return self._model

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = await self._ensure_model()

        def _encode():
            vectors = model.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype=np.float32)

        return await asyncio.to_thread(_encode)

    async def embed_one(self, text: str) -> np.ndarray:
        arr = await self.embed([text])
        return arr[0]


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
