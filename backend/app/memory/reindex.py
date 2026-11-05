"""Safe versioned re-index of memories onto multilingual E5 embeddings."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import select

from app.config import (
    ACTIVE_EMBEDDING_MODEL,
    EMBEDDING_INDEX_VERSION,
    get_settings,
    reload_settings,
)
from app.database.models import Memory
from app.database.session import init_db, session_scope
from app.memory.embeddings import EmbeddingService, reset_embedding_services
from app.memory.vector_store import VectorStore, get_vector_store, reset_vector_stores


async def backup_database() -> Path:
    settings = get_settings()
    settings.ensure_dirs()
    db_path = Path(str(settings.database_url).split("///")[-1])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest = settings.data_dir / "backups" / f"localvoiceai.pre-e5-reindex.{stamp}.db"
    if db_path.exists():
        shutil.copy2(db_path, dest)
        logger.info("Database backed up to {}", dest)
    return dest


async def reindex_memories_to_e5(
    *,
    user_id: str | None = None,
    activate: bool = True,
) -> dict[str, Any]:
    """Rebuild vectors under e5-small-v1 from original memory text. Never deletes rows."""
    settings = get_settings()
    await backup_database()
    await init_db(settings)

    embedder = EmbeddingService(ACTIVE_EMBEDDING_MODEL, version=EMBEDDING_INDEX_VERSION)
    probe = await embedder.embed_passage("probe")
    dim = int(probe.shape[0])

    stats: dict[str, Any] = {
        "embedding_model": ACTIVE_EMBEDDING_MODEL,
        "index_version": EMBEDDING_INDEX_VERSION,
        "dimension": dim,
        "users": {},
        "activated": False,
    }

    async with session_scope(settings) as db:
        stmt = select(Memory).where(Memory.active.is_(True))
        if user_id:
            stmt = stmt.where(Memory.user_id == user_id)
        memories = list((await db.execute(stmt)).scalars().all())

        by_user: dict[str, list[Memory]] = {}
        for m in memories:
            by_user.setdefault(m.user_id, []).append(m)

        for uid, items in by_user.items():
            store = VectorStore(uid, dimension=dim, index_version=EMBEDDING_INDEX_VERSION)
            store.clear()
            indexed = 0
            for mem in items:
                if not mem.vector_id:
                    continue
                vec = await embedder.embed_passage(mem.content)
                if vec.shape[0] != dim:
                    raise RuntimeError("Unexpected embedding dimension during reindex")
                store.add(mem.vector_id, vec)
                mem.meta = {
                    **(mem.meta or {}),
                    "embedding_model": ACTIVE_EMBEDDING_MODEL,
                    "embedding_index_version": EMBEDDING_INDEX_VERSION,
                }
                indexed += 1
            stats["users"][uid] = {"memories": len(items), "indexed": indexed, "store_count": store.count()}
            if store.count() != indexed:
                raise RuntimeError(
                    f"Index count mismatch for {uid}: store={store.count()} indexed={indexed}"
                )

    if activate:
        settings.save_overrides(
            {
                "embedding_model": ACTIVE_EMBEDDING_MODEL,
                "embedding_index_version": EMBEDDING_INDEX_VERSION,
                "bilingual_memory": True,
            }
        )
        reset_embedding_services()
        reset_vector_stores()
        reload_settings()
        stats["activated"] = True

    # Quick bilingual smoke retrieval if any memories exist
    async with session_scope(settings) as db:
        from app.memory.service import MemoryService

        svc = MemoryService(db, reload_settings(), embedder)
        uid = user_id or settings.default_user_id
        bn, _ = await svc.retrieve(uid, "আমি কীভাবে উত্তর পছন্দ করি?", min_score=0.1)
        en, _ = await svc.retrieve(uid, "How do I prefer explanations?", min_score=0.1)
        stats["smoke_bn_hits"] = len(bn)
        stats["smoke_en_hits"] = len(en)

    logger.info("E5 reindex complete: {}", stats)
    return stats
