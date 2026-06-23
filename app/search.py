"""
app/search.py — brute-force NumPy cosine search with metadata boosts.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from app import config, db
from app.embeddings import Embedder

_embedder: Embedder | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


@dataclass
class SearchResult:
    media_id: int
    tg_document_id: str
    media_kind: str          # sticker | gif
    sticker_format: str | None
    local_path: str | None
    bot_file_id: str | None
    bot_send_method: str | None
    bot_cache_status: str
    set_short_name: str | None
    set_title: str | None
    emoji: str | None
    is_favorite: bool
    is_recent: bool
    score: float


def _load_all_embeddings(model_name: str) -> tuple[list[int], np.ndarray]:
    """
    Load all frame embeddings for the given model.
    Returns (list of media_ids, matrix of shape [N, dim]).
    """
    conn = db.get_conn()
    rows = conn.execute(
        """
        SELECT mi.id AS media_id, fe.vector
        FROM frame_embeddings fe
        JOIN media_frames mf ON mf.id = fe.frame_id
        JOIN media_items mi ON mi.id = mf.media_id
        WHERE fe.model_name = ?
          AND mi.embed_status = 'ok'
        ORDER BY mi.id, mf.frame_index
        """,
        (model_name,),
    ).fetchall()

    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)

    dim = len(rows[0]["vector"]) // 4  # float32 = 4 bytes
    vecs = np.zeros((len(rows), dim), dtype=np.float32)
    media_ids: list[int] = []

    for i, row in enumerate(rows):
        buf = row["vector"]
        vecs[i] = np.frombuffer(buf, dtype=np.float32)
        media_ids.append(row["media_id"])

    return media_ids, vecs


def search(query: str, top_k: int | None = None) -> list[SearchResult]:
    """
    Embed query text, cosine-search all frame embeddings, apply metadata boosts,
    deduplicate per media_item, group max 3 per sticker set, return top_k.
    """
    if top_k is None:
        top_k = config.TOP_K

    embedder = _get_embedder()
    query_vec = embedder.embed_text(query)  # shape (dim,), already normalized

    model_name = embedder.model_name
    frame_media_ids, frame_vecs = _load_all_embeddings(model_name)

    if len(frame_media_ids) == 0:
        return []

    # Cosine similarity = dot product (vecs are normalized)
    scores = frame_vecs @ query_vec  # shape (N,)

    # Aggregate: max score per media_item
    media_best: dict[int, float] = {}
    for media_id, score in zip(frame_media_ids, scores.tolist()):
        if media_id not in media_best or score > media_best[media_id]:
            media_best[media_id] = score

    # Fetch metadata for all candidates
    if not media_best:
        return []

    placeholders = ",".join("?" * len(media_best))
    rows = db.get_conn().execute(
        f"""
        SELECT id, tg_document_id, media_kind, sticker_format,
               local_path, bot_file_id, bot_send_method, bot_cache_status,
               set_short_name, set_title, emoji,
               is_favorite, is_recent
        FROM media_items
        WHERE id IN ({placeholders})
        """,
        list(media_best.keys()),
    ).fetchall()

    # Apply metadata boosts
    results: list[SearchResult] = []
    query_lower = query.lower()

    for row in rows:
        score = media_best[row["id"]]

        if row["is_favorite"]:
            score += 0.030
        if row["is_recent"]:
            score += 0.015

        emoji = row["emoji"] or ""
        if emoji and emoji in query:
            score += 0.050

        set_title = row["set_title"] or ""
        set_short = row["set_short_name"] or ""
        if query_lower in set_title.lower() or query_lower in set_short.lower():
            score += 0.020

        results.append(
            SearchResult(
                media_id=row["id"],
                tg_document_id=row["tg_document_id"],
                media_kind=row["media_kind"],
                sticker_format=row["sticker_format"],
                local_path=row["local_path"],
                bot_file_id=row["bot_file_id"],
                bot_send_method=row["bot_send_method"],
                bot_cache_status=row["bot_cache_status"],
                set_short_name=row["set_short_name"],
                set_title=row["set_title"],
                emoji=emoji or None,
                is_favorite=bool(row["is_favorite"]),
                is_recent=bool(row["is_recent"]),
                score=score,
            )
        )

    # Sort descending
    results.sort(key=lambda r: r.score, reverse=True)

    # Group: max 3 results per sticker set
    set_count: dict[str, int] = {}
    deduped: list[SearchResult] = []
    for r in results:
        key = r.set_short_name or f"__single_{r.media_id}"
        if set_count.get(key, 0) >= 3:
            continue
        set_count[key] = set_count.get(key, 0) + 1
        deduped.append(r)
        if len(deduped) >= top_k:
            break

    return deduped
