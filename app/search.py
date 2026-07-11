"""
app/search.py — hybrid BM25+vector search with Reciprocal Rank Fusion.

Pipeline:
  1. Embed query text -> cosine scores over cached frame embeddings (image-semantic leg)
  2. Embed query text -> cosine scores over cached OCR-text embeddings (text-semantic leg)
  3. FTS5 BM25 over set_title / set_short_name / emoji (metadata leg)
  4. FTS5 BM25 over OCR text extracted from frames (ocr lexical leg)
  5. Fuse ranked lists via weighted RRF, normalize, apply boosts, dedup, return top_k

If lexical legs return no results the search falls back to semantic legs only.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np

from app import config, db
from app.embeddings import Embedder
from app.ocr import normalize_text as _ocr_normalize


def _get_embedder() -> Embedder:
    from app.embeddings import get_shared_embedder
    return get_shared_embedder()


@dataclass
class SearchResult:
    media_id: int
    tg_document_id: str
    media_kind: str
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


@dataclass
class _EmbedCache:
    count: int
    text_count: int
    media_ids: list[int]
    vecs: np.ndarray
    text_media_ids: list[int]
    text_vecs: np.ndarray
    meta: dict[int, Any]


_cache: dict[str, _EmbedCache] = {}


def invalidate_cache() -> None:
    _cache.clear()


def _load(model_name: str) -> _EmbedCache:
    count = db.count_embeddings_for_model(model_name)
    count_text = db.count_text_embeddings_for_model(model_name)
    cached = _cache.get(model_name)
    if cached is not None and cached.count == count and cached.text_count == count_text:
        return cached

    conn = db.get_conn()
    rows = conn.execute(
        """
        SELECT mi.id AS media_id, fe.vector, fe.dim,
               mi.tg_document_id, mi.media_kind, mi.sticker_format,
               mi.local_path, mi.bot_file_id, mi.bot_send_method, mi.bot_cache_status,
               mi.set_short_name, mi.set_title, mi.emoji,
               mi.is_favorite, mi.is_recent
        FROM frame_embeddings fe
        JOIN media_frames mf ON mf.id = fe.frame_id
        JOIN media_items mi  ON mi.id = mf.media_id
        WHERE fe.model_name = ?
          AND mi.embed_status = 'ok'
        ORDER BY mi.id, mf.frame_index
        """,
        (model_name,),
    ).fetchall()

    text_rows = conn.execute(
        """
        SELECT mte.media_id, mte.vector, mte.dim
        FROM media_text_embeddings mte
        JOIN media_items mi ON mi.id = mte.media_id
        WHERE mte.model_name = ?
          AND mi.embed_status = 'ok'
        ORDER BY mte.media_id
        """,
        (model_name,),
    ).fetchall()

    if not rows and not text_rows:
        entry = _EmbedCache(
            count=count,
            text_count=count_text,
            media_ids=[],
            vecs=np.empty((0, 0), dtype=np.float32),
            text_media_ids=[],
            text_vecs=np.empty((0, 0), dtype=np.float32),
            meta={},
        )
        _cache[model_name] = entry
        return entry

    meta: dict[int, Any] = {}
    media_ids: list[int] = []
    vecs = np.empty((0, 0), dtype=np.float32)
    if rows:
        dim = rows[0]["dim"]
        expected_bytes = dim * 4
        if len(rows[0]["vector"]) != expected_bytes:
            raise RuntimeError(
                f"frame_embeddings dtype mismatch for model {model_name!r}: "
                f"dim={dim} expects {expected_bytes} bytes but blob is {len(rows[0]['vector'])} bytes"
            )
        vecs = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32).reshape(len(rows), dim).copy()
        media_ids = [r["media_id"] for r in rows]
        meta.update({r["media_id"]: r for r in rows})

    text_media_ids: list[int] = []
    text_vecs = np.empty((0, 0), dtype=np.float32)
    if text_rows:
        text_dim = text_rows[0]["dim"]
        expected_bytes = text_dim * 4
        if len(text_rows[0]["vector"]) != expected_bytes:
            raise RuntimeError(
                f"media_text_embeddings dtype mismatch for model {model_name!r}: "
                f"dim={text_dim} expects {expected_bytes} bytes but blob is {len(text_rows[0]['vector'])} bytes"
            )
        text_vecs = np.frombuffer(b"".join(r["vector"] for r in text_rows), dtype=np.float32).reshape(len(text_rows), text_dim).copy()
        text_media_ids = [r["media_id"] for r in text_rows]
        if text_media_ids:
            extra = conn.execute(
                """
                SELECT id AS media_id, tg_document_id, media_kind, sticker_format,
                       local_path, bot_file_id, bot_send_method, bot_cache_status,
                       set_short_name, set_title, emoji, is_favorite, is_recent
                FROM media_items
                WHERE id IN (%s)
                """ % ",".join("?" for _ in text_media_ids),
                text_media_ids,
            ).fetchall()
            meta.update({r["media_id"]: r for r in extra})

    entry = _EmbedCache(
        count=count,
        text_count=count_text,
        media_ids=media_ids,
        vecs=vecs,
        text_media_ids=text_media_ids,
        text_vecs=text_vecs,
        meta=meta,
    )
    _cache[model_name] = entry
    return entry


_FTS_SPECIAL = re.compile(r'["\'\+\-\*\(\)\:\^]')


def _fts_query_str(query: str) -> str | None:
    clean = _FTS_SPECIAL.sub(" ", query)
    words = [w for w in clean.split() if w]
    if not words:
        return None
    return " ".join(f'"{w}"*' for w in words)


def _fts_ranked_ids(query: str, candidate_ids: set[int]) -> list[int]:
    fts_q = _fts_query_str(query)
    if not fts_q:
        return []
    try:
        rows = db.get_conn().execute(
            "SELECT rowid FROM media_fts WHERE media_fts MATCH ? ORDER BY rank",
            (fts_q,),
        ).fetchall()
        return [r[0] for r in rows if r[0] in candidate_ids]
    except Exception:
        return []


def _ocr_fts_ranked_ids(query: str, candidate_ids: set[int]) -> list[int]:
    norm_q = _ocr_normalize(query)
    fts_q = _fts_query_str(norm_q)
    if not fts_q:
        return []
    try:
        rows = db.get_conn().execute(
            "SELECT rowid FROM media_ocr_fts WHERE media_ocr_fts MATCH ? ORDER BY rank",
            (fts_q,),
        ).fetchall()
        return [r[0] for r in rows if r[0] in candidate_ids]
    except Exception:
        return []


def _rrf_fuse_weighted(lists: list[tuple[list[int], float]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranked, weight in lists:
        for rank, mid in enumerate(ranked, start=1):
            scores[mid] = scores.get(mid, 0.0) + weight / (k + rank)
    return scores


def _is_text_like_query(query: str) -> bool:
    tokens = [t for t in re.split(r"\s+", query.strip()) if t]
    if not tokens:
        return False
    alpha_tokens = [t for t in tokens if any(ch.isalpha() for ch in t)]
    return bool(alpha_tokens)


def _semantic_only_should_return_empty(
    query: str,
    *,
    best_sem: float,
    second_sem: float,
    best_image_sem: float,
    best_text_sem: float,
) -> bool:
    token_count = len([t for t in re.split(r"\s+", query.strip()) if t])

    if best_sem < 0.18:
        return True
    if best_sem < 0.24 and (best_sem - second_sem) < 0.015:
        return True

    if not _is_text_like_query(query):
        return False

    # OCR-text semantic matches are much noisier than lexical hits. For pure
    # semantic-only queries, be stricter when neither the image leg nor the
    # text-semantic leg is convincingly strong.
    if token_count >= 4 and best_text_sem < 0.93 and best_image_sem < 0.20:
        return True
    if token_count <= 3 and best_text_sem < 0.82 and best_image_sem < 0.14:
        return True

    return False


def search(query: str, top_k: int | None = None) -> list[SearchResult]:
    if top_k is None:
        top_k = config.TOP_K

    embedder = _get_embedder()
    query_vec = embedder.embed_text(query)

    cache = _load(embedder.model_name)
    if not cache.media_ids and not cache.text_media_ids:
        return []

    image_best: dict[int, float] = {}
    if cache.media_ids:
        image_scores = cache.vecs @ query_vec
        for media_id, score in zip(cache.media_ids, image_scores.tolist()):
            if media_id not in image_best or score > image_best[media_id]:
                image_best[media_id] = score

    text_best: dict[int, float] = {}
    if cache.text_media_ids:
        text_scores = cache.text_vecs @ query_vec
        for media_id, score in zip(cache.text_media_ids, text_scores.tolist()):
            if media_id not in text_best or score > text_best[media_id]:
                text_best[media_id] = score

    candidate_ids = set(image_best) | set(text_best)
    if not candidate_ids:
        return []

    vec_ranked = sorted(image_best, key=lambda mid: image_best[mid], reverse=True)
    ocr_sem_ranked = sorted(text_best, key=lambda mid: text_best[mid], reverse=True)

    fts_ranked = _fts_ranked_ids(query, candidate_ids)
    ocr_ranked = _ocr_fts_ranked_ids(query, candidate_ids)

    fts_w = 2.0 if len(query.split()) <= 3 else 1.0
    text_sem_w = 1.15 if len(query.split()) >= 2 else 1.0

    semantic_lists: list[tuple[list[int], float]] = []
    if vec_ranked:
        semantic_lists.append((vec_ranked, 1.0))
    if ocr_sem_ranked:
        semantic_lists.append((ocr_sem_ranked, text_sem_w))

    lexical_lists: list[tuple[list[int], float]] = []
    if fts_ranked:
        lexical_lists.append((fts_ranked, fts_w))
    if ocr_ranked:
        lexical_lists.append((ocr_ranked, fts_w))

    if lexical_lists:
        rrf = _rrf_fuse_weighted(semantic_lists + lexical_lists)
        vals = list(rrf.values())
        lo, span = min(vals), (max(vals) - min(vals)) or 1.0
        base_scores = {mid: (rrf[mid] - lo) / span for mid in rrf}
    else:
        semantic_base = dict(image_best)
        for media_id, score in text_best.items():
            semantic_base[media_id] = max(score, semantic_base.get(media_id, score))

        ranked_sem = sorted(semantic_base.items(), key=lambda kv: kv[1], reverse=True)
        best_sem = ranked_sem[0][1]
        second_sem = ranked_sem[1][1] if len(ranked_sem) > 1 else -1.0
        best_image_sem = max(image_best.values()) if image_best else -1.0
        best_text_sem = max(text_best.values()) if text_best else -1.0
        if _semantic_only_should_return_empty(
            query,
            best_sem=best_sem,
            second_sem=second_sem,
            best_image_sem=best_image_sem,
            best_text_sem=best_text_sem,
        ):
            return []

        vals = list(semantic_base.values())
        lo, span = min(vals), (max(vals) - min(vals)) or 1.0
        base_scores = {mid: (semantic_base[mid] - lo) / span for mid in semantic_base}

    query_lower = query.lower()
    results: list[SearchResult] = []
    for media_id, base in base_scores.items():
        row = cache.meta.get(media_id)
        if row is None:
            continue
        score = base

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
                media_id=media_id,
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

    results.sort(key=lambda r: r.score, reverse=True)

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
