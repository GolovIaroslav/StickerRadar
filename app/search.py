"""
app/search.py — hybrid BM25+vector search with Reciprocal Rank Fusion.

Pipeline:
  1. Embed query text → cosine scores over cached frame embeddings (vector leg)
  2. FTS5 BM25 over set_title / set_short_name / emoji (metadata leg)
  3. FTS5 BM25 over OCR text extracted from frames (ocr leg)
  4. Fuse all ranked lists via weighted RRF (k=60)
     - short queries (≤3 words): FTS legs get 2× weight (exact text matters more)
     - abstract queries: equal weights
  5. Normalize fused scores to [0,1], apply metadata boosts, dedup, return top_k

If FTS / OCR legs return no results the search falls back to pure vector ranking.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

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


@dataclass
class _EmbedCache:
    count: int
    media_ids: list[int]            # one entry per frame row
    vecs: np.ndarray                # shape (N, dim)
    meta: dict[int, sqlite3.Row]    # media_id → metadata row


_cache: dict[str, _EmbedCache] = {}


def invalidate_cache() -> None:
    """Call after /sync so the next query picks up new embeddings."""
    _cache.clear()


def _load(model_name: str) -> _EmbedCache:
    """
    Return cached embeddings + metadata, invalidated when embedding count changes.
    Metadata folded into the JOIN — no second IN(...) query needed.
    """
    count = db.count_embeddings_for_model(model_name)
    cached = _cache.get(model_name)
    if cached is not None and cached.count == count:
        return cached

    rows = db.get_conn().execute(
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

    if not rows:
        entry = _EmbedCache(count=count, media_ids=[], vecs=np.empty((0, 0), dtype=np.float32), meta={})
        _cache[model_name] = entry
        return entry

    dim = rows[0]["dim"]
    expected_bytes = dim * 4
    if len(rows[0]["vector"]) != expected_bytes:
        raise RuntimeError(
            f"frame_embeddings dtype mismatch for model {model_name!r}: "
            f"dim={dim} expects {expected_bytes} bytes but blob is {len(rows[0]['vector'])} bytes"
        )
    vecs = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32).reshape(len(rows), dim).copy()
    media_ids = [r["media_id"] for r in rows]
    meta = {r["media_id"]: r for r in rows}

    entry = _EmbedCache(count=count, media_ids=media_ids, vecs=vecs, meta=meta)
    _cache[model_name] = entry
    return entry


# ---------------------------------------------------------------------------
# FTS5 + RRF helpers
# ---------------------------------------------------------------------------

_FTS_SPECIAL = re.compile(r'["\'\+\-\*\(\)\:\^]')


def _fts_query_str(query: str) -> str | None:
    """Convert user query to a safe FTS5 MATCH expression with prefix matching."""
    clean = _FTS_SPECIAL.sub(" ", query)
    words = [w for w in clean.split() if w]
    if not words:
        return None
    return " ".join(f'"{w}"*' for w in words)


def _fts_ranked_ids(query: str, candidate_ids: set[int]) -> list[int]:
    """
    FTS5 BM25 search over set_title / set_short_name / emoji.
    Returns media_ids in BM25 order (best first), filtered to candidate_ids.
    Returns [] on FTS parse error or no match — caller falls back to vector-only.
    """
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
    """
    FTS5 BM25 search over OCR-extracted text (media_ocr_fts).
    Query is normalized the same way as stored OCR text (ё→е, lowercase, etc.).
    Returns [] when no OCR text is indexed or on error.
    """
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
    """Weighted RRF: score += weight / (k + rank) per list."""
    scores: dict[int, float] = {}
    for ranked, weight in lists:
        for rank, mid in enumerate(ranked, start=1):
            scores[mid] = scores.get(mid, 0.0) + weight / (k + rank)
    return scores


def _rrf_fuse(lists: list[list[int]], k: int = 60) -> dict[int, float]:
    return _rrf_fuse_weighted([(lst, 1.0) for lst in lists], k)


# ---------------------------------------------------------------------------
# Main search entry point
# ---------------------------------------------------------------------------

def search(query: str, top_k: int | None = None) -> list[SearchResult]:
    """
    Hybrid BM25+vector search with RRF fusion, metadata boosts, dedup, top_k.
    """
    if top_k is None:
        top_k = config.TOP_K

    embedder = _get_embedder()
    query_vec = embedder.embed_text(query)

    cache = _load(embedder.model_name)
    if not cache.media_ids:
        return []

    # --- Vector leg ---
    scores_raw = cache.vecs @ query_vec  # shape (N,)
    media_best: dict[int, float] = {}
    for media_id, score in zip(cache.media_ids, scores_raw.tolist()):
        if media_id not in media_best or score > media_best[media_id]:
            media_best[media_id] = score

    if not media_best:
        return []

    vec_ranked = sorted(media_best, key=lambda mid: media_best[mid], reverse=True)

    candidate_ids = set(media_best.keys())

    # --- Lexical legs (FTS5 BM25) ---
    fts_ranked = _fts_ranked_ids(query, candidate_ids)
    ocr_ranked = _ocr_fts_ranked_ids(query, candidate_ids)

    # Short queries (≤3 words): give FTS legs 2× weight — exact text matters more
    # for meme text, profanity, short labels; abstract queries keep equal weights.
    fts_w = 2.0 if len(query.split()) <= 3 else 1.0

    # --- Fusion ---
    has_lexical = fts_ranked or ocr_ranked
    if has_lexical:
        rrf = _rrf_fuse_weighted([
            (vec_ranked, 1.0),
            (fts_ranked, fts_w),
            (ocr_ranked, fts_w),
        ])
        rrf_vals = list(rrf.values())
        lo, span = min(rrf_vals), (max(rrf_vals) - min(rrf_vals)) or 1.0
        base_scores = {mid: (rrf[mid] - lo) / span for mid in rrf}
    else:
        # FTS returned nothing (abstract query) — pure vector path, no regression
        raw_vals = list(media_best.values())
        lo, span = min(raw_vals), (max(raw_vals) - min(raw_vals)) or 1.0
        base_scores = {mid: (media_best[mid] - lo) / span for mid in media_best}

    # --- Metadata boosts ---
    query_lower = query.lower()
    results: list[SearchResult] = []

    for media_id, base in base_scores.items():
        row = cache.meta[media_id]
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
