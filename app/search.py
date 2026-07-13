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
from dataclasses import dataclass
from typing import Any

import numpy as np

from app import config, db
from app.embeddings import Embedder
from app.errors import EmbedError, ModelNotInstalled
from app.ocr import normalize_text as _ocr_normalize


def _get_embedder() -> Embedder:
    from app.embeddings import get_shared_embedder
    return get_shared_embedder()


def _get_text_embedder():
    from app.text_embed import get_shared_text_embedder
    return get_shared_text_embedder()


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
               mi.is_favorite, mi.is_recent, mi.ocr_text
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
                       set_short_name, set_title, emoji, is_favorite, is_recent, ocr_text
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


_FTS_SPECIAL = re.compile(r"[^\w\s]", re.UNICODE)


_OCR_LOOKALIKE = str.maketrans({
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х",
    "y": "у", "k": "к", "m": "м", "t": "т", "b": "в", "i": "і",
})
_OCR_SUFFIXES = (
    "иями", "ами", "ями", "ого", "ему", "ому", "ыми", "ими", "ов", "ев",
    "ам", "ям", "ах", "ях", "ой", "ый", "ий", "ая", "яя", "ое", "ее",
    "ом", "ем", "у", "ю", "а", "я", "ы", "и", "е", "о", "ь",
)


_OCR_STATE_SYNONYMS = {
    # Normalize common colloquial forms of a negative feeling before matching
    # against OCR. This keeps the expansion narrow and avoids a dependency on
    # a heavyweight query-rewrite model for ordinary Russian phrasing.
    "хуево": "плохо",
    "херово": "плохо",
    "дерьмово": "плохо",
    "паршиво": "плохо",
}


def _ocr_token_key(token: str) -> str:
    key = token.lower().translate(_OCR_LOOKALIKE)
    return _OCR_STATE_SYNONYMS.get(key, key)


# Function words are useful for natural language but create broad, meaningless
# fuzzy hits (for example, ``не`` in almost any OCR text). They must not alone
# activate the direct OCR ranking leg.
_OCR_STOPWORDS = {
    "а", "без", "бы", "в", "во", "вы", "да", "для", "до", "его", "ее", "же",
    "за", "и", "из", "их", "как", "ко", "к", "ли", "меня", "мне", "мы", "на",
    "над", "не", "ни", "но", "о", "об", "он", "она", "они", "оно", "от", "по",
    "под", "с", "себя", "со", "та", "те", "тебе", "то", "тот", "ты", "у", "эта",
    "это", "эту", "черт", "что", "я",
}


def _ocr_tokens(text: str) -> list[str]:
    normalized = _ocr_normalize(text)
    return [
        _ocr_token_key(token)
        for token in normalized.split()
        if token and _ocr_token_key(token) not in _OCR_STOPWORDS
    ]


def _ocr_stem(token: str) -> str:
    for suffix in _OCR_SUFFIXES:
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token


def ocr_text_match_score(query: str, ocr_text: str) -> float:
    """Score useful OCR overlap, tolerant of inflection and common OCR lookalikes."""
    from difflib import SequenceMatcher

    query_tokens = [_ocr_stem(t) for t in _ocr_tokens(query)]
    text_tokens = [_ocr_stem(t) for t in _ocr_tokens(ocr_text)]
    if not query_tokens or not text_tokens:
        return 0.0
    matched: list[float] = []
    for query_token in query_tokens:
        best = 0.0
        for text_token in text_tokens:
            if query_token == text_token:
                best = 1.0
                break
            if (
                min(len(query_token), len(text_token)) >= 4
                and (query_token in text_token or text_token in query_token)
            ):
                best = max(best, 0.82)
            elif min(len(query_token), len(text_token)) >= 4:
                ratio = SequenceMatcher(None, query_token, text_token).ratio()
                if ratio >= 0.82:
                    best = max(best, 0.55)
        matched.append(best)
    informative = [score for token, score in zip(query_tokens, matched) if len(token) >= 4]
    if not informative:
        informative = matched
    coverage = sum(score >= 0.70 for score in informative) / len(informative)
    quality = sum(informative) / len(informative)
    phrase_bonus = 0.15 if _ocr_tokens(query) and " ".join(_ocr_tokens(query)) in " ".join(_ocr_tokens(ocr_text)) else 0.0
    # Keep function words for this very specific adjacent-token recovery: OCR
    # frequently turns the initial ``м`` in ``мне плохо`` into ``те плохо``.
    # They are intentionally excluded from generic fuzzy scoring above.
    state_query = " ".join(_ocr_token_key(t) for t in _ocr_normalize(query).split())
    state_text = " ".join(_ocr_token_key(t) for t in _ocr_normalize(ocr_text).split())
    state_bonus = 0.55 if "плох" in state_query and re.search(r"(?:мне|те)\s+плох", state_text) else 0.0
    return min(1.0, 0.65 * coverage + 0.25 * quality + phrase_bonus + state_bonus)


def _fts_query_str(query: str, *, operator: str = "OR") -> str | None:
    clean = _FTS_SPECIAL.sub(" ", query)
    words = [
        word
        for word in clean.split()
        if word and _ocr_token_key(word) not in _OCR_STOPWORDS
    ]
    if not words:
        return None
    if operator not in {"AND", "OR"}:
        raise ValueError(f"Unsupported FTS operator: {operator}")
    return f" {operator} ".join(f'"{word}"*' for word in words)


def _fts_ranked_ids(
    query: str,
    candidate_ids: set[int],
    *,
    operator: str = "OR",
) -> list[int]:
    fts_q = _fts_query_str(query, operator=operator)
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


def _ocr_fts_ranked_ids(
    query: str,
    candidate_ids: set[int],
    *,
    operator: str = "OR",
) -> list[int]:
    norm_q = _ocr_normalize(query)
    fts_q = _fts_query_str(norm_q, operator=operator)
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

    image_cache = _load(embedder.model_name)
    text_cache = image_cache
    text_query_vec = query_vec
    dedicated_text_active = False
    if config.TEXT_EMBED_ENABLED:
        try:
            text_embedder = _get_text_embedder()
            dedicated_cache = _load(text_embedder.model_id())
            if dedicated_cache is not image_cache and dedicated_cache.text_media_ids:
                dedicated_query = text_embedder.embed_text(query, is_query=True)
                if (
                    dedicated_cache.text_vecs.ndim == 2
                    and dedicated_cache.text_vecs.shape[1] == dedicated_query.shape[0]
                ):
                    text_query_vec = dedicated_query
                    text_cache = dedicated_cache
                    dedicated_text_active = True
        except (EmbedError, ModelNotInstalled):
            # The dedicated branch is opt-in and must not make the existing
            # SigLIP2 search unavailable when its local server is down.
            pass

    if not image_cache.media_ids and not text_cache.text_media_ids:
        return []

    image_best: dict[int, float] = {}
    if image_cache.media_ids:
        image_scores = image_cache.vecs @ query_vec
        for media_id, score in zip(image_cache.media_ids, image_scores.tolist()):
            if media_id not in image_best or score > image_best[media_id]:
                image_best[media_id] = score

    text_best: dict[int, float] = {}
    if text_cache.text_media_ids:
        text_scores = text_cache.text_vecs @ text_query_vec
        for media_id, score in zip(text_cache.text_media_ids, text_scores.tolist()):
            if media_id not in text_best or score > text_best[media_id]:
                text_best[media_id] = score

    candidate_ids = set(image_best) | set(text_best)
    if not candidate_ids:
        return []
    metadata = dict(image_cache.meta)
    metadata.update(text_cache.meta)

    vec_ranked = sorted(image_best, key=lambda mid: image_best[mid], reverse=True)
    ocr_sem_ranked = sorted(text_best, key=lambda mid: text_best[mid], reverse=True)

    # Multi-word AND FTS is a high-precision signal. Keep OR-only matches as a
    # deliberately weak recall leg: a ubiquitous word must not suppress an
    # otherwise strong visual semantic match.
    fts_ranked = _fts_ranked_ids(query, candidate_ids, operator="AND")
    ocr_ranked = _ocr_fts_ranked_ids(query, candidate_ids, operator="AND")
    fts_ids = set(fts_ranked)
    ocr_ids = set(ocr_ranked)
    fts_or_ranked = [
        media_id
        for media_id in _fts_ranked_ids(query, candidate_ids, operator="OR")
        if media_id not in fts_ids
    ]
    ocr_or_ranked = [
        media_id
        for media_id in _ocr_fts_ranked_ids(query, candidate_ids, operator="OR")
        if media_id not in ocr_ids
    ]

    # FTS is intentionally strict and misses partial/garbled OCR. Keep a
    # bounded fuzzy text leg over the already-loaded metadata instead of
    # letting generic image similarity decide a text query.
    ocr_lexical: dict[int, float] = {}
    for media_id in candidate_ids:
        row = metadata.get(media_id)
        if row is None:
            continue
        try:
            ocr_text = row["ocr_text"] or ""
        except (KeyError, IndexError):
            ocr_text = ""
        if ocr_text:
            score = ocr_text_match_score(query, ocr_text)
            if score >= 0.20:
                ocr_lexical[media_id] = score

    fts_w = 2.0 if len(query.split()) <= 3 else 1.0
    text_sem_w = (
        2.0 if dedicated_text_active
        else (1.15 if len(query.split()) >= 2 else 1.0)
    )
    weak_fts_w = 0.15

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
    if fts_or_ranked:
        lexical_lists.append((fts_or_ranked, weak_fts_w))
    if ocr_or_ranked:
        lexical_lists.append((ocr_or_ranked, weak_fts_w))

    if ocr_lexical:
        # Qwen's dedicated text space is the strongest text signal. Preserve
        # fuzzy OCR as corroboration, but do not let a partial lexical match
        # erase a higher-quality semantic paraphrase/cross-lingual match.
        max_ocr = max(ocr_lexical.values()) or 1.0
        max_img = max(image_best.values()) if image_best else 0.0
        min_img = min(image_best.values()) if image_best else 0.0
        img_span = (max_img - min_img) or 1.0
        max_text = max(text_best.values()) if text_best else 0.0
        min_text = min(text_best.values()) if text_best else 0.0
        text_span = (max_text - min_text) or 1.0
        base_scores = {}
        for media_id in candidate_ids:
            ocr_score = ocr_lexical.get(media_id, 0.0) / max_ocr
            img_score = (image_best.get(media_id, min_img) - min_img) / img_span
            text_score = text_best.get(media_id, -1.0)
            if dedicated_text_active:
                text_score = (text_score - min_text) / text_span
                base_scores[media_id] = 0.60 * text_score + 0.30 * ocr_score + 0.10 * img_score
            else:
                # Calibrated fusion: direct OCR evidence dominates weak visual noise.
                base_scores[media_id] = 0.78 * ocr_score + 0.17 * img_score + 0.05 * max(text_score, 0.0)
    elif lexical_lists:
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

        # Cosine scores from image and OCR-text embeddings are not calibrated to
        # one another (their ranges differ by an order of magnitude here).
        # Use their ranks for the final fusion instead of ``max(raw_score)``.
        rrf = _rrf_fuse_weighted(semantic_lists)
        vals = list(rrf.values())
        lo, span = min(vals), (max(vals) - min(vals)) or 1.0
        base_scores = {mid: (rrf[mid] - lo) / span for mid in rrf}

    query_lower = query.lower()
    results: list[SearchResult] = []
    for media_id, base in base_scores.items():
        row = metadata.get(media_id)
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
