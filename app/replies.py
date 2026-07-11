"""Rank library stickers/GIFs as a reaction to an incoming media item.

This intentionally does not run ordinary nearest-neighbour search.  It uses the
same multimodal embedding space as StickerRadar's regular search, but first
infers a *source reaction type* from visual frames and OCR text, then ranks
library media against the corresponding counter-reaction intents.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.search import SearchResult, _EmbedCache, _get_embedder, _load

# (what the inbound media expresses, a strong reply direction).  These are
# short retrieval prompts, not generated reply text: all sent media comes from
# the user's already-indexed local collection.
_RESPONSE_PROFILES: tuple[tuple[str, str], ...] = (
    ("протест отказ возражение недовольство", "уверенная победа саркастичный ответ доминирование"),
    ("угроза злость агрессия конфликт", "невозмутимый сильный ответ спокойная победа"),
    ("хвастовство самоуверенность понты", "ирония обесценивание смешной подкол"),
    ("грусть плач поражение разочарование", "поддержка сочувствие неловкая смешная реакция"),
    ("удивление шок непонимание", "саркастичное согласие смешная реакция"),
    ("флирт любовь романтика", "дерзкий флирт уверенный смешной ответ"),
)
_GENERIC_REPLY = "смешная сильная ответная реакция сарказм уверенный подкол"


def _best_by_media(media_ids: Sequence[int], scores: np.ndarray) -> dict[int, float]:
    best: dict[int, float] = {}
    for media_id, score in zip(media_ids, scores.tolist(), strict=True):
        best[media_id] = max(best.get(media_id, -1.0), float(score))
    return best


def _result(media_id: int, row, score: float) -> SearchResult:
    return SearchResult(
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
        emoji=row["emoji"],
        is_favorite=bool(row["is_favorite"]),
        is_recent=bool(row["is_recent"]),
        score=score,
    )


def _incoming_score(query_vectors: Sequence[np.ndarray], candidate_vectors: np.ndarray, media_ids: Sequence[int]) -> dict[int, float]:
    if len(query_vectors) == 0 or candidate_vectors.size == 0:
        return {}
    query = np.stack(query_vectors)
    return _best_by_media(media_ids, (candidate_vectors @ query.T).max(axis=1))


def rank_reply_candidates(
    *,
    image_vectors: Sequence[np.ndarray],
    ocr_text: str,
    top_k: int = 10,
    intent: dict | None = None,
) -> list[SearchResult]:
    """Return diverse, counter-reaction candidates for already embedded media.

    Image-frame embeddings are used to infer the incoming reaction type and OCR
    text gets a separate semantic leg. Direct visual similarity is deliberately
    only a small relevance signal, so a duplicate of the incoming sticker cannot
    outrank a stronger response reaction.
    """
    embedder = _get_embedder()
    cache: _EmbedCache = _load(embedder.model_name)
    if not cache.media_ids and not cache.text_media_ids:
        return []

    source_image = _incoming_score(image_vectors, cache.vecs, cache.media_ids)
    source_text: dict[int, float] = {}
    ocr_vector: np.ndarray | None = None
    if ocr_text.strip() and cache.text_media_ids:
        ocr_vector = embedder.embed_text(ocr_text)
        source_text = _best_by_media(cache.text_media_ids, cache.text_vecs @ ocr_vector)

    trigger_prompts = [source for source, _ in _RESPONSE_PROFILES]
    trigger_vectors = embedder.embed_texts(trigger_prompts)
    trigger_scores: list[float] = []
    for vector in trigger_vectors:
        scores: list[float] = []
        if image_vectors:
            scores.append(float(np.max(np.stack(image_vectors) @ vector)))
        if ocr_vector is not None:
            scores.append(float(ocr_vector @ vector))
        trigger_scores.append(max(scores, default=-1.0))
    best_profile = int(np.argmax(trigger_scores)) if trigger_scores else 0

    # Do not embed "response to <incoming OCR text>" directly: in a shared
    # image/text space it often retrieves a visual duplicate of that text.
    # OCR still influences the selected source profile above; retrieval then
    # uses only its counter-reaction direction.
    reply_prompts = [_RESPONSE_PROFILES[best_profile][1], _GENERIC_REPLY]
    if intent:
        reply_prompts.extend(
            str(query).strip() for query in intent.get("retrieval_queries", [])
            if str(query).strip()
        )
        role = str(intent.get("desired_reply_role", "")).strip()
        if role:
            reply_prompts.append(f"смешная реакция в роли: {role}")
    reply_vectors = embedder.embed_texts(list(dict.fromkeys(reply_prompts)))

    reaction_image = _incoming_score(reply_vectors, cache.vecs, cache.media_ids)
    reaction_text = _incoming_score(reply_vectors, cache.text_vecs, cache.text_media_ids)
    candidate_ids = set(source_image) | set(source_text) | set(reaction_image) | set(reaction_text)

    ranked: list[SearchResult] = []
    for media_id in candidate_ids:
        row = cache.meta.get(media_id)
        if row is None:
            continue
        # The source only chooses a response profile. Rewarding source overlap
        # turns the feature back into ordinary nearest-neighbour search, so it
        # is explicitly a duplicate penalty rather than a relevance bonus.
        image_reply = reaction_image.get(media_id, 0.0)
        text_reply = reaction_text.get(media_id, 0.0)
        image_duplicate = source_image.get(media_id, 0.0)
        text_duplicate = source_text.get(media_id, 0.0)
        score = 0.65 * image_reply + 0.25 * text_reply - 0.20 * image_duplicate - 0.10 * text_duplicate
        if row["is_favorite"]:
            score += 0.025
        if row["is_recent"]:
            score += 0.010
        ranked.append(_result(media_id, row, score))

    ranked.sort(key=lambda item: item.score, reverse=True)
    # One item per sticker pack gives actual alternatives, not ten neighboring
    # frames from a single pack. GIFs without a pack remain individually eligible.
    seen_packs: set[str] = set()
    diverse: list[SearchResult] = []
    for item in ranked:
        key = item.set_short_name or f"__single_{item.media_id}"
        if key in seen_packs:
            continue
        seen_packs.add(key)
        diverse.append(item)
        if len(diverse) >= top_k:
            break
    return diverse
