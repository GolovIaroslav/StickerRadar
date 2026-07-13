from __future__ import annotations

import numpy as np


def _meta(media_id: int) -> dict:
    return {
        "tg_document_id": f"doc-{media_id}",
        "media_kind": "sticker",
        "sticker_format": "static",
        "local_path": None,
        "bot_file_id": None,
        "bot_send_method": None,
        "bot_cache_status": "missing",
        "set_short_name": None,
        "set_title": None,
        "emoji": None,
        "is_favorite": 0,
        "is_recent": 0,
        "ocr_text": "caption",
    }


def test_enabled_text_branch_uses_dedicated_query_and_vectors(monkeypatch):
    from app import config, search

    class FakeImageEmbedder:
        model_name = "siglip2"

        def embed_text(self, _query):
            return np.array([0.0, 1.0], dtype=np.float32)

    class FakeTextEmbedder:
        def model_id(self):
            return "qwen3-embedding-0.6b-q8"

        def embed_text(self, _query, *, is_query):
            assert is_query is True
            return np.array([1.0, 0.0], dtype=np.float32)

    image_cache = search._EmbedCache(
        count=2,
        text_count=0,
        media_ids=[1, 2],
        vecs=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        text_media_ids=[],
        text_vecs=np.empty((0, 0), dtype=np.float32),
        meta={1: _meta(1), 2: _meta(2)},
    )
    text_cache = search._EmbedCache(
        count=0,
        text_count=2,
        media_ids=[],
        vecs=np.empty((0, 0), dtype=np.float32),
        text_media_ids=[1, 2],
        text_vecs=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        meta={1: _meta(1), 2: _meta(2)},
    )

    monkeypatch.setattr(config, "TEXT_EMBED_ENABLED", True)
    monkeypatch.setattr(search, "_get_embedder", lambda: FakeImageEmbedder())
    monkeypatch.setattr(search, "_get_text_embedder", lambda: FakeTextEmbedder())
    monkeypatch.setattr(search, "_load", lambda name: image_cache if name == "siglip2" else text_cache)
    monkeypatch.setattr(search, "_fts_ranked_ids", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(search, "_ocr_fts_ranked_ids", lambda *_args, **_kwargs: [])

    results = search.search("meaning", top_k=2)

    assert [result.media_id for result in results] == [1, 2]


def test_dedicated_text_semantics_are_not_overridden_by_weak_ocr_fuzzy_match(monkeypatch):
    from app import config, search

    class FakeImageEmbedder:
        model_name = "siglip2"

        def embed_text(self, _query):
            return np.array([0.0, 1.0], dtype=np.float32)

    class FakeTextEmbedder:
        def model_id(self):
            return "qwen3-embedding-0.6b-q8"

        def embed_text(self, _query, *, is_query):
            assert is_query is True
            return np.array([1.0, 0.0], dtype=np.float32)

    target = _meta(1)
    lexical = _meta(2)
    target["ocr_text"] = "semantic target"
    lexical["ocr_text"] = "weak fuzzy match"
    image_cache = search._EmbedCache(
        count=2,
        text_count=0,
        media_ids=[1, 2],
        vecs=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        text_media_ids=[],
        text_vecs=np.empty((0, 0), dtype=np.float32),
        meta={1: target, 2: lexical},
    )
    text_cache = search._EmbedCache(
        count=0,
        text_count=2,
        media_ids=[],
        vecs=np.empty((0, 0), dtype=np.float32),
        text_media_ids=[1, 2],
        text_vecs=np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
        meta={1: target, 2: lexical},
    )

    monkeypatch.setattr(config, "TEXT_EMBED_ENABLED", True)
    monkeypatch.setattr(search, "_get_embedder", lambda: FakeImageEmbedder())
    monkeypatch.setattr(search, "_get_text_embedder", lambda: FakeTextEmbedder())
    monkeypatch.setattr(search, "_load", lambda name: image_cache if name == "siglip2" else text_cache)
    monkeypatch.setattr(search, "_fts_ranked_ids", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(search, "_ocr_fts_ranked_ids", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        search,
        "ocr_text_match_score",
        lambda _query, text: 0.20 if text == "semantic target" else 1.0,
    )

    results = search.search("meaning", top_k=2)

    assert [result.media_id for result in results] == [1, 2]
