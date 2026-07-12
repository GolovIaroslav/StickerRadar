from __future__ import annotations

import numpy as np


def test_search_returns_empty_on_low_confidence_semantic_only(monkeypatch):
    from app import search

    class FakeEmbedder:
        model_name = 'test-model'

        def embed_text(self, text: str):
            return np.array([1.0, 0.0], dtype=np.float32)

    meta = {
        1: {
            'tg_document_id': 'doc-1',
            'media_kind': 'sticker',
            'sticker_format': 'static',
            'local_path': '/tmp/1.webp',
            'bot_file_id': None,
            'bot_send_method': None,
            'bot_cache_status': None,
            'set_short_name': 'alpha',
            'set_title': 'Alpha',
            'emoji': None,
            'is_favorite': 0,
            'is_recent': 0,
        },
        2: {
            'tg_document_id': 'doc-2',
            'media_kind': 'sticker',
            'sticker_format': 'static',
            'local_path': '/tmp/2.webp',
            'bot_file_id': None,
            'bot_send_method': None,
            'bot_cache_status': None,
            'set_short_name': 'beta',
            'set_title': 'Beta',
            'emoji': None,
            'is_favorite': 0,
            'is_recent': 0,
        },
    }

    cache = search._EmbedCache(
        count=2,
        text_count=0,
        media_ids=[1, 2],
        vecs=np.array([
            [0.08, 0.92],
            [0.07, 0.93],
        ], dtype=np.float32),
        text_media_ids=[],
        text_vecs=np.empty((0, 0), dtype=np.float32),
        meta=meta,
    )

    monkeypatch.setattr(search, '_get_embedder', lambda: FakeEmbedder())
    monkeypatch.setattr(search, '_load', lambda model_name: cache)
    monkeypatch.setattr(search, '_fts_ranked_ids', lambda query, candidate_ids, **kwargs: [])
    monkeypatch.setattr(search, '_ocr_fts_ranked_ids', lambda query, candidate_ids, **kwargs: [])

    results = search.search('совсем случайный бессмысленный запрос', top_k=5)
    assert results == []


def test_search_returns_empty_on_weak_short_semantic_only_text_query(monkeypatch):
    from app import search

    class FakeEmbedder:
        model_name = 'test-model'

        def embed_text(self, text: str):
            return np.array([1.0, 0.0], dtype=np.float32)

    meta = {
        1: {
            'tg_document_id': 'doc-1',
            'media_kind': 'sticker',
            'sticker_format': 'static',
            'local_path': '/tmp/1.webp',
            'bot_file_id': None,
            'bot_send_method': None,
            'bot_cache_status': None,
            'set_short_name': 'alpha',
            'set_title': 'Alpha',
            'emoji': None,
            'is_favorite': 0,
            'is_recent': 0,
        },
        2: {
            'tg_document_id': 'doc-2',
            'media_kind': 'sticker',
            'sticker_format': 'static',
            'local_path': '/tmp/2.webp',
            'bot_file_id': None,
            'bot_send_method': None,
            'bot_cache_status': None,
            'set_short_name': 'beta',
            'set_title': 'Beta',
            'emoji': None,
            'is_favorite': 0,
            'is_recent': 0,
        },
    }

    cache = search._EmbedCache(
        count=2,
        text_count=2,
        media_ids=[1, 2],
        vecs=np.array([
            [0.12, 0.88],
            [0.11, 0.89],
        ], dtype=np.float32),
        text_media_ids=[1, 2],
        text_vecs=np.array([
            [0.79, 0.21],
            [0.78, 0.22],
        ], dtype=np.float32),
        meta=meta,
    )

    monkeypatch.setattr(search, '_get_embedder', lambda: FakeEmbedder())
    monkeypatch.setattr(search, '_load', lambda model_name: cache)
    monkeypatch.setattr(search, '_fts_ranked_ids', lambda query, candidate_ids, **kwargs: [])
    monkeypatch.setattr(search, '_ocr_fts_ranked_ids', lambda query, candidate_ids, **kwargs: [])

    results = search.search('абракадабра ъуъ', top_k=5)
    assert results == []


def test_search_keeps_strong_short_semantic_only_text_query(monkeypatch):
    from app import search

    class FakeEmbedder:
        model_name = 'test-model'

        def embed_text(self, text: str):
            return np.array([1.0, 0.0], dtype=np.float32)

    meta = {
        1: {
            'tg_document_id': 'doc-1',
            'media_kind': 'sticker',
            'sticker_format': 'static',
            'local_path': '/tmp/1.webp',
            'bot_file_id': None,
            'bot_send_method': None,
            'bot_cache_status': None,
            'set_short_name': 'alpha',
            'set_title': 'Alpha',
            'emoji': None,
            'is_favorite': 0,
            'is_recent': 0,
        },
        2: {
            'tg_document_id': 'doc-2',
            'media_kind': 'sticker',
            'sticker_format': 'static',
            'local_path': '/tmp/2.webp',
            'bot_file_id': None,
            'bot_send_method': None,
            'bot_cache_status': None,
            'set_short_name': 'beta',
            'set_title': 'Beta',
            'emoji': None,
            'is_favorite': 0,
            'is_recent': 0,
        },
    }

    cache = search._EmbedCache(
        count=2,
        text_count=2,
        media_ids=[1, 2],
        vecs=np.array([
            [0.12, 0.88],
            [0.11, 0.89],
        ], dtype=np.float32),
        text_media_ids=[1, 2],
        text_vecs=np.array([
            [0.95, 0.05],
            [0.81, 0.19],
        ], dtype=np.float32),
        meta=meta,
    )

    monkeypatch.setattr(search, '_get_embedder', lambda: FakeEmbedder())
    monkeypatch.setattr(search, '_load', lambda model_name: cache)
    monkeypatch.setattr(search, '_fts_ranked_ids', lambda query, candidate_ids, **kwargs: [])
    monkeypatch.setattr(search, '_ocr_fts_ranked_ids', lambda query, candidate_ids, **kwargs: [])

    results = search.search('я идиот', top_k=5)
    assert [r.media_id for r in results] == [1, 2]
