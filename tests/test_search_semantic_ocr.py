from __future__ import annotations

import numpy as np


def test_search_uses_semantic_ocr_text_leg_when_lexical_text_misses(monkeypatch):
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
            'emoji': '🙂',
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
            'emoji': '🙂',
            'is_favorite': 0,
            'is_recent': 0,
        },
    }

    cache = search._EmbedCache(
        count=2,
        text_count=2,
        media_ids=[1, 2],
        vecs=np.array([
            [0.05, 0.95],
            [0.04, 0.96],
        ], dtype=np.float32),
        text_media_ids=[1, 2],
        text_vecs=np.array([
            [0.99, 0.01],
            [0.10, 0.90],
        ], dtype=np.float32),
        meta=meta,
    )
    cache.text_count = 2
    cache.text_media_ids = [1, 2]
    cache.text_vecs = np.array([
        [0.99, 0.01],
        [0.20, 0.80],
    ], dtype=np.float32)

    monkeypatch.setattr(search, '_get_embedder', lambda: FakeEmbedder())
    monkeypatch.setattr(search, '_load', lambda model_name: cache)
    monkeypatch.setattr(search, '_fts_ranked_ids', lambda query, candidate_ids: [])
    monkeypatch.setattr(search, '_ocr_fts_ranked_ids', lambda query, candidate_ids: [])

    results = search.search('я идиот', top_k=2)

    assert [r.media_id for r in results][:2] == [1, 2]
    assert results[0].score > results[1].score
