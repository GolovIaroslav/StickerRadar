from __future__ import annotations

import numpy as np


def _meta(mid: int, pack: str) -> dict[str, object]:
    return {
        "tg_document_id": f"doc-{mid}",
        "media_kind": "sticker",
        "sticker_format": "static",
        "local_path": f"/tmp/{mid}.webp",
        "bot_file_id": None,
        "bot_send_method": None,
        "bot_cache_status": "missing",
        "set_short_name": pack,
        "set_title": pack,
        "emoji": None,
        "is_favorite": 0,
        "is_recent": 0,
    }


def test_reply_ranking_prefers_response_intent_over_visual_duplicate(monkeypatch):
    from app import replies, search

    # item 1 is visually identical to the inbound protest image; item 2 is the
    # confident/sarcastic reaction intent. A reply search must pick 2 first.
    cache = search._EmbedCache(
        count=3,
        text_count=0,
        media_ids=[1, 2, 3],
        vecs=np.array([[1.0, 0.0], [0.25, 0.97], [0.2, 0.8]], dtype=np.float32),
        text_media_ids=[],
        text_vecs=np.empty((0, 0), dtype=np.float32),
        meta={1: _meta(1, "same-source"), 2: _meta(2, "replies"), 3: _meta(3, "other")},
    )

    class FakeEmbedder:
        model_name = "test"

        def embed_texts(self, texts):
            vectors = []
            for text in texts:
                if "уверенная победа" in text:
                    vectors.append([0.0, 1.0])
                elif "протест" in text:
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 0.5])
            return np.array(vectors, dtype=np.float32)

        def embed_text(self, text):
            return self.embed_texts([text])[0]

    monkeypatch.setattr(replies, "_get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(replies, "_load", lambda _: cache)

    results = replies.rank_reply_candidates(
        image_vectors=[np.array([1.0, 0.0], dtype=np.float32)],
        ocr_text="протестую",
        top_k=2,
    )

    assert [item.media_id for item in results] == [2, 3]


def test_reply_ranking_keeps_one_result_per_pack(monkeypatch):
    from app import replies, search

    cache = search._EmbedCache(
        count=3,
        text_count=0,
        media_ids=[1, 2, 3],
        vecs=np.array([[0.0, 0.99], [0.0, 0.98], [0.0, 0.97]], dtype=np.float32),
        text_media_ids=[],
        text_vecs=np.empty((0, 0), dtype=np.float32),
        meta={1: _meta(1, "one-pack"), 2: _meta(2, "one-pack"), 3: _meta(3, "second-pack")},
    )

    class FakeEmbedder:
        model_name = "test"
        def embed_texts(self, texts):
            return np.array([[0.0, 1.0] for _ in texts], dtype=np.float32)

    monkeypatch.setattr(replies, "_get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(replies, "_load", lambda _: cache)

    results = replies.rank_reply_candidates(
        image_vectors=[np.array([1.0, 0.0], dtype=np.float32)], ocr_text="", top_k=10
    )

    assert [item.media_id for item in results] == [1, 3]
