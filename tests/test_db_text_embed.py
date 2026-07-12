from __future__ import annotations

import numpy as np


def test_media_text_embeddings_support_multiple_models(monkeypatch, tmp_path):
    from app import config, db

    db.close()
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.sqlite")
    media_id = db.upsert_media_item(
        tg_document_id="text-model-test",
        access_hash="hash",
        file_reference=b"ref",
        media_kind="sticker",
        sticker_format="static",
        mime_type="image/webp",
        file_ext=".webp",
        emoji=None,
        set_id=None,
        set_access_hash=None,
        set_short_name=None,
        set_title=None,
    )
    vector = np.array([1.0, 0.0], dtype=np.float32)
    for model_name in ("siglip2", "qwen3-embedding-0.6b-q8"):
        db.upsert_media_text_embedding(
            media_id=media_id,
            model_name=model_name,
            dim=2,
            vector_bytes=vector.tobytes(),
            source_text="hello",
        )

    rows = db.get_conn().execute(
        "SELECT model_name FROM media_text_embeddings WHERE media_id=? ORDER BY model_name",
        (media_id,),
    ).fetchall()
    assert [row["model_name"] for row in rows] == ["qwen3-embedding-0.6b-q8", "siglip2"]
    db.close()
