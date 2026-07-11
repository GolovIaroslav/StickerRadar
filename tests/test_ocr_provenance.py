from __future__ import annotations

import importlib
from pathlib import Path


def _prepare_modules(monkeypatch, tmp_path):
    monkeypatch.setenv("STICKERRADAR_ENV_FILE", str(tmp_path / ".env"))
    monkeypatch.chdir(tmp_path)

    import app.config as config
    import app.db as db
    import app.ocr as ocr
    import app.scanner as scanner

    importlib.reload(config)
    importlib.reload(db)
    importlib.reload(ocr)
    importlib.reload(scanner)

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.sqlite")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PREVIEWS_DIR", tmp_path / "previews")
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(config, "SESSION_PATH", tmp_path / "sessions" / "user")
    monkeypatch.setattr(config, "OCR_ENABLED", True)
    monkeypatch.setattr(config, "OCR_BACKEND", "glm-ocr")
    monkeypatch.setattr(config, "OCR_FALLBACK_ENABLED", False)
    monkeypatch.setattr(config, "OCR_LLM_REPO", "ggml-org/GLM-OCR-GGUF:Q8_0")

    config.ensure_dirs()
    conn = db.get_conn()
    return config, db, ocr, scanner, conn


def _seed_media(db, conn, tmp_path):
    media_id = db.upsert_media_item(
        tg_document_id="doc-1",
        access_hash="hash",
        file_reference=b"ref",
        media_kind="sticker",
        sticker_format="static",
        mime_type="image/webp",
        file_ext=".webp",
        emoji="🙂",
        set_id="set1",
        set_access_hash="sa",
        set_short_name="pack",
        set_title="Pack",
        is_installed=True,
    )
    conn.execute(
        "UPDATE media_items SET download_status='ok', preview_status='ok', embed_status='ok', ocr_status='ok', ocr_text='старый текст' WHERE id=?",
        (media_id,),
    )
    conn.commit()
    preview_path = tmp_path / "frame.png"
    preview_path.write_bytes(b"fake-image")
    frame_id = db.upsert_frame(
        media_id=media_id,
        frame_index=0,
        frame_pos=0.0,
        preview_path=str(preview_path),
        width=512,
        height=512,
    )
    conn.execute(
        "INSERT OR REPLACE INTO frame_ocr(frame_id, raw_text, norm_text, confidence) VALUES (?, ?, ?, ?)",
        (frame_id, "Старый Текст", "старый текст", 0.25),
    )
    conn.commit()
    return media_id, frame_id, preview_path


def test_run_ocr_persists_backend_pipeline_and_timestamp(monkeypatch, tmp_path):
    config, db, ocr, scanner, conn = _prepare_modules(monkeypatch, tmp_path)
    media_id, frame_id, _ = _seed_media(db, conn, tmp_path)
    conn.execute("UPDATE media_items SET ocr_status='pending', ocr_text=NULL WHERE id=?", (media_id,))
    conn.execute("DELETE FROM frame_ocr WHERE frame_id=?", (frame_id,))
    conn.commit()

    def fake_glm(paths: list[Path]):
        return [("Я долбоеб", "я долбоеб", 0.65)]

    monkeypatch.setattr(ocr, "is_backend_available", lambda backend: backend == "glm-ocr")
    monkeypatch.setattr(ocr, "_ocr_glm", fake_glm)

    scanner._run_ocr(limit=None)

    media_row = conn.execute(
        "SELECT ocr_status, ocr_text, ocr_backend, ocr_model_ref, ocr_pipeline_version, ocr_generated_at FROM media_items WHERE id=?",
        (media_id,),
    ).fetchone()
    assert media_row["ocr_status"] == "ok"
    assert media_row["ocr_text"] == "я долбоеб"
    assert media_row["ocr_backend"] == "glm-ocr"
    assert media_row["ocr_model_ref"] == config.OCR_LLM_REPO
    assert media_row["ocr_pipeline_version"]
    assert media_row["ocr_generated_at"]

    frame_row = conn.execute(
        "SELECT raw_text, norm_text, confidence, backend, model_ref, pipeline_version, generated_at FROM frame_ocr WHERE frame_id=?",
        (frame_id,),
    ).fetchone()
    assert frame_row["raw_text"] == "Я долбоеб"
    assert frame_row["norm_text"] == "я долбоеб"
    assert frame_row["confidence"] == 0.65
    assert frame_row["backend"] == "glm-ocr"
    assert frame_row["model_ref"] == config.OCR_LLM_REPO
    assert frame_row["pipeline_version"]
    assert frame_row["generated_at"]


def test_force_reocr_clears_old_ocr_payload_and_marks_items_pending(monkeypatch, tmp_path):
    _config, db, _ocr, _scanner, conn = _prepare_modules(monkeypatch, tmp_path)
    media_id, _frame_id, _preview_path = _seed_media(db, conn, tmp_path)
    conn.execute(
        "UPDATE media_items SET ocr_backend='easyocr', ocr_model_ref='legacy', ocr_pipeline_version='old', ocr_generated_at='2026-01-01 00:00:00' WHERE id=?",
        (media_id,),
    )
    conn.commit()

    reset_count = db.force_reocr()
    assert reset_count == 1

    media_row = conn.execute(
        "SELECT ocr_status, ocr_text, ocr_backend, ocr_model_ref, ocr_pipeline_version, ocr_generated_at FROM media_items WHERE id=?",
        (media_id,),
    ).fetchone()
    assert media_row["ocr_status"] == "pending"
    assert media_row["ocr_text"] is None
    assert media_row["ocr_backend"] is None
    assert media_row["ocr_model_ref"] is None
    assert media_row["ocr_pipeline_version"] is None
    assert media_row["ocr_generated_at"] is None

    count_frame_ocr = conn.execute("SELECT COUNT(*) FROM frame_ocr").fetchone()[0]
    assert count_frame_ocr == 0
