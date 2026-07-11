from __future__ import annotations

import importlib
from pathlib import Path


def test_runtime_profile_recommendation_prefers_glm_when_llama_cpp_is_available():
    from app.setup_wizard import RuntimeProfile, choose_default_ocr_profile

    runtime = RuntimeProfile(
        has_gpu=True,
        gpu_name="RTX 3060",
        gpu_total_gb=6.0,
        gpu_free_gb=3.5,
        llama_cpp_available=True,
        ffmpeg_available=True,
    )

    ocr = choose_default_ocr_profile(runtime)
    assert ocr.key == "off"


def test_run_ocr_uses_glm_backend_when_configured_primary_is_available(monkeypatch, tmp_path, capsys):
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
    monkeypatch.setattr(config, "OCR_GLM_PREFILTER_BACKEND", "auto")

    config.ensure_dirs()
    conn = db.get_conn()
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
    conn.execute("UPDATE media_items SET download_status='ok' WHERE id=?", (media_id,))
    conn.commit()
    db.mark_preview_ok(media_id)

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

    called = []

    def fake_glm(paths: list[Path]):
        called.append([p.name for p in paths])
        return [("Я долбоеб", "я долбоеб", 0.65)]

    monkeypatch.setattr(ocr, "is_backend_available", lambda backend: backend == "glm-ocr")
    monkeypatch.setattr(ocr, "_ocr_glm", fake_glm)

    scanner._run_ocr(limit=None)

    out = capsys.readouterr().out
    assert "OCR skipped" not in out
    assert called == [["frame.png"]]

    row = conn.execute(
        "SELECT ocr_status, ocr_text FROM media_items WHERE id=?",
        (media_id,),
    ).fetchone()
    assert row["ocr_status"] == "ok"
    assert row["ocr_text"] == "я долбоеб"

    frame_row = conn.execute(
        "SELECT raw_text, norm_text, confidence FROM frame_ocr WHERE frame_id=?",
        (frame_id,),
    ).fetchone()
    assert frame_row["raw_text"] == "Я долбоеб"
    assert frame_row["norm_text"] == "я долбоеб"
    assert frame_row["confidence"] == 0.65


def test_glm_primary_uses_fast_detector_and_only_escalates_low_confidence_paths(monkeypatch):
    from app import config
    from app import ocr

    monkeypatch.setattr(config, "OCR_BACKEND", "glm-ocr")
    monkeypatch.setattr(config, "OCR_FALLBACK_ENABLED", False)
    monkeypatch.setattr(config, "OCR_FALLBACK_CONFIDENCE", 0.45)
    monkeypatch.setattr(config, "OCR_FALLBACK_MIN_CHARS", 3)

    detector_calls = []
    glm_calls = []

    def fake_detector(paths):
        detector_calls.append([p.name for p in paths])
        return [
            ("", "", 0.0),
            ("good text", "good text", 0.91),
            ("xo", "xo", 0.91),
            ("bad read", "bad read", 0.2),
        ]

    def fake_glm(paths):
        glm_calls.append([p.name for p in paths])
        return [
            ("fixed short", "fixed short", 0.71),
            ("fixed low confidence", "fixed low confidence", 0.72),
        ]

    monkeypatch.setattr(ocr, "is_backend_available", lambda backend: backend in {"glm-ocr", "rapidocr"})
    monkeypatch.setattr(ocr, "_ocr_rapidocr", fake_detector)
    monkeypatch.setattr(ocr, "_ocr_glm", fake_glm)

    paths = [Path("no-text.png"), Path("good.png"), Path("short.png"), Path("bad.png")]
    out = ocr.ocr_frames(paths)

    assert detector_calls == [["no-text.png", "good.png", "short.png", "bad.png"]]
    assert glm_calls == [["short.png", "bad.png"]]
    assert out == [
        ("", "", 0.0),
        ("good text", "good text", 0.91),
        ("fixed short", "fixed short", 0.71),
        ("fixed low confidence", "fixed low confidence", 0.72),
    ]


def test_glm_primary_reports_final_backend_for_every_frame(monkeypatch):
    from app import config
    from app import ocr

    monkeypatch.setattr(config, "OCR_BACKEND", "glm-ocr")
    monkeypatch.setattr(config, "OCR_FALLBACK_ENABLED", False)
    monkeypatch.setattr(config, "OCR_FALLBACK_CONFIDENCE", 0.45)
    monkeypatch.setattr(config, "OCR_FALLBACK_MIN_CHARS", 3)
    monkeypatch.setattr(ocr, "is_backend_available", lambda backend: backend in {"glm-ocr", "rapidocr"})
    monkeypatch.setattr(
        ocr,
        "_ocr_rapidocr",
        lambda paths: [("fast", "fast", 0.91), ("bad", "bad", 0.2)],
    )
    monkeypatch.setattr(ocr, "_ocr_glm", lambda paths: [("rescued", "rescued", 0.72)])

    results = ocr.ocr_frames_with_provenance([Path("fast.png"), Path("bad.png")])

    assert [(r.norm_text, r.backend) for r in results] == [
        ("fast", "rapidocr"),
        ("rescued", "glm-ocr"),
    ]
