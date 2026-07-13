from __future__ import annotations

import os
import sys

import pytest


def _add_media(db, conn, tmp_path, document_id: str, ocr_text: str) -> tuple[int, object]:
    media_id = db.upsert_media_item(
        tg_document_id=document_id,
        access_hash="hash",
        file_reference=b"ref",
        media_kind="sticker",
        sticker_format="static",
        mime_type="image/png",
        file_ext=".png",
        emoji=None,
        set_id=None,
        set_access_hash=None,
        set_short_name=None,
        set_title=None,
    )
    image = tmp_path / f"{document_id}.png"
    image.write_bytes(b"not-decoded-in-test")
    conn.execute(
        "UPDATE media_items SET local_path=?, ocr_status='ok', ocr_text=? WHERE id=?",
        (str(image), ocr_text, media_id),
    )
    conn.commit()
    return media_id, image


def test_shadow_ocr_only_writes_its_own_table(monkeypatch, tmp_path):
    from app import config, db, shadow_ocr

    db.close()
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.sqlite")
    conn = db.get_conn()
    corpus_media_id, corpus_image = _add_media(db, conn, tmp_path, "corpus", "canonical OCR")
    empty_media_id, empty_image = _add_media(db, conn, tmp_path, "empty", "")

    monkeypatch.setattr(
        shadow_ocr,
        "_load_corpus",
        lambda: [{"id": corpus_media_id, "visual_text": "ground truth"}],
    )

    def fake_worker(paths):
        assert set(paths) == {corpus_image.resolve(), empty_image.resolve()}
        return [
            shadow_ocr.ShadowOCRResult(path, f"shadow {path.stem}", [[[0, 0]]], [0.9])
            for path in paths
        ]

    monkeypatch.setattr(shadow_ocr, "run_ppocr_worker", fake_worker)
    report = shadow_ocr.run_shadow_ocr(limit=2)

    assert report == {
        "processed": 2,
        "skipped": 0,
        "previously_empty": 1,
        "recovered": 1,
        "plausibility_candidates": 1,
        "corpus_reported": 1,
        "mean_current_cer": 1.0,
        "mean_shadow_cer": 11 / 12,
    }
    canonical = conn.execute(
        "SELECT id, ocr_text FROM media_items ORDER BY id"
    ).fetchall()
    assert [(row["id"], row["ocr_text"]) for row in canonical] == [
        (corpus_media_id, "canonical OCR"),
        (empty_media_id, ""),
    ]
    shadow_rows = conn.execute(
        "SELECT media_id, backend, text, boxes_json, scores_json FROM ocr_shadow ORDER BY media_id"
    ).fetchall()
    assert [(row["media_id"], row["backend"], row["text"]) for row in shadow_rows] == [
        (corpus_media_id, "ppocrv5-eslav", "shadow corpus"),
        (empty_media_id, "ppocrv5-eslav", "shadow empty"),
    ]
    assert shadow_rows[0]["boxes_json"] == "[[[0, 0]]]"
    assert shadow_rows[0]["scores_json"] == "[0.9]"
    db.close()


def test_character_error_rate_uses_normalized_plain_levenshtein():
    from app.shadow_ocr import character_error_rate

    assert character_error_rate("Пятница!", "пятница") == 0.0
    assert character_error_rate("abc", "axc") == 1 / 3


def test_shadow_plausibility_heuristic_rejects_punctuation_and_short_noise():
    from app.shadow_ocr import _looks_plausible_text

    assert _looks_plausible_text("пятница")
    assert not _looks_plausible_text("!!!")
    assert not _looks_plausible_text("ok")


def test_shadow_ocr_uses_png_preview_for_webp_stickers(tmp_path):
    from app.shadow_ocr import _image_path

    source = tmp_path / "sticker.webp"
    preview = tmp_path / "frame.png"
    source.write_bytes(b"webp")
    preview.write_bytes(b"png")

    assert _image_path({"local_path": str(source), "preview_path": str(preview)}) == preview.resolve()


def test_shadow_ocr_uses_video_when_no_preview_survives(tmp_path):
    from app.shadow_ocr import _image_path

    source = tmp_path / "sticker.webm"
    source.write_bytes(b"video")

    assert _image_path({"local_path": str(source), "preview_path": None}) == source.resolve()


def test_shadow_ocr_keeps_tgs_source_when_no_preview_survives(tmp_path):
    from app.shadow_ocr import _image_path

    source = tmp_path / "sticker.tgs"
    source.write_bytes(b"tgs")

    assert _image_path({"local_path": str(source), "preview_path": None}) == source.resolve()


def test_shadow_worker_path_may_be_a_venv_python_symlink(monkeypatch, tmp_path):
    from app import config, shadow_ocr

    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are unavailable")
    worker = tmp_path / "venv" / "bin" / "python"
    worker.parent.mkdir(parents=True)
    os.symlink(sys.executable, worker)
    monkeypatch.setattr(config, "OCR_SHADOW_PYTHON", str(worker))

    assert shadow_ocr._worker_python() == str(worker.absolute())
