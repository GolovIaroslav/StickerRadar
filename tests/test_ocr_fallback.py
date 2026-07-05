from __future__ import annotations

from pathlib import Path


def test_should_use_llm_fallback_heuristic():
    from app.ocr import should_use_llm_fallback

    assert should_use_llm_fallback("", "", 0.0, min_confidence=0.45, min_chars=3)
    assert should_use_llm_fallback("yo", "yo", 0.9, min_confidence=0.45, min_chars=3)
    assert should_use_llm_fallback("text", "text", 0.2, min_confidence=0.45, min_chars=3)
    assert not should_use_llm_fallback("нормальный текст", "нормальный текст", 0.8, min_confidence=0.45, min_chars=3)


def test_ocr_frames_applies_fallback_only_to_flagged_items(monkeypatch):
    from app import config
    from app import ocr

    monkeypatch.setattr(config, "OCR_BACKEND", "easyocr")
    monkeypatch.setattr(config, "OCR_FALLBACK_ENABLED", True)
    monkeypatch.setattr(config, "OCR_FALLBACK_BACKEND", "glm-ocr")
    monkeypatch.setattr(config, "OCR_FALLBACK_CONFIDENCE", 0.45)
    monkeypatch.setattr(config, "OCR_FALLBACK_MIN_CHARS", 3)

    primary_calls = []
    fallback_calls = []

    def fake_primary(paths):
        primary_calls.append([p.name for p in paths])
        return [
            ("", "", 0.0),
            ("good text", "good text", 0.91),
            ("xo", "xo", 0.91),
        ]

    def fake_fallback(paths):
        fallback_calls.append([p.name for p in paths])
        return [
            ("fixed one", "fixed one", 0.66),
            ("fixed three", "fixed three", 0.64),
        ]

    monkeypatch.setattr(ocr, "_ocr_easyocr", fake_primary)
    monkeypatch.setattr(ocr, "_ocr_glm", fake_fallback)

    paths = [Path("one.png"), Path("two.png"), Path("three.png")]
    out = ocr.ocr_frames(paths)

    assert primary_calls == [["one.png", "two.png", "three.png"]]
    assert fallback_calls == [["one.png", "three.png"]]
    assert out == [
        ("fixed one", "fixed one", 0.66),
        ("good text", "good text", 0.91),
        ("fixed three", "fixed three", 0.64),
    ]
