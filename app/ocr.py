"""
app/ocr.py — OCR runner for preview frames.

Backends (set OCR_BACKEND in .env):
  easyocr   — EasyOCR (default). 80+ languages, scene text, pip install easyocr.
              Configure languages via OCR_LANGS (e.g. "ru,en,ko,ja,ch_sim").
              "ru,en" already covers Cyrillic + 50+ Latin-script languages.
  rapidocr  — RapidOCR (PaddleOCR models via ONNX, no GPU required, faster).
              pip install rapidocr-onnxruntime. Language via OCR_LANGS first code.

VLM-style models (Baidu Unlimited-OCR, GLM-OCR, Qwen-VL) are NOT used here:
they are designed for multi-page document understanding, not scene text on stickers.
For 1–5 words on a 512×512 image, EasyOCR/RapidOCR are faster and more accurate.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

# ── Normalize ────────────────────────────────────────────────────────────────

_YO = str.maketrans("ёЁ", "еЕ")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")
_REPEATS = re.compile(r"(.)\1{2,}")  # aaaa → aa


def normalize_text(text: str) -> str:
    """Lowercase, ё→е, strip punctuation, collapse 3+ repeated chars."""
    if not text:
        return ""
    text = text.translate(_YO)
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    text = _REPEATS.sub(r"\1\1", text)
    return _SPACES.sub(" ", text).strip()


# ── Backend detection ─────────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True if the configured OCR backend is installed."""
    from app import config
    backend = config.OCR_BACKEND.lower()
    if backend == "easyocr":
        try:
            import easyocr  # noqa: F401
            return True
        except ImportError:
            return False
    if backend == "rapidocr":
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: F401
            return True
        except ImportError:
            return False
    return False


# ── EasyOCR backend ───────────────────────────────────────────────────────────

_easyocr_reader = None
_easyocr_lock = threading.Lock()


def _get_easyocr_reader():
    global _easyocr_reader
    with _easyocr_lock:
        if _easyocr_reader is None:
            import easyocr
            from app import config
            langs = [l.strip() for l in config.OCR_LANGS.split(",") if l.strip()]
            gpu = config.OCR_USE_GPU
            print(
                f"Loading EasyOCR ({','.join(langs)}, gpu={gpu})"
                " — first run downloads models (~300 MB) …"
            )
            _easyocr_reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
            print("EasyOCR ready.")
    return _easyocr_reader


def _ocr_easyocr(paths: list[Path]) -> list[tuple[str, str, float]]:
    reader = _get_easyocr_reader()
    out: list[tuple[str, str, float]] = []
    for path in paths:
        try:
            detections = reader.readtext(str(path), detail=1, paragraph=False)
            texts, confs = [], []
            for _, text, conf in detections:
                if conf >= 0.3 and text.strip():
                    texts.append(text.strip())
                    confs.append(conf)
            if not texts:
                out.append(("", "", 0.0))
            else:
                raw = " ".join(texts)
                out.append((raw, normalize_text(raw), sum(confs) / len(confs)))
        except Exception:
            out.append(("", "", 0.0))
    return out


# ── RapidOCR backend ──────────────────────────────────────────────────────────

_rapidocr_engine = None
_rapidocr_lock = threading.Lock()


def _get_rapidocr_engine():
    global _rapidocr_engine
    with _rapidocr_lock:
        if _rapidocr_engine is None:
            from rapidocr_onnxruntime import RapidOCR
            print("Loading RapidOCR (ONNX, cpu) …")
            _rapidocr_engine = RapidOCR()
            print("RapidOCR ready.")
    return _rapidocr_engine


def _ocr_rapidocr(paths: list[Path]) -> list[tuple[str, str, float]]:
    engine = _get_rapidocr_engine()
    out: list[tuple[str, str, float]] = []
    for path in paths:
        try:
            result, _ = engine(str(path))
            if not result:
                out.append(("", "", 0.0))
                continue
            texts, confs = [], []
            for item in result:
                # result rows: [bbox, text, confidence]
                text = item[1] if len(item) > 1 else ""
                conf = float(item[2]) if len(item) > 2 else 0.5
                if conf >= 0.3 and text.strip():
                    texts.append(text.strip())
                    confs.append(conf)
            if not texts:
                out.append(("", "", 0.0))
            else:
                raw = " ".join(texts)
                out.append((raw, normalize_text(raw), sum(confs) / len(confs)))
        except Exception:
            out.append(("", "", 0.0))
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def ocr_frames(paths: list[Path]) -> list[tuple[str, str, float]]:
    """
    Run OCR on image paths using the configured backend.
    Returns (raw_text, norm_text, confidence) per path.
    Returns ("", "", 0.0) when no text is detected or on error.
    """
    from app import config
    backend = config.OCR_BACKEND.lower()
    if backend == "rapidocr":
        return _ocr_rapidocr(paths)
    return _ocr_easyocr(paths)
