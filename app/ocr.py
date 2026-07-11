"""
app/ocr.py — OCR runner for preview frames.

Backends (set OCR_BACKEND in .env):
  easyocr   — EasyOCR. 80+ languages, scene text, pip install easyocr.
              Configure languages via OCR_LANGS (e.g. "ru,en,ko,ja,ch_sim").
              "ru,en" already covers Cyrillic + 50+ Latin-script languages.
  rapidocr  — RapidOCR (PaddleOCR models via ONNX, no GPU required, faster).
              pip install rapidocr-onnxruntime.
  glm-ocr   — Experimental GLM-OCR via llama.cpp (`llama-cli`) using a GGUF repo.
              Configure OCR_LLM_REPO (default: ggml-org/GLM-OCR-GGUF:Q8_0).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

# ── Normalize ────────────────────────────────────────────────────────────────

_YO = str.maketrans("ёЁ", "еЕ")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")
_REPEATS = re.compile(r"(.)\1{2,}")  # aaaa → aa
OCR_PIPELINE_VERSION = "frame-ocr-v2"


@dataclass(frozen=True)
class OCRFrameResult:
    """OCR payload together with the backend that produced the final text."""

    raw_text: str
    norm_text: str
    confidence: float
    backend: str


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


def backend_install_hint() -> str:
    from app import config

    backend = config.OCR_BACKEND.lower()
    if backend == "rapidocr":
        return "uv add rapidocr-onnxruntime"
    if backend == "glm-ocr":
        return "Install llama.cpp and ensure `llama-cli` is in PATH"
    return "uv add easyocr"


def is_backend_available(backend: str) -> bool:
    backend = backend.lower()
    if backend in {"", "off", "none"}:
        return False
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
    if backend == "glm-ocr":
        return shutil.which("llama-cli") is not None
    return False


def effective_backend_name() -> str | None:
    from app import config

    primary = config.OCR_BACKEND.lower()
    if is_backend_available(primary):
        return primary
    if config.OCR_FALLBACK_ENABLED:
        fallback = config.OCR_FALLBACK_BACKEND.lower()
        if fallback != primary and is_backend_available(fallback):
            return fallback
    return None


def is_available() -> bool:
    """Return True if any effective OCR backend is installed."""
    return effective_backend_name() is not None


def current_provenance(backend: str | None = None) -> tuple[str, str | None, str]:
    from app import config

    chosen = (backend or effective_backend_name() or config.OCR_BACKEND).lower()
    model_ref = config.OCR_LLM_REPO if chosen == "glm-ocr" else None
    return chosen, model_ref, OCR_PIPELINE_VERSION


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


# ── GLM-OCR via llama.cpp ─────────────────────────────────────────────────────


def _clean_llama_output(text: str, prompt: str) -> str:
    cleaned = text.replace("\r", "")
    if "Loaded media from" in cleaned:
        cleaned = cleaned.split("Loaded media from", 1)[1]
    if f"> {prompt}" in cleaned:
        cleaned = cleaned.split(f"> {prompt}", 1)[1]
    if "[ Prompt:" in cleaned:
        cleaned = cleaned.split("[ Prompt:", 1)[0]
    cleaned = cleaned.strip()
    prefixes = [prompt, "assistant", "Assistant", "OCR result:"]
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    cleaned = cleaned.replace("<|assistant|>", "").replace("<|user|>", "")
    return cleaned.strip()


def _prepare_glm_image(path: Path) -> tuple[Path, Path | None]:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        return path, None

    from PIL import Image

    tmp = tempfile.NamedTemporaryFile(prefix="stickerradar-glm-", suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    Image.open(path).convert("RGB").save(tmp_path)
    return tmp_path, tmp_path


def _ocr_glm(paths: list[Path]) -> list[tuple[str, str, float]]:
    from app import config

    prompt = "OCR"
    out: list[tuple[str, str, float]] = []
    repo = config.OCR_LLM_REPO
    from app.model_artifacts import resolve_local_path
    model_path = config.OCR_MODEL_PATH or str(resolve_local_path(repo) or "")
    if not model_path:
        return [("", "", 0.0) for _ in paths]
    for path in paths:
        cleanup_path: Path | None = None
        try:
            prepared_path, cleanup_path = _prepare_glm_image(path)
            proc = subprocess.run(
                [
                    "llama-cli",
                    "--model",
                    model_path,
                    "--image",
                    str(prepared_path),
                    "--prompt",
                    prompt,
                    "--single-turn",
                    "--simple-io",
                    "--log-disable",
                    "--no-display-prompt",
                    "--verbosity",
                    "0",
                    "--temp",
                    "0",
                    "--top-k",
                    "1",
                    "--ctx-size",
                    "4096",
                    "--n-predict",
                    "128",
                ],
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            text = _clean_llama_output(proc.stdout or "", prompt)
            if proc.returncode != 0 or not text:
                out.append(("", "", 0.0))
                continue
            out.append((text, normalize_text(text), 0.65))
        except Exception:
            out.append(("", "", 0.0))
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)
    return out


def should_use_llm_fallback(
    raw_text: str,
    norm_text: str,
    confidence: float,
    *,
    min_confidence: float,
    min_chars: int,
) -> bool:
    if not raw_text.strip() or not norm_text.strip():
        return True
    if len(norm_text.strip()) < min_chars:
        return True
    return confidence < min_confidence


def _select_glm_prefilter_backend() -> str | None:
    from app import config

    preferred = (config.OCR_GLM_PREFILTER_BACKEND or "auto").strip().lower()
    candidates = [preferred] if preferred not in {"", "auto"} else ["rapidocr", "easyocr"]
    for candidate in candidates:
        if candidate == "glm-ocr":
            continue
        if is_backend_available(candidate):
            return candidate
    return None


def _ocr_glm_with_prefilter_results(
    paths: list[Path],
) -> tuple[list[tuple[str, str, float]], list[str]]:
    """Run a fast detector first and preserve the final source of every frame."""
    from app import config

    detector_backend = _select_glm_prefilter_backend()
    if detector_backend is None:
        results = _ocr_glm(paths)
        return results, ["glm-ocr"] * len(results)

    detector_results = _ocr_backend(detector_backend, paths)
    sources = [detector_backend] * len(detector_results)
    retry_indexes = [
        idx
        for idx, (raw, norm, conf) in enumerate(detector_results)
        if raw.strip()
        and norm.strip()
        and should_use_llm_fallback(
            raw,
            norm,
            conf,
            min_confidence=config.OCR_FALLBACK_CONFIDENCE,
            min_chars=config.OCR_FALLBACK_MIN_CHARS,
        )
    ]
    if not retry_indexes:
        return detector_results, sources

    glm_results = _ocr_glm([paths[idx] for idx in retry_indexes])
    merged = list(detector_results)
    for idx, replacement in zip(retry_indexes, glm_results):
        raw, norm, _ = replacement
        if raw.strip() and norm.strip():
            merged[idx] = replacement
            sources[idx] = "glm-ocr"
    return merged, sources


def _ocr_glm_with_prefilter(paths: list[Path]) -> list[tuple[str, str, float]]:
    return _ocr_glm_with_prefilter_results(paths)[0]


def _ocr_backend(backend: str, paths: list[Path]) -> list[tuple[str, str, float]]:
    backend = backend.lower()
    if backend == "rapidocr":
        return _ocr_rapidocr(paths)
    if backend == "glm-ocr":
        return _ocr_glm_with_prefilter(paths)
    return _ocr_easyocr(paths)


def _apply_optional_fallback(
    primary_backend: str,
    paths: list[Path],
    primary_results: list[tuple[str, str, float]],
) -> list[tuple[str, str, float]]:
    from app import config

    if not config.OCR_FALLBACK_ENABLED:
        return primary_results
    fallback_backend = config.OCR_FALLBACK_BACKEND.lower()
    if fallback_backend == primary_backend.lower():
        return primary_results
    if not is_backend_available(fallback_backend):
        return primary_results

    retry_indexes = [
        idx
        for idx, (raw, norm, conf) in enumerate(primary_results)
        if should_use_llm_fallback(
            raw,
            norm,
            conf,
            min_confidence=config.OCR_FALLBACK_CONFIDENCE,
            min_chars=config.OCR_FALLBACK_MIN_CHARS,
        )
    ]
    if not retry_indexes:
        return primary_results

    fallback_paths = [paths[idx] for idx in retry_indexes]
    fallback_results = (
        _ocr_glm(fallback_paths)
        if fallback_backend == "glm-ocr"
        else _ocr_backend(fallback_backend, fallback_paths)
    )
    merged = list(primary_results)
    for idx, replacement in zip(retry_indexes, fallback_results):
        raw, norm, _ = replacement
        if raw.strip() and norm.strip():
            merged[idx] = replacement
    return merged


# ── Public API ────────────────────────────────────────────────────────────────


def ocr_frames(paths: list[Path]) -> list[tuple[str, str, float]]:
    """
    Run OCR on image paths using the configured backend.
    Returns (raw_text, norm_text, confidence) per path.
    Returns ("", "", 0.0) when no text is detected or on error.
    """
    return [
        (result.raw_text, result.norm_text, result.confidence)
        for result in ocr_frames_with_provenance(paths)
    ]


def ocr_frames_with_provenance(paths: list[Path]) -> list[OCRFrameResult]:
    """Run OCR and report the backend that produced every final frame result."""
    from app import config

    backend = effective_backend_name() or config.OCR_BACKEND.lower()
    if backend == "glm-ocr":
        results, sources = _ocr_glm_with_prefilter_results(paths)
    else:
        results = _ocr_backend(backend, paths)
        sources = [backend] * len(results)

    if (
        config.OCR_FALLBACK_ENABLED
        and (fallback_backend := config.OCR_FALLBACK_BACKEND.lower()) != backend
        and is_backend_available(fallback_backend)
    ):
        retry_indexes = [
            idx
            for idx, (raw, norm, confidence) in enumerate(results)
            if should_use_llm_fallback(
                raw,
                norm,
                confidence,
                min_confidence=config.OCR_FALLBACK_CONFIDENCE,
                min_chars=config.OCR_FALLBACK_MIN_CHARS,
            )
        ]
        fallback_paths = [paths[idx] for idx in retry_indexes]
        fallback_results = (
            _ocr_glm(fallback_paths)
            if fallback_backend == "glm-ocr"
            else _ocr_backend(fallback_backend, fallback_paths)
        )
        for idx, replacement in zip(retry_indexes, fallback_results):
            raw, norm, _ = replacement
            if raw.strip() and norm.strip():
                results[idx] = replacement
                sources[idx] = fallback_backend

    return [
        OCRFrameResult(raw, norm, confidence, source)
        for (raw, norm, confidence), source in zip(results, sources, strict=True)
    ]
