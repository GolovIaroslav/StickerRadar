"""Run local PP-OCRv5 models in the separate PaddleOCR environment.

The parent process consumes JSON Lines from stdout. Keep all PaddleOCR chatter
on stderr so it cannot corrupt that protocol.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _plain(value: Any) -> Any:
    """Convert NumPy/Paddle values returned by PaddleOCR into JSON values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _page_data(page: Any) -> dict[str, Any]:
    data = page if isinstance(page, dict) else getattr(page, "json", {})
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        return {}
    result = data.get("res", data)
    return result if isinstance(result, dict) else {}


def _prepare_input(path: Path) -> tuple[Path, Path | None]:
    suffix = path.suffix.lower()
    if suffix not in {".gif", ".webp", ".mp4", ".webm"}:
        return path, None

    with tempfile.NamedTemporaryFile(prefix="stickerradar-ppocr-", suffix=".png", delete=False) as tmp:
        converted = Path(tmp.name)
    if suffix in {".gif", ".webp"}:
        from PIL import Image

        Image.open(path).convert("RGB").save(converted)
        return converted, converted

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        converted.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg is required to prepare video media for PP-OCR")
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i", str(path),
            "-frames:v", "1",
            "-vf", "scale=512:512:force_original_aspect_ratio=decrease",
            str(converted),
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0 or not converted.exists() or converted.stat().st_size == 0:
        converted.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not extract a frame from {path.name}")
    return converted, converted


def _result_for_path(ocr: Any, path: Path) -> dict[str, Any]:
    prepared, cleanup = _prepare_input(path)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            prediction = ocr.predict(str(prepared))

        texts: list[str] = []
        boxes: list[Any] = []
        scores: list[float] = []
        for page in prediction:
            data = _page_data(page)
            texts.extend(str(text) for text in data.get("rec_texts", []) if str(text).strip())
            boxes.extend(_plain(box) for box in data.get("dt_polys", []))
            for score in data.get("rec_scores", []):
                try:
                    scores.append(float(score))
                except (TypeError, ValueError):
                    continue
        return {
            "path": str(path),
            "text": " ".join(texts),
            "boxes": boxes,
            "rec_scores": scores,
        }
    finally:
        if cleanup is not None:
            cleanup.unlink(missing_ok=True)


def main() -> int:
    if sys.version_info[:2] > (3, 12):
        raise SystemExit("PP-OCR worker requires Python 3.12 or lower")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("PP-OCR worker must run inside a separate virtual environment")

    parser = argparse.ArgumentParser(description="Local PP-OCRv5 JSONL worker")
    parser.add_argument("--det-dir", required=True)
    parser.add_argument("--rec-dir", required=True)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()

    det_dir = Path(args.det_dir)
    rec_dir = Path(args.rec_dir)
    if not det_dir.is_dir() or not rec_dir.is_dir():
        raise SystemExit("PP-OCR model directories must exist and be passed explicitly")

    # PaddleOCR's model-dir arguments force use of these local folders; this
    # worker has no downloader and does not accept remote model identifiers.
    with contextlib.redirect_stdout(sys.stderr):
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_detection_model_dir=str(det_dir),
            text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
            text_recognition_model_dir=str(rec_dir),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )

    for path in args.images:
        try:
            print(json.dumps(_result_for_path(ocr, path), ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"path": str(path), "error": str(exc)[:300]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
