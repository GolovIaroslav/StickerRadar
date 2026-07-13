"""PP-OCRv5 shadow OCR orchestration.

This module deliberately stores all output in ``ocr_shadow``. It does not
write ``media_items.ocr_text`` or any canonical OCR provenance fields.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app import config, db
from app.errors import ModelNotInstalled
from app.ocr import normalize_text

_BACKEND = "ppocrv5-eslav"
_CORPUS_PATH = config.PROJECT_ROOT / "research" / "diagnostics" / "text_sticker_corpus_v1.json"
_WORKER_PATH = config.PROJECT_ROOT / "scripts" / "ppocr_worker.py"
_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".mp4", ".png", ".webm", ".webp"}


@dataclass(frozen=True)
class ShadowOCRResult:
    path: Path
    text: str
    boxes: list[object]
    scores: list[float]
    error: str | None = None


def _worker_python() -> str:
    raw = config.OCR_SHADOW_PYTHON.strip()
    if not raw:
        raise ModelNotInstalled(
            "OCR_SHADOW_PYTHON is not set. Create a Python <=3.12 PaddleOCR venv and set it."
        )
    path = Path(raw).expanduser()
    # Keep the venv executable path itself: resolving it follows the usual
    # ``venv/bin/python -> base-python`` symlink and loses the venv prefix.
    resolved = str(path.absolute()) if path.is_file() else shutil.which(raw)
    if resolved:
        return resolved
    raise ModelNotInstalled(f"OCR_SHADOW_PYTHON does not exist: {path}")


def _model_dir(value: str, key: str) -> Path:
    path = Path(value).expanduser()
    if not value or not path.is_dir():
        raise ModelNotInstalled(
            f"{key} is not an installed local directory. "
            "Install explicitly with: python -m app model install --id ocr:ppocrv5-eslav"
        )
    return path.resolve()


def _validate_worker_runtime(worker_python: str) -> None:
    """Require the configured interpreter to be a separate Python <=3.12 venv."""
    probe = (
        "import json, sys; "
        "print(json.dumps({'version': list(sys.version_info[:2]), "
        "'prefix': sys.prefix, 'base_prefix': sys.base_prefix}))"
    )
    try:
        proc = subprocess.run(
            [worker_python, "-c", probe], capture_output=True, text=True, timeout=10, check=False
        )
        runtime = json.loads(proc.stdout) if proc.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise ModelNotInstalled(f"Could not inspect OCR_SHADOW_PYTHON: {exc}") from exc
    if not isinstance(runtime, dict):
        raise ModelNotInstalled("Could not inspect OCR_SHADOW_PYTHON runtime.")
    version = runtime.get("version", [])
    if not isinstance(version, list) or len(version) != 2 or tuple(version) > (3, 12):
        raise ModelNotInstalled("OCR_SHADOW_PYTHON must use Python 3.12 or lower for PaddleOCR.")
    if runtime.get("prefix") == runtime.get("base_prefix"):
        raise ModelNotInstalled("OCR_SHADOW_PYTHON must be inside a separate virtual environment.")
    if Path(str(runtime.get("prefix", ""))).absolute() == Path(sys.prefix).absolute():
        raise ModelNotInstalled(
            "OCR_SHADOW_PYTHON must point to a separate PaddleOCR virtual environment."
        )


def run_ppocr_worker(paths: list[Path]) -> list[ShadowOCRResult]:
    """Run the separate PaddleOCR worker and decode its JSONL protocol."""
    if not paths:
        return []
    if not config.OCR_SHADOW_ENABLED:
        raise ModelNotInstalled("Shadow OCR is disabled (set OCR_SHADOW_ENABLED=true).")

    worker_python = _worker_python()
    _validate_worker_runtime(worker_python)
    det_dir = _model_dir(config.OCR_SHADOW_DET_DIR, "OCR_SHADOW_DET_DIR")
    rec_dir = _model_dir(config.OCR_SHADOW_REC_DIR, "OCR_SHADOW_REC_DIR")
    command = [
        worker_python,
        str(_WORKER_PATH),
        "--det-dir", str(det_dir),
        "--rec-dir", str(rec_dir),
        *(str(path.resolve()) for path in paths),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    except OSError as exc:
        raise ModelNotInstalled(f"Could not start OCR_SHADOW_PYTHON: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("PP-OCR shadow worker timed out after 600 seconds") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["unknown worker failure"]
        raise RuntimeError(f"PP-OCR shadow worker failed: {detail[0][:300]}")

    expected = {path.resolve() for path in paths}
    results: dict[Path, ShadowOCRResult] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            path = Path(item["path"]).resolve()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("PP-OCR shadow worker emitted invalid JSONL") from exc
        if path not in expected:
            raise RuntimeError(f"PP-OCR shadow worker returned an unexpected path: {path}")
        results[path] = ShadowOCRResult(
            path=path,
            text=str(item.get("text", "")),
            boxes=list(item.get("boxes") or []),
            scores=[float(score) for score in (item.get("rec_scores") or [])],
            error=str(item["error"]) if item.get("error") else None,
        )

    return [
        results.get(path.resolve(), ShadowOCRResult(path.resolve(), "", [], [], "worker returned no result"))
        for path in paths
    ]


def _load_corpus() -> list[dict[str, object]]:
    return json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))


def _levenshtein(left: str, right: str) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(ground_truth: str, hypothesis: str) -> float:
    """CER using a tiny local Levenshtein implementation and OCR normalization."""
    ground_truth = normalize_text(ground_truth)
    hypothesis = normalize_text(hypothesis)
    if not ground_truth:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(ground_truth, hypothesis) / len(ground_truth)


def _image_path(row) -> Path | None:
    preview = Path(row["preview_path"]) if row["preview_path"] else None
    if preview and preview.is_file():
        return preview.resolve()
    local = Path(row["local_path"]) if row["local_path"] else None
    if local and local.suffix.lower() in _IMAGE_SUFFIXES and local.is_file():
        return local.resolve()
    return None


def _looks_plausible_text(text: str) -> bool:
    """Conservative reporting heuristic; a human still decides promotion."""
    normalized = normalize_text(text)
    characters = normalized.replace(" ", "")
    return len(characters) >= 3 and any(char.isalnum() for char in characters)


def run_shadow_ocr(*, limit: int = 40, only_empty: bool = False, seed: int = 42) -> dict[str, int | float]:
    """Run PP-OCRv5 as a measurement-only batch and print its comparison report."""
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    config.ensure_dirs()
    db.get_conn()

    corpus = _load_corpus()
    corpus_by_id = {int(case["id"]): case for case in corpus}
    corpus_ids = [] if only_empty else [int(case["id"]) for case in corpus][:limit]
    remaining = max(0, limit - len(corpus_ids))
    empty_pool = [
        media_id
        for media_id in db.list_empty_ocr_media_ids()
        if only_empty or media_id not in corpus_by_id
    ]
    sample = random.Random(seed).sample(empty_pool, min(remaining, len(empty_pool)))
    selected_ids = corpus_ids + sample

    rows = {int(row["id"]): row for row in db.list_media_for_ocr_shadow(selected_ids)}
    path_to_media: dict[Path, int] = {}
    skipped = 0
    for media_id in selected_ids:
        row = rows.get(media_id)
        image = _image_path(row) if row is not None else None
        if image is None:
            skipped += 1
            print(f"  Skip {media_id}: no local image or preview frame")
            continue
        path_to_media[image] = media_id

    results = run_ppocr_worker(list(path_to_media)) if path_to_media else []
    by_media: dict[int, ShadowOCRResult] = {}
    for result in results:
        media_id = path_to_media[result.path]
        by_media[media_id] = result
        db.upsert_ocr_shadow(
            media_id=media_id,
            backend=_BACKEND,
            text=result.text,
            boxes_json=json.dumps(result.boxes, ensure_ascii=False),
            scores_json=json.dumps(result.scores, ensure_ascii=False),
        )

    empty_selected = [
        media_id for media_id, row in rows.items()
        if media_id in by_media and not (row["ocr_text"] or "").strip()
    ]
    recovered = sum(bool((by_media[media_id].text or "").strip()) for media_id in empty_selected)
    plausible = sum(_looks_plausible_text(by_media[media_id].text) for media_id in empty_selected)
    denominator = len(empty_selected) or 1
    print(
        f"PP-OCRv5 shadow: processed={len(by_media)}, skipped={skipped}; "
        f"previously-empty with text={recovered}/{len(empty_selected)} "
        f"({recovered / denominator:.0%}); plausibility candidates={plausible}/{len(empty_selected)}"
    )
    if empty_selected:
        print("Previously-empty outputs (review plausibility before any promotion):")
        for media_id in empty_selected:
            result = by_media[media_id]
            label = "candidate" if _looks_plausible_text(result.text) else "review"
            print(f"  #{media_id} [{label}]: {result.text}" + (f"  [error: {result.error}]" if result.error else ""))

    corpus_reported = 0
    current_cers: list[float] = []
    shadow_cers: list[float] = []
    for media_id in corpus_ids:
        row = rows.get(media_id)
        case = corpus_by_id[media_id]
        result = by_media.get(media_id)
        if row is None or result is None:
            continue
        ground_truth = str(case["visual_text"])
        current_text = str(row["ocr_text"] or "")
        shadow_text = result.text
        current_cer = character_error_rate(ground_truth, current_text)
        shadow_cer = character_error_rate(ground_truth, shadow_text)
        current_cers.append(current_cer)
        shadow_cers.append(shadow_cer)
        print(f"\n#{media_id} CER current={current_cer:.3f} shadow={shadow_cer:.3f}")
        print(f"  ground truth: {ground_truth}")
        print(f"  current OCR : {current_text}")
        print(f"  shadow OCR  : {shadow_text}" + (f"  [error: {result.error}]" if result.error else ""))
        corpus_reported += 1
    mean_current_cer = sum(current_cers) / len(current_cers) if current_cers else 0.0
    mean_shadow_cer = sum(shadow_cers) / len(shadow_cers) if shadow_cers else 0.0
    if corpus_reported:
        print(
            f"\nCorpus mean CER: current={mean_current_cer:.3f} shadow={mean_shadow_cer:.3f} "
            f"(cases={corpus_reported})"
        )

    return {
        "processed": len(by_media),
        "skipped": skipped,
        "previously_empty": len(empty_selected),
        "recovered": recovered,
        "plausibility_candidates": plausible,
        "corpus_reported": corpus_reported,
        "mean_current_cer": mean_current_cer,
        "mean_shadow_cer": mean_shadow_cer,
    }
