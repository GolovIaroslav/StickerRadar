"""Model artifact metadata and local-readiness checks.

This module is deliberately read-only. It never downloads, creates cache entries,
or initializes a model runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Artifact:
    key: str
    capability: str
    label: str
    source: str
    size: str
    license: str
    files: tuple[str, ...] = ()
    notes: str = ""
    revision: str | None = None


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(
        key="embedding:siglip2-base",
        capability="embedding",
        label="SigLIP2 Base 224",
        source="google/siglip2-base-patch16-224",
        size="~1.5 GB",
        license="Apache-2.0",
        notes="Multilingual image+text retrieval model.",
    ),
    Artifact(
        key="text-embed:qwen3-0.6b",
        capability="text-embedding",
        label="Qwen3-Embedding-0.6B GGUF Q8_0",
        source="Qwen/Qwen3-Embedding-0.6B-GGUF",
        size="~640 MB",
        license="Apache-2.0",
        files=("Qwen3-Embedding-0.6B-Q8_0.gguf",),
        notes="Dedicated multilingual OCR-text/query embeddings via llama.cpp.",
        revision="370f27d7550e0def9b39c1f16d3fbaa13aa67728",
    ),
    Artifact(
        key="ocr:rapidocr",
        capability="ocr",
        label="RapidOCR",
        source="rapidocr-onnxruntime package assets",
        size="~150–250 MB",
        license="see package/model licenses",
        notes="Light CPU OCR; optional dependency and model assets.",
    ),
    Artifact(
        key="ocr:easyocr",
        capability="ocr",
        label="EasyOCR ru,en",
        source="EasyOCR model assets",
        size="~300 MB",
        license="see EasyOCR/model licenses",
        notes="Better scene-text quality, especially on stylized Russian text.",
    ),
    Artifact(
        key="ocr:glm-ocr",
        capability="ocr",
        label="GLM-OCR Q8_0",
        source="ggml-org/GLM-OCR-GGUF:Q8_0",
        size="~1.43 GB including mmproj",
        license="see model card",
        notes="Heavy rescue/all-frames OCR; never a default.",
    ),
    Artifact(
        key="vlm:internvl2.5-1b-q8",
        capability="vlm",
        label="InternVL2.5-1B Q8_0 + mmproj",
        source="ggml-org/InternVL2_5-1B-GGUF",
        size="~1.30 GB combined (local files)",
        license="MIT (check bundled component licenses)",
        files=("InternVL2_5-1B-Q8_0.gguf", "mmproj-InternVL2_5-1B-f16.gguf"),
        notes="Experimental compact multilingual image understanding.",
    ),
    Artifact(
        key="vlm:qwen2-vl-2b-q4",
        capability="vlm",
        label="Qwen2-VL-2B Q4_K_M + mmproj",
        source="bartowski/Qwen2-VL-2B-Instruct-GGUF",
        size="~2.32 GB combined (Q4_K_M + f16 mmproj)",
        license="Apache-2.0 (verify quantization repo terms)",
        files=("Qwen2-VL-2B-Instruct-Q4_K_M.gguf", "mmproj-Qwen2-VL-2B-Instruct-f16.gguf"),
        notes="Potentially stronger reasoning; total is above 1 GB after mmproj.",
    ),
)


def get_artifact(key: str) -> Artifact | None:
    return next((item for item in ARTIFACTS if item.key == key), None)


def _model_cache_roots() -> list[Path]:
    roots = []
    explicit = os.environ.get("HF_HOME", "").strip()
    if explicit:
        roots.append(Path(explicit).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def _hf_cache_path(repo: str) -> Path | None:
    if "/" not in repo:
        return None
    owner, name = repo.split("/", 1)
    prefix = f"models--{owner}--{name}"
    for root in _model_cache_roots():
        candidate = root / prefix
        snapshots = sorted((p for p in candidate.glob("snapshots/*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshots:
            return snapshots[0]
    return None


def resolve_local_path(source: str, configured_path: str | Path | None = None) -> Path | None:
    """Resolve only an existing local path or an existing HF cache snapshot."""
    if configured_path:
        path = Path(configured_path).expanduser()
        if path.exists():
            return path.resolve()
    path = Path(source).expanduser()
    if path.exists():
        return path.resolve()
    return _hf_cache_path(source)


def artifact_ready(artifact: Artifact, configured_path: str | Path | None = None) -> bool:
    if artifact.capability in {"ocr"} and artifact.key in {"ocr:easyocr", "ocr:rapidocr"}:
        return False  # dependency-specific checks live in the backend, never download here
    path = resolve_local_path(artifact.source, configured_path)
    if path is None:
        return False
    if not artifact.files:
        return True
    files = list(path.rglob("*")) if path.is_dir() else [path]
    names = " ".join(item.name.lower() for item in files)
    return all(part.lower() in names for part in artifact.files if part)


def readiness_message(artifact: Artifact, destination: str | Path | None = None) -> str:
    target = str(Path(destination).expanduser()) if destination else "MODEL_ROOT"
    return (
        f"Artifact {artifact.label!r} is not installed locally.\n"
        f"  Size: {artifact.size}\n  Destination: {target}\n"
        f"  Source: {artifact.source}\n"
        f"Install explicitly with: python -m app model install --id {artifact.key} --path {target}"
    )
