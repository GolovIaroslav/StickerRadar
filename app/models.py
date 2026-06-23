"""
app/models.py — embedding model registry.

StickerRadar compares image vectors and text vectors in the SAME shared
embedding space, which requires a CLIP-style multimodal model.

Text-only models (Gemini Embedding, Qwen3-text, etc.) cannot embed sticker
images and are NOT compatible with this architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelEntry:
    key: str             # value for MODEL_NAME in .env
    text_model: str      # HuggingFace model id for text encoding
    image_model: str     # HuggingFace model id for image encoding
    unified: bool        # True = same model handles both (Jina-style)
    dim: int             # embedding dimension
    quality: str         # "fast" | "good" | "best"
    size_hint: str       # approximate download size
    multilingual: bool
    langs_note: str      # brief language coverage
    notes: str           # English install/usage instructions
    experimental: bool = False


REGISTRY: list[ModelEntry] = [
    ModelEntry(
        key="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        text_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        image_model="clip-ViT-B-32",
        unified=False,
        dim=512,
        quality="fast",
        size_hint="~380 MB",
        multilingual=True,
        langs_note="50+ languages (Russian, Chinese included)",
        notes=(
            "Default. Fast, lightweight. 2021 model, ViT-B/32 — adequate for quick tests.\n"
            "No extra install needed. Downloads automatically on first use."
        ),
    ),
    ModelEntry(
        key="jinaai/jina-clip-v2",
        text_model="jinaai/jina-clip-v2",
        image_model="jinaai/jina-clip-v2",
        unified=True,
        dim=1024,
        quality="best",
        size_hint="~3.5 GB",
        multilingual=True,
        langs_note="89 languages (strong Russian & Chinese)",
        notes=(
            "Recommended upgrade. 2024 model, significantly better search quality.\n"
            "Extra install:  uv add 'transformers>=4.36' einops\n"
            "Activate:       MODEL_NAME=jinaai/jina-clip-v2  in .env\n"
            "Re-embed:       python -m app sync --reindex"
        ),
    ),
    ModelEntry(
        key="openai/clip-vit-large-patch14",
        text_model="openai/clip-vit-large-patch14",
        image_model="openai/clip-vit-large-patch14",
        unified=True,
        dim=768,
        quality="good",
        size_hint="~900 MB",
        multilingual=False,
        langs_note="English only",
        notes=(
            "OpenAI ViT-L/14. Good quality but English-only text queries.\n"
            "No extra install needed. Downloads automatically on first use."
        ),
        experimental=True,
    ),
]


def get(key: str) -> ModelEntry | None:
    """Return registry entry for the given MODEL_NAME, or None."""
    for m in REGISTRY:
        if m.key == key:
            return m
    return None


def default() -> ModelEntry:
    return REGISTRY[0]
