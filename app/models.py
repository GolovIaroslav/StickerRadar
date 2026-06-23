"""
app/models.py — embedding model registry.

StickerRadar compares image vectors and text vectors in the SAME shared
embedding space, which requires a CLIP-style multimodal model.

Text-only models (Gemini Embedding, Qwen3 text embeddings, etc.) cannot embed
sticker images and are NOT compatible with this architecture.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelEntry:
    key: str             # value for MODEL_NAME in .env
    loader: str          # "hf" | "st" | "open_clip"
    dim: int             # embedding dimension
    size_mb: int         # approximate download size in MB
    quality: str         # "fast" | "good" | "best"
    multilingual: bool
    langs_note: str      # brief language coverage
    notes: str           # English install/usage instructions
    text_padding: str = "max_length"   # used by the "hf" backend (siglip needs max_length)
    trust_remote_code: bool = False
    experimental: bool = False


def fmt_size(mb: int) -> str:
    if mb >= 1024:
        return f"~{mb / 1024:.1f} GB"
    return f"~{mb} MB"


REGISTRY: list[ModelEntry] = [
    ModelEntry(
        key="google/siglip2-base-patch16-224",
        loader="hf",
        dim=768,
        size_mb=375,
        quality="good",
        multilingual=True,
        langs_note="multilingual (Russian, Chinese, 100+)",
        text_padding="max_length",
        notes=(
            "Default. Google SigLIP2 (2025), strong multilingual quality at a small size.\n"
            "No extra install needed (transformers). Downloads automatically on first use."
        ),
    ),
    ModelEntry(
        key="jinaai/jina-clip-v2",
        loader="st",
        dim=1024,
        size_mb=3500,
        quality="best",
        multilingual=True,
        langs_note="89 languages (strong Russian & Chinese)",
        trust_remote_code=True,
        notes=(
            "Highest verified quality among the lighter options. Jina CLIP v2 (2024).\n"
            "Extra install:  uv add einops\n"
            "Activate:       MODEL_NAME=jinaai/jina-clip-v2  in .env\n"
            "Re-embed:       python -m app sync --reindex"
        ),
    ),
    ModelEntry(
        key="google/siglip2-large-patch16-256",
        loader="hf",
        dim=1024,
        size_mb=1500,
        quality="best",
        multilingual=True,
        langs_note="multilingual (Russian, Chinese, 100+)",
        text_padding="max_length",
        notes=(
            "Larger SigLIP2 (2025). Better quality than base, still multilingual.\n"
            "No extra install needed (transformers)."
        ),
    ),
    ModelEntry(
        key="openai/clip-vit-large-patch14",
        loader="hf",
        dim=768,
        size_mb=1700,
        quality="good",
        multilingual=False,
        langs_note="English only",
        text_padding="max_length",
        notes=(
            "OpenAI CLIP ViT-L/14. Good quality but English-only text queries.\n"
            "No extra install needed (transformers)."
        ),
    ),
    ModelEntry(
        key="apple/MobileCLIP2-S2",
        loader="open_clip",
        dim=512,
        size_mb=200,
        quality="fast",
        multilingual=False,
        langs_note="English-leaning",
        experimental=True,
        notes=(
            "Apple MobileCLIP2 (2025). Tiny and very fast, English-leaning.\n"
            "Extra install:  uv add open_clip_torch\n"
            "Experimental loader — verify it works on your machine."
        ),
    ),
    ModelEntry(
        key="jinaai/jina-embeddings-v4",
        loader="st",
        dim=2048,
        size_mb=7500,
        quality="best",
        multilingual=True,
        langs_note="multilingual",
        trust_remote_code=True,
        experimental=True,
        notes=(
            "Jina Embeddings v4 (2025), multimodal. Very large (~7.5 GB), slow on CPU.\n"
            "Extra install:  uv add peft einops\n"
            "Experimental — heavy; use only with a GPU."
        ),
    ),
    ModelEntry(
        key="Qwen/Qwen3-VL-Embedding-2B",
        loader="st",
        dim=2048,
        size_mb=4500,
        quality="best",
        multilingual=True,
        langs_note="multilingual",
        trust_remote_code=True,
        experimental=True,
        notes=(
            "Qwen3-VL embedding (2B). Large (~4.5 GB) and may require custom code.\n"
            "Extra install:  uv add transformers accelerate\n"
            "Experimental — NOT verified in this build; may need a custom loader."
        ),
    ),
]


# Models people often ask about that DO NOT work here (text-only / cloud API).
INCOMPATIBLE: list[tuple[str, str]] = [
    ("Gemini Embedding 2 (gemini-embedding-*)",
     "Text-only AND cloud API — cannot encode sticker images, and breaks the local/private design."),
    ("Qwen3-Embedding (text)",
     "Text-only — no image tower. Use the multimodal Qwen3-VL variant instead."),
    ("OpenAI text-embedding-3 / BGE / E5",
     "Text-only — cannot encode images into the shared space."),
]


def get(key: str) -> ModelEntry | None:
    for m in REGISTRY:
        if m.key == key:
            return m
    return None


def default() -> ModelEntry:
    return REGISTRY[0]
