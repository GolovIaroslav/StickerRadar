"""
app/models.py — embedding model registry.

StickerRadar compares image vectors and text vectors in the SAME shared
embedding space, which requires a CLIP-style multimodal model.

Text-only models (Gemini Embedding, Qwen3 text embeddings, etc.) cannot embed
sticker images and are NOT compatible with this architecture.

Sizes/params/languages below reflect each model's public model card. Where a
language list is not officially published, we say so — test your languages
locally rather than trusting a number.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelEntry:
    key: str             # value for MODEL_NAME in .env
    loader: str          # "hf" | "st" | "open_clip"
    params: str          # parameter count, e.g. "2B"
    size: str            # approximate download size, e.g. "~4-5 GB"
    license: str         # "Apache-2.0" | "MIT" | "CC-BY-NC (non-commercial)" ...
    quality: str         # "fast" | "good" | "better" | "best"
    langs: str           # honest language note
    notes: str           # English install/usage instructions
    text_padding: str = "max_length"   # used by the "hf" backend (siglip needs max_length)
    trust_remote_code: bool = False
    verified: bool = False     # True = load tested in this project
    experimental: bool = False


REGISTRY: list[ModelEntry] = [
    ModelEntry(
        key="Qwen/Qwen3-VL-Embedding-2B",
        loader="st",
        params="2B",
        size="~4.26 GB",
        license="Apache-2.0",
        quality="best",
        langs="30+ languages; multimodal (text/image/video)",
        trust_remote_code=True,
        verified=True,
        notes=(
            "Default. Strongest open-license (Apache-2.0) option. Multimodal, 30+ languages.\n"
            "Loads via sentence-transformers; needs torchvision (already a dependency).\n"
            "Heavy: ~2B params. Does NOT fit small GPUs (<8 GB) — runs on CPU automatically.\n"
            "On a small/busy GPU set  DEVICE=cpu  in .env (slower but works; uses RAM)."
        ),
    ),
    ModelEntry(
        key="google/siglip2-base-patch16-224",
        loader="hf",
        params="0.4B",
        size="~1.5 GB",
        license="Apache-2.0",
        quality="good",
        langs="multilingual; exact language list not published — test RU/ZH locally",
        verified=True,
        notes=(
            "Lightweight, VERIFIED fallback (load-tested). Google SigLIP2 (2025), Apache-2.0.\n"
            "No extra install needed (transformers). Best choice if the default fails."
        ),
    ),
    ModelEntry(
        key="google/siglip2-large-patch16-256",
        loader="hf",
        params="0.9B",
        size="~3.53 GB",
        license="Apache-2.0",
        quality="better",
        langs="multilingual; exact language list not published — test RU/ZH locally",
        notes=(
            "Larger SigLIP2 (2025), Apache-2.0. Same loader as the verified base model.\n"
            "No extra install needed (transformers)."
        ),
    ),
    ModelEntry(
        key="jinaai/jina-clip-v2",
        loader="st",
        params="0.9B",
        size="~1.73 GB",
        license="CC-BY-NC-4.0",
        quality="best",
        langs="89 languages; multimodal (image+text)",
        trust_remote_code=True,
        notes=(
            "Proven quality option (2024). NON-COMMERCIAL license.\n"
            "Needs:  uv add einops   (optional on CUDA: xFormers / FlashAttention)\n"
            "Loads via sentence-transformers with trust_remote_code."
        ),
    ),
    ModelEntry(
        key="jinaai/jina-embeddings-v5-omni-nano-retrieval",
        loader="st",
        params="~0.95B",
        size="~1.9 GB",
        license="CC-BY-NC-4.0",
        quality="best",
        langs="multimodal (text/image/video/audio); language count unconfirmed",
        trust_remote_code=True,
        experimental=True,
        notes=(
            "Newest (2026), multimodal, light for its quality. NON-COMMERCIAL license.\n"
            "Experimental — verify availability on Hugging Face and that it loads.\n"
            "Needs:  uv add einops (and possibly more — see the model card)."
        ),
    ),
    ModelEntry(
        key="jinaai/jina-embeddings-v5-omni-small-retrieval",
        loader="st",
        params="~1.56B",
        size="~3.12 GB",
        license="CC-BY-NC-4.0",
        quality="best",
        langs="multimodal (text/image/video/audio/PDF); language count unconfirmed",
        trust_remote_code=True,
        experimental=True,
        notes=(
            "Newest (2026), heavier than nano, higher quality. NON-COMMERCIAL license.\n"
            "Experimental — verify availability and loading. Needs: uv add einops."
        ),
    ),
    ModelEntry(
        key="openai/clip-vit-large-patch14",
        loader="hf",
        params="0.4B",
        size="~1.71 GB",
        license="MIT",
        quality="good",
        langs="English only",
        notes=(
            "OpenAI CLIP ViT-L/14. Good quality but English-only text queries.\n"
            "No extra install needed (transformers)."
        ),
    ),
    ModelEntry(
        key="apple/MobileCLIP2-S2",
        loader="open_clip",
        params="~99M",
        size="~398 MB",
        license="Apple AMLR (research)",
        quality="fast",
        langs="small; English-leaning not formally confirmed",
        experimental=True,
        notes=(
            "Apple MobileCLIP2 (2025). Tiny and very fast, English-leaning.\n"
            "Needs:  uv add open_clip_torch\n"
            "Experimental loader — verify it works on your machine."
        ),
    ),
    ModelEntry(
        key="jinaai/jina-embeddings-v4",
        loader="st",
        params="4B",
        size="~7.89 GB",
        license="Qwen Research License",
        quality="best",
        langs="30+ languages; multimodal",
        trust_remote_code=True,
        experimental=True,
        notes=(
            "LEGACY (superseded by v5-omni). Very large (~7.89 GB), GPU recommended.\n"
            "Needs:  uv add peft einops"
        ),
    ),
]


# Models people often ask about that DO NOT work here (text-only / cloud API).
INCOMPATIBLE: list[tuple[str, str]] = [
    ("Gemini Embedding (gemini-embedding-*)",
     "Text-only AND cloud API — cannot encode sticker images, and breaks the local/private design."),
    ("Qwen3-Embedding (text)",
     "Text-only — no image tower. Use the multimodal Qwen3-VL-Embedding instead."),
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
