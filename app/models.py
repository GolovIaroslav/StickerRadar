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
    install_command: str = "No extra install needed"
    open_clip_pretrained: str | None = None


REGISTRY: list[ModelEntry] = [
    # ── Default: best clean-license multilingual option ───────────────────────
    ModelEntry(
        key="google/siglip2-so400m-patch16-384",
        loader="hf",
        params="~1.1B",
        size="~2 GB",
        license="Apache-2.0",
        quality="best",
        langs="109 langs incl. RU/ZH (documented)",
        verified=True,
        notes=(
            "Higher-memory Apache-2.0 multilingual option. SigLIP2 SO400M, 384px.\n"
            "In local StickerRadar retrieval tests on a 6 GB GPU it matched siglip2-base\n"
            "closely on quality, but used much more RAM/VRAM. Choose it for A/B testing\n"
            "or if you want a heavier open-license option and have headroom."
        ),
    ),
    # ── Quality: heavier, strongest multilingual multimodal ───────────────────
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
            "Heavy quality option (Apache-2.0). Multimodal, 30+ languages.\n"
            "GPU benchmark on this 6 GB card failed with CUDA OOM during image retrieval,\n"
            "but CPU mode worked and gave the strongest quality on a small sample.\n"
            "Requires trust_remote_code — review the model card before using."
        ),
    ),
    # ── Fallbacks: lighter Apache-2.0 options ────────────────────────────────
    ModelEntry(
        key="google/siglip2-base-patch16-224",
        loader="hf",
        params="0.4B",
        size="~1.5 GB",
        license="Apache-2.0",
        quality="good",
        langs="109 langs incl. RU/ZH (documented)",
        verified=True,
        notes=(
            "Recommended default for most local installs. Google SigLIP2, Apache-2.0.\n"
            "In local StickerRadar retrieval benchmarks it stayed very close to siglip2-so400m\n"
            "on quality while using much less RAM/VRAM, and it works well on CPU-only hosts."
        ),
    ),
    ModelEntry(
        key="google/siglip2-large-patch16-256",
        loader="hf",
        params="0.9B",
        size="~3.53 GB",
        license="Apache-2.0",
        quality="better",
        langs="109 langs incl. RU/ZH (documented)",
        notes=(
            "Larger SigLIP2 (2025), Apache-2.0. Same loader as the verified base model.\n"
            "No extra install needed (transformers)."
        ),
    ),
    # ── Non-commercial options ────────────────────────────────────────────────
    ModelEntry(
        key="jinaai/jina-clip-v2",
        loader="st",
        params="0.9B",
        size="~1.73 GB",
        license="CC-BY-NC-4.0",
        quality="best",
        langs="89 languages incl. RU/ZH; multimodal (image+text)",
        trust_remote_code=True,
        notes=(
            "Proven quality option (2024). NON-COMMERCIAL license.\n"
            "Needs extra runtime deps in this project: uv add einops timm requests\n"
            "Current local benchmark attempt is blocked by a transformers compatibility break\n"
            "(`clip_loss` import in remote code). Treat as experimental until that stack mismatch is resolved."
        ),
        install_command="uv add einops timm requests",
        experimental=True,
    ),
    ModelEntry(
        key="facebook/metaclip-2-worldwide-huge",
        loader="hf",
        params="~2B",
        size="~4+ GB",
        license="CC-BY-NC-4.0",
        quality="best",
        langs="300+ languages (multilingual SOTA)",
        experimental=True,
        notes=(
            "MetaCLIP 2 worldwide (2025). Best multilingual coverage, 300+ langs.\n"
            "NON-COMMERCIAL license. Heavy: ~2B params. No extra install needed.\n"
            "Current local benchmark attempt is blocked because this repo id was not publicly\n"
            "resolvable from the current environment (401 / invalid model identifier)."
        ),
    ),
    # ── Experimental / A-B candidates ────────────────────────────────────────
    ModelEntry(
        key="visheratin/mexma-siglip2",
        loader="hf",
        params="~1B",
        size="~2 GB",
        license="MIT",
        quality="best",
        langs="80 languages incl. RU/ZH",
        trust_remote_code=True,
        experimental=True,
        notes=(
            "MEXMA-SigLIP2 (MIT). Strong multilingual RU/ZH candidate for A/B testing.\n"
            "Requires trust_remote_code — review the model repo before enabling.\n"
            "Current local benchmark attempt is blocked by a remote-code compatibility break\n"
            "(`SiglipVisionModel` missing `vision_model` in the current transformers stack)."
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
        verified=True,
        notes=(
            "Newest (2026), multimodal, light for its quality. NON-COMMERCIAL license.\n"
            "Load-tested here on a 6 GB GPU, but retrieval quality stayed behind siglip2-base\n"
            "while using more RAM/VRAM and much more time.\n"
            "Needs:  uv add einops"
        ),
        install_command="uv add einops",
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
        verified=True,
        notes=(
            "Newest (2026), heavier than nano, higher quality. NON-COMMERCIAL license.\n"
            "Load-tested here on a 6 GB GPU, but still failed to beat siglip2-base\n"
            "convincingly while using much more VRAM and latency. Needs: uv add einops."
        ),
        install_command="uv add einops",
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
        verified=True,
        notes=(
            "Apple MobileCLIP2 (2025). Tiny and fast to download, English-leaning.\n"
            "Measured here on 20 real stickers: decent quality (MRR ~0.833) but slower\n"
            "and not better than siglip2-base for Russian-heavy retrieval.\n"
            "Needs:  uv add open_clip_torch"
        ),
        install_command="uv add open_clip_torch",
        open_clip_pretrained="dfndr2b",
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
        install_command="uv add peft einops",
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
    return next(m for m in REGISTRY if m.key == "google/siglip2-base-patch16-224")


def get_install_command(key: str) -> str:
    model = get(key)
    if model is None:
        return "See the model card / repository for install requirements"
    return model.install_command
