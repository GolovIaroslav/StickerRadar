"""
app/embeddings.py — multimodal image/text embedder with multiple backends.

Backends (selected from the model registry's `loader` field):
  "hf"        — transformers AutoModel (SigLIP2, OpenAI CLIP). get_*_features API.
  "st"        — sentence-transformers (jina-clip-v2, packaged CLIP models).
  "open_clip" — open_clip_torch (MobileCLIP). Experimental.

All backends return L2-normalized float32 vectors in a shared image/text space.
Custom models not in the registry fall back to the "st" backend.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app import config

_EMBED_BATCH_SIZE = 32
_TEXT_MAX_LEN = 64


def _l2(arr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return (arr / norm).astype(np.float32)


def _resolve_device(pref: str | None) -> str:
    pref = (pref or "auto").lower()
    if pref in ("cpu", "cuda"):
        return pref
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower() or "CUDA out of memory" in str(exc)


def _params_to_count(params: str) -> float | None:
    """Parse a params string like '2B', '~0.95B', '~99M' into a number, or None."""
    import re
    m = re.search(r"([\d.]+)\s*([BM])", params or "", re.IGNORECASE)
    if not m:
        return None
    n = float(m.group(1))
    return n * (1e9 if m.group(2).upper() == "B" else 1e6)


def _estimate_vram_gb(params: str) -> float | None:
    """Rough GPU memory need in fp16 (2 bytes/param) plus ~30% overhead, in GB."""
    n = _params_to_count(params)
    if n is None:
        return None
    return (n * 2 / 1e9) * 1.3


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class _HFBackend:
    """transformers AutoModel backend (SigLIP2, CLIP)."""

    def __init__(self, model_id: str, text_padding: str, trust_remote_code: bool, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device
        print(f"Loading model: {model_id}  (transformers, device={self.device})")
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self.text_padding = text_padding

    @staticmethod
    def _tensor(out):
        """Coerce a transformers output (tensor or wrapped object) to a feature tensor."""
        if hasattr(out, "cpu"):
            return out
        for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
            val = getattr(out, attr, None)
            if val is not None and hasattr(val, "cpu"):
                return val
        if isinstance(out, (tuple, list)) and out:
            return out[0]
        raise RuntimeError(f"Unexpected model output type: {type(out)}")

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            feats = self._tensor(self.model.get_image_features(**inputs))
        return _l2(feats.cpu().numpy())

    def encode_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts,
            padding=self.text_padding,
            max_length=_TEXT_MAX_LEN,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.no_grad():
            feats = self._tensor(self.model.get_text_features(**inputs))
        return _l2(feats.cpu().numpy())


class _STBackend:
    """sentence-transformers backend (jina-clip-v2, packaged CLIP)."""

    def __init__(self, text_model: str, image_model: str, trust_remote_code: bool, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        kw = {"device": device}
        if trust_remote_code:
            kw["trust_remote_code"] = True
        if device == "cuda":
            # Half precision halves VRAM use, so 2B+ models fit on smaller GPUs.
            import torch
            kw["model_kwargs"] = {"torch_dtype": torch.float16}
        dt = "fp16" if device == "cuda" else "fp32"
        print(f"Loading text model: {text_model}  (sentence-transformers, device={device}, {dt})")
        self.text = SentenceTransformer(text_model, **kw)
        if image_model == text_model:
            self.img = self.text
        else:
            print(f"Loading image model: {image_model}  (sentence-transformers, device={device}, {dt})")
            self.img = SentenceTransformer(image_model, **kw)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        v = self.img.encode(
            images, batch_size=_EMBED_BATCH_SIZE,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return v.astype(np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        v = self.text.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
        )
        return v.astype(np.float32)


class _OpenClipBackend:
    """open_clip backend (MobileCLIP). Experimental."""

    def __init__(self, model_id: str, device: str) -> None:
        try:
            import open_clip
        except ImportError:
            raise RuntimeError(
                "open_clip is required for this model. Install it:\n"
                "  uv add open_clip_torch"
            )
        import torch

        self.torch = torch
        self.open_clip = open_clip
        self.device = device
        name = model_id.split("/")[-1]
        print(f"Loading model: {model_id}  (open_clip, device={self.device})")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            name, pretrained=model_id
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(name)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        batch = self.torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_image(batch)
        return _l2(feats.cpu().numpy())

    def encode_text(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_text(tokens)
        return _l2(feats.cpu().numpy())


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """Lazily loads the active model on first use."""

    def __init__(self) -> None:
        self._backend = None

    @property
    def model_name(self) -> str:
        return config.MODEL_NAME

    @staticmethod
    def _warn_if_low_vram(needed_gb: float | None) -> None:
        """Print free GPU memory and warn only if the model likely won't fit."""
        try:
            import torch
            free, total = torch.cuda.mem_get_info()
            free_gb, total_gb = free / 1e9, total / 1e9
            print(f"GPU memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
            if needed_gb and free_gb < needed_gb:
                print(
                    f"   Model needs ~{needed_gb:.1f} GB (fp16) but only {free_gb:.1f} GB is free.\n"
                    "   If it runs out of memory it will fall back to CPU. To skip the GPU\n"
                    "   attempt, set DEVICE=cpu in .env, or pick a smaller model (python -m app models)."
                )
        except Exception:
            pass

    def _build(self, device: str):
        from app.models import get as registry_get
        entry = registry_get(config.MODEL_NAME)

        if entry is None:
            # Custom model not in registry — a Hugging Face id OR a local path.
            # Loaded via sentence-transformers; must be CLIP-style (image + text).
            image_model = config.IMAGE_MODEL_NAME or config.MODEL_NAME
            return _STBackend(config.MODEL_NAME, image_model, True, device)
        if entry.loader == "hf":
            return _HFBackend(entry.key, entry.text_padding, entry.trust_remote_code, device)
        if entry.loader == "open_clip":
            return _OpenClipBackend(entry.key, device)
        image_model = config.IMAGE_MODEL_NAME or entry.key
        return _STBackend(entry.key, image_model, entry.trust_remote_code, device)

    def _get(self):
        if self._backend is not None:
            return self._backend

        device = _resolve_device(config.DEVICE)
        if device == "cuda":
            from app.models import get as registry_get
            entry = registry_get(config.MODEL_NAME)
            self._warn_if_low_vram(_estimate_vram_gb(entry.params) if entry else None)
        try:
            self._backend = self._build(device)
        except Exception as exc:
            # Out of GPU memory → automatically retry on CPU.
            if device == "cuda" and _is_oom(exc):
                print(
                    "\n" + "!" * 64 + "\n"
                    "!  GPU OUT OF MEMORY — falling back to CPU (uses RAM, slower).\n"
                    "!  Set DEVICE=cpu in .env to skip the GPU attempt next time,\n"
                    "!  or switch to a smaller model (e.g. siglip2-base).\n"
                    + "!" * 64 + "\n"
                )
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self._backend = self._build("cpu")
                return self._backend
            raise RuntimeError(
                f"Failed to load embedding model '{config.MODEL_NAME}':\n  {exc}\n\n"
                "It may need extra dependencies or a custom loader. Run `python -m app models`\n"
                "for install notes, or use the verified fallback in .env:\n"
                "  MODEL_NAME=google/siglip2-base-patch16-224"
            ) from exc

        return self._backend

    def embed_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return self._get().encode_images([img])[0]

    def embed_images(self, paths: list[Path]) -> list[np.ndarray]:
        if not paths:
            return []
        imgs = [Image.open(p).convert("RGB") for p in paths]
        backend = self._get()
        out: list[np.ndarray] = []
        for i in range(0, len(imgs), _EMBED_BATCH_SIZE):
            chunk = imgs[i: i + _EMBED_BATCH_SIZE]
            out.extend(list(backend.encode_images(chunk)))
        return out

    def embed_text(self, text: str) -> np.ndarray:
        return self._get().encode_text([text])[0]
