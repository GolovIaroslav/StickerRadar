"""
app/embeddings.py — CLIP image and text embedder with model registry support.

Supported setups:
  Dual-model (default):  separate text tower + image encoder sharing one space.
  Unified model (jina):  single model handles both text and images.

Custom models can be used by setting MODEL_NAME (text) and IMAGE_MODEL_NAME
(image) in .env without adding them to the registry.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app import config

_EMBED_BATCH_SIZE = 32


class Embedder:
    """Lazily loads CLIP models on first use. Thread-safe after initialization."""

    def __init__(self) -> None:
        self._text_model = None
        self._img_model = None
        self._entry = None

    def _resolve_entry(self):
        if self._entry is None:
            from app.models import get as registry_get
            self._entry = registry_get(config.MODEL_NAME)
        return self._entry

    @property
    def model_name(self) -> str:
        return config.MODEL_NAME

    def _text(self):
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer
            name = config.MODEL_NAME
            print(f"Loading text model: {name}")
            self._text_model = SentenceTransformer(name)
        return self._text_model

    def _img(self):
        if self._img_model is not None:
            return self._img_model

        entry = self._resolve_entry()
        if entry is not None and entry.unified:
            # Unified model: image tower is the same as text tower
            self._img_model = self._text()
            return self._img_model

        # Dual-model setup: use IMAGE_MODEL_NAME env var or registry image_model or fallback
        if config.IMAGE_MODEL_NAME:
            img_name = config.IMAGE_MODEL_NAME
        elif entry is not None:
            img_name = entry.image_model
        else:
            # Custom text model without explicit image model: assume same
            img_name = config.MODEL_NAME

        from sentence_transformers import SentenceTransformer
        print(f"Loading image model: {img_name}")
        self._img_model = SentenceTransformer(img_name)
        return self._img_model

    def embed_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        vec = self._img().encode(
            [img], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return vec.astype(np.float32)

    def embed_images(self, paths: list[Path]) -> list[np.ndarray]:
        """Batch-encode multiple images. More efficient than calling embed_image N times."""
        if not paths:
            return []
        imgs = [Image.open(p).convert("RGB") for p in paths]
        vecs = self._img().encode(
            imgs,
            batch_size=_EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [v.astype(np.float32) for v in vecs]

    def embed_text(self, text: str) -> np.ndarray:
        vec = self._text().encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return vec.astype(np.float32)
