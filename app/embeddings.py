"""
app/embeddings.py — CLIP image and text embedder.

Image model : clip-ViT-B-32          (original CLIP, handles PIL images)
Text model  : config.MODEL_NAME       (multilingual text tower, same 512-d space)

Both return L2-normalized float32 vectors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app import config

# The multilingual text model shares the embedding space with the original
# OpenAI CLIP image encoder, so we keep both loaded separately.
_IMAGE_MODEL_NAME = "clip-ViT-B-32"


class Embedder:
    """Lazily loads CLIP models on first use."""

    def __init__(self) -> None:
        self._text_model = None
        self._img_model = None

    @property
    def model_name(self) -> str:
        return config.MODEL_NAME

    def _text(self):
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading text model: {config.MODEL_NAME}")
            self._text_model = SentenceTransformer(config.MODEL_NAME)
        return self._text_model

    def _img(self):
        if self._img_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading image model: {_IMAGE_MODEL_NAME}")
            self._img_model = SentenceTransformer(_IMAGE_MODEL_NAME)
        return self._img_model

    def embed_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        vec = self._img().encode(
            [img], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return vec.astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        vec = self._text().encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return vec.astype(np.float32)
