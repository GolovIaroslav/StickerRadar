"""Client for the optional local llama.cpp text-embedding server."""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from app import config
from app.errors import EmbedError, ModelNotInstalled

_BATCH_SIZE = 32
_MODEL_ID = "qwen3-embedding-0.6b-q8"
_INSTALL_COMMAND = "python -m app model install --id text-embed:qwen3-0.6b"


class TextEmbedder:
    """Lazily connect to, or start, the configured local embedding server."""

    def __init__(self) -> None:
        if config.TEXT_EMBED_BACKEND.lower() != "llamacpp":
            raise EmbedError(
                f"Unsupported text embedding backend {config.TEXT_EMBED_BACKEND!r}; "
                "only 'llamacpp' is supported."
            )
        if config.TEXT_EMBED_SERVER_HOST != "127.0.0.1":
            raise EmbedError("TEXT_EMBED_SERVER_HOST must be 127.0.0.1")
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        atexit.register(self.unload)

    def model_id(self) -> str:
        return _MODEL_ID

    @property
    def model_name(self) -> str:
        """Compatibility name for the existing embedding/index interfaces."""
        return self.model_id()

    def embed_texts(self, texts: list[str], *, is_query: bool) -> list[np.ndarray]:
        if not texts:
            return []

        prefix = config.TEXT_EMBED_QUERY_PREFIX.replace("\\n", "\n") if is_query else ""
        # config values are stripped to normalize ordinary string settings, so
        # the semantic Qwen prompt's required space after ``Query:`` cannot
        # survive in an unquoted .env value. Restore that exact prompt form.
        if prefix.endswith("Query:"):
            prefix += " "
        prepared = [prefix + text for text in texts]
        self._ensure_server()

        vectors: list[np.ndarray] = []
        for start in range(0, len(prepared), _BATCH_SIZE):
            chunk = prepared[start : start + _BATCH_SIZE]
            body = json.dumps({"input": chunk, "model": self.model_id()}).encode()
            request = urllib.request.Request(
                f"{self._base_url()}/v1/embeddings",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    payload = json.loads(response.read())
            except (OSError, ValueError) as exc:
                raise EmbedError(f"Text embedding server request failed: {exc}") from exc

            rows = sorted(payload.get("data", []), key=lambda row: row.get("index", 0))
            if len(rows) != len(chunk):
                raise EmbedError(
                    f"Text embedding server returned {len(rows)} vectors for {len(chunk)} inputs"
                )
            for row in rows:
                vector = np.asarray(row["embedding"], dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                if vector.ndim != 1 or not np.isfinite(vector).all() or norm < 1e-8:
                    raise EmbedError("Text embedding server returned an invalid vector")
                vectors.append((vector / norm).astype(np.float32, copy=False))
        return vectors

    def embed_text(self, text: str, *, is_query: bool = False) -> np.ndarray:
        return self.embed_texts([text], is_query=is_query)[0]

    def unload(self) -> bool:
        """Terminate the llama-server process started by this client, if any."""
        with self._lock:
            process = self._process
            self._process = None
        if process is None or process.poll() is not None:
            return False
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return True

    def _base_url(self) -> str:
        return f"http://{config.TEXT_EMBED_SERVER_HOST}:{config.TEXT_EMBED_SERVER_PORT}"

    def _health(self) -> bool:
        request = urllib.request.Request(f"{self._base_url()}/health", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read())
        except (OSError, ValueError):
            return False
        return payload.get("status") == "ok"

    def _ensure_server(self) -> None:
        if self._health():
            return
        if not config.TEXT_EMBED_SERVER_AUTOSTART:
            raise EmbedError(
                f"Text embedding server is not healthy at {self._base_url()}; "
                "start llama-server or enable TEXT_EMBED_SERVER_AUTOSTART."
            )

        with self._lock:
            if self._health():
                return
            if self._process is None or self._process.poll() is not None:
                binary = Path(config.TEXT_EMBED_LLAMA_SERVER_PATH).expanduser()
                model = Path(config.TEXT_EMBED_MODEL_PATH).expanduser()
                if not binary.is_file() or not os.access(binary, os.X_OK):
                    raise ModelNotInstalled(
                        f"llama-server is missing at {binary}. Install the model with {_INSTALL_COMMAND} "
                        "and set TEXT_EMBED_LLAMA_SERVER_PATH to the local binary."
                    )
                if not model.is_file():
                    raise ModelNotInstalled(
                        f"Text embedding model is missing at {model}. Install it with {_INSTALL_COMMAND}."
                    )

                env = os.environ.copy()
                library_dir = str(binary.parent)
                old_library_path = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = (
                    f"{library_dir}{os.pathsep}{old_library_path}" if old_library_path else library_dir
                )
                self._process = subprocess.Popen(
                    [
                        str(binary),
                        "-m", str(model),
                        "--embedding",
                        "--pooling", "last",
                        "-ngl", "0",
                        "-t", "4",
                        "-c", "2048",
                        "-b", "2048",
                        "-ub", "2048",
                        "--host", config.TEXT_EMBED_SERVER_HOST,
                        "--port", str(config.TEXT_EMBED_SERVER_PORT),
                    ],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if self._health():
                    return
                if self._process is not None and self._process.poll() is not None:
                    raise EmbedError("llama-server exited before its health endpoint became ready")
                time.sleep(0.25)
            raise EmbedError(f"Timed out waiting for llama-server health at {self._base_url()}")


_shared: TextEmbedder | None = None
_shared_lock = threading.Lock()


def get_shared_text_embedder() -> TextEmbedder:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = TextEmbedder()
    return _shared


def unload_shared_text_embedder() -> bool:
    """Unload only the llama-server process owned by the shared client."""
    with _shared_lock:
        embedder = _shared
    return embedder.unload() if embedder is not None else False
