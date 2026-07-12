from __future__ import annotations

import json

import numpy as np


class _Response:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self._body


def test_text_embedder_prefixes_queries_batches_and_normalizes(monkeypatch):
    from app import text_embed

    monkeypatch.setattr(text_embed.config, "TEXT_EMBED_BACKEND", "llamacpp")
    monkeypatch.setattr(text_embed.config, "TEXT_EMBED_QUERY_PREFIX", "Q\\n")
    monkeypatch.setattr(text_embed.config, "TEXT_EMBED_SERVER_AUTOSTART", False)
    requests: list[dict] = []

    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/health"):
            return _Response({"status": "ok"})
        payload = json.loads(request.data)
        requests.append(payload)
        return _Response({
            "data": [
                {"index": index, "embedding": [3.0, 4.0]}
                for index, _ in enumerate(payload["input"])
            ]
        })

    monkeypatch.setattr(text_embed.urllib.request, "urlopen", fake_urlopen)
    embedder = text_embed.TextEmbedder()
    vectors = embedder.embed_texts([str(i) for i in range(33)], is_query=True)

    assert len(requests) == 2
    assert requests[0]["input"][0] == "Q\n0"
    assert len(requests[0]["input"]) == 32
    assert np.allclose(vectors[0], [0.6, 0.8])
    assert embedder.model_id() == "qwen3-embedding-0.6b-q8"


def test_text_embedder_reports_missing_autostart_artifacts(monkeypatch, tmp_path):
    from app import text_embed
    from app.errors import ModelNotInstalled

    monkeypatch.setattr(text_embed.config, "TEXT_EMBED_BACKEND", "llamacpp")
    monkeypatch.setattr(text_embed.config, "TEXT_EMBED_SERVER_AUTOSTART", True)
    monkeypatch.setattr(text_embed.config, "TEXT_EMBED_LLAMA_SERVER_PATH", str(tmp_path / "llama-server"))
    monkeypatch.setattr(text_embed.config, "TEXT_EMBED_MODEL_PATH", str(tmp_path / "model.gguf"))
    monkeypatch.setattr(text_embed.TextEmbedder, "_health", lambda _self: False)

    try:
        text_embed.TextEmbedder().embed_texts(["hello"], is_query=False)
    except ModelNotInstalled as exc:
        assert "python -m app model install --id text-embed:qwen3-0.6b" in str(exc)
    else:
        raise AssertionError("missing llama-server should raise ModelNotInstalled")


def test_text_embedder_restores_space_after_qwen_query_marker(monkeypatch):
    from app import text_embed

    monkeypatch.setattr(text_embed.config, "TEXT_EMBED_BACKEND", "llamacpp")
    monkeypatch.setattr(
        text_embed.config,
        "TEXT_EMBED_QUERY_PREFIX",
        "Instruct: retrieve a sticker caption\\nQuery:",
    )
    requests: list[dict] = []

    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/health"):
            return _Response({"status": "ok"})
        payload = json.loads(request.data)
        requests.append(payload)
        return _Response({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    monkeypatch.setattr(text_embed.urllib.request, "urlopen", fake_urlopen)

    text_embed.TextEmbedder().embed_text("sad", is_query=True)

    assert requests[0]["input"] == ["Instruct: retrieve a sticker caption\nQuery: sad"]
