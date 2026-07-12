from __future__ import annotations

from pathlib import Path


def test_qwen_text_artifact_is_pinned_to_the_requested_file_and_revision():
    from app.model_artifacts import get_artifact

    artifact = get_artifact("text-embed:qwen3-0.6b")

    assert artifact is not None
    assert artifact.source == "Qwen/Qwen3-Embedding-0.6B-GGUF"
    assert artifact.files == ("Qwen3-Embedding-0.6B-Q8_0.gguf",)
    assert artifact.revision == "370f27d7550e0def9b39c1f16d3fbaa13aa67728"


def test_qwen_text_install_defaults_to_model_root_and_prints_env_paths(monkeypatch, tmp_path, capsys):
    from app import config, model_installer

    monkeypatch.setattr(config, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(config, "TEXT_EMBED_MODEL_PATH", "")
    monkeypatch.setattr(model_installer, "artifact_ready", lambda *_args: False)
    monkeypatch.setattr(
        model_installer.shutil,
        "which",
        lambda name: "/usr/bin/hf" if name == "hf" else None,
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        model_installer.subprocess,
        "run",
        lambda command, check: commands.append(command),
    )

    assert model_installer.install_artifact("text-embed:qwen3-0.6b", yes=True) == 0

    target = tmp_path / "text-embed-qwen3-0.6b"
    command = commands[0]
    assert command[:3] == ["hf", "download", "Qwen/Qwen3-Embedding-0.6B-GGUF"]
    assert "Qwen3-Embedding-0.6B-Q8_0.gguf" in command
    assert command[command.index("--revision") + 1] == "370f27d7550e0def9b39c1f16d3fbaa13aa67728"
    assert command[command.index("--local-dir") + 1] == str(target.resolve())

    output = capsys.readouterr().out
    assert f"TEXT_EMBED_MODEL_PATH={target / 'Qwen3-Embedding-0.6B-Q8_0.gguf'}" in output
    assert "TEXT_EMBED_LLAMA_SERVER_PATH=" in output


def test_env_example_contains_all_text_embed_keys():
    env_example = Path(__file__).parents[1] / ".env.example"
    text = env_example.read_text(encoding="utf-8")

    for key in (
        "TEXT_EMBED_ENABLED",
        "TEXT_EMBED_BACKEND",
        "TEXT_EMBED_MODEL_PATH",
        "TEXT_EMBED_SERVER_HOST",
        "TEXT_EMBED_SERVER_PORT",
        "TEXT_EMBED_SERVER_AUTOSTART",
        "TEXT_EMBED_LLAMA_SERVER_PATH",
        "TEXT_EMBED_QUERY_PREFIX",
    ):
        assert f"{key}=" in text


def test_ppocr_shadow_artifact_installs_both_local_model_folders(monkeypatch, tmp_path, capsys):
    from app import config, model_installer

    monkeypatch.setattr(config, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(config, "OCR_SHADOW_DET_DIR", "")
    monkeypatch.setattr(config, "OCR_SHADOW_REC_DIR", "")
    monkeypatch.setattr(model_installer, "artifact_ready", lambda *_args: False)
    monkeypatch.setattr(model_installer.shutil, "which", lambda name: "/usr/bin/hf" if name == "hf" else None)
    commands: list[list[str]] = []
    monkeypatch.setattr(model_installer.subprocess, "run", lambda command, check: commands.append(command))

    assert model_installer.install_artifact("ocr:ppocrv5-eslav", yes=True) == 0

    target = tmp_path / "ocr-ppocrv5-eslav"
    assert commands == [
        ["hf", "download", "PaddlePaddle/PP-OCRv5_mobile_det", "--local-dir", str(target / "det")],
        ["hf", "download", "PaddlePaddle/eslav_PP-OCRv5_mobile_rec", "--local-dir", str(target / "rec")],
    ]
    output = capsys.readouterr().out
    assert f"OCR_SHADOW_DET_DIR={target / 'det'}" in output
    assert f"OCR_SHADOW_REC_DIR={target / 'rec'}" in output


def test_env_example_contains_all_ppocr_shadow_keys():
    env_example = Path(__file__).parents[1] / ".env.example"
    text = env_example.read_text(encoding="utf-8")

    for key in ("OCR_SHADOW_ENABLED", "OCR_SHADOW_PYTHON", "OCR_SHADOW_DET_DIR", "OCR_SHADOW_REC_DIR"):
        assert f"{key}=" in text
