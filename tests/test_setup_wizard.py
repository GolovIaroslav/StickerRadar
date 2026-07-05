from __future__ import annotations

import importlib
import sys

import pytest


def _reload_config_module(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in [
        "TG_API_ID",
        "TG_API_HASH",
        "BOT_TOKEN",
        "OWNER_USER_ID",
        "MODEL_NAME",
        "DEVICE",
        "OCR_ENABLED",
        "OCR_BACKEND",
    ]:
        monkeypatch.setenv(key, "")
    sys.modules.pop("app.config", None)
    return importlib.import_module("app.config")


def test_config_import_allows_missing_credentials(monkeypatch, tmp_path):
    config = _reload_config_module(monkeypatch, tmp_path)

    assert config.TG_API_ID is None
    assert config.TG_API_HASH == ""
    assert config.BOT_TOKEN == ""
    assert config.OWNER_USER_ID is None

    with pytest.raises(config.ConfigError):
        config.require_login_config()

    with pytest.raises(config.ConfigError):
        config.require_bot_runtime_config()


def test_update_env_text_updates_and_appends_keys():
    from app.setup_wizard import update_env_text

    source = "# Comment\nMODEL_NAME=old/model\nOCR_ENABLED=true\n"
    updated = update_env_text(
        source,
        {
            "MODEL_NAME": "new/model",
            "OCR_ENABLED": "false",
            "OCR_BACKEND": "glm-ocr",
        },
    )

    assert "# Comment" in updated
    assert "MODEL_NAME=new/model" in updated
    assert "OCR_ENABLED=false" in updated
    assert "OCR_BACKEND=glm-ocr" in updated
    assert updated.count("MODEL_NAME=") == 1


def test_runtime_profile_recommendation_prefers_lightweight_models_without_gpu():
    from app.setup_wizard import RuntimeProfile, choose_default_embedding_model, choose_default_ocr_profile

    runtime = RuntimeProfile(
        has_gpu=False,
        gpu_name=None,
        gpu_total_gb=None,
        gpu_free_gb=None,
        llama_cpp_available=False,
        ffmpeg_available=True,
    )

    embedding = choose_default_embedding_model(runtime)
    ocr = choose_default_ocr_profile(runtime)

    assert embedding.key == "google/siglip2-base-patch16-224"
    assert ocr.key == "rapidocr"


def test_runtime_profile_recommendation_prefers_easyocr_when_gpu_is_available():
    from app.setup_wizard import RuntimeProfile, choose_default_ocr_profile

    runtime = RuntimeProfile(
        has_gpu=True,
        gpu_name="RTX 3060",
        gpu_total_gb=6.0,
        gpu_free_gb=3.0,
        llama_cpp_available=True,
        ffmpeg_available=True,
    )

    ocr = choose_default_ocr_profile(runtime)
    assert ocr.key == "easyocr"


def test_quick_setup_keeps_llm_fallback_disabled_even_when_llama_is_available(monkeypatch, tmp_path):
    import app.setup_wizard as wizard

    _reload_config_module(monkeypatch, tmp_path)
    (tmp_path / ".env.example").write_text(
        "MODEL_NAME=google/siglip2-large-patch16-256\n"
        "DEVICE=auto\n"
        "OCR_ENABLED=true\n"
        "OCR_BACKEND=easyocr\n"
        "OCR_USE_GPU=false\n"
        "OCR_LLM_REPO=ggml-org/GLM-OCR-GGUF:Q8_0\n",
        encoding="utf-8",
    )

    runtime = wizard.RuntimeProfile(
        has_gpu=True,
        gpu_name="Test GPU",
        gpu_total_gb=6.0,
        gpu_free_gb=4.0,
        ffmpeg_available=True,
        llama_cpp_available=True,
    )
    monkeypatch.setattr(wizard, "detect_runtime_profile", lambda: runtime)

    wizard.run_setup_wizard(quick=True)
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "OCR_FALLBACK_ENABLED=false" in env_text


def test_quick_setup_enables_easyocr_gpu_on_gpu_hosts(monkeypatch, tmp_path):
    import app.setup_wizard as wizard

    _reload_config_module(monkeypatch, tmp_path)
    (tmp_path / ".env.example").write_text(
        "MODEL_NAME=google/siglip2-large-patch16-256\n"
        "DEVICE=auto\n"
        "OCR_ENABLED=true\n"
        "OCR_BACKEND=easyocr\n"
        "OCR_USE_GPU=false\n",
        encoding="utf-8",
    )

    runtime = wizard.RuntimeProfile(
        has_gpu=True,
        gpu_name="RTX 3060",
        gpu_total_gb=6.0,
        gpu_free_gb=4.0,
        ffmpeg_available=True,
        llama_cpp_available=False,
    )
    monkeypatch.setattr(wizard, "detect_runtime_profile", lambda: runtime)

    wizard.run_setup_wizard(quick=True)
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "OCR_BACKEND=easyocr" in env_text
    assert "OCR_USE_GPU=true" in env_text
