from __future__ import annotations


def test_embedding_registry_default_prefers_siglip2_base():
    from app.models import default

    assert default().key == "google/siglip2-base-patch16-224"


def test_open_clip_registry_entry_exposes_pretrained_tag():
    from app.models import get

    entry = get("apple/MobileCLIP2-S2")
    assert entry is not None
    assert entry.loader == "open_clip"
    assert entry.open_clip_pretrained == "dfndr2b"


def test_embedding_install_command_defaults_to_none_needed():
    from app.models import get_install_command

    assert get_install_command("google/siglip2-base-patch16-224") == "No extra install needed"


def test_embedding_install_command_surfaces_extra_dependencies():
    from app.models import get_install_command

    assert get_install_command("jinaai/jina-clip-v2") == "uv add einops timm requests"


def test_ocr_install_plan_includes_selected_embedding_and_ocr_choices():
    from app.models import get_install_command
    from app.setup_wizard import OCR_PROFILES, install_plan_lines

    embedding_cmd = get_install_command("jinaai/jina-clip-v2")
    ocr = next(p for p in OCR_PROFILES if p.key == "easyocr")

    lines = install_plan_lines(embedding_install=embedding_cmd, ocr_install=ocr.install_hint)

    assert any("Embedding model:" in line for line in lines)
    assert any("uv add einops" in line for line in lines)
    assert any("OCR:" in line for line in lines)
    assert any("uv add easyocr" in line for line in lines)


def test_ocr_install_plan_skips_noop_lines():
    from app.setup_wizard import install_plan_lines

    lines = install_plan_lines(embedding_install="No extra install needed", ocr_install="No extra install")

    assert lines == ["No extra installs are required for the selected embedding + OCR setup."]
