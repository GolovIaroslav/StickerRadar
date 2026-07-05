from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app import config
from app.models import ModelEntry, REGISTRY, default as default_embedding_model, get_install_command


@dataclass(frozen=True)
class RuntimeProfile:
    has_gpu: bool
    gpu_name: str | None
    gpu_total_gb: float | None
    gpu_free_gb: float | None
    llama_cpp_available: bool
    ffmpeg_available: bool


@dataclass(frozen=True)
class OcrProfile:
    key: str
    label: str
    backend: str
    install_hint: str
    approx_size: str
    speed: str
    best_for: str
    notes: str
    enabled: bool = True
    use_gpu: bool = False
    supported: bool = True


OCR_PROFILES: list[OcrProfile] = [
    OcrProfile(
        key="rapidocr",
        label="RapidOCR (fast CPU path)",
        backend="rapidocr",
        install_hint="uv add rapidocr-onnxruntime",
        approx_size="~150-250 MB runtime assets",
        speed="Fast on CPU (~302 ms/image in local tests)",
        best_for="Thousands of stickers, low overhead, good default for CPU-only machines",
        notes="Measured here on 20 real stickers. Lower overhead than EasyOCR, but much weaker on Russian meme text.",
    ),
    OcrProfile(
        key="easyocr",
        label="EasyOCR scene-text",
        backend="easyocr",
        install_hint="uv add easyocr",
        approx_size="~300 MB model download",
        speed="Best default here (~161 ms/image on GPU, ~611 ms/image on CPU)",
        best_for="Decorative fonts and scene text, especially Russian sticker text",
        notes="Measured here on 20 real stickers. Better accuracy than RapidOCR in this project, especially on Russian meme text.",
    ),
    OcrProfile(
        key="glm-ocr",
        label="GLM-OCR via llama.cpp (experimental)",
        backend="glm-ocr",
        install_hint="Install llama.cpp and keep `llama-cli` in PATH",
        approx_size="~1.43 GB (Q8_0 + mmproj)",
        speed="Very slow (~3973 ms/image in local tests)",
        best_for="Hard stickers where classic OCR fails and the user accepts much slower indexing",
        notes="Worked after converting StickerRadar WEBP stickers to PNG first. Useful for rescue passes, but too slow and too verbose for default bulk OCR.",
        use_gpu=True,
    ),
    OcrProfile(
        key="off",
        label="Disable OCR",
        backend="easyocr",
        install_hint="No extra install",
        approx_size="0",
        speed="Fastest setup",
        best_for="Image-semantic search only",
        notes="Use this if the user does not care about exact text printed on stickers.",
        enabled=False,
    ),
]


def detect_runtime_profile() -> RuntimeProfile:
    gpu_name = None
    gpu_total = None
    gpu_free = None
    has_gpu = False
    try:
        import torch

        has_gpu = torch.cuda.is_available()
        if has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            gpu_total = round(total_bytes / (1024**3), 2)
            gpu_free = round(free_bytes / (1024**3), 2)
    except Exception:
        has_gpu = False

    if not has_gpu and shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            ).strip()
            if out:
                name, total_mb, free_mb = [part.strip() for part in out.split(",")[:3]]
                has_gpu = True
                gpu_name = name
                gpu_total = round(float(total_mb) / 1024, 2)
                gpu_free = round(float(free_mb) / 1024, 2)
        except Exception:
            pass

    return RuntimeProfile(
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        gpu_total_gb=gpu_total,
        gpu_free_gb=gpu_free,
        llama_cpp_available=shutil.which("llama-cli") is not None,
        ffmpeg_available=config.check_ffmpeg() is not None,
    )


def choose_default_embedding_model(runtime: RuntimeProfile) -> ModelEntry:
    if not runtime.has_gpu:
        return next(m for m in REGISTRY if m.key == "google/siglip2-base-patch16-224")
    if runtime.gpu_free_gb is not None and runtime.gpu_free_gb < 8.0:
        return next(m for m in REGISTRY if m.key == "google/siglip2-base-patch16-224")
    return next(m for m in REGISTRY if m.key == "google/siglip2-so400m-patch16-384")


def choose_default_ocr_profile(runtime: RuntimeProfile) -> OcrProfile:
    if runtime.has_gpu:
        return next(p for p in OCR_PROFILES if p.key == "easyocr")
    return next(p for p in OCR_PROFILES if p.key == "rapidocr")


def ocr_use_gpu(profile: OcrProfile, runtime: RuntimeProfile) -> bool:
    if not profile.enabled:
        return False
    if profile.key == "rapidocr":
        return False
    if profile.key == "easyocr":
        return runtime.has_gpu
    return profile.use_gpu and runtime.has_gpu


def install_plan_lines(*, embedding_install: str, ocr_install: str) -> list[str]:
    lines: list[str] = []
    if embedding_install != "No extra install needed":
        lines.append(f"Embedding model: {embedding_install}")
    if ocr_install != "No extra install":
        lines.append(f"OCR:            {ocr_install}")
    if not lines:
        return ["No extra installs are required for the selected embedding + OCR setup."]
    return lines


def ocr_profile_lines(runtime: RuntimeProfile) -> list[str]:
    recommended = choose_default_ocr_profile(runtime)
    lines: list[str] = []
    for idx, profile in enumerate(OCR_PROFILES, 1):
        tag = " (recommended)" if profile.key == recommended.key else ""
        lines.append(f"{idx}. {profile.label} — {profile.approx_size}, {profile.speed}{tag}")
        lines.append(f"   Best for: {profile.best_for}")
        lines.append(f"   Install:  {profile.install_hint}")
        lines.append(f"   Notes:    {profile.notes}")
    return lines


def update_env_text(text: str, updates: dict[str, str]) -> str:
    lines = text.splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        replaced = False
        for key, value in updates.items():
            prefix = f"{key}="
            if line.startswith(prefix):
                out.append(f"{key}={value}")
                seen.add(key)
                replaced = True
                break
        if not replaced:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            if out and out[-1] != "":
                out.append("")
            out.append(f"{key}={value}")
    return "\n".join(out).rstrip() + "\n"


def ensure_env_file() -> Path:
    env_path = config.ENV_FILE.resolve()
    if env_path.exists():
        return env_path
    example = Path(".env.example").resolve()
    if example.exists():
        env_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        env_path.write_text("", encoding="utf-8")
    return env_path


def wizard_needed() -> bool:
    return (not config.ENV_FILE.exists()) or (not config.SETUP_WIZARD_COMPLETED)


def _prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{text}{suffix}: ").strip()
    except EOFError:
        return default
    return value or default


def _select_embedding_model(runtime: RuntimeProfile) -> ModelEntry:
    recommended = choose_default_embedding_model(runtime)
    print("\nEmbedding model selection")
    print("These models must embed BOTH images and text. Recommended options are listed first.")
    shortlist = [
        next(m for m in REGISTRY if m.key == recommended.key),
        next(m for m in REGISTRY if m.key == "google/siglip2-so400m-patch16-384"),
        next(m for m in REGISTRY if m.key == "google/siglip2-large-patch16-256"),
        next(m for m in REGISTRY if m.key == "Qwen/Qwen3-VL-Embedding-2B"),
    ]
    seen: set[str] = set()
    options: list[ModelEntry] = []
    for item in shortlist + REGISTRY:
        if item.key not in seen:
            seen.add(item.key)
            options.append(item)
    for idx, model in enumerate(options, 1):
        tag = " (recommended)" if model.key == recommended.key else ""
        print(f"  {idx}. {model.key} — {model.size}, {model.license}, {model.langs}{tag}")
        print(f"     Quality:  {model.quality}")
        print(f"     Install:  {model.install_command}")
    raw = _prompt("Choose embedding model number", str(options.index(recommended) + 1))
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    return recommended


def _select_ocr_profile(runtime: RuntimeProfile) -> OcrProfile:
    recommended = choose_default_ocr_profile(runtime)
    print("\nOCR selection")
    print("Classic OCR is faster for large sticker libraries. VLM OCR is slower but may help on hard text.")
    for line in ocr_profile_lines(runtime):
        print(f"  {line}")
    raw = _prompt("Choose OCR profile number", str(OCR_PROFILES.index(recommended) + 1))
    if raw.isdigit() and 1 <= int(raw) <= len(OCR_PROFILES):
        return OCR_PROFILES[int(raw) - 1]
    return recommended


def _choose_device(runtime: RuntimeProfile, model: ModelEntry) -> str:
    recommended = "auto"
    if not runtime.has_gpu:
        recommended = "cpu"
    elif runtime.gpu_free_gb is not None and runtime.gpu_free_gb < 4 and model.size not in {"~398 MB", "~1.5 GB"}:
        recommended = "cpu"
    print("\nDevice selection")
    print("  auto = use GPU when possible, otherwise CPU")
    print("  cpu  = slow but safest")
    print("  cuda = force GPU (fails if VRAM is insufficient)")
    raw = _prompt("Device", recommended).lower()
    return raw if raw in {"auto", "cpu", "cuda"} else recommended


def _print_runtime_summary(runtime: RuntimeProfile) -> None:
    print("\nDetected machine")
    if runtime.has_gpu:
        print(f"  GPU      : {runtime.gpu_name or 'GPU detected'}")
        print(f"  VRAM     : total {runtime.gpu_total_gb or '?'} GB, free {runtime.gpu_free_gb or '?'} GB")
    else:
        print("  GPU      : not detected (CPU-only mode is still supported)")
    print(f"  ffmpeg   : {'found' if runtime.ffmpeg_available else 'missing'}")
    print(f"  llama.cpp: {'found' if runtime.llama_cpp_available else 'missing'}")


def _collect_credentials() -> dict[str, str]:
    print("\nTelegram credentials")
    print("You can leave these blank for now and fill them later before login/run.")
    return {
        "TG_API_ID": _prompt("TG_API_ID", os.environ.get("TG_API_ID", "")),
        "TG_API_HASH": _prompt("TG_API_HASH", os.environ.get("TG_API_HASH", "")),
        "BOT_TOKEN": _prompt("BOT_TOKEN", os.environ.get("BOT_TOKEN", "")),
        "OWNER_USER_ID": _prompt("OWNER_USER_ID", os.environ.get("OWNER_USER_ID", "")),
    }


def run_setup_wizard(*, quick: bool = False) -> dict[str, str]:
    runtime = detect_runtime_profile()
    env_path = ensure_env_file()
    text = env_path.read_text(encoding="utf-8")
    if not sys.stdin.isatty() and not quick:
        quick = True

    print("\nStickerRadar first-run setup")
    print("This wizard writes a beginner-friendly .env so new users can start quickly.")
    print("You can skip later with `python -m app setup --quick` or by keeping defaults.")
    _print_runtime_summary(runtime)

    if not quick:
        action = _prompt("Press Enter for guided setup, or type 'skip' to keep current .env", "").lower()
        if action == "skip":
            print("Skipping changes.")
            return {}

    embedding = choose_default_embedding_model(runtime) if quick else _select_embedding_model(runtime)
    ocr = choose_default_ocr_profile(runtime) if quick else _select_ocr_profile(runtime)
    device = _choose_device(runtime, embedding) if not quick else ("cpu" if not runtime.has_gpu else "auto")
    credentials = {} if quick else _collect_credentials()

    updates = {
        "MODEL_NAME": embedding.key,
        "DEVICE": device,
        "OCR_ENABLED": "true" if ocr.enabled else "false",
        "OCR_BACKEND": ocr.backend,
        "OCR_USE_GPU": "true" if ocr_use_gpu(ocr, runtime) else "false",
        "OCR_LLM_REPO": "ggml-org/GLM-OCR-GGUF:Q8_0" if ocr.key == "glm-ocr" else config.OCR_LLM_REPO,
        "OCR_FALLBACK_ENABLED": "false",
        "OCR_FALLBACK_BACKEND": "glm-ocr",
        "OCR_FALLBACK_CONFIDENCE": "0.45",
        "OCR_FALLBACK_MIN_CHARS": "3",
        "SETUP_WIZARD_COMPLETED": "true",
    }
    updates.update(credentials)

    updated = update_env_text(text, {k: v for k, v in updates.items() if v != ""})
    env_path.write_text(updated, encoding="utf-8")
    config.reload()

    print("\nSaved configuration to .env")
    print(f"  Embeddings : {embedding.key}")
    print(f"  OCR        : {ocr.label}")
    print(f"  Device     : {device}")
    print("Install commands for this selection:")
    for line in install_plan_lines(
        embedding_install=get_install_command(embedding.key),
        ocr_install=ocr.install_hint,
    ):
        print(f"  - {line}")
    print("Next steps:")
    print("  1. Install any missing extras shown above")
    print("  2. Run `python -m app login`")
    print("  3. Run `python -m app sync`")
    return updates
