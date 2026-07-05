from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from app.errors import ConfigError


ENV_FILE = Path(".env")


def load_env(*, override: bool = False) -> None:
    load_dotenv(dotenv_path=ENV_FILE, override=override)


load_env()


def _require(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise ConfigError(
            f"Required environment variable {name!r} is not set.\n"
            f"Copy .env.example to .env and fill in the values."
        )
    return v


def _str(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _optional_str(name: str) -> str:
    return os.environ.get(name, "").strip()


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        raise ConfigError(f"Environment variable {name!r} must be an integer, got: {raw!r}")


def _optional_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"Environment variable {name!r} must be an integer, got: {raw!r}")


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


# ---------------------------------------------------------------------------
# Telegram (loaded lazily so `python -m app setup` can run before credentials)
# ---------------------------------------------------------------------------

TG_API_ID: int | None
TG_API_HASH: str
BOT_TOKEN: str
OWNER_USER_ID: int | None

# Override image model separately (only needed for custom dual-model setups)
IMAGE_MODEL_NAME: str

# Compute device for embeddings: "auto" | "cpu" | "cuda"
DEVICE: str

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR: Path
SESSION_PATH: Path
DB_PATH: Path

MEDIA_DIR: Path
PREVIEWS_DIR: Path
LOGS_DIR: Path
EVAL_DIR: Path

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_NAME: str

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

TOP_K: int
FRAME_COUNT: int
SCAN_CONCURRENCY: int
BOT_SEND_DELAY_MS: int
MODEL_IDLE_UNLOAD_SEC: int

# ---------------------------------------------------------------------------
# Optional features
# ---------------------------------------------------------------------------

ENABLE_CAPTIONS: bool
CAPTION_PROVIDER: str
OCR_ENABLED: bool
OCR_BACKEND: str
OCR_USE_GPU: bool
OCR_LANGS: str
OCR_LLM_REPO: str
OCR_FALLBACK_ENABLED: bool
OCR_FALLBACK_BACKEND: str
OCR_FALLBACK_CONFIDENCE: float
OCR_FALLBACK_MIN_CHARS: int
ALLOW_REMOTE_CODE: bool
SETUP_WIZARD_COMPLETED: bool


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        raise ConfigError(f"Environment variable {name!r} must be a float, got: {raw!r}")


def reload() -> None:
    load_env(override=True)
    _assign_from_env()


def _assign_from_env() -> None:
    global TG_API_ID, TG_API_HASH, BOT_TOKEN, OWNER_USER_ID
    global IMAGE_MODEL_NAME, DEVICE
    global DATA_DIR, SESSION_PATH, DB_PATH, MEDIA_DIR, PREVIEWS_DIR, LOGS_DIR, EVAL_DIR
    global MODEL_NAME
    global TOP_K, FRAME_COUNT, SCAN_CONCURRENCY, BOT_SEND_DELAY_MS, MODEL_IDLE_UNLOAD_SEC
    global ENABLE_CAPTIONS, CAPTION_PROVIDER, OCR_ENABLED, OCR_BACKEND, OCR_USE_GPU, OCR_LANGS, OCR_LLM_REPO
    global OCR_FALLBACK_ENABLED, OCR_FALLBACK_BACKEND, OCR_FALLBACK_CONFIDENCE, OCR_FALLBACK_MIN_CHARS
    global ALLOW_REMOTE_CODE, SETUP_WIZARD_COMPLETED

    TG_API_ID = _optional_int("TG_API_ID")
    TG_API_HASH = _optional_str("TG_API_HASH")
    BOT_TOKEN = _optional_str("BOT_TOKEN")
    OWNER_USER_ID = _optional_int("OWNER_USER_ID")

    IMAGE_MODEL_NAME = _str("IMAGE_MODEL_NAME", "")
    DEVICE = _str("DEVICE", "auto")

    DATA_DIR = Path(_str("DATA_DIR", "./data")).resolve()
    SESSION_PATH = Path(_str("SESSION_PATH", str(DATA_DIR / "sessions" / "user"))).resolve()
    DB_PATH = Path(_str("DB_PATH", str(DATA_DIR / "app.sqlite"))).resolve()

    MEDIA_DIR = DATA_DIR / "media"
    PREVIEWS_DIR = DATA_DIR / "previews"
    LOGS_DIR = DATA_DIR / "logs"
    EVAL_DIR = DATA_DIR / "eval"

    MODEL_NAME = _str("MODEL_NAME", "google/siglip2-large-patch16-256")

    TOP_K = _int("TOP_K", 10)
    FRAME_COUNT = _int("FRAME_COUNT", 5)
    SCAN_CONCURRENCY = _int("SCAN_CONCURRENCY", 4)
    BOT_SEND_DELAY_MS = _int("BOT_SEND_DELAY_MS", 250)
    MODEL_IDLE_UNLOAD_SEC = _int("MODEL_IDLE_UNLOAD_SEC", 180)

    ENABLE_CAPTIONS = _bool("ENABLE_CAPTIONS", False)
    CAPTION_PROVIDER = _str("CAPTION_PROVIDER", "none")
    OCR_ENABLED = _bool("OCR_ENABLED", True)
    OCR_BACKEND = _str("OCR_BACKEND", "easyocr")
    OCR_USE_GPU = _bool("OCR_USE_GPU", False)
    OCR_LANGS = _str("OCR_LANGS", "ru,en")
    OCR_LLM_REPO = _str("OCR_LLM_REPO", "ggml-org/GLM-OCR-GGUF:Q8_0")
    OCR_FALLBACK_ENABLED = _bool("OCR_FALLBACK_ENABLED", False)
    OCR_FALLBACK_BACKEND = _str("OCR_FALLBACK_BACKEND", "glm-ocr")
    OCR_FALLBACK_CONFIDENCE = _float("OCR_FALLBACK_CONFIDENCE", 0.45)
    OCR_FALLBACK_MIN_CHARS = _int("OCR_FALLBACK_MIN_CHARS", 3)
    ALLOW_REMOTE_CODE = _bool("ALLOW_REMOTE_CODE", False)
    SETUP_WIZARD_COMPLETED = _bool("SETUP_WIZARD_COMPLETED", False)


_assign_from_env()


def _format_missing_message(missing: list[str]) -> str:
    joined = ", ".join(missing)
    return (
        f"Missing required configuration: {joined}.\n"
        f"Run `python -m app setup` to create/update .env, or fill the values manually."
    )


def require_login_config() -> tuple[int, str]:
    missing: list[str] = []
    if TG_API_ID is None:
        missing.append("TG_API_ID")
    if not TG_API_HASH:
        missing.append("TG_API_HASH")
    if missing:
        raise ConfigError(_format_missing_message(missing))
    assert TG_API_ID is not None
    return TG_API_ID, TG_API_HASH


def require_bot_runtime_config() -> tuple[str, int]:
    missing: list[str] = []
    if not BOT_TOKEN:
        missing.append("BOT_TOKEN")
    if OWNER_USER_ID is None:
        missing.append("OWNER_USER_ID")
    if missing:
        raise ConfigError(_format_missing_message(missing))
    assert OWNER_USER_ID is not None
    return BOT_TOKEN, OWNER_USER_ID


# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create all required local data directories."""
    dirs = [
        DATA_DIR,
        SESSION_PATH.parent,
        MEDIA_DIR / "stickers" / "static",
        MEDIA_DIR / "stickers" / "animated",
        MEDIA_DIR / "stickers" / "video",
        MEDIA_DIR / "gifs",
        PREVIEWS_DIR,
        LOGS_DIR,
        EVAL_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def set_profile(name: str) -> None:
    """Switch active session/DB/media dirs to the named profile. Call before db.get_conn()."""
    import re
    if not re.fullmatch(r"[\w\-]+", name):
        raise ConfigError(
            f"Invalid profile name {name!r}: only letters, digits, hyphens and underscores are allowed."
        )
    global SESSION_PATH, DB_PATH, MEDIA_DIR, PREVIEWS_DIR
    SESSION_PATH = (DATA_DIR / "sessions" / name).resolve()
    DB_PATH = (DATA_DIR / f"{name}.sqlite").resolve()
    # Namespace media/preview dirs per profile so two profiles don't share
    # autoincrement row IDs that would collide on the same paths.
    MEDIA_DIR = DATA_DIR / name / "media"
    PREVIEWS_DIR = DATA_DIR / name / "previews"


def check_ffmpeg() -> str | None:
    """Return ffmpeg binary path or None if not found."""
    import shutil
    return shutil.which("ffmpeg")


def require_ffmpeg() -> str:
    """Return ffmpeg binary path or raise ConfigError."""
    path = check_ffmpeg()
    if not path:
        raise ConfigError(
            "ffmpeg not found in PATH.\n"
            "Install it:\n"
            "  Linux (apt):  sudo apt install ffmpeg\n"
            "  Linux (arch): sudo pacman -S ffmpeg\n"
            "  macOS:        brew install ffmpeg\n"
            "  Windows:      https://ffmpeg.org/download.html  (add to PATH)"
        )
    return path
