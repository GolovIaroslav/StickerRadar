from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.errors import ConfigError

load_dotenv()


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


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
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
# Telegram
# ---------------------------------------------------------------------------

def _load_api_id() -> int:
    raw = _require("TG_API_ID")
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"TG_API_ID must be an integer, got: {raw!r}")


TG_API_ID: int = _load_api_id()
TG_API_HASH: str = _require("TG_API_HASH")
BOT_TOKEN: str = _require("BOT_TOKEN")
OWNER_USER_ID: int = int(_require("OWNER_USER_ID"))

# Override image model separately (only needed for custom dual-model setups)
IMAGE_MODEL_NAME: str = _str("IMAGE_MODEL_NAME", "")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR: Path = Path(_str("DATA_DIR", "./data")).resolve()
SESSION_PATH: Path = Path(_str("SESSION_PATH", str(DATA_DIR / "sessions" / "user"))).resolve()
DB_PATH: Path = Path(_str("DB_PATH", str(DATA_DIR / "app.sqlite"))).resolve()

MEDIA_DIR: Path = DATA_DIR / "media"
PREVIEWS_DIR: Path = DATA_DIR / "previews"
LOGS_DIR: Path = DATA_DIR / "logs"
EVAL_DIR: Path = DATA_DIR / "eval"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_NAME: str = _str(
    "MODEL_NAME",
    "sentence-transformers/clip-ViT-B-32-multilingual-v1",
)

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

TOP_K: int = _int("TOP_K", 10)
FRAME_COUNT: int = _int("FRAME_COUNT", 5)
SCAN_CONCURRENCY: int = _int("SCAN_CONCURRENCY", 4)
BOT_SEND_DELAY_MS: int = _int("BOT_SEND_DELAY_MS", 250)

# ---------------------------------------------------------------------------
# Optional features
# ---------------------------------------------------------------------------

ENABLE_CAPTIONS: bool = _bool("ENABLE_CAPTIONS", False)
CAPTION_PROVIDER: str = _str("CAPTION_PROVIDER", "none")


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
    """Switch active session/DB to the named profile. Call before db.get_conn()."""
    global SESSION_PATH, DB_PATH
    SESSION_PATH = (DATA_DIR / "sessions" / name).resolve()
    DB_PATH = (DATA_DIR / f"{name}.sqlite").resolve()


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
