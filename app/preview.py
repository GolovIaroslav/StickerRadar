"""
app/preview.py — extract preview PNG frames from downloaded media.

Output convention:
    data/previews/{media_id}/frame_000.png
    data/previews/{media_id}/frame_001.png
    data/previews/{media_id}/frame_002.png

Supported formats:
    .webp (static)  → 1 frame  (pos 0.0)
    .tgs            → 3 frames via rlottie-python, then python-lottie CLI
    .webm/.mp4/.gif → 3 frames via ffmpeg at pos 0.2 / 0.5 / 0.8
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path

from PIL import Image

from app import config, db

_FRAME_POSITIONS = [0.2, 0.5, 0.8]
_BG = (255, 255, 255)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preview_dir(media_id: int) -> Path:
    d = config.PREVIEWS_DIR / str(media_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _composite(img: Image.Image) -> Image.Image:
    """Flatten transparency onto white background."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, _BG)
        rgba = img.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert("RGB")


def _img_size(path: Path) -> tuple[int, int] | tuple[None, None]:
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None, None


def _probe_duration(src: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        r = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(src),
            ],
            capture_output=True, text=True, timeout=15,
        )
        val = r.stdout.strip()
        return float(val) if val else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Format-specific extractors
# Return list of (frame_pos, output_path) for frames that were produced.
# ---------------------------------------------------------------------------

def _extract_webp(row: sqlite3.Row) -> list[tuple[float, Path]]:
    src = Path(row["local_path"])
    out = _preview_dir(row["id"]) / "frame_000.png"
    with Image.open(src) as img:
        try:
            img.seek(0)
        except EOFError:
            pass
        _composite(img).save(out)
    return [(0.0, out)]


def _extract_video(row: sqlite3.Row, ffmpeg: str) -> list[tuple[float, Path]]:
    src = Path(row["local_path"])
    out_dir = _preview_dir(row["id"])
    duration = _probe_duration(src)

    results: list[tuple[float, Path]] = []
    for i, pos in enumerate(_FRAME_POSITIONS):
        out = out_dir / f"frame_{i:03d}.png"
        ts = (duration * pos) if duration else 0.0
        cmd = [
            ffmpeg, "-y",
            "-ss", str(ts),
            "-i", str(src),
            "-frames:v", "1",
            "-vf", "scale=512:512:force_original_aspect_ratio=decrease",
            str(out),
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30, check=True)
            results.append((pos, out))
        except subprocess.CalledProcessError:
            # fallback: first frame, no seek
            if i == 0:
                try:
                    subprocess.run(
                        [ffmpeg, "-y", "-i", str(src), "-frames:v", "1", str(out)],
                        capture_output=True, timeout=30, check=True,
                    )
                    results.append((pos, out))
                except Exception:
                    pass
    return results


def _extract_tgs_rlottie(row: sqlite3.Row) -> list[tuple[float, Path]] | None:
    try:
        import rlottie_python as rl
    except ImportError:
        return None

    src = Path(row["local_path"])
    out_dir = _preview_dir(row["id"])
    try:
        anim = rl.LottieAnimation.from_tgs(str(src))
        n = anim.lottie_animation_get_totalframe()
        if n <= 0:
            return None
        w, h = anim.lottie_animation_get_size()
        results: list[tuple[float, Path]] = []
        for i, pos in enumerate(_FRAME_POSITIONS):
            fidx = max(0, min(int(pos * n), n - 1))
            out = out_dir / f"frame_{i:03d}.png"
            buf = anim.lottie_animation_render(fidx, w, h)
            img = Image.frombuffer("RGBA", (w, h), buf, "raw", "RGBA", 0, 1)
            _composite(img).save(out)
            results.append((pos, out))
        return results
    except Exception:
        return None


def _extract_tgs_lottie_cli(row: sqlite3.Row) -> list[tuple[float, Path]] | None:
    cli = shutil.which("lottie_convert.py") or shutil.which("lottie_convert")
    if not cli:
        return None

    src = Path(row["local_path"])
    out = _preview_dir(row["id"]) / "frame_001.png"
    try:
        subprocess.run(
            [cli, str(src), str(out), "--frame", "0"],
            capture_output=True, timeout=30, check=True,
        )
        return [(0.5, out)]
    except Exception:
        return None


def _extract_tgs(row: sqlite3.Row) -> list[tuple[float, Path]]:
    frames = _extract_tgs_rlottie(row)
    if frames:
        return frames
    frames = _extract_tgs_lottie_cli(row)
    if frames:
        return frames
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_previews(row: sqlite3.Row, ffmpeg: str) -> None:
    """Extract frames for one media item and update DB. Never raises."""
    media_id: int = row["id"]
    local_path_str: str | None = row["local_path"]

    if not local_path_str:
        db.mark_preview_failed(media_id, "no local_path")
        return

    src = Path(local_path_str)
    if not src.exists():
        db.mark_preview_failed(media_id, f"file not found: {src.name}")
        return

    try:
        suffix = src.suffix.lower()
        fmt: str | None = row["sticker_format"]
        mime: str = row["mime_type"] or ""

        if fmt == "static" or suffix == ".webp":
            frames = _extract_webp(row)
        elif fmt == "animated" or suffix == ".tgs":
            frames = _extract_tgs(row)
        elif fmt == "video" or suffix in (".webm", ".mp4", ".gif") or "video" in mime:
            frames = _extract_video(row, ffmpeg)
        else:
            # unknown — try Pillow as best-effort
            frames = _extract_webp(row)

        if not frames:
            db.mark_preview_failed(media_id, "no frames extracted")
            return

        for idx, (pos, path) in enumerate(frames):
            w, h = _img_size(path)
            db.upsert_frame(
                media_id=media_id,
                frame_index=idx,
                frame_pos=pos,
                preview_path=str(path),
                width=w,
                height=h,
            )

        db.mark_preview_ok(media_id)

    except Exception as exc:
        db.mark_preview_failed(media_id, str(exc)[:500])
