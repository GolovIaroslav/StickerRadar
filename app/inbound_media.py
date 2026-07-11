"""Temporary frame extraction and multimodal analysis for forwarded media.

Nothing in this module writes an incoming file to the StickerRadar SQLite index.
It is intentionally an ephemeral query against the existing library.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image

from app import config
from app.ocr import ocr_frames
from app.replies import rank_reply_candidates


def _flatten(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        background = Image.new("RGB", image.size, (255, 255, 255))
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def _positions(count: int) -> list[float]:
    return [0.5] if count <= 1 else [(i + 0.5) / count for i in range(count)]


def _pillow_frames(source: Path, destination: Path) -> list[Path]:
    with Image.open(source) as image:
        total = int(getattr(image, "n_frames", 1))
        indexes = [0] if total <= 1 else [min(total - 1, int(total * pos)) for pos in _positions(config.FRAME_COUNT)]
        paths: list[Path] = []
        for index, frame_index in enumerate(dict.fromkeys(indexes)):
            image.seek(frame_index)
            out = destination / f"frame_{index:03d}.png"
            _flatten(image).save(out)
            paths.append(out)
        return paths


def _video_frames(source: Path, destination: Path) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to read an incoming video sticker or GIF")
    ffprobe = shutil.which("ffprobe")
    duration = 0.0
    if ffprobe:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(source)],
            capture_output=True, text=True, timeout=15,
        )
        try:
            duration = float(probe.stdout.strip())
        except ValueError:
            pass
    paths: list[Path] = []
    for index, position in enumerate(_positions(config.FRAME_COUNT)):
        out = destination / f"frame_{index:03d}.png"
        command = [ffmpeg, "-y"]
        if duration:
            command += ["-ss", str(duration * position)]
        command += ["-i", str(source), "-frames:v", "1", "-vf", "scale=512:512:force_original_aspect_ratio=decrease", str(out)]
        completed = subprocess.run(command, capture_output=True, timeout=30)
        if completed.returncode == 0 and out.exists() and out.stat().st_size:
            paths.append(out)
    if not paths:
        raise RuntimeError(f"could not extract frames from {source.name}")
    return paths


def _tgs_frames(source: Path, destination: Path) -> list[Path]:
    try:
        import rlottie_python as rl
        animation = rl.LottieAnimation.from_tgs(str(source))
        total = animation.lottie_animation_get_totalframe()
        width, height = animation.lottie_animation_get_size()
        if total <= 0:
            raise RuntimeError("the TGS animation has no frames")
        paths: list[Path] = []
        for index, position in enumerate(_positions(config.FRAME_COUNT)):
            buffer = animation.lottie_animation_render(min(int(position * total), total - 1), width, height)
            image = Image.frombuffer("RGBA", (width, height), buffer, "raw", "RGBA", 0, 1)
            out = destination / f"frame_{index:03d}.png"
            _flatten(image).save(out)
            paths.append(out)
        return paths
    except ImportError as exc:
        raise RuntimeError("TGS support requires rlottie-python, already used by the indexer") from exc


def extract_inbound_frames(source: Path, destination: Path) -> list[Path]:
    """Render incoming WEBP/TGS/video/GIF to temporary RGB PNG frames."""
    destination.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix == ".tgs":
        return _tgs_frames(source, destination)
    if suffix in {".webm", ".mp4"}:
        return _video_frames(source, destination)
    # Pillow handles static WEBP, animated WEBP and GIF without ffmpeg.
    return _pillow_frames(source, destination)


def analyze_incoming_media(source: Path, workdir: Path, top_k: int = 10):
    """Use image semantics + OCR-text semantics to retrieve strong reply media."""
    frames = extract_inbound_frames(source, workdir / "frames")
    from app.embeddings import get_shared_embedder

    embedder = get_shared_embedder()
    image_vectors = embedder.embed_images(frames)
    text = ""
    if config.OCR_ENABLED:
        # Repeat text across frames is unhelpful; preserve first occurrence order.
        text = " ".join(dict.fromkeys(raw.strip() for raw, _norm, _confidence in ocr_frames(frames) if raw.strip()))
    intent = {}
    if config.VLM_ENABLED:
        from app.reply_intent import infer_reply_intent
        intent = infer_reply_intent(frames, workdir)
    return rank_reply_candidates(image_vectors=image_vectors, ocr_text=text, top_k=top_k, intent=intent), text
