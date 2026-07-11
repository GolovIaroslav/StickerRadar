"""Audit hybrid OCR on a reproducible, isolated sample of local media.

The script never updates app.sqlite OCR rows. It renders one temporary audit frame
per selected media directly from the local original and writes JSON/Markdown OCR
transcripts including the final backend for every result.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from app import config
from app.ocr import OCRFrameResult, ocr_frames_with_provenance


def _media_rows(conn: sqlite3.Connection, kind: str) -> list[sqlite3.Row]:
    extra = (
        "AND m.media_kind = 'sticker' AND m.sticker_format IN ('static', 'video')"
        if kind == "sticker" else "AND m.media_kind = 'gif'"
    )
    return conn.execute(
        f"""
        SELECT m.id, m.tg_document_id, m.media_kind, m.sticker_format, m.local_path,
               COALESCE(NULLIF(m.set_short_name, ''), '(без пака)') AS pack,
               COALESCE(NULLIF(m.set_title, ''), '(без названия)') AS pack_title
        FROM media_items AS m
        WHERE m.preview_status = 'ok'
          AND m.local_path IS NOT NULL
          {extra}
        ORDER BY m.id
        """
    ).fetchall()


def _existing(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    return [row for row in rows if Path(row["local_path"]).exists()]


def _stratified_stickers(rows: list[sqlite3.Row], count: int, rng: random.Random) -> list[sqlite3.Row]:
    by_pack: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        by_pack[row["pack"]].append(row)
    packs = list(by_pack)
    rng.shuffle(packs)
    for items in by_pack.values():
        rng.shuffle(items)
    selected: list[sqlite3.Row] = []
    offset = 0
    while len(selected) < count:
        added = False
        for pack in packs:
            if offset < len(by_pack[pack]):
                selected.append(by_pack[pack][offset])
                added = True
                if len(selected) == count:
                    return selected
        if not added:
            break
        offset += 1
    return selected


def _render_preview(row: sqlite3.Row, *, position: float, output_dir: Path) -> Path:
    """Create one audit-only PNG frame; it does not touch app preview paths/DB."""
    source = Path(row["local_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{row['id']}.png"
    if row["sticker_format"] == "static" or source.suffix.lower() == ".webp":
        with Image.open(source) as image:
            if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                canvas = Image.new("RGB", image.size, (255, 255, 255))
                rgba = image.convert("RGBA")
                canvas.paste(rgba, mask=rgba.getchannel("A"))
                canvas.save(out)
            else:
                image.convert("RGB").save(out)
        return out

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to render a video sticker/GIF audit frame")
    duration = 0.0
    if ffprobe:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(source)],
            capture_output=True, text=True, timeout=15, check=False,
        )
        try:
            duration = float(probe.stdout.strip())
        except ValueError:
            duration = 0.0
    timestamp = duration * position if duration else 0.0
    command = [ffmpeg, "-y", "-ss", str(timestamp), "-i", str(source), "-frames:v", "1", "-vf", "scale=512:512:force_original_aspect_ratio=decrease", str(out)]
    rendered = subprocess.run(command, capture_output=True, timeout=30, check=False)
    if rendered.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        # Some Telegram video stickers expose a near-zero container duration. In
        # that case seeking can land after EOF although the first frame decodes.
        rendered = subprocess.run(
            [ffmpeg, "-y", "-i", str(source), "-frames:v", "1", "-vf", "scale=512:512:force_original_aspect_ratio=decrease", str(out)],
            capture_output=True, timeout=30, check=False,
        )
    if rendered.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        error = rendered.stderr.decode(errors="replace").strip()[-500:]
        raise RuntimeError(f"ffmpeg could not render {source.name}: {error}")
    return out


def _result_row(row: sqlite3.Row, preview_path: Path, position: float, result: OCRFrameResult) -> dict[str, Any]:
    return {
        "media_id": row["id"], "tg_document_id": row["tg_document_id"],
        "kind": row["media_kind"], "format": row["sticker_format"] or "gif",
        "pack": row["pack"], "pack_title": row["pack_title"], "frame_position": position,
        "source_path": row["local_path"], "preview_path": str(preview_path),
        "backend": result.backend, "confidence": round(result.confidence, 4),
        "raw_text": result.raw_text, "norm_text": result.norm_text,
    }


def _short(value: str, limit: int = 110) -> str:
    value = value.replace("\n", " ↵ ").replace("|", "\\|")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _write_markdown(path: Path, audit: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = audit["rows"]
    counts = Counter(row["backend"] for row in rows)
    kinds = Counter(row["kind"] for row in rows)
    lines = [
        "# Hybrid OCR sample audit", "",
        f"- Generated: `{audit['generated_at']}`", f"- Seed: `{audit['seed']}`",
        f"- Sample: **{len(rows)} media / one rendered frame per media**",
        "- Selection: sticker media are sampled round-robin across shuffled packs; GIF media are uniformly shuffled.",
        "- This is isolated: audit preview files and this report do not update OCR records in `app.sqlite`.", "",
        "## Actual final OCR source", "", "| Backend | Frames | Share |", "|---|---:|---:|",
    ]
    for backend, count in sorted(counts.items()):
        lines.append(f"| `{backend}` | {count} | {count / len(rows):.1%} |")
    lines.extend(["", "## Sample composition", "", "| Kind | Media |", "|---|---:|"])
    for kind, count in sorted(kinds.items()):
        lines.append(f"| {kind} | {count} |")
    lines.extend(["", "## Per-frame OCR transcript", "", "| # | Kind | Pack | Backend | Conf. | Raw OCR text |", "|---:|---|---|---|---:|---|"])
    for number, row in enumerate(rows, start=1):
        raw = _short(row["raw_text"]) if row["raw_text"].strip() else "*(empty)*"
        lines.append(f"| {number} | {row['kind']} | {_short(row['pack'], 36)} | `{row['backend']}` | {row['confidence']:.3f} | {raw} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sticker-count", type=int, default=100)
    parser.add_argument("--gif-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output-dir", type=Path, default=config.EVAL_DIR)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    stickers = _stratified_stickers(_existing(_media_rows(conn, "sticker")), args.sticker_count, rng)
    gifs = _existing(_media_rows(conn, "gif"))
    rng.shuffle(gifs)
    rows = stickers + gifs[:args.gif_count]
    conn.close()
    if not rows:
        raise SystemExit("No local media available for OCR audit")

    frame_dir = args.output_dir / f"hybrid-ocr-audit-frames-{args.seed}"
    prepared: list[tuple[sqlite3.Row, Path, float]] = []
    for row in rows:
        position = rng.uniform(0.1, 0.9) if row["media_kind"] == "gif" or row["sticker_format"] == "video" else 0.0
        prepared.append((row, _render_preview(row, position=position, output_dir=frame_dir), position))

    output_rows: list[dict[str, Any]] = []
    for start in range(0, len(prepared), args.batch_size):
        batch = prepared[start : start + args.batch_size]
        results = ocr_frames_with_provenance([path for _, path, _ in batch])
        output_rows.extend(_result_row(row, path, position, result) for (row, path, position), result in zip(batch, results, strict=True))
        print(f"OCR audit {min(start + len(batch), len(prepared))}/{len(prepared)}")

    now = datetime.now(UTC)
    audit = {"generated_at": now.isoformat(), "seed": args.seed, "config_backend": config.OCR_BACKEND, "prefilter_backend": config.OCR_GLM_PREFILTER_BACKEND, "rows": output_rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"hybrid-ocr-audit-{now:%Y%m%dT%H%M%SZ}"
    json_path, markdown_path = args.output_dir / f"{stem}.json", args.output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(markdown_path, audit)
    counts = Counter(row["backend"] for row in output_rows)
    print(f"Wrote {markdown_path}\nWrote {json_path}")
    print("Actual source counts: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))


if __name__ == "__main__":
    main()
