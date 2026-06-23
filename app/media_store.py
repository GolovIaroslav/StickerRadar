from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from pathlib import Path

from telethon.errors import FloodWaitError

from app import config, db
from app.errors import DownloadError


def local_path(
    media_kind: str,
    sticker_format: str | None,
    tg_document_id: str,
    file_ext: str,
) -> Path:
    if media_kind == "gif":
        return config.MEDIA_DIR / "gifs" / f"{tg_document_id}{file_ext}"
    fmt = sticker_format if sticker_format in ("static", "animated", "video") else "static"
    return config.MEDIA_DIR / "stickers" / fmt / f"{tg_document_id}{file_ext}"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


async def download_one(
    client,
    row: sqlite3.Row,
    sem: asyncio.Semaphore,
    label: str = "",
) -> None:
    media_id: int = row["id"]
    dest = local_path(
        row["media_kind"],
        row["sticker_format"],
        row["tg_document_id"],
        row["file_ext"],
    )

    async with sem:
        if label:
            print(label)
        for attempt in range(2):
            try:
                await client.download_document(
                    tg_document_id=row["tg_document_id"],
                    access_hash=row["access_hash"],
                    file_reference=bytes(row["file_reference"]),
                    dest_path=dest,
                )
                sha = compute_sha256(dest)
                size = dest.stat().st_size
                db.mark_download_ok(media_id, str(dest), sha, size)
                return
            except DownloadError as exc:
                flood = exc.__cause__
                if isinstance(flood, FloodWaitError) and attempt == 0:
                    await asyncio.sleep(flood.seconds + 1)
                    continue
                db.mark_download_failed(media_id, str(exc))
                return
