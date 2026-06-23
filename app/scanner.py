"""
app/scanner.py — metadata sync, media downloads, and preview extraction.

Usage:
    python -m app.scanner --metadata-only [--limit N]
    python -m app.scanner --download [--limit N]
    python -m app.scanner --preview [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from app import config, db, media_store, preview
from app.tg_user import GifDoc, StickerDoc, TgUserClient


def _upsert_sticker(doc: StickerDoc, *, is_installed: bool = False,
                    is_favorite: bool = False, is_recent: bool = False) -> None:
    sm = doc.set_meta
    db.upsert_media_item(
        tg_document_id=doc.tg_document_id,
        access_hash=doc.access_hash,
        file_reference=doc.file_reference,
        media_kind="sticker",
        sticker_format=doc.sticker_format,
        mime_type=doc.mime_type,
        file_ext=doc.file_ext,
        emoji=doc.emoji,
        set_id=str(sm.id) if sm else None,
        set_access_hash=str(sm.access_hash) if sm else None,
        set_short_name=sm.short_name if sm else None,
        set_title=sm.title if sm else None,
        is_installed=is_installed,
        is_favorite=is_favorite,
        is_recent=is_recent,
        is_saved_gif=False,
    )


def _upsert_gif(doc: GifDoc) -> None:
    db.upsert_media_item(
        tg_document_id=doc.tg_document_id,
        access_hash=doc.access_hash,
        file_reference=doc.file_reference,
        media_kind="gif",
        sticker_format=None,
        mime_type=doc.mime_type,
        file_ext=doc.file_ext,
        emoji=None,
        set_id=None,
        set_access_hash=None,
        set_short_name=None,
        set_title=None,
        is_installed=False,
        is_favorite=False,
        is_recent=False,
        is_saved_gif=True,
    )


async def _run_metadata_sync(limit: int | None) -> None:
    config.ensure_dirs()
    db.get_conn()

    async with TgUserClient(config.TG_API_ID, config.TG_API_HASH, config.SESSION_PATH) as client:
        # ── installed packs ────────────────────────────────────────────────
        sets = await client.get_installed_sticker_sets()
        if limit is not None:
            sets = sets[:limit]

        total_sets = len(sets)
        sticker_count = 0

        for i, set_meta in enumerate(sets, start=1):
            print(f"Pack {i}/{total_sets}: {set_meta.title!r} ({set_meta.count} stickers)")
            docs = await client.get_sticker_set_documents(set_meta)
            for doc in docs:
                _upsert_sticker(doc, is_installed=True)
                sticker_count += 1

        # ── favorites ─────────────────────────────────────────────────────
        print("Syncing favorite stickers…")
        favs = await client.get_favorite_stickers()
        for doc in favs:
            _upsert_sticker(doc, is_favorite=True)
        print(f"  {len(favs)} favorites synced")

        # ── recents ───────────────────────────────────────────────────────
        print("Syncing recent stickers…")
        recents = await client.get_recent_stickers()
        for doc in recents:
            _upsert_sticker(doc, is_recent=True)
        print(f"  {len(recents)} recents synced")

        # ── saved GIFs ────────────────────────────────────────────────────
        print("Syncing saved GIFs…")
        gifs = await client.get_saved_gifs()
        for doc in gifs:
            _upsert_gif(doc)
        print(f"  {len(gifs)} GIFs synced")

    # ── summary ───────────────────────────────────────────────────────────
    counts = db.get_status_counts()
    print("\n── Sync complete ──────────────────────────────")
    print(f"  Total media items : {counts['total']}")
    print(f"  Downloaded        : {counts['downloaded']}")
    print(f"  Previewed         : {counts['previewed']}")
    print(f"  Embedded          : {counts['embedded']}")
    print(f"  Failed            : {counts['failed']}")
    print("───────────────────────────────────────────────")


async def _run_download(limit: int | None) -> None:
    config.ensure_dirs()
    db.get_conn()

    rows = db.list_pending_downloads(limit if limit is not None else 10 ** 9)
    total = len(rows)
    if not total:
        print("No pending downloads.")
        _print_counts()
        return

    sem = asyncio.Semaphore(config.SCAN_CONCURRENCY)
    async with TgUserClient(config.TG_API_ID, config.TG_API_HASH, config.SESSION_PATH) as client:
        tasks = [
            media_store.download_one(
                client, row, sem, label=f"File {i}/{total}: {row['tg_document_id']}"
            )
            for i, row in enumerate(rows, start=1)
        ]
        await asyncio.gather(*tasks)

    _print_counts()


def _run_embed(limit: int | None) -> None:
    config.ensure_dirs()
    db.get_conn()

    rows = db.list_pending_embeddings(limit if limit is not None else 10 ** 9)
    total = len(rows)
    if not total:
        print("No pending embeddings.")
        _print_counts()
        return

    from app.embeddings import Embedder
    from pathlib import Path

    embedder = Embedder()

    for i, row in enumerate(rows, start=1):
        media_id = row["id"]
        print(f"Embed {i}/{total}: {row['tg_document_id']}")
        try:
            frames = db.list_frames_for_media(media_id)
            if not frames:
                db.mark_embed_failed(media_id, "no frames")
                print("  SKIP: no frames")
                continue
            for frame in frames:
                path = Path(frame["preview_path"])
                if not path.exists():
                    raise FileNotFoundError(f"Preview missing: {path}")
                vec = embedder.embed_image(path)
                db.upsert_frame_embedding(
                    frame_id=frame["id"],
                    model_name=embedder.model_name,
                    dim=len(vec),
                    vector_bytes=vec.tobytes(),
                )
            db.mark_embed_ok(media_id)
        except Exception as exc:
            db.mark_embed_failed(media_id, str(exc)[:200])
            print(f"  ERROR: {exc}")

    _print_counts()


def _run_preview(limit: int | None) -> None:
    config.ensure_dirs()
    db.get_conn()
    ffmpeg = config.require_ffmpeg()

    rows = db.list_pending_previews(limit if limit is not None else 10 ** 9)
    total = len(rows)
    if not total:
        print("No pending previews.")
        _print_counts()
        return

    for i, row in enumerate(rows, start=1):
        label = f"Preview {i}/{total}: [{row['sticker_format'] or row['mime_type']}] {row['tg_document_id']}"
        print(label)
        preview.extract_previews(row, ffmpeg)

    _print_counts()


def _print_counts() -> None:
    counts = db.get_status_counts()
    print("\n── Status ──────────────────────────────────────")
    print(f"  Total         : {counts['total']}")
    print(f"  Downloaded    : {counts['downloaded']}")
    print(f"  Previewed     : {counts['previewed']}")
    print(f"  Embedded      : {counts['embedded']}")
    print(f"  Failed        : {counts['failed']}")
    print("────────────────────────────────────────────────")


def main() -> None:
    parser = argparse.ArgumentParser(description="StickerRadar scanner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--metadata-only",
        action="store_true",
        help="Sync metadata only (no file downloads)",
    )
    group.add_argument(
        "--download",
        action="store_true",
        help="Download pending media files",
    )
    group.add_argument(
        "--preview",
        action="store_true",
        help="Extract preview frames for downloaded media",
    )
    group.add_argument(
        "--embed",
        action="store_true",
        help="Compute CLIP embeddings for preview frames",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N items",
    )
    args = parser.parse_args()

    try:
        if args.metadata_only:
            asyncio.run(_run_metadata_sync(args.limit))
        elif args.download:
            asyncio.run(_run_download(args.limit))
        elif args.preview:
            _run_preview(args.limit)
        else:
            _run_embed(args.limit)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    finally:
        db.close()


if __name__ == "__main__":
    main()
