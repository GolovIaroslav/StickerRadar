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
from concurrent.futures import ThreadPoolExecutor, as_completed

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


async def _run_metadata_sync(limit: int | None, client: TgUserClient | None = None) -> None:
    config.ensure_dirs()
    db.get_conn()

    async def _sync(c: TgUserClient) -> None:
        installed_ids: set[str] = set()
        favorite_ids: set[str] = set()
        recent_ids: set[str] = set()
        saved_gif_ids: set[str] = set()

        sets = await c.get_installed_sticker_sets()
        if limit is not None:
            sets = sets[:limit]
        for i, set_meta in enumerate(sets, start=1):
            print(f"Pack {i}/{len(sets)}: {set_meta.title!r} ({set_meta.count} stickers)")
            docs = await c.get_sticker_set_documents(set_meta)
            for doc in docs:
                _upsert_sticker(doc, is_installed=True)
                installed_ids.add(doc.tg_document_id)

        print("Syncing favorite stickers…")
        favs = await c.get_favorite_stickers()
        for doc in favs:
            _upsert_sticker(doc, is_favorite=True)
            favorite_ids.add(doc.tg_document_id)
        print(f"  {len(favs)} favorites synced")

        print("Syncing recent stickers…")
        recents = await c.get_recent_stickers()
        for doc in recents:
            _upsert_sticker(doc, is_recent=True)
            recent_ids.add(doc.tg_document_id)
        print(f"  {len(recents)} recents synced")

        print("Syncing saved GIFs…")
        gifs = await c.get_saved_gifs()
        for doc in gifs:
            _upsert_gif(doc)
            saved_gif_ids.add(doc.tg_document_id)
        print(f"  {len(gifs)} GIFs synced")

        db.clear_stale_source_flags(installed_ids, favorite_ids, recent_ids, saved_gif_ids)

    if client is not None:
        await _sync(client)
    else:
        async with TgUserClient(config.TG_API_ID, config.TG_API_HASH, config.SESSION_PATH) as c:
            await _sync(c)

    counts = db.get_status_counts()
    print("\n── Sync complete ──────────────────────────────")
    print(f"  Total media items : {counts['total']}")
    print(f"  Downloaded        : {counts['downloaded']}")
    print(f"  Previewed         : {counts['previewed']}")
    print(f"  Embedded          : {counts['embedded']}")
    print(f"  Failed            : {counts['failed']}")
    print("───────────────────────────────────────────────")


async def _run_download(limit: int | None, client: TgUserClient | None = None) -> None:
    config.ensure_dirs()
    db.get_conn()

    rows = db.list_pending_downloads(limit if limit is not None else 10 ** 9)
    total = len(rows)
    if not total:
        print("No pending downloads.")
        _print_counts()
        return

    sem = asyncio.Semaphore(config.SCAN_CONCURRENCY)

    async def _download(c: TgUserClient) -> None:
        tasks = [
            media_store.download_one(
                c, row, sem, label=f"File {i}/{total}: {row['tg_document_id']}"
            )
            for i, row in enumerate(rows, start=1)
        ]
        await asyncio.gather(*tasks)

    if client is not None:
        await _download(client)
    else:
        async with TgUserClient(config.TG_API_ID, config.TG_API_HASH, config.SESSION_PATH) as c:
            await _download(c)

    _print_counts()


def _run_ocr(limit: int | None) -> None:
    from pathlib import Path
    from app import ocr as ocr_mod

    config.ensure_dirs()
    db.get_conn()

    if not ocr_mod.is_available():
        print("OCR skipped — easyocr not installed. To enable:  uv add easyocr")
        return

    rows = db.list_pending_ocr(limit if limit is not None else 10 ** 9)
    total = len(rows)
    if not total:
        print("No pending OCR.")
        _print_counts()
        return

    print(f"Running OCR on {total} items …")
    done = skipped = 0
    for i, row in enumerate(rows, start=1):
        media_id = row["id"]
        frames = db.list_frames_for_media(media_id)
        if not frames:
            db.mark_ocr_ok(media_id, "")
            skipped += 1
            continue

        paths = [Path(f["preview_path"]) for f in frames]
        existing = [p for p in paths if p.exists()]
        if not existing:
            # Previews already deleted (item was embedded before OCR was enabled).
            db.mark_ocr_ok(media_id, "")
            skipped += 1
            continue

        try:
            results = ocr_mod.ocr_frames(existing)
            norm_parts: list[str] = []
            for frame, (raw, norm, conf) in zip(frames, results):
                if raw:
                    db.upsert_frame_ocr(frame["id"], raw, norm, conf)
                if norm.strip():
                    norm_parts.append(norm)
            db.mark_ocr_ok(media_id, " ".join(norm_parts) if norm_parts else "")
            done += 1
        except Exception as exc:
            db.mark_ocr_failed(media_id, str(exc)[:200])
            print(f"  OCR ERROR {media_id}: {exc}")
            continue

        if i % 100 == 0 or i == total:
            print(f"  OCR {i}/{total}  (text found: {done}, no-preview: {skipped}) …")

    _print_counts()



def _run_embed(limit: int | None, keep_previews: bool = False) -> None:
    config.ensure_dirs()
    db.get_conn()

    rows = db.list_pending_embeddings(limit if limit is not None else 10 ** 9)
    total = len(rows)
    if not total:
        print("No pending embeddings.")
        _print_counts()
        return

    from app.embeddings import get_shared_embedder
    from app.preview import delete_previews
    from pathlib import Path

    embedder = get_shared_embedder()
    try:
        embedder.embed_text("warmup")
    except Exception as exc:
        print(f"  Model failed to load — aborting embed run:\n  {exc}")
        return

    print(f"Embedding {total} items …")

    done = 0
    freed = 0
    for row in rows:
        media_id = row["id"]
        frames = db.list_frames_for_media(media_id)
        if not frames:
            db.mark_embed_failed(media_id, "no frames")
            continue

        paths = [Path(f["preview_path"]) for f in frames]
        if not all(p.exists() for p in paths):
            db.mark_embed_failed(media_id, "preview frames missing (re-run preview stage)")
            continue

        try:
            vecs = embedder.embed_images(paths)
            for frame, vec in zip(frames, vecs):
                db.upsert_frame_embedding(
                    frame_id=frame["id"],
                    model_name=embedder.model_name,
                    dim=len(vec),
                    vector_bytes=vec.tobytes(),
                )
            db.mark_embed_ok(media_id)
            if not keep_previews:
                freed += delete_previews(media_id)
        except Exception as exc:
            db.mark_embed_failed(media_id, str(exc)[:200])
            print(f"  ERROR {media_id}: {exc}")
            continue

        done += 1
        if done % 50 == 0 or done == total:
            print(f"  Embedded {done}/{total} …")

    if freed and not keep_previews:
        print(f"  Freed {freed / 1024 / 1024:.1f} MB of preview frames.")

    _print_counts()


def _run_preview(limit: int | None) -> None:
    config.ensure_dirs()
    db.get_conn()
    ffmpeg = config.require_ffmpeg()

    # Auto-detect FRAME_COUNT change and force a full reindex if it happened.
    current_fc = str(config.FRAME_COUNT)
    stored_fc = db.get_app_state("frame_count")
    if stored_fc is not None and stored_fc != current_fc:
        print(
            f"FRAME_COUNT changed {stored_fc} → {current_fc}: forcing full reindex.\n"
            "  All preview frames and embeddings will be regenerated."
        )
        db.force_reindex()

    rows = db.list_pending_previews(limit if limit is not None else 10 ** 9)
    total = len(rows)
    if not total:
        print("No pending previews.")
        db.set_app_state("frame_count", current_fc)
        _print_counts()
        return

    done = 0
    with ThreadPoolExecutor(max_workers=config.SCAN_CONCURRENCY) as pool:
        futures = {
            pool.submit(preview.extract_previews, row, ffmpeg): (i, row)
            for i, row in enumerate(rows, start=1)
        }
        for fut in as_completed(futures):
            i, row = futures[fut]
            done += 1
            fmt = row["sticker_format"] or row["mime_type"]
            print(f"Preview {done}/{total}: [{fmt}] {row['tg_document_id']}")
            exc = fut.exception()
            if exc:
                print(f"  ERROR: {exc}")

    db.set_app_state("frame_count", current_fc)
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
        "--ocr",
        action="store_true",
        help="Run OCR on preview frames (needs: uv add easyocr)",
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
        elif args.ocr:
            _run_ocr(args.limit)
        else:
            _run_embed(args.limit)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    finally:
        db.close()


if __name__ == "__main__":
    main()
