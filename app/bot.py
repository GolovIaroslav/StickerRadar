"""
app/bot.py — aiogram bot handlers (Phase G).
"""
from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Message

from app import config, db

log = logging.getLogger(__name__)
router = Router()


def _owner_only(message: Message) -> bool:
    return message.from_user is not None and message.from_user.id == config.OWNER_USER_ID


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    if not _owner_only(message):
        return
    await message.answer(
        "StickerRadar\n\n"
        "Send me a text query and I'll find matching stickers from your collection.\n\n"
        "Commands:\n"
        "  /sync   — scan and index your stickers\n"
        "  /status — show index stats\n"
        "  /help   — usage examples\n\n"
        "Run /sync first to build the index."
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    if not _owner_only(message):
        return
    await message.answer(
        "Example queries:\n"
        "  laughing\n"
        "  tired\n"
        "  happy\n"
        "  angry\n"
        "  hug\n"
        "  sad cat\n\n"
        "Any text message is treated as a search query.\n"
        "Queries in any language work — quality depends on the embedding model."
    )


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    if not _owner_only(message):
        return
    counts = db.get_status_counts()
    await message.answer(
        f"Index status:\n"
        f"  Total media:  {counts['total']}\n"
        f"  Downloaded:   {counts['downloaded']}\n"
        f"  Previewed:    {counts['previewed']}\n"
        f"  Embedded:     {counts['embedded']}\n"
        f"  Failed:       {counts['failed']}"
    )


@router.message(Command("sync"))
async def cmd_sync(message: Message, bot: Bot, tg_client) -> None:
    if not _owner_only(message):
        return

    status_msg = await message.answer("Sync started…")

    async def update(text: str) -> None:
        try:
            await status_msg.edit_text(text)
        except Exception:
            pass

    loop = asyncio.get_event_loop()
    try:
        from app.scanner import _run_metadata_sync, _run_download, _run_preview, _run_embed

        await update("Syncing metadata…")
        await _run_metadata_sync(None, client=tg_client)
        c = db.get_status_counts()
        await update(f"Metadata: {c['total']} items\nDownloading files…")

        await _run_download(None, client=tg_client)
        c = db.get_status_counts()

        # Flag items that lack embeddings for the active model (handles model change).
        db.mark_items_for_model(config.MODEL_NAME)

        await update(
            f"Metadata: {c['total']} items\n"
            f"Downloaded: {c['downloaded']}\n"
            f"Extracting previews…"
        )

        await loop.run_in_executor(None, _run_preview, None)
        c = db.get_status_counts()
        await update(
            f"Metadata: {c['total']} items\n"
            f"Downloaded: {c['downloaded']}  Previewed: {c['previewed']}\n"
            f"Building embeddings…"
        )

        await loop.run_in_executor(None, _run_embed, None)
        c = db.get_status_counts()
        await update(
            f"Sync complete.\n"
            f"  Total:    {c['total']}\n"
            f"  Embedded: {c['embedded']}\n"
            f"  Failed:   {c['failed']}"
        )
    except Exception as e:
        log.exception("Sync failed")
        await update(f"Sync failed: {e}")


@router.message()
async def handle_query(message: Message, bot: Bot) -> None:
    if not _owner_only(message):
        return
    if not message.text:
        return

    query = message.text.strip()
    if not query:
        return

    counts = db.get_status_counts()
    if counts["embedded"] == 0:
        await message.answer("Index is empty. Run /sync first.")
        return

    from app import search as search_mod
    from app.sender import send_results

    try:
        results = await asyncio.get_event_loop().run_in_executor(
            None, search_mod.search, query
        )
    except Exception as e:
        log.exception("Search error")
        await message.answer(f"Search error: {e}")
        return

    if not results:
        await message.answer("No results found.")
        return

    sent = await send_results(bot, message.chat.id, results)
    if sent == 0:
        await message.answer("Found results but failed to send them.")


def build_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.include_router(router)
    return dp
