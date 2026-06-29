"""
app/bot.py — aiogram bot handlers (Phase G).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import (
    BotCommand, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup,
    Message, TelegramObject,
)

from app import config, db

log = logging.getLogger(__name__)
router = Router()

_sync_lock = asyncio.Lock()

# button_msg_id → (query, already_shown)
_more_state: dict[int, tuple[str, int]] = {}
_MORE_STATE_MAX = 200


async def _send_more_button(bot: Bot, chat_id: int, query: str, shown: int) -> None:
    markup = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🔍 More 10", callback_data="more"),
    ]])
    msg = await bot.send_message(chat_id, f"Shown: {shown}", reply_markup=markup)
    _more_state[msg.message_id] = (query, shown)
    if len(_more_state) > _MORE_STATE_MAX:
        _more_state.pop(next(iter(_more_state)))


class OwnerOnlyMiddleware(BaseMiddleware):
    """Silently drop all updates not originating from OWNER_USER_ID."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        if user is None or user.id != config.OWNER_USER_ID:
            return
        return await handler(event, data)


def _owner_only(message: Message) -> bool:
    return message.from_user is not None and message.from_user.id == config.OWNER_USER_ID


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    if not _owner_only(message):
        return
    await message.answer(
        "👋 StickerRadar\n\n"
        "Semantic search over your Telegram stickers and saved GIFs.\n\n"
        "First time? Run /sync to build the index — it downloads your stickers "
        "and builds a vector search index. Takes a few minutes.\n\n"
        "After that, just type anything to search:\n"
        "  <i>crying cat</i>  ·  <i>smug</i>  ·  <i>laughing</i>  ·  <i>facepalm</i>\n\n"
        "/sync — index stickers\n"
        "/status — index stats\n"
        "/help — tips",
        parse_mode="HTML",
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    if not _owner_only(message):
        return
    ocr_tip = (
        "\n\n🔤 <b>Text on stickers</b> — OCR is enabled. "
        "Search by words printed on stickers:\n"
        "  <i>lol</i>  ·  <i>nope</i>  ·  <i>omg</i>  ·  <i>why</i>"
        if config.OCR_ENABLED else ""
    )
    await message.answer(
        "💡 <b>How to search</b>\n\n"
        "Any text message = search query. Examples:\n\n"
        "<b>By mood / action:</b>\n"
        "  <i>laughing</i>  ·  <i>crying</i>  ·  <i>angry</i>  ·  <i>hug</i>\n\n"
        "<b>By description:</b>\n"
        "  <i>sad cat</i>  ·  <i>thumbs up</i>  ·  <i>anime girl surprised</i>"
        f"{ocr_tip}\n\n"
        "Results are ranked by visual similarity + text match. "
        "Up to 3 stickers per pack per query.",
        parse_mode="HTML",
    )


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    if not _owner_only(message):
        return
    counts = db.get_status_counts()
    ocr_line = f"  OCR text:     {counts['ocr']}\n" if config.OCR_ENABLED else ""
    await message.answer(
        f"Index status:\n"
        f"  Total media:  {counts['total']}\n"
        f"  Downloaded:   {counts['downloaded']}\n"
        f"  Previewed:    {counts['previewed']}\n"
        f"  Embedded:     {counts['embedded']}\n"
        f"{ocr_line}"
        f"  Failed:       {counts['failed']}"
    )


@router.message(Command("sync"))
async def cmd_sync(message: Message, bot: Bot, tg_client) -> None:
    if not _owner_only(message):
        return

    if _sync_lock.locked():
        await message.answer("Sync already in progress.")
        return

    status_msg = await message.answer("Sync started…")

    async def update(text: str) -> None:
        try:
            await status_msg.edit_text(text)
        except Exception:
            pass

    loop = asyncio.get_running_loop()
    async with _sync_lock:
        try:
            from app.scanner import _run_metadata_sync, _run_download, _run_preview, _run_ocr, _run_embed

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

            if config.OCR_ENABLED:
                await update(
                    f"Metadata: {c['total']} items\n"
                    f"Downloaded: {c['downloaded']}  Previewed: {c['previewed']}\n"
                    f"Running OCR…"
                )
                await loop.run_in_executor(None, _run_ocr, None)
                c = db.get_status_counts()

            await update(
                f"Metadata: {c['total']} items\n"
                f"Downloaded: {c['downloaded']}  Previewed: {c['previewed']}\n"
                f"Building embeddings…"
            )

            await loop.run_in_executor(None, _run_embed, None)
            c = db.get_status_counts()
            from app.search import invalidate_cache
            invalidate_cache()
            ocr_line = f"  OCR:      {c['ocr']}\n" if config.OCR_ENABLED else ""
            await update(
                f"Sync complete.\n"
                f"  Total:    {c['total']}\n"
                f"  Embedded: {c['embedded']}\n"
                f"{ocr_line}"
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
        results = await asyncio.get_running_loop().run_in_executor(
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
        return
    await _send_more_button(bot, message.chat.id, query, sent)


@router.callback_query(F.data == "more")
async def handle_more(callback: CallbackQuery, bot: Bot) -> None:
    state = _more_state.pop(callback.message.message_id, None)
    if not state:
        await callback.answer("Search again — the session expired.")
        return

    query, shown = state
    await callback.answer()

    try:
        await callback.message.delete()
    except Exception:
        pass

    from app import search as search_mod
    from app.sender import send_results as _send

    results = await asyncio.get_running_loop().run_in_executor(
        None, search_mod.search, query, shown + 10
    )
    batch = results[shown:]
    if not batch:
        await bot.send_message(callback.message.chat.id, "No more results.")
        return

    sent = await _send(bot, callback.message.chat.id, batch)
    await _send_more_button(bot, callback.message.chat.id, query, shown + sent)


async def register_commands(bot: Bot) -> None:
    """Register the command list shown in the Telegram chat input menu."""
    await bot.set_my_commands([
        BotCommand(command="sync",   description="Scan and index your stickers / GIFs"),
        BotCommand(command="status", description="Show index statistics"),
        BotCommand(command="help",   description="Usage examples and tips"),
    ])


def build_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.update.outer_middleware(OwnerOnlyMiddleware())
    dp.include_router(router)
    return dp
