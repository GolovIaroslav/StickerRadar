"""
app/bot.py — aiogram bot handlers (Phase G).
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from html import escape
from pathlib import Path
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import (
    Animation, BotCommand, CallbackQuery, Document, InlineKeyboardButton,
    InlineKeyboardMarkup, Message, Sticker, TelegramObject,
)

from app import config, db

log = logging.getLogger(__name__)
router = Router()

_sync_lock = asyncio.Lock()

# button_msg_id → (query, candidate_cursor, already_shown)
_more_state: dict[int, tuple[str, int, int]] = {}
_MORE_STATE_MAX = 200


async def _send_more_button(bot: Bot, chat_id: int, query: str, cursor: int, shown: int) -> None:
    markup = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="🔍 More 10", callback_data="more"),
    ]])
    msg = await bot.send_message(chat_id, f"Shown: {shown}", reply_markup=markup)
    _more_state[msg.message_id] = (query, cursor, shown)
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
    active_embeddings = db.count_embeddings_for_model(config.MODEL_NAME)
    ocr_line = f"  OCR text:     {counts['ocr']}\n" if config.OCR_ENABLED else ""
    await message.answer(
        f"Index status:\n"
        f"  Total media:  {counts['total']}\n"
        f"  Downloaded:   {counts['downloaded']}\n"
        f"  Previewed:    {counts['previewed']}\n"
        f"  Embedded:     {counts['embedded']}\n"
        f"  Active model: {active_embeddings} frames for {config.MODEL_NAME}\n"
        f"{ocr_line}"
        f"  Failed:       {counts['failed']}"
    )


@router.message(Command("models"))
async def cmd_models(message: Message) -> None:
    if not _owner_only(message):
        return
    from app.model_artifacts import ARTIFACTS, artifact_ready
    lines = ["<b>Model artifacts (local-only runtime)</b>"]
    for artifact in ARTIFACTS:
        marker = "✅" if artifact_ready(artifact) else "⬜"
        lines.append(f"{marker} <code>{artifact.key}</code> — {artifact.size} — {artifact.license}")
    lines.append("\nNothing is downloaded automatically. Install explicitly with the CLI.")
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("unload"))
async def cmd_unload(message: Message) -> None:
    if not _owner_only(message):
        return

    def unload_models() -> tuple[bool, bool]:
        from app.embeddings import unload_shared_embedder
        from app.text_embed import unload_shared_text_embedder

        return unload_shared_embedder(), unload_shared_text_embedder()

    image_unloaded, text_unloaded = await asyncio.get_running_loop().run_in_executor(
        None, unload_models
    )
    unloaded = []
    if image_unloaded:
        unloaded.append("image embedding model")
    if text_unloaded:
        unloaded.append("text embedding server")
    await message.answer(
        "🧹 Unloaded from RAM/VRAM: " + ", ".join(unloaded) + "."
        if unloaded else "No StickerRadar model was loaded — nothing to unload."
    )


@router.message(Command("pipeline"))
async def cmd_pipeline(message: Message) -> None:
    if not _owner_only(message):
        return
    args = (message.text or "").split()[1:]
    for item in args:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key, value = key.lower(), value.lower()
        if key == "ocr" and value in {"off", "none"}:
            config.OCR_ENABLED, config.OCR_BACKEND = False, "off"
        elif key == "vlm" and value in {"off", "none"}:
            config.VLM_ENABLED, config.VLM_BACKEND = False, "none"
        elif key == "vlm" and value in {"internvl", "qwen"}:
            config.VLM_ENABLED, config.VLM_BACKEND = True, value
    await message.answer(
        "<b>Pipeline</b>\n"
        f"OCR: <code>{config.OCR_BACKEND}</code> ({'on' if config.OCR_ENABLED else 'off'})\n"
        f"VLM: <code>{config.VLM_BACKEND}</code> ({'on' if config.VLM_ENABLED else 'off'})\n"
        "\nExamples: <code>/pipeline ocr=off vlm=off</code>, <code>/pipeline vlm=qwen</code>\n"
        "Runtime changes are temporary; install artifacts explicitly via CLI.",
        parse_mode="HTML",
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
            from app.scanner import (
                _run_download,
                _run_embed,
                _run_metadata_sync,
                _run_ocr,
                _run_preview,
                _run_text_embed_backfill,
            )

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
            if config.TEXT_EMBED_ENABLED:
                await loop.run_in_executor(None, _run_text_embed_backfill, None)
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


def _is_gif_document(document: Document | None) -> bool:
    if document is None:
        return False
    return document.mime_type == "image/gif" or (document.file_name or "").lower().endswith(".gif")


def _select_visual_target(message: Message) -> Sticker | Animation | Document | None:
    """Prefer attached media; a text reply may target sticker, animation, or GIF document."""
    targets = (message, getattr(message, "reply_to_message", None))
    for target in targets:
        if target is None:
            continue
        sticker = getattr(target, "sticker", None)
        if sticker:
            return sticker
        animation = getattr(target, "animation", None)
        if animation:
            return animation
        document = getattr(target, "document", None)
        if _is_gif_document(document):
            return document
    return None


def _inbound_suffix(media: Sticker | Animation | Document) -> str:
    """Choose a real extension so the temporary frame renderer identifies Telegram media."""
    if hasattr(media, "is_animated"):
        if media.is_animated:
            return ".tgs"
        if media.is_video:
            return ".webm"
        return ".webp"
    filename = media.file_name
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix:
            return suffix
    return ".gif"


@router.message(F.sticker | F.animation | F.document | F.reply_to_message.sticker | F.reply_to_message.animation | F.reply_to_message.document)
async def handle_reply_media(message: Message, bot: Bot) -> None:
    """Return up to ten diverse local-library reactions to incoming or replied-to media."""
    if not _owner_only(message):
        return
    media = _select_visual_target(message)
    if media is None:
        return
    if db.count_embeddings_for_model(config.MODEL_NAME) == 0:
        await message.answer("Index for the active model is empty. Run /sync first.")
        return

    status = await message.answer("Сканирую смысл, кадры и текст…")
    try:
        with tempfile.TemporaryDirectory(prefix="stickerradar-reply-") as temp:
            incoming = Path(temp) / f"incoming{_inbound_suffix(media)}"
            await bot.download(media, destination=incoming)
            from app.inbound_media import analyze_incoming_media
            results, ocr_text = await asyncio.get_running_loop().run_in_executor(
                None, analyze_incoming_media, incoming, Path(temp), 10
            )
        if not results:
            await status.edit_text("Не нашёл достаточно сильных ответочек в индексированной коллекции.")
            return
        from app.sender import send_results
        sent = await send_results(bot, message.chat.id, results)
        if sent == 0:
            await status.edit_text("Ответочки нашлись, но Telegram не смог их отправить.")
            return
        text_note = f" Текст: <code>{escape(ocr_text[:160])}</code>" if ocr_text else ""
        await status.edit_text(
            f"Готово: {sent} разных ответочек. Я искал реакцию, а не дубликаты сюжета.{text_note}",
            parse_mode="HTML",
        )
    except Exception as exc:
        log.exception("Reply-media analysis error")
        await status.edit_text(f"Не удалось разобрать медиа: {exc}")


async def _send_query_page(
    bot: Bot,
    chat_id: int,
    query: str,
    cursor: int = 0,
) -> tuple[int, int, bool]:
    """Send one full page, expanding the candidate window around skipped items."""
    from app import search as search_mod
    from app.sender import select_sendable_page, send_results

    page_size = config.TOP_K
    candidate_limit = max(page_size, cursor + page_size)
    sent_total = 0
    next_cursor = cursor
    results: list = []
    while sent_total < page_size:
        results = await asyncio.get_running_loop().run_in_executor(
            None, search_mod.search, query, candidate_limit
        )
        # Text search must not hide the best match merely because it was
        # sent in a previous query. The candidate cursor prevents repeats
        # between pages of this query.
        recent_ids: set[int] = set()
        recent_packs: set[str] = set()
        batch, next_cursor = select_sendable_page(
            results,
            cursor=next_cursor,
            page_size=page_size - sent_total,
            recent_media_ids=recent_ids,
            recent_packs=recent_packs,
        )
        if batch:
            sent_total += await send_results(
                bot, chat_id, batch, exclude_recent=False
            )
        log.info(
            "Text search page query=%r candidate_limit=%d selected=%d sent_total=%d cursor=%d",
            query,
            candidate_limit,
            len(batch),
            sent_total,
            next_cursor,
        )
        if sent_total >= page_size:
            break
        if next_cursor >= len(results) and len(results) >= candidate_limit:
            candidate_limit += page_size
            continue
        break

    has_more = next_cursor < len(results) or len(results) >= candidate_limit
    return sent_total, next_cursor, has_more


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
    active_embeddings = db.count_embeddings_for_model(config.MODEL_NAME)
    if counts["embedded"] == 0 or active_embeddings == 0:
        await message.answer(
            "Index for the active model is empty. Run /sync first.\n"
            f"Active model: {config.MODEL_NAME}"
        )
        return

    try:
        sent, cursor, has_more = await _send_query_page(bot, message.chat.id, query)
    except Exception as e:
        log.exception("Search error")
        await message.answer(f"Search error: {e}")
        return

    if sent == 0:
        await message.answer("Found results but failed to send them.")
        return
    if has_more:
        await _send_more_button(bot, message.chat.id, query, cursor, sent)


@router.callback_query(F.data == "more")
async def handle_more(callback: CallbackQuery, bot: Bot) -> None:
    state = _more_state.pop(callback.message.message_id, None)
    if not state:
        await callback.answer("Search again — the session expired.")
        return

    query, cursor, shown = state
    await callback.answer()

    try:
        await callback.message.delete()
    except Exception:
        pass

    try:
        sent, next_cursor, has_more = await _send_query_page(
            bot, callback.message.chat.id, query, cursor
        )
    except Exception as e:
        log.exception("More search error")
        await bot.send_message(callback.message.chat.id, f"Search error: {e}")
        return

    if sent == 0:
        await bot.send_message(callback.message.chat.id, "No more results.")
        return
    if has_more:
        await _send_more_button(
            bot, callback.message.chat.id, query, next_cursor, shown + sent
        )


async def register_commands(bot: Bot) -> None:
    """Register the command list shown in the Telegram chat input menu."""
    await bot.set_my_commands([
        BotCommand(command="sync",   description="Scan and index your stickers / GIFs"),
        BotCommand(command="status", description="Show index statistics"),
        BotCommand(command="models", description="Show local model artifacts"),
        BotCommand(command="pipeline", description="Configure OCR/VLM for this run"),
        BotCommand(command="unload", description="Unload StickerRadar models from RAM now"),
        BotCommand(command="help",   description="Usage examples and tips"),
    ])


def build_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.update.outer_middleware(OwnerOnlyMiddleware())
    dp.include_router(router)
    return dp
