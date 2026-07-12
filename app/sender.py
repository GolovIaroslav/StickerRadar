"""
app/sender.py — send stickers/GIFs via aiogram, caching bot_file_id in DB.

First send: upload local file -> receive bot file_id -> cache in DB.
Subsequent sends: reuse cached file_id.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from aiogram import Bot
from aiogram.types import FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup

from app import config, db
from app.search import SearchResult

log = logging.getLogger(__name__)


def _pack_markup(result: SearchResult) -> InlineKeyboardMarkup | None:
    """Inline button showing the sticker's pack; opens it when tapped."""
    if not result.set_short_name:
        return None
    title = result.set_title or result.set_short_name
    if len(title) > 60:
        title = title[:57] + "…"
    button = InlineKeyboardButton(
        text=f"📦 {title}",
        url=f"https://t.me/addstickers/{result.set_short_name}",
    )
    return InlineKeyboardMarkup(inline_keyboard=[[button]])


async def send_result(bot: Bot, chat_id: int, result: SearchResult) -> bool:
    """
    Send a single SearchResult to chat_id.
    Returns True on success, False on failure.
    """
    if result.bot_cache_status == "cached" and result.bot_file_id:
        success = await _send_cached(bot, chat_id, result)
        if success:
            return True
        # cache may be stale — fall through to upload

    if result.local_path and Path(result.local_path).exists():
        return await _send_upload(bot, chat_id, result)

    log.warning("No file to send for media_id=%s", result.media_id)
    return False


async def _send_cached(bot: Bot, chat_id: int, result: SearchResult) -> bool:
    markup = _pack_markup(result)
    try:
        if result.bot_send_method == "sticker":
            await bot.send_sticker(chat_id, result.bot_file_id, reply_markup=markup)
        else:
            await bot.send_animation(chat_id, result.bot_file_id, reply_markup=markup)
        return True
    except Exception as e:
        log.warning("Cached send failed for media_id=%s: %s", result.media_id, e)
        return False


async def _send_upload(bot: Bot, chat_id: int, result: SearchResult) -> bool:
    path = Path(result.local_path)
    send_method = _pick_send_method(result)
    markup = _pack_markup(result)
    try:
        file = FSInputFile(path)
        if send_method == "sticker":
            msg = await bot.send_sticker(chat_id, file, reply_markup=markup)
            file_id = msg.sticker.file_id
            file_unique_id = msg.sticker.file_unique_id
        else:
            msg = await bot.send_animation(chat_id, file, reply_markup=markup)
            file_id = msg.animation.file_id
            file_unique_id = msg.animation.file_unique_id

        db.save_bot_file_id(result.media_id, file_id, file_unique_id, send_method)
        return True
    except Exception as e:
        log.error("Upload failed for media_id=%s path=%s: %s", result.media_id, path, e)
        return False


def _pick_send_method(result: SearchResult) -> str:
    if result.bot_send_method:
        return result.bot_send_method
    if result.media_kind == "gif":
        return "animation"
    # sticker: animated (.tgs) and video (.webm) also use send_sticker
    return "sticker"


def select_sendable_page(
    results: list[SearchResult],
    *,
    cursor: int,
    page_size: int,
    recent_media_ids: set[int],
    recent_packs: set[str],
) -> tuple[list[SearchResult], int]:
    """Select one page and return the next cursor over *candidates consumed*.

    A cursor based on successfully sent items is incorrect: recent-history
    filtering can skip candidates, so the next page would repeat earlier
    candidates.  The cursor therefore advances for every inspected result.
    """
    page: list[SearchResult] = []
    seen_ids: set[int] = set()
    seen_packs: set[str] = set()
    index = max(0, cursor)
    while index < len(results) and len(page) < page_size:
        result = results[index]
        index += 1
        pack = result.set_short_name or ""
        if result.media_id in recent_media_ids or result.media_id in seen_ids:
            continue
        if pack and (pack in recent_packs or pack in seen_packs):
            continue
        page.append(result)
        seen_ids.add(result.media_id)
        if pack:
            seen_packs.add(pack)
    return page, index


async def send_results(
    bot: Bot,
    chat_id: int,
    results: list[SearchResult],
    delay_ms: int | None = None,
    *,
    exclude_recent: bool = True,
) -> int:
    """
    Send all results with optional inter-message delay.
    Returns count of successfully sent items.
    """
    if delay_ms is None:
        delay_ms = config.BOT_SEND_DELAY_MS

    recent_ids = db.recent_sent_media(chat_id) if exclude_recent else set()
    recent_packs = db.recent_sent_packs(chat_id) if exclude_recent else set()
    seen_ids: set[int] = set()
    seen_packs: set[str] = set()
    sent = 0
    for result in results:
        pack = result.set_short_name or ""
        if result.media_id in recent_ids or result.media_id in seen_ids:
            continue
        if pack and (pack in recent_packs or pack in seen_packs):
            continue
        ok = await send_result(bot, chat_id, result)
        if ok:
            db.record_sent_media(chat_id, result.media_id)
            seen_ids.add(result.media_id)
            if pack:
                seen_packs.add(pack)
            sent += 1
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)
    return sent
