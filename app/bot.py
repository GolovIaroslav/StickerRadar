"""
aiogram bot handlers — Phase G stub.
Handlers will be filled in during Phase G.
"""

from __future__ import annotations

from aiogram import Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Message

from app import config

router = Router()


def _owner_only(message: Message) -> bool:
    """Return True if message is from the owner. Silently ignore others."""
    return message.from_user is not None and message.from_user.id == config.OWNER_USER_ID


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    if not _owner_only(message):
        return
    await message.answer(
        "StickerRadar 🔍\n\n"
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
        "  кот устал\n"
        "  я устал\n"
        "  радость\n"
        "  злой\n"
        "  обнимаю\n\n"
        "Any text message is treated as a search query."
    )


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    if not _owner_only(message):
        return
    from app import db
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
async def cmd_sync(message: Message) -> None:
    if not _owner_only(message):
        return
    await message.answer("⏳ Sync not yet implemented. Coming in Phase B–E.")


@router.message()
async def handle_query(message: Message) -> None:
    if not _owner_only(message):
        return
    if not message.text:
        return
    await message.answer("🔍 Search not yet implemented. Run /sync first, then Phase F–G.")


def build_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.include_router(router)
    return dp
