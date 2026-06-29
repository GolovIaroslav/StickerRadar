"""
Entry point for StickerRadar.

Starts both the Telethon user-client (keepalive) and the aiogram bot
in the same asyncio event loop.

Usage:
    python -m app.main
"""

from __future__ import annotations

import asyncio
import logging
import sys

from app import config
from app.errors import ConfigError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stickerradar")


async def _idle_unload_task() -> None:
    """Unload the embedding model after MODEL_IDLE_UNLOAD_SEC of inactivity."""
    if config.MODEL_IDLE_UNLOAD_SEC <= 0:
        return
    try:
        while True:
            await asyncio.sleep(30)
            from app.embeddings import get_shared_embedder
            emb = get_shared_embedder()
            idle = emb.idle_seconds()
            if idle > 0 and idle >= config.MODEL_IDLE_UNLOAD_SEC:
                emb.unload()
    except asyncio.CancelledError:
        pass


async def _run() -> None:
    config.ensure_dirs()

    # Import lazily to avoid heavy startup cost when running scripts
    from app.tg_user import TgUserClient
    from aiogram import Bot

    from app.bot import build_dispatcher, register_commands

    bot = Bot(token=config.BOT_TOKEN)
    dp = build_dispatcher()

    await register_commands(bot)

    tg = TgUserClient(config.TG_API_ID, config.TG_API_HASH, config.SESSION_PATH)

    log.info("Starting StickerRadar …")
    log.info("Bot: polling   |   Telethon: user-client")

    async with tg:
        await asyncio.gather(
            dp.start_polling(bot, handle_signals=False, tg_client=tg),
            tg.keepalive(),
            _idle_unload_task(),
        )


def cli_main() -> None:
    try:
        asyncio.run(_run())
    except ConfigError as e:
        print(f"Configuration error:\n{e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    cli_main()
