"""
Phase A — Telegram access inventory.

Logs into the Telegram account via Telethon and prints counts for:
  - Installed sticker sets
  - Stickers in first 3 packs (sample)
  - Favorite stickers
  - Recent stickers
  - Saved GIFs / animations

Usage:
    python scripts/inventory.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make sure project root is on path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.errors import ConfigError
from app.tg_user import TgUserClient


async def run_inventory() -> None:
    config.ensure_dirs()

    tg = TgUserClient(config.TG_API_ID, config.TG_API_HASH, config.SESSION_PATH)

    async with tg:
        print("\n=== StickerRadar — Telegram Inventory ===\n")

        # --- Installed sticker sets ---
        sets = await tg.get_installed_sticker_sets()
        print(f"[OK] Installed sticker sets:  {len(sets)}")

        if sets:
            sample = sets[:3]
            print(f"     Sampling first {len(sample)} packs …")
            total_sampled = 0
            for s in sample:
                docs = await tg.get_sticker_set_documents(s)
                total_sampled += len(docs)
                formats = {}
                for d in docs:
                    formats[d.sticker_format] = formats.get(d.sticker_format, 0) + 1
                fmt_str = ", ".join(f"{k}:{v}" for k, v in sorted(formats.items()))
                print(f"       • {s.title!r} ({s.short_name}) — {len(docs)} stickers  [{fmt_str}]")

            estimated_total = sum(s.count for s in sets)
            print(f"     Estimated total stickers (all packs): ~{estimated_total}")
        else:
            print("     No installed packs found.")

        # --- Favorite stickers ---
        faved = await tg.get_favorite_stickers()
        print(f"\n[OK] Favorite stickers:       {len(faved)}")
        if faved:
            sample_emojis = [d.emoji for d in faved[:10] if d.emoji]
            print(f"     Sample emojis: {' '.join(sample_emojis)}")

        # --- Recent stickers ---
        recent = await tg.get_recent_stickers()
        print(f"[OK] Recent stickers:         {len(recent)}")

        # --- Saved GIFs ---
        gifs = await tg.get_saved_gifs()
        print(f"[OK] Saved GIFs/animations:   {len(gifs)}")
        if gifs:
            mime_counts: dict[str, int] = {}
            for g in gifs:
                mime_counts[g.mime_type] = mime_counts.get(g.mime_type, 0) + 1
            mime_str = ", ".join(f"{k}:{v}" for k, v in mime_counts.items())
            print(f"     Mime types: {mime_str}")

        print("\n=== Done ===\n")


def main() -> None:
    try:
        asyncio.run(run_inventory())
    except ConfigError as e:
        print(f"\nConfiguration error:\n{e}\n", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
