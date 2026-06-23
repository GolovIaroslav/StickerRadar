# StickerRadar

**Local semantic search for your Telegram stickers and saved GIFs.**

Type a Russian (or any language) phrase into your private bot and get the most relevant stickers back — without manual tagging, without a cloud service, without giving anyone access to your Telegram account.

```
You → bot: "кот устал"
Bot → you: [🐱 5-10 matching stickers / GIFs]
```

---

## How it works

1. **Telethon** logs into your personal Telegram account (user-client) and scans your installed sticker packs, favorites, recent stickers, and saved GIFs.
2. The app downloads all media locally and extracts preview frames.
3. A multilingual CLIP model turns every frame into a vector embedding.
4. When you send a text query to the private bot, the same model embeds your text and finds the closest sticker/GIF frames by cosine similarity.
5. The bot sends the top results — as real Telegram stickers and animations.

Everything runs on your own machine. No data leaves except normal Telegram API calls.

---

## Requirements

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/download.html) — must be available in `PATH`
- ~1 GB RAM for the CLIP model
- A Telegram account with stickers installed

---

## Quick start

### 1. Get Telegram credentials

- **API ID and API Hash**: go to [my.telegram.org/apps](https://my.telegram.org/apps), create an app, copy the values.
- **Bot token**: message [@BotFather](https://t.me/BotFather), create a bot, copy the token.
- **Your user ID**: message [@userinfobot](https://t.me/userinfobot) or [@getidsbot](https://t.me/getidsbot).

### 2. Install

```bash
git clone https://github.com/golov-j/StickerRadar.git
cd StickerRadar

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -e .
```

For animated `.tgs` sticker support (optional, may require compilation):

```bash
pip install -e ".[tgs]"
# or fallback renderer:
pip install -e ".[lottie]"
```

### 3. Configure

```bash
cp .env.example .env
```

Open `.env` and fill in `TG_API_ID`, `TG_API_HASH`, `BOT_TOKEN`, `OWNER_USER_ID`.

### 4. Verify Telegram access

```bash
python scripts/inventory.py
```

This logs in (first run asks for your phone number and code) and prints counts of your sticker packs, favorites, and saved GIFs.

### 5. Run

```bash
python -m app.main
```

Then open your bot in Telegram:

```
/sync          — scan and index your stickers (takes a few minutes)
/status        — show index stats
кот устал      — search
```

---

## Bot commands

| Command | Description |
|---|---|
| `/start` | Welcome message and usage hint |
| `/sync` | Scan Telegram account and rebuild index |
| `/status` | Show counts: indexed, pending, failed |
| `/help` | Usage examples |
| Any text | Search query — returns top 10 results |

---

## Project layout

```
app/
  config.py       — env loading, paths
  db.py           — SQLite schema and queries
  tg_user.py      — Telethon user-client
  scanner.py      — sync orchestration
  media_store.py  — download and local storage
  preview.py      — frame extraction (webp/tgs/webm/gif)
  embeddings.py   — CLIP model wrapper
  search.py       — vector search and ranking
  bot.py          — aiogram handlers
  sender.py       — sendSticker/sendAnimation + file_id cache
scripts/
  inventory.py    — Telegram access test
  eval_queries.py — search quality evaluation
data/             — local only, gitignored
  app.sqlite
  sessions/
  media/
  previews/
```

---

## Security

> **Your session file gives full access to your Telegram account.**
> It lives in `data/sessions/` and is gitignored.
> Never share it, never put it in cloud storage.

The bot is protected by `OWNER_USER_ID` — it ignores all users except you.

This project uses Telegram data only to build a local retrieval index for personal use.
It does not train or fine-tune any model on Telegram content.

---

## Switching the search model

Edit `.env`:

```env
# Default — smaller, faster, good Russian support
MODEL_NAME=sentence-transformers/clip-ViT-B-32-multilingual-v1

# Larger, better quality
MODEL_NAME=jinaai/jina-clip-v2
```

After changing the model, re-run `/sync` (or `python -m app.scanner --reindex`) to rebuild embeddings.

---

## License

MIT
