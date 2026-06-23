# StickerRadar

**Local semantic search for your Telegram stickers and saved GIFs.**

Type a phrase into your private bot and get the most relevant stickers back — no manual tagging, no cloud service, no third-party access to your account.

```
You → bot: "tired cat"
Bot → you: [top 10 matching stickers / GIFs]
```

Works for any language. Search quality depends on the chosen [embedding model](#embedding-models).

---

## How it works

1. **Telethon** (user-client) scans your sticker packs, favorites, recents, and saved GIFs.
2. Each sticker / GIF is downloaded locally and preview frames are extracted.
3. A CLIP-style multimodal model encodes every frame as a vector.
4. When you send a text query to your private bot, the same model encodes the text and finds the closest frames by cosine similarity.
5. The bot sends the top results as real Telegram stickers and animations.

Everything runs on your machine. No data leaves except standard Telegram API calls.

---

## Requirements

| Dependency | Notes |
|---|---|
| Python 3.11+ | |
| [ffmpeg](https://ffmpeg.org/download.html) | Must be in `PATH` |
| ~500 MB RAM | For the default model |
| A Telegram account | With sticker packs installed |

---

## Install

```bash
git clone https://github.com/golov-j/StickerRadar.git
cd StickerRadar

# Recommended: uv (fast)
pip install uv
uv sync

# Or with pip
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Optional: animated `.tgs` sticker support (may require native libs):

```bash
uv add rlottie-python   # primary renderer
# or: uv add lottie     # fallback renderer
```

---

## Quick start

### 1. Get credentials

| Credential | Where to get |
|---|---|
| `TG_API_ID` + `TG_API_HASH` | [my.telegram.org/apps](https://my.telegram.org/apps) — create an app |
| `BOT_TOKEN` | [@BotFather](https://t.me/BotFather) — create a bot |
| `OWNER_USER_ID` | [@userinfobot](https://t.me/userinfobot) or [@getidsbot](https://t.me/getidsbot) |

### 2. Configure

```bash
cp .env.example .env
# Edit .env: fill in TG_API_ID, TG_API_HASH, BOT_TOKEN, OWNER_USER_ID
```

### 3. Log in

```bash
python -m app login
```

You'll be asked to choose between **QR code** (recommended) or **phone number**.

**QR code login:** Open Telegram on your phone → Settings → Devices → Link Desktop Device → scan the QR that appears in the terminal.

**Phone code login:** Enter your number. The code arrives as a message from the official *Telegram* account (user 777000) in your Telegram app — not as SMS. If it doesn't arrive, use QR login instead.

### 4. Sync and start

```bash
python -m app sync    # downloads stickers, extracts previews, builds embeddings
python -m app run     # starts the bot
```

Then open your bot in Telegram and send any text. Or use `/sync` from the bot to trigger a full sync without touching the terminal.

---

## CLI reference

```
python -m app <command> [--profile NAME]
```

| Command | Description |
|---|---|
| `login [--method qr\|phone]` | Authenticate Telegram account |
| `sync` | Run full pipeline: metadata → download → preview → embed |
| `sync --metadata` | Metadata stage only |
| `sync --download` | Download stage only |
| `sync --preview` | Preview extraction only |
| `sync --embed` | Embedding stage only |
| `sync --reindex` | Re-embed all items (after changing model) |
| `sync --full-reindex` | Re-extract previews and re-embed (after changing `FRAME_COUNT`) |
| `sync --frames N` | Override frame count for this run |
| `status` | Show pipeline status counts |
| `stats` | Show disk usage, model cache size, failure reasons |
| `models` | List available embedding models and upgrade instructions |
| `session list` | Show all saved profiles |
| `session use <name>` | Switch active profile |
| `session delete <name>` | Remove a profile |
| `session reset` | Delete session file (re-login required) |
| `search "<query>"` | CLI search for testing |
| `run` | Start the Telegram bot |

---

## Bot commands

| Command | Description |
|---|---|
| `/start` | Welcome and usage hint |
| `/sync` | Full sync (shows live progress) |
| `/status` | Index stats |
| `/help` | Search examples |
| Any text | Search query — returns top results |

---

## Incremental updates

Re-running `sync` is safe and efficient. The pipeline only processes items that haven't been handled yet:

- Added a new sticker pack? Run `sync` → only the new pack is downloaded and embedded.
- Removed a pack from Telegram? The local copy stays (use `stats` to track disk usage).
- Existing items with `download_status=ok` are never re-downloaded.

To force a full rebuild (e.g. after changing the model or frame count):

```bash
python -m app sync --reindex          # re-embed only
python -m app sync --full-reindex     # re-extract previews + re-embed
```

---

## Embedding models

StickerRadar uses **CLIP-style multimodal models** that encode both images and text into the same vector space. Text-only models (Gemini Embedding, Qwen3-text, etc.) are not compatible — they cannot encode sticker images.

```bash
python -m app models    # list all options with install instructions
```

| Model | Quality | Size | Languages |
|---|---|---|---|
| `sentence-transformers/clip-ViT-B-32-multilingual-v1` | fast | ~380 MB | 50+ |
| `jinaai/jina-clip-v2` | **best** | ~3.5 GB | 89 (incl. Russian, Chinese) |
| `openai/clip-vit-large-patch14` | good | ~900 MB | English only |

**Default** (`clip-ViT-B-32-multilingual-v1`) is a 2021 ViT-B/32 model — fast and lightweight, adequate for testing. For better quality, upgrade to `jina-clip-v2`:

```bash
uv add 'transformers>=4.36' einops
# In .env:
MODEL_NAME=jinaai/jina-clip-v2
# Re-embed:
python -m app sync --reindex
```

**Custom model:** set `MODEL_NAME` (text encoder) and `IMAGE_MODEL_NAME` (image encoder) in `.env`. Both must share the same embedding space.

---

## Multiple accounts (profiles)

Each profile has its own session file and database:

```bash
python -m app login --profile work
python -m app sync --profile work
python -m app session use work       # set as default
python -m app session list
python -m app session delete work
```

The active profile is stored in `data/.active_profile`.

---

## Disk usage

```bash
python -m app stats
```

Shows: media downloads, preview frames, database, HF model cache, total. Downloaded files are kept locally — they are needed for the first bot send (Telegram requires an upload to mint a `file_id`; subsequent sends reuse the cached ID).

---

## Security

> **Your session file gives full access to your Telegram account.**
> It lives in `data/sessions/` and is gitignored. Keep it local.
> Never share it, never put it in cloud storage or version control.

- The session file is created with `600` permissions (owner read/write only) on POSIX systems.
- The bot ignores all users except the one matching `OWNER_USER_ID`.
- `.env` (containing credentials) is gitignored.
- Telegram media is used only to build a local retrieval index. No model training on Telegram content.

---

## Project layout

```
app/
  __main__.py     — unified CLI entrypoint (python -m app)
  auth.py         — QR and phone login helpers
  models.py       — embedding model registry
  config.py       — env loading, paths, profile support
  db.py           — SQLite schema and queries
  tg_user.py      — Telethon user-client wrapper
  scanner.py      — sync pipeline orchestration
  media_store.py  — download and local path logic
  preview.py      — frame extraction (webp / tgs / webm / gif)
  embeddings.py   — CLIP model wrapper with batch encoding
  search.py       — vector search and metadata ranking
  bot.py          — aiogram bot handlers
  sender.py       — sendSticker/sendAnimation + file_id cache
  main.py         — asyncio entrypoint for the bot
scripts/
  inventory.py    — quick Telegram access test
  eval_queries.py — search quality evaluation (HTML report)
data/             — local only, gitignored
  sessions/       — Telethon session files (sensitive)
  media/          — downloaded stickers and GIFs
  previews/       — extracted PNG frames
  *.sqlite        — per-profile databases
```

---

## Troubleshooting

**Login code not arriving via phone** → use QR login: `python -m app login --method qr`

**"database is locked"** → only one process can use the session at a time. Stop any running `python -m app run` before running `sync` manually, or use the bot's `/sync` command which reuses the active connection.

**Download fails: "file_reference may be expired"** → run `sync --metadata` first to refresh references, then `sync --download`.

**Search returns nothing after sync** → check `python -m app status`: `embedded` count should be > 0. If it's 0, run `python -m app sync --embed`.

---

## License

MIT
