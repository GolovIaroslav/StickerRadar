# StickerRadar

**Find your Telegram stickers by meaning, not by scrolling.**

StickerRadar indexes every sticker and saved GIF in your Telegram account with a
local AI vision model, then lets you search them in plain language through your
own private bot. Type *"tired cat"* and get the cat stickers — even if you have
4,000 stickers across hundreds of packs and never tagged a single one.

```
You → bot: "tired cat"
Bot → you: [the matching stickers, each with a button to open its pack]
```

100% local and private: it runs on your machine and the only network traffic is
the normal Telegram API. Your stickers are never uploaded anywhere or used to
train anything.

> 🔍 Works in **any language** (Russian, Chinese, English, …) — quality depends
> on the chosen [embedding model](#embedding-models).

<!-- Tip: add a demo.gif here, e.g.  ![demo](docs/demo.gif) -->

### Features

- 🧠 **Semantic search** over stickers, favorites, recents and saved GIFs
- 🌍 **Multilingual** queries out of the box (SigLIP2 default)
- 🔒 **Private** — local index, owner-only bot, session never leaves your disk
- 🔄 **Incremental** — re-`sync` only processes new stickers
- 🔌 **Swappable models** — one line in `.env`, from 0.2 GB to 7 GB
- 👤 **Multiple accounts** via named profiles
- 🖥️ **Cross-platform** — Linux, macOS, Windows

---

## How it works

1. **Telethon** (user-client) scans your sticker packs, favorites, recents, and saved GIFs.
2. Each sticker / GIF is downloaded locally and preview frames are extracted.
3. A CLIP-style multimodal model encodes every frame as a vector.
4. When you send a text query to your private bot, the same model encodes the text and finds the closest frames by cosine similarity.
5. The bot sends the top results as real Telegram stickers — each with a button that opens its sticker pack.

Everything runs on your machine. No data leaves except standard Telegram API calls.

---

## Requirements

| Dependency | Notes |
|---|---|
| Python 3.11+ | 3.12 recommended |
| [ffmpeg](https://ffmpeg.org/download.html) | Must be in `PATH` (used for animated/video stickers) |
| ~1 GB RAM | For the default model on CPU |
| A Telegram account | With sticker packs installed |

Runs on **Linux, macOS, and Windows**. On Windows, use PowerShell or Windows
Terminal (for clean QR rendering) and make sure `ffmpeg` is on your `PATH`.

A GPU is **optional**: if you install a CUDA build of PyTorch, embedding is used
on the GPU automatically. Plain CPU is fine for one-time indexing of a few
thousand stickers.

---

## Install

```bash
git clone https://github.com/GolovIaroslav/StickerRadar.git
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

## Quick start (5 steps)

```bash
# 1. Get credentials (see table below) and configure
cp .env.example .env          # then edit .env

# 2. Log in to Telegram (QR recommended)
python -m app login

# 3. Build the index (downloads stickers, embeds them)
python -m app sync

# 4. Start the bot
python -m app run

# 5. Open your bot in Telegram and type any phrase
```

That's it. After the first setup you only ever need `python -m app run`, and `/sync` from inside the bot to pick up new stickers.

### Credentials

| Credential | Where to get |
|---|---|
| `TG_API_ID` + `TG_API_HASH` | [my.telegram.org/apps](https://my.telegram.org/apps) — create an app |
| `BOT_TOKEN` | [@BotFather](https://t.me/BotFather) — create a bot |
| `OWNER_USER_ID` | [@userinfobot](https://t.me/userinfobot) or [@getidsbot](https://t.me/getidsbot) |

Put all four into `.env`.

### Logging in

`python -m app login` asks you to choose:

- **QR code (recommended):** Open Telegram on your phone → Settings → Devices → Link Desktop Device → scan the QR in the terminal. The QR auto-refreshes.
- **Phone number:** The code arrives as a message from the official *Telegram* account (user 777000) **inside the Telegram app — not as SMS**. If it never arrives, use QR login instead.

If you have 2FA enabled, you'll be asked for your password.

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
| `sync --reindex` | Re-extract previews and re-embed everything (after changing `FRAME_COUNT`) |
| `sync --frames N` | Override frame count for this run |
| `sync --keep-previews` | Don't auto-delete preview frames after embedding |
| `status` | Show pipeline status counts |
| `stats` | Show disk usage, model cache size, failure reasons |
| `prune` | Delete local media for stickers already sent once (frees disk) |
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

**Changing the model is automatic:** if you switch `MODEL_NAME` and run `sync`, StickerRadar detects that existing items have no vectors for the new model and re-embeds them (using the kept media — no re-download). Unchanged model → only genuinely new items are processed.

To force a full rebuild after changing `FRAME_COUNT` (frame structure changes):

```bash
python -m app sync --reindex
```

---

## Embedding models

StickerRadar uses **CLIP-style multimodal models** that encode both images and text into the same vector space. Text-only models (Gemini Embedding, Qwen3-text, etc.) are not compatible — they cannot encode sticker images.

```bash
python -m app models    # list all options + per-model install instructions
```

| Model | Quality | Size | Languages |
|---|---|---|---|
| `google/siglip2-base-patch16-224` *(default)* | good | ~0.4 GB | multilingual (RU, ZH, 100+) |
| `google/siglip2-large-patch16-256` | best | ~1.5 GB | multilingual |
| `jinaai/jina-clip-v2` | best | ~3.5 GB | 89 (incl. RU, ZH) |
| `openai/clip-vit-large-patch14` | good | ~1.7 GB | English only |
| `apple/MobileCLIP2-S2` *(experimental)* | fast | ~0.2 GB | English-leaning |
| `jinaai/jina-embeddings-v4` *(experimental)* | best | ~7.5 GB | multilingual |
| `Qwen/Qwen3-VL-Embedding-2B` *(experimental)* | best | ~4.5 GB | multilingual |

**Default** is Google SigLIP2 (2025) — modern, multilingual, small. To upgrade quality:

```bash
# In .env:
MODEL_NAME=jinaai/jina-clip-v2
uv add einops          # this model needs einops
python -m app sync --reindex
```

**Not compatible:** text-only embedders (Gemini Embedding, Qwen3 *text* embedding, OpenAI text-embedding-3, BGE, E5) cannot encode sticker images, so they cannot be used for this kind of image search.

**Custom model:** set `MODEL_NAME` in `.env` to any CLIP-style model on Hugging Face (loaded via sentence-transformers).

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

Shows: media downloads, preview frames, database, model cache, total, and top failure reasons.

**Automatic cleanup:** preview frames (`data/previews/`) are deleted automatically after each item is embedded — they're disposable intermediates and often larger than the original stickers. Use `sync --keep-previews` to keep them.

**Downloaded media is kept** because it's needed for the *first* send of each sticker: Telegram only returns a reusable `file_id` after you upload the file once. Once a sticker has been sent (its `file_id` is cached), its local copy is no longer needed:

```bash
python -m app prune    # delete media for already-sent stickers
```

Pruned stickers still send instantly via their cached `file_id`. (If you later switch models, pruned items can't be re-embedded without their media — re-run `sync --download` first.)

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
