# StickerRadar

**Find your Telegram stickers by meaning, not by scrolling.**

StickerRadar indexes every sticker and saved GIF in your Telegram account with a
local AI vision model, then lets you search them in plain language through your
own private bot. Type *"tired cat"* and get the cat stickers — even if you have
4,000 stickers across hundreds of packs and never tagged a single one.

```
You:  tired cat
Bot:  matching stickers, each with a button to open its pack
```

100% local and private: it runs on your machine and the only network traffic is
the normal Telegram API. Your stickers are never uploaded anywhere or used to
train anything.

Works in any language (Russian, Chinese, English, ...) — quality depends on the
chosen [embedding model](#embedding-models).

<!-- Tip: add a demo.gif here, e.g.  ![demo](docs/demo.gif) -->

### Features

- **Semantic search** over stickers, favorites, recents and saved GIFs
- **Multilingual** queries out of the box
- **Private** — local index, owner-only bot, session never leaves your disk
- **Incremental** — re-`sync` only processes new stickers
- **Swappable models** — change one line in `.env`
- **Cross-platform** — Linux, macOS, Windows

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
| ffmpeg | Must be in `PATH` (used for animated/video stickers) |
| RAM | ~2–3 GB for `siglip2-base`; more for heavier A/B models |
| A Telegram account | With sticker packs installed |

Runs on **Linux, macOS, and Windows**. On Windows, use PowerShell or Windows
Terminal (for clean QR rendering) and make sure `ffmpeg` is on your `PATH`.

**GPU vs CPU:** embedding runs on the GPU automatically if a CUDA PyTorch is
installed. The recommended default `google/siglip2-base-patch16-224` is the
best local balance we measured here: on real StickerRadar retrieval tests it was
very close to `siglip2-so400m`, but with much lower RAM/VRAM use. On GPUs with
more headroom you can A/B heavier models such as `siglip2-so400m`. The
`Qwen/Qwen3-VL-Embedding-2B` option is stronger on quality, but on this 6 GB GPU
it hit CUDA OOM for image retrieval and only worked reliably on CPU.

---

## Install

```bash
git clone https://github.com/GolovIaroslav/StickerRadar.git
cd StickerRadar

# Recommended: uv (fast, no venv activation needed)
# Install uv first if you don't have it:
#   Linux / macOS:  curl -LsSf https://astral.sh/uv/install.sh | sh
#   Arch Linux:     sudo pacman -S uv
#   Windows:        winget install astral-sh.uv
uv sync

# Or with pip (must activate the venv before running app commands)
python -m venv .venv
source .venv/bin/activate        # bash / zsh
# source .venv/bin/activate.fish  # fish shell
# .venv\Scripts\activate          # Windows PowerShell
pip install -e .
```

### Text embedding branch (optional)

The dedicated Qwen3 text branch improves paraphrase and Russian/English matching
for stickers with OCR text. Install it locally, then add these lines to `.env`:

```bash
python -m app model install --id text-embed:qwen3-0.6b

TEXT_EMBED_ENABLED=true
TEXT_EMBED_MODEL_PATH=/path/to/Qwen3-Embedding-0.6B-Q8_0.gguf
TEXT_EMBED_LLAMA_SERVER_PATH=/path/to/llama-server
```

Run `python -m app sync --text-embed-backfill` after OCR text is already indexed.

Optional: animated `.tgs` sticker support (may require native libs):

```bash
uv add rlottie-python   # primary renderer
# or: uv add lottie cairosvg  # safe fallback renderer
```

---

## First-run setup wizard

StickerRadar includes a guided **English setup wizard** for new users:

```bash
uv run python -m app setup
```

It helps the user choose:

- the **embedding model** (with approximate size and license notes)
- the **OCR profile** (GLM hybrid-rescue by default when `llama-cli` is available, with lighter classic OCR options still available)
- the best **device** (`auto`, `cpu`, `cuda`) based on detected hardware

Current built-in OCR guidance comes from local tests on **20 real StickerRadar stickers**:

- **EasyOCR + GPU** — best default here (~161 ms/image, best Russian accuracy)
- **EasyOCR + CPU** — same quality, much slower (~611 ms/image)
- **RapidOCR + CPU** — lower RAM and no GPU required, but much weaker on Russian meme text (~302 ms/image)
- **GLM-OCR** — useful as a gated rescue pass after a fast detector says text is present but confidence is weak; brute-forcing every image is too slow

It also creates or updates `.env`, and it can be skipped quickly if the user
already knows what they want. If someone runs `login`, `sync`, or `run` before
completing setup, StickerRadar auto-launches the wizard once.

---

## Quick start (5 steps)

```bash
# 1. Run the setup wizard (recommended)
uv run python -m app setup

# 2. Log in to Telegram (QR recommended)
uv run python -m app login

# 3. Build the index (downloads stickers, embeds them)
uv run python -m app sync

# 4. Start the bot
uv run python -m app run

# 5. Open your bot in Telegram and type any phrase
```

That's it. After the first setup you only ever need `uv run python -m app run`, and `/sync` from inside the bot to pick up new stickers.

### Credentials

| Credential | Where to get |
|---|---|
| `TG_API_ID` + `TG_API_HASH` | [my.telegram.org/apps](https://my.telegram.org/apps) — create an app |
| `BOT_TOKEN` | [@BotFather](https://t.me/BotFather) — create a bot |
| `OWNER_USER_ID` | [@userinfobot](https://t.me/userinfobot) or [@getidsbot](https://t.me/getidsbot) |

Put all four into `.env`.

### Logging in

`uv run python -m app login` asks you to choose:

- **QR code (recommended):** On your phone, open Telegram, go to Settings > Devices > Link Desktop Device, and scan the QR shown in the terminal. The QR auto-refreshes.
- **Phone number:** The code arrives as a message from the official *Telegram* account (user 777000) **inside the Telegram app — not as SMS**. If it never arrives, use QR login instead.

If you have 2FA enabled, you'll be asked for your password.

---

## CLI reference

```
uv run python -m app <command>
```

| Command | Description |
|---|---|
| `login [--method qr\|phone]` | Authenticate Telegram account |
| `sync` | Run the full pipeline: metadata, download, preview, OCR, image embed, OCR-text embed |
| `sync --metadata` | Metadata stage only |
| `sync --download` | Download stage only |
| `sync --preview` | Preview extraction only |
| `sync --ocr` | OCR stage only |
| `sync --embed` | Image embedding stage only |
| `sync --ocr-text-embed` | Backfill semantic OCR-text embeddings from persisted OCR text |
| `sync --reocr` | Safe batched re-OCR/re-embed pass for the existing local library; with `OCR_BACKEND=glm-ocr` it now prefilters frames and sends only low-confidence text-like cases into GLM |
| `sync --reindex` | Re-extract previews and re-embed everything (after changing `FRAME_COUNT`) |
| `sync --frames N` | Override frame count for this run |
| `sync --keep-previews` | Don't auto-delete preview frames after embedding |
| `status` | Show pipeline status counts |
| `stats` | Show disk usage, model cache size, failure reasons |
| `prune` | Delete local media for stickers already sent once (frees disk) |
| `models` | List available embedding models and upgrade instructions |
| `ocr-models` | List OCR profile options, trade-offs, and install commands |
| `ocr-benchmark --backend ...` | Benchmark one OCR backend on real local stickers |
| `retrieval-benchmark --model ...` | Benchmark retrieval quality for one embedding model on real stickers |
| `setup` | Run the first-run setup wizard again |
| `session reset` | Delete the session file (re-login required) |
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

- Added a new sticker pack? Run `sync` and only the new pack is downloaded and embedded.
- Removed a pack from Telegram? The local copy stays (use `stats` to track disk usage).
- Existing items with `download_status=ok` are never re-downloaded.

**Changing the model is explicit:** install the selected model artifact first with
`python -m app model install ... --yes`, then switch `MODEL_NAME` and run `sync`.
Runtime never downloads model weights, even during `sync`.

To force a full rebuild after changing `FRAME_COUNT` (frame structure changes):

```bash
uv run python -m app sync --reindex
```

---

## Embedding models

StickerRadar uses **CLIP-style multimodal models** that encode both images and text into the same vector space. Text-only models (Gemini Embedding, Qwen3-text, etc.) are not compatible — they cannot encode sticker images.

```bash
uv run python -m app models    # list all options + per-model install instructions
```

| Model | Params | Size | License | Notes |
|---|---|---|---|---|
| `google/siglip2-base-patch16-224` | 0.4B | ~1.5 GB | Apache-2.0 | recommended default; multilingual, low-memory |
| `google/siglip2-so400m-patch16-384` | ~1.1B | ~2 GB | Apache-2.0 | multilingual, heavier A/B option |
| `Qwen/Qwen3-VL-Embedding-2B` | 2B | ~4.26 GB | Apache-2.0 | strongest quality candidate; CPU-friendly, GPU-heavy |
| `google/siglip2-large-patch16-256` | 0.9B | ~3.53 GB | Apache-2.0 | multilingual; benchmark on your machine |
| `jinaai/jina-clip-v2` | 0.9B | ~1.73 GB | CC BY-NC 4.0 | blocked by current transformers compatibility |
| `jinaai/jina-embeddings-v5-omni-nano-retrieval` | ~0.95B | ~1.9 GB | CC BY-NC 4.0 | verified; slower/heavier than siglip2-base |
| `jinaai/jina-embeddings-v5-omni-small-retrieval` | ~1.56B | ~3.12 GB | CC BY-NC 4.0 | verified; still not worth the extra cost here |
| `facebook/metaclip-2-worldwide-huge` | ~2B | ~4+ GB | CC BY-NC 4.0 | repo/access blocked in current environment |
| `visheratin/mexma-siglip2` | ~1B | ~2 GB | MIT | remote-code compatibility break in current stack |
| `openai/clip-vit-large-patch14` | 0.4B | ~1.71 GB | MIT | English only |
| `apple/MobileCLIP2-S2` | ~99M | ~398 MB | Apple AMLR (research) | verified; small, but not better than siglip2-base |
| `jinaai/jina-embeddings-v4` | 4B | ~7.89 GB | Qwen Research License | heavy, multimodal, 30+ langs |

Sizes are approximate (core model weights). Where a model card doesn't publish an
exact language list, test your languages locally rather than trusting a number.

**Recommended default** is `google/siglip2-base-patch16-224` for most local installs. In local StickerRadar retrieval benchmarks it stayed very close to `siglip2-so400m` on quality while using much less RAM/VRAM. If you have more headroom and want to A/B a heavier open-license option, try `google/siglip2-so400m-patch16-384`.

`Qwen/Qwen3-VL-Embedding-2B` remains a strong quality candidate, but on this 6 GB GPU it failed with CUDA OOM during image retrieval. CPU mode worked, but was much slower.

Among the experimental/non-commercial options tested here, none beat `siglip2-base` as a practical default: `MobileCLIP2-S2`, `jina-v5-omni-nano`, and `jina-v5-omni-small` all loaded successfully, but each was slower and/or heavier without a clear retrieval-quality win on this corpus. Several other candidates are currently blocked by stack or access issues (`jina-clip-v2`, `mexma-siglip2`, `metaclip-2-worldwide-huge`).

```bash
# In .env:
MODEL_NAME=google/siglip2-base-patch16-224   # recommended default
uv run python -m app sync --reindex
```

To switch to any model, set `MODEL_NAME` in `.env`, install any extra deps shown by `uv run python -m app models`, then run `uv run python -m app sync --reindex`.

**Licenses matter:** several strong models (Jina) are **non-commercial**. For commercial use, prefer the Apache-2.0 models (Qwen3-VL, SigLIP2) or MIT (OpenAI CLIP).

### Local retrieval benchmark results

StickerRadar now includes a retrieval benchmark that builds a temporary eval DB from real local stickers and runs the actual search pipeline:

```bash
uv run python -m app retrieval-benchmark --model google/siglip2-base-patch16-224 --device auto --sample-size 20 --seed 42
```

Measured here on this machine / local corpus:

| Model | Device | Sample / Queries | Time | MRR | Hit@1 | Hit@5 | RAM delta | VRAM delta | Practical note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `google/siglip2-base-patch16-224` | auto (GPU) | 20 / 60 | 9.22 s | 0.847 | 0.717 | 1.0 | ~2988 MB | ~1009 MB | Best local balance |
| `google/siglip2-so400m-patch16-384` | auto (GPU) | 20 / 60 | 11.22 s | 0.845 | 0.717 | 1.0 | ~7346 MB | ~2805 MB | Similar quality, much heavier |
| `apple/MobileCLIP2-S2` | auto (GPU) | 20 / 60 | 19.90 s | 0.833 | 0.717 | 0.983 | ~1728 MB | ~971 MB | Loads fine, but no quality win |
| `jinaai/jina-embeddings-v5-omni-nano-retrieval` | auto (GPU) | 20 / 60 | 45.27 s | 0.833 | 0.717 | 0.967 | ~4544 MB | ~2165 MB | Verified, but worse trade-off than siglip2-base |
| `jinaai/jina-embeddings-v5-omni-small-retrieval` | auto (GPU) | 15 / 45 | 49.74 s | 0.837 | 0.733 | 0.956 | ~4153 MB | ~3379 MB | Verified, still too heavy for the gain |
| `google/siglip2-base-patch16-224` | cpu | 30 / 90 | 21.93 s | 0.786 | 0.667 | 0.956 | ~2008 MB | 0 MB | Good CPU-only fallback |
| `Qwen/Qwen3-VL-Embedding-2B` | cpu | 15 / 45 | 407.80 s | 0.826 | 0.711 | 0.978 | ~7369 MB | 0 MB | Strong but very slow on CPU |

Observed failure mode:

- `Qwen/Qwen3-VL-Embedding-2B` on this **6 GB GPU** failed with a real `CUDA out of memory` during image retrieval benchmark.

So for practical local use on modest hardware, `siglip2-base` is the current recommendation.

---

**Custom model:** `MODEL_NAME` accepts either a Hugging Face id **or a local folder path**:

```bash
MODEL_NAME=org/some-clip-model          # Hugging Face
MODEL_NAME=/home/me/models/my-clip      # local path (C:\models\my-clip on Windows)
```

It's loaded via sentence-transformers and must be a CLIP-style model that embeds **both** images and text.

---

## OCR profiles and local benchmark command

StickerRadar lets the user choose **whether OCR is worth downloading at all**, and if yes, which OCR path to use.

```bash
uv run python -m app ocr-models
uv run python -m app ocr-benchmark --backend easyocr --limit 20 --seed 42
uv run --with rapidocr-onnxruntime python -m app ocr-benchmark --backend rapidocr --limit 20 --seed 42 --cpu
uv run --with easyocr python -m app ocr-benchmark --backend easyocr --limit 20 --seed 42 --cpu
uv run python -m app ocr-benchmark --backend glm-ocr --limit 20 --seed 42
```

Measured on this project with **20 random real stickers** (same seed across runs):

| OCR backend | Device | Approx model/runtime footprint | Speed | Quality summary |
|---|---|---:|---:|---|
| `easyocr` | GPU | ~300 MB download, ~469 MB observed VRAM delta | ~161 ms/image | Best overall balance here; strongest Russian sticker-text accuracy |
| `easyocr` | CPU | ~300 MB download, ~1.3 GB RAM peak | ~611 ms/image | Same quality as GPU, but much slower |
| `rapidocr` | CPU | ~150–250 MB assets, ~261 MB RAM peak | ~302 ms/image | Lower overhead, but often transliterates or mangles Cyrillic |
| `glm-ocr` | GPU + llama.cpp | ~1.43 GB GGUF + mmproj | historically ~3973 ms/image when brute-forced; now used as gated rescue pass | Best reserved for low-confidence text-like frames after a cheap detector, not for every sticker |

Practical recommendation:

- **Want the best default for Russian / mixed-language sticker text?** Use `easyocr`.
- **Need the lightest CPU-only option?** Use `rapidocr`.
- **Need an experimental rescue pass for hard cases and accept slow indexing?** Try `glm-ocr` manually or as a carefully limited fallback.
- **Do not care about exact printed text on stickers?** Disable OCR entirely and keep image-semantic search only.

Notes:

- `glm-ocr` first converts StickerRadar `.webp` stickers to temporary PNGs before sending them to `llama.cpp`, because direct WEBP decoding was unreliable in the tested path.
- The benchmark command is intentionally single-backend per run so CPU/GPU contention does not distort results.
- For `glm-ocr`, the benchmark's RAM/VRAM numbers mainly reflect the Python wrapper process; the important practical metric is wall-clock time.

---

## Disk usage

```bash
uv run python -m app stats
```

Shows: media downloads, preview frames, database, model cache, total, and top failure reasons.

**Automatic cleanup:** preview frames (`data/previews/`) are deleted automatically after each item is embedded — they're disposable intermediates and often larger than the original stickers. Use `sync --keep-previews` to keep them.

**Downloaded media is kept** because it's needed for the *first* send of each sticker: Telegram only returns a reusable `file_id` after you upload the file once. Once a sticker has been sent (its `file_id` is cached), its local copy is no longer needed:

```bash
uv run python -m app prune    # delete media for already-sent stickers
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

## Troubleshooting

**Login code not arriving via phone:** use QR login, `uv run python -m app login --method qr`

**"database is locked":** only one process can use the session at a time. Stop any running `uv run python -m app run` before running `sync` manually, or use the bot's `/sync` command which reuses the active connection.

**Download fails ("file_reference may be expired"):** run `uv run python -m app sync --metadata` first to refresh references, then `uv run python -m app sync --download`.

**Search returns nothing after sync:** check `uv run python -m app status` — the `embedded` count should be above 0. If it's 0, run `uv run python -m app sync --embed`.

---

## License

MIT
