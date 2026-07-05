"""
python -m app — StickerRadar unified CLI.

Usage:
    python -m app login   [--method qr|phone] [--profile NAME]
    python -m app ocr-models
    python -m app ocr-benchmark --backend easyocr|rapidocr|glm-ocr [--limit N] [--seed N] [--cpu]
    python -m app sync    [--metadata|--download|--preview|--embed]
                          [--reindex] [--full-reindex] [--frames N] [--limit N]
    python -m app status  [--profile NAME]
    python -m app stats   [--profile NAME]
    python -m app models
    python -m app session list|use <name>|delete <name>|reset
    python -m app search  "<query>" [--top N]
    python -m app run     [--profile NAME]

Global option:
    --profile NAME   Named session/DB profile (default: reads data/.active_profile or 'main')
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def _active_profile_file() -> Path:
    from app import config
    return config.DATA_DIR / ".active_profile"


def _read_active_profile() -> str:
    f = _active_profile_file()
    if f.exists():
        return f.read_text().strip() or "main"
    return "main"


def _write_active_profile(name: str) -> None:
    f = _active_profile_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(name)


def _apply_profile(args: argparse.Namespace) -> None:
    from app import config
    profile = getattr(args, "profile", None) or _read_active_profile()
    # "main" keeps the env-configured SESSION_PATH/DB_PATH (backward-compatible).
    # Any other named profile overrides both paths.
    if profile and profile != "main":
        config.set_profile(profile)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_login(args: argparse.Namespace) -> None:
    _apply_profile(args)
    from app import config
    api_id, api_hash = config.require_login_config()
    from app.auth import ensure_logged_in
    asyncio.run(ensure_logged_in(
        config.SESSION_PATH,
        api_id,
        api_hash,
        method=getattr(args, "method", None),
    ))


def _interactive_sync_config() -> bool:
    """Show settings menu before sync. Returns True to proceed, False to abort."""
    from app import config
    from app.models import REGISTRY

    def _gpu_label() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                return f"GPU {free / 1e9:.1f} GB free"
            return "CPU only"
        except Exception:
            return ""

    while True:
        ocr_label = (("on (gpu)" if config.OCR_USE_GPU else "on (cpu)") if config.OCR_ENABLED else "off")
        print()
        print("── Sync settings ──────────────────────────────────")
        print(f"  1  Model       : {config.MODEL_NAME.split('/')[-1]}")
        print(f"  2  Device      : {config.DEVICE}  ({_gpu_label()})")
        print(f"  3  Frames      : {config.FRAME_COUNT}")
        print(f"  4  Concurrency : {config.SCAN_CONCURRENCY}")
        print(f"  5  OCR         : {ocr_label}")
        print("────────────────────────────────────────────────────")

        try:
            choice = input("  Number to change, Enter to start, q to quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False

        if not choice:
            return True
        if choice == "q":
            return False

        if choice == "1":
            print()
            for i, m in enumerate(REGISTRY, 1):
                mark = "→" if m.key == config.MODEL_NAME else " "
                print(f"  {mark} {i:2}. {m.key.split('/')[-1]:<46} {m.params:>6}  {m.quality}")
            try:
                sel = input("  Model number (Enter to keep): ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if sel.isdigit() and 1 <= int(sel) <= len(REGISTRY):
                config.MODEL_NAME = REGISTRY[int(sel) - 1].key

        elif choice == "2":
            opts = ["auto", "cpu", "cuda"]
            cur = config.DEVICE if config.DEVICE in opts else "auto"
            config.DEVICE = opts[(opts.index(cur) + 1) % len(opts)]

        elif choice == "3":
            try:
                val = input(f"  Frames per item [{config.FRAME_COUNT}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if val.isdigit() and int(val) >= 1:
                config.FRAME_COUNT = int(val)

        elif choice == "4":
            try:
                val = input(f"  Concurrency [{config.SCAN_CONCURRENCY}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if val.isdigit() and int(val) >= 1:
                config.SCAN_CONCURRENCY = int(val)

        elif choice == "5":
            # Cycle: off → on(cpu) → on(gpu) → off
            if not config.OCR_ENABLED:
                config.OCR_ENABLED = True
                config.OCR_USE_GPU = False
            elif not config.OCR_USE_GPU:
                config.OCR_USE_GPU = True
            else:
                config.OCR_ENABLED = False
                config.OCR_USE_GPU = False


def cmd_sync(args: argparse.Namespace) -> None:
    from app import config, db
    _apply_profile(args)
    config.require_login_config()

    if getattr(args, "frames", None):
        config.FRAME_COUNT = args.frames

    from app.scanner import _run_metadata_sync, _run_download, _run_preview, _run_ocr, _run_embed

    keep_previews = getattr(args, "keep_previews", False)
    limit = getattr(args, "limit", None)
    run_all = not any([
        getattr(args, "metadata", False),
        getattr(args, "download", False),
        getattr(args, "preview", False),
        getattr(args, "embed", False),
    ])

    if run_all and not getattr(args, "yes", False):
        if not _interactive_sync_config():
            return

    config.ensure_dirs()
    db.get_conn()

    if getattr(args, "reindex", False):
        n = db.force_reindex()
        print(f"Reindex: reset {n} items to re-preview and re-embed.")

    try:
        if run_all or getattr(args, "metadata", False):
            asyncio.run(_run_metadata_sync(limit))
        if run_all or getattr(args, "download", False):
            asyncio.run(_run_download(limit))

        # Model-aware: flag downloaded items that lack embeddings for the active
        # model so a plain `sync` re-does only what's needed after a model change.
        if run_all:
            flagged = db.mark_items_for_model(config.MODEL_NAME)
            if flagged:
                print(f"{flagged} item(s) need embeddings for model '{config.MODEL_NAME}'.")

        if run_all or getattr(args, "preview", False):
            _run_preview(limit)
        if run_all and config.OCR_ENABLED:
            _run_ocr(limit)
        if run_all or getattr(args, "embed", False):
            _run_embed(limit, keep_previews=keep_previews)

        if run_all:
            _print_stats(config, db, getattr(args, "profile", None) or _read_active_profile())
    finally:
        db.close()


def cmd_status(args: argparse.Namespace) -> None:
    from app import config, db
    _apply_profile(args)
    db.get_conn()
    profile = getattr(args, "profile", None) or _read_active_profile()
    counts = db.get_status_counts()
    print(f"\nProfile    : {profile}")
    print(f"Total      : {counts['total']}")
    print(f"Downloaded : {counts['downloaded']}")
    print(f"Previewed  : {counts['previewed']}")
    print(f"Embedded   : {counts['embedded']}")
    print(f"Failed     : {counts['failed']}\n")
    db.close()


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _print_stats(config, db, profile: str) -> None:
    import os
    media = _dir_size(config.MEDIA_DIR)
    previews = _dir_size(config.PREVIEWS_DIR)
    db_size = config.DB_PATH.stat().st_size if config.DB_PATH.exists() else 0

    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    model_cache = _dir_size(hf_cache / "hub") if hf_cache.exists() else 0

    counts = db.get_status_counts()
    print(f"\n── Stats ({profile}) ─────────────────────────────")
    print(f"  Media (downloads)  : {_fmt_bytes(media)}")
    print(f"  Preview frames     : {_fmt_bytes(previews)}")
    print(f"  Database           : {_fmt_bytes(db_size)}")
    print(f"  Model cache (all)  : {_fmt_bytes(model_cache)}")
    print(f"  Total on disk      : {_fmt_bytes(media + previews + db_size)}")
    print()
    print(f"  Total items        : {counts['total']}")
    print(f"  Downloaded         : {counts['downloaded']}")
    print(f"  Previewed          : {counts['previewed']}")
    print(f"  Embedded           : {counts['embedded']}")
    print(f"  Failed             : {counts['failed']}")
    print(f"  Active model       : {config.MODEL_NAME}")
    print(f"  Vectors (model)    : {db.count_embeddings_for_model(config.MODEL_NAME)}")
    print("─────────────────────────────────────────────────")

    errors = db.get_conn().execute(
        "SELECT last_error, COUNT(*) n FROM media_items "
        "WHERE last_error IS NOT NULL GROUP BY last_error ORDER BY n DESC LIMIT 5"
    ).fetchall()
    if errors:
        print("\nTop failure reasons:")
        for row in errors:
            print(f"  ({row[1]}×) {(row[0] or '')[:90]}")
    print()


def cmd_stats(args: argparse.Namespace) -> None:
    from app import config, db
    _apply_profile(args)
    db.get_conn()
    profile = getattr(args, "profile", None) or _read_active_profile()
    _print_stats(config, db, profile)
    db.close()


def cmd_fts_rebuild(args: argparse.Namespace) -> None:
    """Rebuild the FTS5 index (needed once when upgrading from a pre-FTS build)."""
    from app import config, db
    _apply_profile(args)
    db.get_conn()
    print("Rebuilding FTS index …")
    db.fts_rebuild()
    count = db.get_conn().execute("SELECT COUNT(*) FROM media_fts").fetchone()[0]
    print(f"Done. FTS index contains {count} row(s).")
    db.close()


def cmd_prune(args: argparse.Namespace) -> None:
    """Delete local media files for stickers that already have a cached bot file_id."""
    from app import config, db
    _apply_profile(args)
    db.get_conn()

    rows = db.list_prunable_media()
    if not rows:
        print("Nothing to prune. (Media is removed only after a sticker has been")
        print("sent at least once, so its Telegram file_id is cached.)")
        db.close()
        return

    freed = 0
    count = 0
    for row in rows:
        p = Path(row["local_path"])
        if p.exists():
            freed += p.stat().st_size
            try:
                p.unlink()
                count += 1
            except Exception:
                continue
        db.clear_local_path(row["id"])

    print(f"Pruned {count} media file(s), freed {_fmt_bytes(freed)}.")
    print("These stickers can still be sent via their cached Telegram file_id.")
    db.close()


def cmd_models(_args: argparse.Namespace) -> None:
    from app.models import REGISTRY, INCOMPATIBLE
    from app import config

    active = config.MODEL_NAME
    print("\nAvailable embedding models")
    print("─" * 118)
    print(f"  {'MODEL KEY':<46} {'PARAMS':<7} {'SIZE':<9} {'LICENSE':<22} {'INSTALL'}")
    print("─" * 118)
    for m in REGISTRY:
        tags = []
        if m.key == active:
            tags.append("← active")
        if m.verified:
            tags.append("verified")
        if m.experimental:
            tags.append("experimental")
        tag = ("   " + ", ".join(tags)) if tags else ""
        print(f"  {m.key:<46} {m.params:<7} {m.size:<9} {m.license:<22} {m.install_command}{tag}")
    print()
    print("To switch model:")
    print("  1. Set MODEL_NAME=<key> in .env  (install any extra deps shown below)")
    print("  2. Run: python -m app sync --reindex")
    print()
    print("Custom CLIP-style model not in this list — MODEL_NAME accepts either:")
    print("  • a Hugging Face id   e.g.  MODEL_NAME=org/model-name")
    print("  • a local folder path e.g.  MODEL_NAME=/home/me/models/my-clip")
    print("  Loaded via sentence-transformers; it must embed BOTH images and text.")
    print()
    print("⚠  NOT compatible (cannot encode sticker images):")
    for name, why in INCOMPATIBLE:
        print(f"     • {name} — {why}")
    print()
    for m in REGISTRY:
        if m.notes:
            label = m.key.split("/")[-1]
            print(f"[{label}]  {m.params} · {m.size} · {m.license}")
            for line in m.notes.splitlines():
                print(f"  {line}")
            print()


def cmd_session(args: argparse.Namespace) -> None:
    from app import config, db
    action = getattr(args, "action", None)

    if action == "list" or action is None:
        sessions_dir = config.DATA_DIR / "sessions"
        if not sessions_dir.exists():
            print("No sessions found.")
            return
        files = sorted(sessions_dir.glob("*.session"))
        if not files:
            print("No sessions found.")
            return
        active = _read_active_profile()
        print("Profiles:")
        for f in files:
            name = f.stem
            marker = " ← active" if name == active else ""
            db_path = config.DATA_DIR / f"{name}.sqlite"
            db_note = f"  (DB: {db_path.name})" if db_path.exists() else ""
            print(f"  {name}{marker}{db_note}")

    elif action == "use":
        name = getattr(args, "name", None)
        if not name:
            print("Usage: python -m app session use <name>")
            return
        session_file = config.DATA_DIR / "sessions" / f"{name}.session"
        if not session_file.exists():
            print(f"Profile '{name}' not found.")
            print(f"Run: python -m app login --profile {name}")
            return
        _write_active_profile(name)
        print(f"Active profile set to: {name}")

    elif action == "delete":
        name = getattr(args, "name", None)
        if not name:
            print("Usage: python -m app session delete <name>")
            return
        confirm = input(f"Delete profile '{name}'? This removes the session file. [y/N]: ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return
        sessions_dir = config.DATA_DIR / "sessions"
        for f in [sessions_dir / f"{name}.session", sessions_dir / f"{name}.session-journal"]:
            if f.exists():
                f.unlink()
                print(f"Deleted: {f.name}")
        db_file = config.DATA_DIR / f"{name}.sqlite"
        if db_file.exists():
            del_db = input(f"Also delete database {db_file.name}? [y/N]: ").strip().lower()
            if del_db == "y":
                db_file.unlink()
                print(f"Deleted: {db_file.name}")
        if _read_active_profile() == name:
            _write_active_profile("main")
        print(f"Profile '{name}' removed.")

    elif action == "reset":
        profile = getattr(args, "profile", None) or _read_active_profile()
        sessions_dir = config.DATA_DIR / "sessions"
        for f in [sessions_dir / f"{profile}.session", sessions_dir / f"{profile}.session-journal"]:
            if f.exists():
                f.unlink()
                print(f"Deleted: {f.name}")
        print(f"Session '{profile}' cleared.")
        print(f"Run: python -m app login --profile {profile}")

    else:
        print("Usage: python -m app session [list | use <name> | delete <name> | reset]")


def cmd_search(args: argparse.Namespace) -> None:
    from app import config, db
    _apply_profile(args)
    db.get_conn()
    from app.search import search
    top_k = getattr(args, "top", None) or config.TOP_K
    results = search(args.query, top_k=top_k)
    if not results:
        print("No results.")
    else:
        for i, r in enumerate(results, 1):
            pack = r.set_title or r.set_short_name or r.media_kind
            print(f"{i:2}. [{r.score:.3f}] {pack}  {r.emoji or ''}  id={r.tg_document_id}")
    db.close()


def cmd_run(args: argparse.Namespace) -> None:
    _apply_profile(args)
    from app import config
    config.require_login_config()
    config.require_bot_runtime_config()
    from app.main import cli_main
    cli_main()


def cmd_setup(args: argparse.Namespace) -> None:
    from app.setup_wizard import run_setup_wizard
    run_setup_wizard(quick=getattr(args, "quick", False))


def cmd_ocr_benchmark(args: argparse.Namespace) -> None:
    from app.ocr_benchmark import run_benchmark

    summary = run_benchmark(
        backend=args.backend,
        limit=args.limit,
        seed=args.seed,
        use_gpu=not getattr(args, "cpu", False) and args.backend in {"easyocr", "glm-ocr"},
        llm_repo=args.llm_repo,
    )
    import json

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_ocr_models(_args: argparse.Namespace) -> None:
    from app.setup_wizard import detect_runtime_profile, ocr_profile_lines

    runtime = detect_runtime_profile()
    print("\nAvailable OCR profiles")
    print("Choose based on speed, language quality, and how much extra software you want to install.")
    for line in ocr_profile_lines(runtime):
        print(f"  {line}")


def _maybe_run_first_time_setup(args: argparse.Namespace) -> None:
    if getattr(args, "command", None) not in {"login", "sync", "run", "setup"}:
        return
    from app.setup_wizard import wizard_needed, run_setup_wizard
    if getattr(args, "command", None) == "setup":
        return
    if wizard_needed():
        print("\nStickerRadar setup has not been completed yet.")
        run_setup_wizard(quick=getattr(args, "yes", False))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="StickerRadar — semantic sticker & GIF search for Telegram",
    )
    parser.add_argument(
        "--profile", default=None, metavar="NAME",
        help="Named session profile (default: reads data/.active_profile or 'main')",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # login
    p = sub.add_parser("login", help="Authenticate Telegram account (QR or phone)")
    p.add_argument("--method", choices=["qr", "phone"], default=None,
                   help="Login method (default: prompt)")

    # sync
    p = sub.add_parser("sync", help="Run the sync pipeline (metadata→download→preview→embed)")
    p.add_argument("--metadata", action="store_true", help="Metadata stage only")
    p.add_argument("--download", action="store_true", help="Download stage only")
    p.add_argument("--preview", action="store_true", help="Preview stage only")
    p.add_argument("--embed", action="store_true", help="Embed stage only")
    p.add_argument("--reindex", action="store_true",
                   help="Re-extract previews and re-embed everything (after changing FRAME_COUNT)")
    p.add_argument("--keep-previews", action="store_true", dest="keep_previews",
                   help="Do not auto-delete preview frames after embedding")
    p.add_argument("--frames", type=int, default=None, metavar="N",
                   help="Override FRAME_COUNT for this run")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Process only the first N items per stage (for testing)")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip interactive settings and start immediately")

    # status / stats
    sub.add_parser("status", help="Show pipeline status counts")
    sub.add_parser("stats", help="Show disk usage and index details")

    # prune
    sub.add_parser("prune", help="Delete local media for stickers already sent once (frees disk)")

    # fts-rebuild
    sub.add_parser("fts-rebuild", help="Rebuild FTS5 full-text index (run once after upgrading)")

    # models
    sub.add_parser("models", help="List available embedding models and upgrade instructions")

    # ocr-models
    sub.add_parser("ocr-models", help="List OCR profile options, trade-offs, and install commands")

    # ocr-benchmark
    p = sub.add_parser("ocr-benchmark", help="Benchmark one OCR backend on real StickerRadar stickers")
    p.add_argument("--backend", required=True, choices=["easyocr", "rapidocr", "glm-ocr"])
    p.add_argument("--limit", type=int, default=20, metavar="N", help="Number of sticker files to sample")
    p.add_argument("--seed", type=int, default=42, metavar="N", help="Sampling seed for reproducible runs")
    p.add_argument("--cpu", action="store_true", help="Force CPU mode where supported")
    p.add_argument("--llm-repo", default="ggml-org/GLM-OCR-GGUF:Q8_0", help="GGUF repo for glm-ocr")

    # setup
    p = sub.add_parser("setup", help="Run the first-run setup wizard")
    p.add_argument("--quick", action="store_true", help="Write recommended defaults with minimal prompts")

    # session
    p = sub.add_parser("session", help="Manage named profiles")
    p.add_argument("action", choices=["list", "use", "delete", "reset"], nargs="?")
    p.add_argument("name", nargs="?", default=None, help="Profile name (for use/delete)")

    # search
    p = sub.add_parser("search", help="Run a CLI search query (for testing)")
    p.add_argument("query", help="Search query text")
    p.add_argument("--top", type=int, default=None, metavar="N")

    # run
    sub.add_parser("run", help="Start the Telegram bot")

    args = parser.parse_args()

    if not getattr(args, "profile", None):
        args.profile = _read_active_profile()

    _maybe_run_first_time_setup(args)

    dispatch = {
        "setup": cmd_setup,
        "login": cmd_login,
        "sync": cmd_sync,
        "status": cmd_status,
        "stats": cmd_stats,
        "prune": cmd_prune,
        "fts-rebuild": cmd_fts_rebuild,
        "models": cmd_models,
        "ocr-models": cmd_ocr_models,
        "ocr-benchmark": cmd_ocr_benchmark,
        "session": cmd_session,
        "search": cmd_search,
        "run": cmd_run,
    }

    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
