"""Explicit model installation commands.

Nothing in this module is imported by runtime model loaders. Installation is an
intentional, user-visible side effect and requires --yes.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from app.model_artifacts import ARTIFACTS, artifact_ready, get_artifact, readiness_message


def _configured_path(item) -> Path | None:
    from app import config
    if item.key == "vlm:internvl2.5-1b-q8" and config.VLM_MODEL_PATH:
        return Path(config.VLM_MODEL_PATH).expanduser()
    if item.key == "vlm:qwen2-vl-2b-q4" and config.VLM_MODEL_PATH:
        configured = Path(config.VLM_MODEL_PATH).expanduser()
        if "qwen" in str(configured).lower():
            return configured
    if item.key == "text-embed:qwen3-0.6b" and config.TEXT_EMBED_MODEL_PATH:
        return Path(config.TEXT_EMBED_MODEL_PATH).expanduser()
    return Path(config.MODEL_ROOT).expanduser() / item.key.replace(":", "-")


def status_lines() -> list[str]:
    lines = []
    for item in ARTIFACTS:
        ready = artifact_ready(item, _configured_path(item))
        lines.append(f"{'READY' if ready else 'NOT INSTALLED':13} {item.key:32} {item.size:24} {item.source}")
    return lines


def _print_text_embed_env_lines(target: Path, item) -> None:
    if item.key != "text-embed:qwen3-0.6b":
        return
    model_file = target if target.is_file() else target / item.files[0]
    llama_server = shutil.which("llama-server") or ""
    print("Set these two path values in .env:")
    print(f"TEXT_EMBED_MODEL_PATH={model_file}")
    print(f"TEXT_EMBED_LLAMA_SERVER_PATH={llama_server}")


def install_artifact(key: str, destination: str | Path | None = None, *, yes: bool = False) -> int:
    item = get_artifact(key)
    if item is None:
        raise ValueError(f"Unknown artifact: {key}")
    target = (Path(destination).expanduser() if destination else _configured_path(item)).resolve()
    if artifact_ready(item, target):
        print(f"Already installed: {target}")
        _print_text_embed_env_lines(target, item)
        return 0
    print(f"About to install: {item.label}")
    print(f"  Source: {item.source}\n  Size: {item.size}\n  Destination: {target}")
    if not yes:
        print("Nothing downloaded. Re-run with --yes to confirm this explicit installation.")
        return 2
    if item.source.startswith("ggml-org/") or "GGUF" in item.source or item.capability == "embedding":
        target.mkdir(parents=True, exist_ok=True)
        repo = item.source.split(":", 1)[0]
        base = ["download", repo]
        if item.files:
            base.extend(item.files)
        if item.revision:
            base.extend(["--revision", item.revision])
        base.extend(["--local-dir", str(target)])
        command = (["hf"] + base) if shutil.which("hf") else [
            sys.executable, "-c",
            "from huggingface_hub.cli.hf import main; main()",
            "hf", *base,
        ]
        print("Running explicit installer command:", " ".join(command))
        subprocess.run(command, check=True)
        _print_text_embed_env_lines(target, item)
        return 0
    raise RuntimeError(
        f"No safe installer recipe exists for {item.source!r}. "
        "Use a local path and configure it explicitly instead of guessing files."
    )


def describe(key: str) -> None:
    item = get_artifact(key)
    if item is None:
        raise ValueError(f"Unknown artifact: {key}")
    print(item.label)
    print(f"  id: {item.key}\n  source: {item.source}\n  size: {item.size}")
    print(f"  license: {item.license}\n  notes: {item.notes}")
    print(readiness_message(item))
