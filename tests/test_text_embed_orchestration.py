from __future__ import annotations

from types import SimpleNamespace

import pytest


def _stub_sync_dependencies(monkeypatch):
    import app.__main__ as cli
    from app import config, db, scanner

    calls: list[tuple[str, int | None]] = []
    monkeypatch.setattr(cli, "_apply_profile", lambda _args: None)
    monkeypatch.setattr(config, "require_login_config", lambda: (1, "hash"))
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    monkeypatch.setattr(config, "TEXT_EMBED_ENABLED", True)
    monkeypatch.setattr(db, "get_conn", lambda: None)
    monkeypatch.setattr(db, "close", lambda: None)
    monkeypatch.setattr(scanner, "_run_metadata_sync", lambda _limit: None)
    monkeypatch.setattr(scanner, "_run_download", lambda _limit: None)
    monkeypatch.setattr(scanner, "_run_preview", lambda _limit: None)
    monkeypatch.setattr(scanner, "_run_embed", lambda _limit, keep_previews=False: None)
    monkeypatch.setattr(scanner, "_run_ocr_text_embed", lambda _limit: None)
    monkeypatch.setattr(scanner, "_run_ocr", lambda limit: calls.append(("ocr", limit)))
    monkeypatch.setattr(
        scanner,
        "_run_text_embed_backfill",
        lambda limit: calls.append(("text-backfill", limit)),
    )
    return cli, config, db, calls


def test_sync_ocr_updates_dedicated_text_embeddings(monkeypatch):
    cli, _config, _db, calls = _stub_sync_dependencies(monkeypatch)

    cli.cmd_sync(SimpleNamespace(profile="main", ocr=True, limit=7))

    assert calls == [("ocr", 7), ("text-backfill", None)]


def test_sync_reocr_updates_all_dedicated_text_embeddings(monkeypatch):
    cli, _config, db, calls = _stub_sync_dependencies(monkeypatch)
    monkeypatch.setattr(db, "force_reocr", lambda: 0)
    monkeypatch.setattr(db, "force_reindex", lambda: 0)
    monkeypatch.setattr(
        cli,
        "_run_reocr_in_safe_batches",
        lambda **_kwargs: calls.append(("reocr", None)),
    )

    cli.cmd_sync(SimpleNamespace(profile="main", reocr=True, limit=7))

    assert calls == [("reocr", None), ("text-backfill", None)]


def test_model_install_subcommand_is_the_explicit_download_confirmation(monkeypatch):
    import app.__main__ as cli
    from app import model_installer

    received: list[tuple[str, str | None, bool]] = []
    monkeypatch.setattr(
        model_installer,
        "install_artifact",
        lambda artifact_id, path, *, yes: received.append((artifact_id, path, yes)) or 0,
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_model(
            SimpleNamespace(
                model_action="install",
                artifact_id="text-embed:qwen3-0.6b",
                path=None,
                yes=False,
            )
        )

    assert exit_info.value.code == 0
    assert received == [("text-embed:qwen3-0.6b", None, True)]
