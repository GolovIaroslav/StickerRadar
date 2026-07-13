from __future__ import annotations

import threading
from types import SimpleNamespace


class _Message:
    def __init__(self, user_id: int, text: str | None = None) -> None:
        self.from_user = SimpleNamespace(id=user_id)
        self.chat = SimpleNamespace(id=123)
        self.text = text
        self.answers: list[str] = []

    async def answer(self, text: str) -> None:
        self.answers.append(text)


async def test_unload_command_is_owner_only(monkeypatch):
    from app import bot, config, embeddings, text_embed

    monkeypatch.setattr(config, "OWNER_USER_ID", 10)
    monkeypatch.setattr(
        embeddings,
        "unload_shared_embedder",
        lambda: (_ for _ in ()).throw(AssertionError("must not unload for non-owner")),
    )
    monkeypatch.setattr(
        text_embed,
        "unload_shared_text_embedder",
        lambda: (_ for _ in ()).throw(AssertionError("must not unload for non-owner")),
    )
    message = _Message(user_id=11)

    await bot.cmd_unload(message)

    assert message.answers == []


async def test_unload_command_releases_both_models_off_event_loop(monkeypatch):
    from app import bot, config, embeddings, text_embed

    monkeypatch.setattr(config, "OWNER_USER_ID", 10)
    main_thread = threading.get_ident()
    worker_threads: list[int] = []

    def unload() -> bool:
        worker_threads.append(threading.get_ident())
        return True

    monkeypatch.setattr(embeddings, "unload_shared_embedder", unload)
    monkeypatch.setattr(text_embed, "unload_shared_text_embedder", unload)
    message = _Message(user_id=10)

    await bot.cmd_unload(message)

    assert len(worker_threads) == 2
    assert all(thread_id != main_thread for thread_id in worker_threads)
    assert message.answers == [
        "🧹 Unloaded from RAM/VRAM: image embedding model, text embedding server."
    ]


async def test_query_reports_no_results_instead_of_a_send_failure(monkeypatch):
    from app import bot, config, db

    monkeypatch.setattr(config, "OWNER_USER_ID", 10)
    monkeypatch.setattr(db, "get_status_counts", lambda: {"embedded": 1})
    monkeypatch.setattr(db, "count_embeddings_for_model", lambda _model: 1)

    async def no_candidates(*_args, **_kwargs):
        return 0, 0, False

    monkeypatch.setattr(bot, "_send_query_page", no_candidates)
    message = _Message(user_id=10, text="несуществующий запрос")

    await bot.handle_query(message, SimpleNamespace())

    assert message.answers == ["Не нашёл результатов в индексированной коллекции."]
