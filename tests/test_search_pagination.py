from __future__ import annotations

from dataclasses import replace

from app.search import SearchResult
from app.sender import select_sendable_page


def _result(media_id: int, pack: str) -> SearchResult:
    return SearchResult(
        media_id=media_id,
        tg_document_id=f"doc-{media_id}",
        media_kind="sticker",
        sticker_format="static",
        local_path=f"/tmp/{media_id}.webp",
        bot_file_id=None,
        bot_send_method=None,
        bot_cache_status=None,
        set_short_name=pack,
        set_title=pack,
        emoji=None,
        is_favorite=False,
        is_recent=False,
        score=1.0,
    )


def test_page_fills_ten_items_beyond_recent_results_and_returns_candidate_cursor():
    results = [_result(i, f"pack-{i}") for i in range(1, 16)]

    page, next_cursor = select_sendable_page(
        results,
        cursor=0,
        page_size=10,
        recent_media_ids=set(range(1, 7)),
        recent_packs=set(),
    )

    assert [item.media_id for item in page] == list(range(7, 15 + 1))
    assert len(page) == 9
    assert next_cursor == 15


def test_next_page_uses_candidate_cursor_not_number_successfully_sent():
    results = [_result(i, f"pack-{i}") for i in range(1, 31)]

    first, cursor = select_sendable_page(
        results,
        cursor=0,
        page_size=10,
        recent_media_ids=set(range(1, 7)),
        recent_packs=set(),
    )
    second, next_cursor = select_sendable_page(
        results,
        cursor=cursor,
        page_size=10,
        recent_media_ids={item.media_id for item in first},
        recent_packs={item.set_short_name for item in first},
    )

    assert len(first) == 10
    assert [item.media_id for item in first] == list(range(7, 17))
    assert [item.media_id for item in second] == list(range(17, 27))
    assert next_cursor == 26
