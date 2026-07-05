from __future__ import annotations


def test_choose_queries_prefers_ocr_then_metadata_then_emoji():
    from app.retrieval_benchmark import choose_queries_for_row

    row = {
        "ocr_text": "хохлы что с ебалом",
        "set_title": "Funny Pack",
        "set_short_name": "funny_pack",
        "emoji": "😂",
    }

    queries = choose_queries_for_row(row)

    assert queries[0] == "хохлы что с ебалом"
    assert "Funny Pack" in queries
    assert "funny pack" in queries
    assert "😂" in queries


def test_choose_queries_deduplicates_and_skips_empty_values():
    from app.retrieval_benchmark import choose_queries_for_row

    row = {
        "ocr_text": "",
        "set_title": "Same",
        "set_short_name": "same",
        "emoji": "",
    }

    queries = choose_queries_for_row(row)

    assert queries == ["Same", "same"]


def test_score_ranks_counts_hits_at_top_k_thresholds():
    from app.retrieval_benchmark import score_ranks

    metrics = score_ranks([1, 2, 5, None], top_ks=(1, 3, 5))

    assert metrics["mrr"] == 0.425
    assert metrics["hit@1"] == 0.25
    assert metrics["hit@3"] == 0.5
    assert metrics["hit@5"] == 0.75
