from __future__ import annotations


def test_ocr_text_score_finds_partial_russian_meaningful_overlap():
    from app.search import ocr_text_match_score

    score = ocr_text_match_score(
        "Я чувствую себя плохо",
        "те плохо бнись 0б стенку и сuает хоппо",
    )

    assert score >= 0.35


def test_ocr_text_score_prioritizes_exact_help_phrase_over_generic_visual_match():
    from app.search import ocr_text_match_score

    target = ocr_text_match_score(
        "помогите мне чёрт!",
        "да помогите блять жак фреско",
    )
    unrelated = ocr_text_match_score(
        "помогите мне чёрт!",
        "мм ооаоаа",
    )

    assert target > unrelated
    assert target >= 0.35


def test_ocr_text_score_handles_yo_punctuation_and_latin_lookalikes():
    from app.search import ocr_text_match_score

    assert ocr_text_match_score("чёрт помогите", "да помогите блять") > 0.0
    assert ocr_text_match_score("плохо", "плoxo") > 0.0


def test_ocr_text_score_recognizes_garbled_mne_plokho_context():
    from app.search import ocr_text_match_score

    contextual = ocr_text_match_score("Я чувствую себя плохо", "те плохо бнись")
    unrelated = ocr_text_match_score("Я чувствую себя плохо", "плохое место")

    assert contextual > unrelated


def test_ocr_text_score_recognizes_colloquial_bad_feeling_synonym():
    from app.search import ocr_text_match_score

    contextual = ocr_text_match_score(
        "я чувствую себя хуёво",
        "те плохо бнись об стенку и станет хорошо",
    )
    unrelated = ocr_text_match_score("я чувствую себя хуёво", "плохое место")

    assert contextual >= 0.55
    assert contextual > unrelated


def test_ocr_text_score_ignores_function_words_that_create_false_matches():
    from app.search import ocr_text_match_score

    assert ocr_text_match_score("не ругайся матом", "а ты хороший мальчик") == 0.0


def test_ocr_text_score_does_not_match_one_character_ocr_noise_as_a_word():
    from app.search import ocr_text_match_score

    assert ocr_text_match_score("не ругайся матом", "m") == 0.0


def test_fts_query_excludes_russian_function_words():
    from app.search import _fts_query_str

    assert _fts_query_str("не ругайся матом") == '"ругайся"* OR "матом"*'
