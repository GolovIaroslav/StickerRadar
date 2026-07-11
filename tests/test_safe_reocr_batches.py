from __future__ import annotations

from types import SimpleNamespace


def test_run_reocr_in_safe_batches_processes_pipeline_in_chunks(monkeypatch):
    import app.__main__ as main

    calls: list[tuple[str, int | None]] = []

    config = SimpleNamespace(
        OCR_ENABLED=True,
        SCAN_CONCURRENCY=4,
        MODEL_NAME='google/siglip2-base-patch16-224',
    )

    state = {
        'preview': 5,
        'ocr': 5,
        'embed': 5,
        'text': 5,
    }

    db = SimpleNamespace(
        list_pending_previews=lambda limit=1: [object()] * min(limit, state['preview']),
        list_pending_ocr=lambda limit=1: [object()] * min(limit, state['ocr']),
        list_pending_embeddings=lambda limit=1: [object()] * min(limit, state['embed']),
        list_media_missing_text_embeddings=lambda model_name, limit=1: [object()] * min(limit, state['text']),
    )

    def run_preview(limit):
        calls.append(('preview', limit))
        state['preview'] = max(0, state['preview'] - (limit or 0))

    def run_ocr(limit):
        calls.append(('ocr', limit))
        state['ocr'] = max(0, state['ocr'] - (limit or 0))

    def run_embed(limit, keep_previews=False):
        calls.append(('embed', limit))
        state['embed'] = max(0, state['embed'] - (limit or 0))

    def run_text(limit):
        calls.append(('text', limit))
        state['text'] = max(0, state['text'] - (limit or 0))

    main._run_reocr_in_safe_batches(
        config=config,
        db=db,
        run_preview=run_preview,
        run_ocr=run_ocr,
        run_embed=run_embed,
        run_ocr_text_embed=run_text,
        batch_size=2,
        preview_workers=1,
        keep_previews=False,
    )

    assert calls == [
        ('preview', 2), ('ocr', 2), ('embed', 2), ('text', 2),
        ('preview', 2), ('ocr', 2), ('embed', 2), ('text', 2),
        ('preview', 2), ('ocr', 2), ('embed', 2), ('text', 2),
    ]
    assert config.SCAN_CONCURRENCY == 4


def test_run_reocr_in_safe_batches_restores_concurrency_after_failure(monkeypatch):
    import app.__main__ as main

    config = SimpleNamespace(
        OCR_ENABLED=True,
        SCAN_CONCURRENCY=4,
        MODEL_NAME='google/siglip2-base-patch16-224',
    )
    db = SimpleNamespace(
        list_pending_previews=lambda limit=1: [object()],
        list_pending_ocr=lambda limit=1: [],
        list_pending_embeddings=lambda limit=1: [],
        list_media_missing_text_embeddings=lambda model_name, limit=1: [],
    )

    def boom(limit):
        raise RuntimeError('preview failed')

    try:
        main._run_reocr_in_safe_batches(
            config=config,
            db=db,
            run_preview=boom,
            run_ocr=lambda limit: None,
            run_embed=lambda limit, keep_previews=False: None,
            run_ocr_text_embed=lambda limit: None,
            batch_size=3,
            preview_workers=1,
            keep_previews=False,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError('expected RuntimeError')

    assert config.SCAN_CONCURRENCY == 4
