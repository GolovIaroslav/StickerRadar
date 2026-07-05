from __future__ import annotations


def test_summarize_benchmark_counts_exact_and_nonempty_matches():
    from app.ocr_benchmark import BenchmarkRow, summarize_benchmark

    rows = [
        BenchmarkRow(path="one.webp", expected_text="Привет", expected_norm="привет"),
        BenchmarkRow(path="two.webp", expected_text="мир", expected_norm="мир"),
        BenchmarkRow(path="three.webp", expected_text="", expected_norm=""),
    ]
    outputs = [
        ("Привет!", "привет", 0.91),
        ("", "", 0.0),
        ("лишний текст", "лишний текст", 0.55),
    ]

    summary = summarize_benchmark(
        backend="easyocr",
        use_gpu=True,
        rows=rows,
        outputs=outputs,
        seconds=3.25,
        ram_peak_mb=512.0,
        vram_peak_delta_mb=256.0,
    )

    assert summary["backend"] == "easyocr"
    assert summary["use_gpu"] is True
    assert summary["count"] == 3
    assert summary["nonempty_outputs"] == 2
    assert summary["exact_norm_matches"] == 1
    assert summary["empty_expected_rows"] == 1
    assert summary["seconds"] == 3.25
    assert summary["per_image_ms"] == round(3.25 / 3 * 1000, 1)
    assert summary["samples"][0]["got"] == "Привет!"


def test_build_benchmark_env_disables_fallback_and_sets_backend_flags():
    from app.ocr_benchmark import build_benchmark_env

    env = build_benchmark_env(
        backend="glm-ocr",
        use_gpu=True,
        llm_repo="ggml-org/GLM-OCR-GGUF:Q8_0",
    )

    assert env["OCR_BACKEND"] == "glm-ocr"
    assert env["OCR_USE_GPU"] == "true"
    assert env["OCR_FALLBACK_ENABLED"] == "false"
    assert env["OCR_LLM_REPO"] == "ggml-org/GLM-OCR-GGUF:Q8_0"
