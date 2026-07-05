from __future__ import annotations


def test_ocr_profile_lines_have_single_recommended_entry_on_gpu_hosts():
    from app.setup_wizard import RuntimeProfile, ocr_profile_lines

    runtime = RuntimeProfile(
        has_gpu=True,
        gpu_name="RTX 3060",
        gpu_total_gb=6.0,
        gpu_free_gb=4.0,
        llama_cpp_available=True,
        ffmpeg_available=True,
    )

    lines = ocr_profile_lines(runtime)
    recommended = [line for line in lines if "recommended" in line]

    assert len(recommended) == 1
    assert "EasyOCR scene-text" in recommended[0]


def test_ocr_profile_lines_have_single_recommended_entry_on_cpu_hosts():
    from app.setup_wizard import RuntimeProfile, ocr_profile_lines

    runtime = RuntimeProfile(
        has_gpu=False,
        gpu_name=None,
        gpu_total_gb=None,
        gpu_free_gb=None,
        llama_cpp_available=False,
        ffmpeg_available=True,
    )

    lines = ocr_profile_lines(runtime)
    recommended = [line for line in lines if "recommended" in line]

    assert len(recommended) == 1
    assert "RapidOCR" in recommended[0]
