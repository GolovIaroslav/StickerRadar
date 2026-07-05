from __future__ import annotations

import argparse
import json
import os
import random
import resource
import shutil
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app import config
from app.ocr import is_backend_available, normalize_text, ocr_frames


@dataclass(frozen=True)
class BenchmarkRow:
    path: str
    expected_text: str
    expected_norm: str


class VramSampler:
    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_mb = self._query_used_mb()
        self.peak_mb = self.baseline_mb

    def _query_used_mb(self) -> float | None:
        if not shutil.which("nvidia-smi"):
            return None
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            ).strip().splitlines()
            if not out:
                return None
            return max(float(line.strip()) for line in out if line.strip())
        except Exception:
            return None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            used = self._query_used_mb()
            if used is not None:
                if self.peak_mb is None:
                    self.peak_mb = used
                else:
                    self.peak_mb = max(self.peak_mb, used)

    def start(self) -> None:
        if self.baseline_mb is None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join(timeout=2)
        if self.baseline_mb is None or self.peak_mb is None:
            return None
        return max(0.0, self.peak_mb - self.baseline_mb)


def build_benchmark_env(*, backend: str, use_gpu: bool, llm_repo: str) -> dict[str, str]:
    env = dict(os.environ)
    env["OCR_BACKEND"] = backend
    env["OCR_USE_GPU"] = "true" if use_gpu else "false"
    env["OCR_FALLBACK_ENABLED"] = "false"
    env["OCR_LLM_REPO"] = llm_repo
    return env


def _maxrss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux returns KiB, macOS returns bytes.
    if usage > 1024 * 1024 * 32:
        return round(usage / (1024 * 1024), 1)
    return round(usage / 1024, 1)


def sample_rows(*, limit: int, seed: int) -> list[BenchmarkRow]:
    conn = sqlite3.connect(config.DB_PATH)
    cur = conn.cursor()
    rows = cur.execute(
        """
        select local_path, ocr_text
        from media_items
        where ocr_text != '' and local_path is not null
        """
    ).fetchall()
    conn.close()

    existing = [
        BenchmarkRow(path=path, expected_text=text, expected_norm=normalize_text(text))
        for path, text in rows
        if path and Path(path).exists()
    ]
    if not existing:
        raise RuntimeError("No benchmarkable sticker files with OCR ground truth were found.")

    rng = random.Random(seed)
    if limit >= len(existing):
        rng.shuffle(existing)
        return existing
    return rng.sample(existing, limit)


def summarize_benchmark(
    *,
    backend: str,
    use_gpu: bool,
    rows: list[BenchmarkRow],
    outputs: list[tuple[str, str, float]],
    seconds: float,
    ram_peak_mb: float | None,
    vram_peak_delta_mb: float | None,
) -> dict[str, object]:
    exact_norm_matches = 0
    nonempty_outputs = 0
    empty_expected_rows = 0
    samples: list[dict[str, object]] = []

    for row, output in zip(rows, outputs, strict=True):
        raw, norm, conf = output
        if raw.strip():
            nonempty_outputs += 1
        if not row.expected_norm:
            empty_expected_rows += 1
        if row.expected_norm and norm == row.expected_norm:
            exact_norm_matches += 1
        if len(samples) < 8:
            samples.append(
                {
                    "path": Path(row.path).name,
                    "expected": row.expected_text[:120],
                    "got": raw[:160],
                    "norm": norm[:160],
                    "conf": round(conf, 3),
                }
            )

    count = len(rows)
    return {
        "backend": backend,
        "use_gpu": use_gpu,
        "count": count,
        "seconds": round(seconds, 3),
        "per_image_ms": round(seconds / count * 1000, 1) if count else None,
        "ram_peak_mb": ram_peak_mb,
        "vram_peak_delta_mb": round(vram_peak_delta_mb, 1) if vram_peak_delta_mb is not None else None,
        "resource_note": (
            "RAM / VRAM are measured from the Python benchmark process only; external llama.cpp subprocess usage is not fully captured."
            if backend == "glm-ocr"
            else "RAM / VRAM reflect the OCR process used inside this Python run."
        ),
        "nonempty_outputs": nonempty_outputs,
        "exact_norm_matches": exact_norm_matches,
        "empty_expected_rows": empty_expected_rows,
        "samples": samples,
    }


def run_benchmark(
    *,
    backend: str,
    limit: int,
    seed: int,
    use_gpu: bool,
    llm_repo: str,
) -> dict[str, object]:
    if not is_backend_available(backend):
        raise RuntimeError(f"OCR backend '{backend}' is not available in this environment.")

    rows = sample_rows(limit=limit, seed=seed)

    config.OCR_BACKEND = backend
    config.OCR_USE_GPU = use_gpu
    config.OCR_FALLBACK_ENABLED = False
    config.OCR_LLM_REPO = llm_repo

    paths = [Path(row.path) for row in rows]
    vram = VramSampler()
    vram.start()
    t0 = time.time()
    outputs = ocr_frames(paths)
    seconds = time.time() - t0
    vram_delta_mb = vram.stop()

    return summarize_benchmark(
        backend=backend,
        use_gpu=use_gpu,
        rows=rows,
        outputs=outputs,
        seconds=seconds,
        ram_peak_mb=_maxrss_mb(),
        vram_peak_delta_mb=vram_delta_mb,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app ocr-benchmark",
        description="Benchmark one OCR backend on real StickerRadar stickers.",
    )
    parser.add_argument("--backend", required=True, choices=["easyocr", "rapidocr", "glm-ocr"])
    parser.add_argument("--limit", type=int, default=20, help="Number of real sticker files to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode where supported")
    parser.add_argument(
        "--llm-repo",
        default="ggml-org/GLM-OCR-GGUF:Q8_0",
        help="GGUF repo for glm-ocr backend",
    )
    args = parser.parse_args()

    use_gpu = not args.cpu and args.backend in {"easyocr", "glm-ocr"}
    summary = run_benchmark(
        backend=args.backend,
        limit=args.limit,
        seed=args.seed,
        use_gpu=use_gpu,
        llm_repo=args.llm_repo,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
