from __future__ import annotations

import json
import os
import random
import resource
import sqlite3
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import config


@dataclass
class EvalItem:
    media_id: int
    frame_id: int
    tg_document_id: str
    image_path: str
    set_title: str | None
    set_short_name: str | None
    emoji: str | None
    ocr_text: str | None
    media_kind: str
    sticker_format: str | None
    local_path: str | None
    bot_file_id: str | None
    bot_send_method: str | None
    bot_cache_status: str | None
    is_favorite: int
    is_recent: int


def choose_queries_for_row(row: dict[str, Any]) -> list[str]:
    queries: list[str] = []

    def add(text: str | None) -> None:
        text = (text or "").strip()
        if text and text not in queries:
            queries.append(text)

    add(row.get("ocr_text"))
    add(row.get("set_title"))

    short_name = (row.get("set_short_name") or "").strip()
    if short_name:
        add(short_name.replace("_", " "))
        add(short_name)

    add(row.get("emoji"))
    return queries


def score_ranks(ranks: list[int | None], *, top_ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    total = len(ranks) or 1
    mrr = sum((1.0 / rank) for rank in ranks if rank) / total
    out: dict[str, float] = {"mrr": round(mrr, 3)}
    for k in top_ks:
        hits = sum(1 for rank in ranks if rank is not None and rank <= k)
        out[f"hit@{k}"] = round(hits / total, 3)
    return out


class _VramMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_mb = self._read_mb()
        self.peak_mb = self.start_mb

    def _read_mb(self) -> float:
        try:
            import subprocess

            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if proc.returncode != 0:
                return 0.0
            raw = proc.stdout.strip().splitlines()
            return float(raw[0]) if raw else 0.0
        except Exception:
            return 0.0

    def start(self) -> None:
        def run() -> None:
            while not self._stop.is_set():
                self.peak_mb = max(self.peak_mb, self._read_mb())
                time.sleep(0.2)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        return max(0.0, self.peak_mb - self.start_mb)


@contextmanager
def temporary_eval_config(*, model_name: str, device: str, db_path: Path):
    from app import db, search

    old_model = config.MODEL_NAME
    old_device = config.DEVICE
    old_db_path = config.DB_PATH
    try:
        db.close()
        search.invalidate_cache()
        config.MODEL_NAME = model_name
        config.DEVICE = device
        config.DB_PATH = db_path
        yield
    finally:
        db.close()
        search.invalidate_cache()
        config.MODEL_NAME = old_model
        config.DEVICE = old_device
        config.DB_PATH = old_db_path


def _fetch_eval_items(*, limit: int, seed: int) -> list[EvalItem]:
    source_db = sqlite3.connect(str(config.DB_PATH))
    source_db.row_factory = sqlite3.Row
    rows = source_db.execute(
        """
        SELECT mi.id AS media_id,
               mf.id AS frame_id,
               mi.tg_document_id,
               mf.preview_path,
               mi.local_path,
               mi.set_title,
               mi.set_short_name,
               mi.emoji,
               mi.ocr_text,
               mi.media_kind,
               mi.sticker_format,
               mi.bot_file_id,
               mi.bot_send_method,
               mi.bot_cache_status,
               mi.is_favorite,
               mi.is_recent
        FROM media_items mi
        JOIN media_frames mf ON mf.media_id = mi.id
        WHERE mi.preview_status = 'ok'
          AND mf.frame_index = 0
          AND (
                (mf.preview_path IS NOT NULL AND trim(mf.preview_path) != '')
                OR (mi.local_path IS NOT NULL AND trim(mi.local_path) != '')
              )
        ORDER BY mi.id
        """
    ).fetchall()
    source_db.close()

    def resolve_image_path(row: sqlite3.Row) -> str | None:
        preview_path = (row["preview_path"] or "").strip()
        if preview_path and Path(preview_path).exists():
            return preview_path
        local_path = (row["local_path"] or "").strip()
        if not local_path or not Path(local_path).exists():
            return None
        if (row["sticker_format"] or "") == "static":
            return local_path
        suffix = Path(local_path).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return local_path
        return None

    items = []
    for r in rows:
        image_path = resolve_image_path(r)
        if not image_path:
            continue
        items.append(
            EvalItem(
                media_id=int(r["media_id"]),
                frame_id=int(r["frame_id"]),
                tg_document_id=r["tg_document_id"],
                image_path=image_path,
                set_title=r["set_title"],
                set_short_name=r["set_short_name"],
                emoji=r["emoji"],
                ocr_text=r["ocr_text"],
                media_kind=r["media_kind"],
                sticker_format=r["sticker_format"],
                local_path=r["local_path"],
                bot_file_id=r["bot_file_id"],
                bot_send_method=r["bot_send_method"],
                bot_cache_status=r["bot_cache_status"],
                is_favorite=int(r["is_favorite"] or 0),
                is_recent=int(r["is_recent"] or 0),
            )
        )

    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:limit]


def _build_eval_queries(items: list[EvalItem], *, max_queries_per_item: int = 3) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
    for item in items:
        queries = choose_queries_for_row(
            {
                "ocr_text": item.ocr_text,
                "set_title": item.set_title,
                "set_short_name": item.set_short_name,
                "emoji": item.emoji,
            }
        )
        for query in queries[:max_queries_per_item]:
            pairs.append((item.media_id, query))
    return pairs


def _populate_temp_db(db_path: Path, items: list[EvalItem], vectors: list[bytes], *, model_name: str, dim: int) -> None:
    from app import db as app_db

    with temporary_eval_config(model_name=model_name, device=config.DEVICE, db_path=db_path):
        conn = app_db.get_conn()
        for item, vector in zip(items, vectors):
            conn.execute(
                """
                INSERT INTO media_items (
                    id, tg_document_id, media_kind, sticker_format, local_path,
                    bot_file_id, bot_send_method, bot_cache_status,
                    set_short_name, set_title, emoji,
                    is_favorite, is_recent,
                    preview_status, embed_status, ocr_status, ocr_text,
                    download_status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ok', 'ok', 'ok', ?, 'ok', CURRENT_TIMESTAMP)
                """,
                (
                    item.media_id,
                    item.tg_document_id,
                    item.media_kind,
                    item.sticker_format,
                    item.local_path,
                    item.bot_file_id,
                    item.bot_send_method,
                    item.bot_cache_status or "missing",
                    item.set_short_name,
                    item.set_title,
                    item.emoji,
                    item.is_favorite,
                    item.is_recent,
                    item.ocr_text or "",
                ),
            )
            conn.execute(
                """
                INSERT INTO media_frames (id, media_id, frame_index, frame_pos, preview_path, width, height)
                VALUES (?, ?, 0, 0.0, ?, NULL, NULL)
                """,
                (item.frame_id, item.media_id, item.image_path),
            )
            conn.execute(
                """
                INSERT INTO frame_embeddings (frame_id, model_name, dim, vector)
                VALUES (?, ?, ?, ?)
                """,
                (item.frame_id, model_name, dim, vector),
            )
        conn.commit()


def benchmark_model(*, model_name: str, device: str, sample_size: int, seed: int) -> dict[str, Any]:
    from app import search
    from app.embeddings import Embedder

    items = _fetch_eval_items(limit=sample_size, seed=seed)
    queries = _build_eval_queries(items)
    if not items or not queries:
        raise RuntimeError("Not enough preview/OCR data to build retrieval benchmark set")

    monitor = _VramMonitor()
    monitor.start()
    ram_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    started = time.perf_counter()

    embedder = None
    tmp_db_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="stickerradar-retrieval-", suffix=".sqlite", delete=False) as tmp:
            tmp_db_path = Path(tmp.name)

        with temporary_eval_config(model_name=model_name, device=device, db_path=config.DB_PATH):
            embedder = Embedder()
            image_vecs = embedder.embed_images([Path(item.image_path) for item in items])

        dim = len(image_vecs[0])
        blobs = [vec.astype("float32").tobytes() for vec in image_vecs]
        _populate_temp_db(tmp_db_path, items, blobs, model_name=model_name, dim=dim)

        ranks: list[int | None] = []
        sample_results: list[dict[str, Any]] = []
        with temporary_eval_config(model_name=model_name, device=device, db_path=tmp_db_path):
            import app.embeddings as embeddings_mod

            old_shared = embeddings_mod._shared
            embeddings_mod._shared = embedder
            try:
                for target_media_id, query in queries:
                    results = search.search(query, top_k=10)
                    rank = next((idx for idx, row in enumerate(results, start=1) if row.media_id == target_media_id), None)
                    ranks.append(rank)
                    if len(sample_results) < 8:
                        sample_results.append(
                            {
                                "query": query,
                                "target_media_id": target_media_id,
                                "rank": rank,
                                "top_results": [r.media_id for r in results[:5]],
                            }
                        )
            finally:
                embeddings_mod._shared = old_shared
    finally:
        if embedder is not None:
            try:
                embedder.unload()
            except Exception:
                pass
        seconds = time.perf_counter() - started
        ram_peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        vram_delta_mb = monitor.stop()
        if tmp_db_path is not None:
            Path(tmp_db_path).unlink(missing_ok=True)

    metrics = score_ranks(ranks)
    return {
        "model_name": model_name,
        "device": device,
        "sample_size": len(items),
        "query_count": len(queries),
        "seconds": round(seconds, 3),
        "per_query_ms": round(seconds / len(queries) * 1000, 1),
        "ram_peak_mb": round(ram_peak_mb, 1),
        "ram_delta_mb": round(max(0.0, ram_peak_mb - ram_before), 1),
        "vram_peak_delta_mb": round(vram_delta_mb, 1),
        **metrics,
        "samples": sample_results,
    }


def run_benchmark(*, model_name: str, device: str, sample_size: int = 120, seed: int = 42) -> dict[str, Any]:
    return benchmark_model(model_name=model_name, device=device, sample_size=sample_size, seed=seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(model_name=args.model, device=args.device, sample_size=args.sample_size, seed=args.seed), ensure_ascii=False, indent=2))
