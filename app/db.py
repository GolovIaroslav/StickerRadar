from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from app import config

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS media_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,

  -- Telegram MTProto identity
  tg_document_id TEXT NOT NULL UNIQUE,
  access_hash TEXT,
  file_reference BLOB,

  -- Classification
  media_kind TEXT NOT NULL,          -- sticker | gif
  sticker_format TEXT,               -- static | animated | video | unknown
  mime_type TEXT,
  file_ext TEXT,

  -- Source flags (same document can appear in multiple sources)
  is_installed INTEGER DEFAULT 0,
  is_favorite  INTEGER DEFAULT 0,
  is_recent    INTEGER DEFAULT 0,
  is_saved_gif INTEGER DEFAULT 0,

  -- Sticker-pack metadata (NULL for GIFs)
  set_id          TEXT,
  set_access_hash TEXT,
  set_short_name  TEXT,
  set_title       TEXT,
  emoji           TEXT,

  -- Local storage
  local_path TEXT,
  sha256     TEXT,
  file_size  INTEGER,

  -- Bot API file_id cache
  bot_file_id        TEXT,
  bot_file_unique_id TEXT,
  bot_send_method    TEXT,              -- sticker | animation
  bot_cache_status   TEXT DEFAULT 'missing',  -- missing | cached | failed

  -- Pipeline state
  download_status TEXT DEFAULT 'pending',  -- pending | ok | failed
  preview_status  TEXT DEFAULT 'pending',
  embed_status    TEXT DEFAULT 'pending',
  ocr_status      TEXT DEFAULT 'pending',
  ocr_text        TEXT,                    -- aggregated normalized OCR text from all frames
  last_error      TEXT,

  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS media_frames (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  media_id    INTEGER NOT NULL REFERENCES media_items(id) ON DELETE CASCADE,
  frame_index INTEGER NOT NULL,
  frame_pos   REAL NOT NULL,           -- 0.0..1.0; static sticker = 0.0
  preview_path TEXT NOT NULL,
  width  INTEGER,
  height INTEGER,
  UNIQUE(media_id, frame_index)
);

CREATE TABLE IF NOT EXISTS frame_embeddings (
  frame_id   INTEGER PRIMARY KEY REFERENCES media_frames(id) ON DELETE CASCADE,
  model_name TEXT NOT NULL,
  dim        INTEGER NOT NULL,
  vector     BLOB NOT NULL,            -- normalized float32 bytes
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS app_state (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_media_sources    ON media_items(is_installed, is_favorite, is_recent, is_saved_gif);
CREATE INDEX IF NOT EXISTS idx_media_set        ON media_items(set_short_name);
CREATE INDEX IF NOT EXISTS idx_media_dl_status  ON media_items(download_status);
CREATE INDEX IF NOT EXISTS idx_media_prev_status ON media_items(preview_status);
CREATE INDEX IF NOT EXISTS idx_media_emb_status ON media_items(embed_status);
CREATE INDEX IF NOT EXISTS idx_media_ocr_status ON media_items(ocr_status);
CREATE INDEX IF NOT EXISTS idx_frame_emb_model  ON frame_embeddings(model_name);

-- FTS5 full-text index over sticker metadata for hybrid BM25+vector search.
CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(
    set_title, set_short_name, emoji,
    content=media_items,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS media_fts_ai AFTER INSERT ON media_items BEGIN
    INSERT INTO media_fts(rowid, set_title, set_short_name, emoji)
    VALUES (new.id, new.set_title, new.set_short_name, new.emoji);
END;

CREATE TRIGGER IF NOT EXISTS media_fts_ad AFTER DELETE ON media_items BEGIN
    INSERT INTO media_fts(media_fts, rowid, set_title, set_short_name, emoji)
    VALUES ('delete', old.id, old.set_title, old.set_short_name, old.emoji);
END;

CREATE TRIGGER IF NOT EXISTS media_fts_au AFTER UPDATE ON media_items BEGIN
    INSERT INTO media_fts(media_fts, rowid, set_title, set_short_name, emoji)
    VALUES ('delete', old.id, old.set_title, old.set_short_name, old.emoji);
    INSERT INTO media_fts(rowid, set_title, set_short_name, emoji)
    VALUES (new.id, new.set_title, new.set_short_name, new.emoji);
END;

-- OCR text per preview frame.
CREATE TABLE IF NOT EXISTS frame_ocr (
  frame_id   INTEGER PRIMARY KEY REFERENCES media_frames(id) ON DELETE CASCADE,
  raw_text   TEXT NOT NULL DEFAULT '',
  norm_text  TEXT NOT NULL DEFAULT '',
  confidence REAL NOT NULL DEFAULT 0.0
);

-- Separate FTS5 for OCR text — kept separate from media_fts so no migration
-- of the existing metadata index is needed when adding this feature.
CREATE VIRTUAL TABLE IF NOT EXISTS media_ocr_fts USING fts5(
    ocr_text,
    content=media_items,
    content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS ocr_fts_ai AFTER INSERT ON media_items BEGIN
    INSERT INTO media_ocr_fts(rowid, ocr_text) VALUES (new.id, new.ocr_text);
END;

CREATE TRIGGER IF NOT EXISTS ocr_fts_au AFTER UPDATE OF ocr_text ON media_items BEGIN
    INSERT INTO media_ocr_fts(media_ocr_fts, rowid, ocr_text)
        VALUES ('delete', old.id, old.ocr_text);
    INSERT INTO media_ocr_fts(rowid, ocr_text) VALUES (new.id, new.ocr_text);
END;

CREATE TRIGGER IF NOT EXISTS ocr_fts_ad AFTER DELETE ON media_items BEGIN
    INSERT INTO media_ocr_fts(media_ocr_fts, rowid, ocr_text)
        VALUES ('delete', old.id, old.ocr_text);
END;
"""

# ---------------------------------------------------------------------------
# Connection — one connection per thread (WAL allows concurrent readers)
# ---------------------------------------------------------------------------

_local = threading.local()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns/tables introduced after the initial schema — idempotent."""
    tbl = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='media_items'"
    ).fetchone()
    if tbl is None:
        return  # fresh DB — _SCHEMA will create everything with all columns

    existing = {r[1] for r in conn.execute("PRAGMA table_info(media_items)")}
    changed = False
    if "ocr_status" not in existing:
        conn.execute("ALTER TABLE media_items ADD COLUMN ocr_status TEXT DEFAULT 'pending'")
        changed = True
    if "ocr_text" not in existing:
        conn.execute("ALTER TABLE media_items ADD COLUMN ocr_text TEXT")
        changed = True
    if changed:
        conn.commit()


def get_conn() -> sqlite3.Connection:
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn is None:
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(config.DB_PATH))
        conn.row_factory = sqlite3.Row
        _migrate(conn)  # add new columns to existing DBs before schema runs
        conn.executescript(_SCHEMA)  # idempotent — CREATE TABLE IF NOT EXISTS
        conn.commit()
        _local.conn = conn
    return conn


def close() -> None:
    conn: sqlite3.Connection | None = getattr(_local, "conn", None)
    if conn:
        conn.close()
        _local.conn = None


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def upsert_media_item(
    *,
    tg_document_id: str,
    access_hash: str,
    file_reference: bytes,
    media_kind: str,
    sticker_format: str | None,
    mime_type: str,
    file_ext: str,
    emoji: str | None,
    set_id: str | None,
    set_access_hash: str | None,
    set_short_name: str | None,
    set_title: str | None,
    # source flags
    is_installed: bool = False,
    is_favorite: bool = False,
    is_recent: bool = False,
    is_saved_gif: bool = False,
) -> int:
    """Insert or update a media item. Returns the row id."""
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO media_items (
            tg_document_id, access_hash, file_reference,
            media_kind, sticker_format, mime_type, file_ext,
            emoji, set_id, set_access_hash, set_short_name, set_title,
            is_installed, is_favorite, is_recent, is_saved_gif,
            updated_at
        ) VALUES (
            :tg_document_id, :access_hash, :file_reference,
            :media_kind, :sticker_format, :mime_type, :file_ext,
            :emoji, :set_id, :set_access_hash, :set_short_name, :set_title,
            :is_installed, :is_favorite, :is_recent, :is_saved_gif,
            CURRENT_TIMESTAMP
        )
        ON CONFLICT(tg_document_id) DO UPDATE SET
            access_hash     = excluded.access_hash,
            file_reference  = excluded.file_reference,
            is_installed    = MAX(is_installed, excluded.is_installed),
            is_favorite     = MAX(is_favorite,  excluded.is_favorite),
            is_recent       = MAX(is_recent,    excluded.is_recent),
            is_saved_gif    = MAX(is_saved_gif, excluded.is_saved_gif),
            emoji           = COALESCE(excluded.emoji, emoji),
            set_id          = COALESCE(excluded.set_id, set_id),
            set_access_hash = COALESCE(excluded.set_access_hash, set_access_hash),
            set_short_name  = COALESCE(excluded.set_short_name, set_short_name),
            set_title       = COALESCE(excluded.set_title, set_title),
            updated_at      = CURRENT_TIMESTAMP
        """,
        {
            "tg_document_id": tg_document_id,
            "access_hash": access_hash,
            "file_reference": file_reference,
            "media_kind": media_kind,
            "sticker_format": sticker_format,
            "mime_type": mime_type,
            "file_ext": file_ext,
            "emoji": emoji,
            "set_id": set_id,
            "set_access_hash": set_access_hash,
            "set_short_name": set_short_name,
            "set_title": set_title,
            "is_installed": int(is_installed),
            "is_favorite": int(is_favorite),
            "is_recent": int(is_recent),
            "is_saved_gif": int(is_saved_gif),
        },
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM media_items WHERE tg_document_id = ?", (tg_document_id,)
    ).fetchone()
    return row["id"]


def mark_download_ok(media_id: int, local_path: str, sha256: str, file_size: int) -> None:
    get_conn().execute(
        """UPDATE media_items
           SET local_path=?, sha256=?, file_size=?, download_status='ok', last_error=NULL, updated_at=CURRENT_TIMESTAMP
           WHERE id=?""",
        (local_path, sha256, file_size, media_id),
    )
    get_conn().commit()


def mark_download_failed(media_id: int, error: str) -> None:
    get_conn().execute(
        "UPDATE media_items SET download_status='failed', last_error=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (error, media_id),
    )
    get_conn().commit()


def mark_preview_ok(media_id: int) -> None:
    get_conn().execute(
        "UPDATE media_items SET preview_status='ok', updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (media_id,),
    )
    get_conn().commit()


def mark_preview_failed(media_id: int, error: str) -> None:
    get_conn().execute(
        "UPDATE media_items SET preview_status='failed', last_error=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (error, media_id),
    )
    get_conn().commit()


def mark_embed_ok(media_id: int) -> None:
    get_conn().execute(
        "UPDATE media_items SET embed_status='ok', updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (media_id,),
    )
    get_conn().commit()


def mark_embed_failed(media_id: int, error: str) -> None:
    get_conn().execute(
        "UPDATE media_items SET embed_status='failed', last_error=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (error, media_id),
    )
    get_conn().commit()


def save_bot_file_id(
    media_id: int, file_id: str, file_unique_id: str, send_method: str
) -> None:
    get_conn().execute(
        """UPDATE media_items
           SET bot_file_id=?, bot_file_unique_id=?, bot_send_method=?, bot_cache_status='cached', updated_at=CURRENT_TIMESTAMP
           WHERE id=?""",
        (file_id, file_unique_id, send_method, media_id),
    )
    get_conn().commit()


def upsert_frame(
    *,
    media_id: int,
    frame_index: int,
    frame_pos: float,
    preview_path: str,
    width: int | None,
    height: int | None,
) -> int:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO media_frames (media_id, frame_index, frame_pos, preview_path, width, height)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(media_id, frame_index) DO UPDATE SET
            preview_path=excluded.preview_path,
            width=excluded.width,
            height=excluded.height
        """,
        (media_id, frame_index, frame_pos, preview_path, width, height),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM media_frames WHERE media_id=? AND frame_index=?",
        (media_id, frame_index),
    ).fetchone()
    return row["id"]


def upsert_frame_embedding(
    *, frame_id: int, model_name: str, dim: int, vector_bytes: bytes
) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO frame_embeddings (frame_id, model_name, dim, vector)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(frame_id) DO UPDATE SET
            model_name=excluded.model_name,
            dim=excluded.dim,
            vector=excluded.vector,
            created_at=CURRENT_TIMESTAMP
        """,
        (frame_id, model_name, dim, vector_bytes),
    )
    conn.commit()


def mark_items_for_model(model_name: str) -> int:
    """
    Find downloaded items that have NO embedding for `model_name` and mark them
    pending for embedding (so a plain `sync` re-does only what's needed when the
    model changes). Preview is re-flagged ONLY when the preview frames are
    missing on disk — so an interrupted run resumes at embed instead of
    re-extracting frames it already has. Returns the number of items flagged.
    """
    from pathlib import Path

    conn = get_conn()
    rows = conn.execute(
        """
        SELECT id, local_path FROM media_items
        WHERE download_status='ok'
          AND id NOT IN (
            SELECT DISTINCT mf.media_id
            FROM media_frames mf
            JOIN frame_embeddings fe ON fe.frame_id = mf.id
            WHERE fe.model_name = ?
          )
        """,
        (model_name,),
    ).fetchall()

    redownload = 0
    for r in rows:
        media_id = r["id"]
        frames = conn.execute(
            "SELECT preview_path FROM media_frames WHERE media_id=?", (media_id,)
        ).fetchall()
        previews_ok = bool(frames) and all(Path(f["preview_path"]).exists() for f in frames)
        if previews_ok:
            # Frames already on disk — just re-embed, skip preview.
            conn.execute(
                "UPDATE media_items SET embed_status='pending', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (media_id,),
            )
        else:
            local_path = r["local_path"]
            media_gone = not local_path or not Path(local_path).exists()
            if media_gone:
                # Media was pruned before re-embedding — must re-download first.
                conn.execute(
                    "UPDATE media_items SET download_status='pending', preview_status='pending', "
                    "embed_status='pending', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (media_id,),
                )
                redownload += 1
            else:
                # Frames missing but file exists — regenerate previews.
                conn.execute(
                    "UPDATE media_items SET preview_status='pending', embed_status='pending', "
                    "updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (media_id,),
                )
    conn.commit()
    if redownload:
        print(
            f"  {redownload} item(s) need re-download (media pruned before model switch)."
        )
    return len(rows)


def force_reindex() -> int:
    """
    Full reindex: drop all frames (cascades embeddings) and reset preview/embed
    status for every downloaded item. Used when FRAME_COUNT changes.
    Returns the number of items reset.
    """
    conn = get_conn()
    conn.execute(
        "DELETE FROM media_frames WHERE media_id IN "
        "(SELECT id FROM media_items WHERE download_status='ok')"
    )
    conn.execute(
        "UPDATE media_items SET preview_status='pending', embed_status='pending', "
        "last_error=NULL, updated_at=CURRENT_TIMESTAMP WHERE download_status='ok'"
    )
    conn.commit()
    return conn.execute(
        "SELECT COUNT(*) FROM media_items WHERE preview_status='pending'"
    ).fetchone()[0]


def count_embeddings_for_model(model_name: str) -> int:
    return get_conn().execute(
        "SELECT COUNT(*) FROM frame_embeddings WHERE model_name=?", (model_name,)
    ).fetchone()[0]


def list_prunable_media() -> list[sqlite3.Row]:
    """Items whose bot_file_id is cached AND still have a local media file."""
    return get_conn().execute(
        "SELECT id, local_path FROM media_items "
        "WHERE bot_cache_status='cached' AND local_path IS NOT NULL"
    ).fetchall()


def clear_local_path(media_id: int) -> None:
    get_conn().execute(
        "UPDATE media_items SET local_path=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (media_id,),
    )
    get_conn().commit()


def list_pending_downloads(limit: int = 100) -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT * FROM media_items WHERE download_status='pending' LIMIT ?", (limit,)
    ).fetchall()


def list_pending_previews(limit: int = 100) -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT * FROM media_items WHERE download_status='ok' AND preview_status='pending' LIMIT ?",
        (limit,),
    ).fetchall()


def list_pending_embeddings(limit: int = 100) -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT * FROM media_items WHERE preview_status='ok' AND embed_status='pending' LIMIT ?",
        (limit,),
    ).fetchall()


def upsert_frame_ocr(frame_id: int, raw_text: str, norm_text: str, confidence: float) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT INTO frame_ocr (frame_id, raw_text, norm_text, confidence)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(frame_id) DO UPDATE SET
               raw_text=excluded.raw_text,
               norm_text=excluded.norm_text,
               confidence=excluded.confidence""",
        (frame_id, raw_text, norm_text, confidence),
    )
    conn.commit()


def mark_ocr_ok(media_id: int, ocr_text: str) -> None:
    get_conn().execute(
        "UPDATE media_items SET ocr_status='ok', ocr_text=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (ocr_text or None, media_id),
    )
    get_conn().commit()


def mark_ocr_failed(media_id: int, error: str) -> None:
    get_conn().execute(
        "UPDATE media_items SET ocr_status='failed', last_error=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (error, media_id),
    )
    get_conn().commit()


def list_pending_ocr(limit: int = 100) -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT * FROM media_items WHERE preview_status='ok' AND ocr_status='pending' LIMIT ?",
        (limit,),
    ).fetchall()


def clear_stale_source_flags(
    installed_ids: set[str],
    favorite_ids: set[str],
    recent_ids: set[str],
    saved_gif_ids: set[str],
) -> None:
    """
    After a full metadata sync, clear source flags for items no longer present
    in that source. Each flag is cleared independently so a sticker that is both
    installed and favorited keeps its is_favorite when removed only from installed.
    """
    conn = get_conn()

    def _clear(flag: str, seen: set[str]) -> None:
        if seen:
            placeholders = ",".join("?" * len(seen))
            conn.execute(
                f"UPDATE media_items SET {flag}=0"
                f" WHERE {flag}=1 AND tg_document_id NOT IN ({placeholders})",
                list(seen),
            )
        else:
            conn.execute(f"UPDATE media_items SET {flag}=0 WHERE {flag}=1")

    _clear("is_installed", installed_ids)
    _clear("is_favorite", favorite_ids)
    _clear("is_recent", recent_ids)
    _clear("is_saved_gif", saved_gif_ids)
    conn.commit()


def fts_rebuild() -> None:
    """Rebuild the FTS indexes from media_items (use after DB upgrade or bulk import)."""
    conn = get_conn()
    conn.execute("INSERT INTO media_fts(media_fts) VALUES('rebuild')")
    conn.execute("INSERT INTO media_ocr_fts(media_ocr_fts) VALUES('rebuild')")
    conn.commit()


def delete_frames_for_media(media_id: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM media_frames WHERE media_id=?", (media_id,))
    conn.commit()


def list_frames_for_media(media_id: int) -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT * FROM media_frames WHERE media_id=? ORDER BY frame_index",
        (media_id,),
    ).fetchall()


def get_status_counts() -> dict[str, int]:
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM media_items").fetchone()[0]
    dl_ok = conn.execute("SELECT COUNT(*) FROM media_items WHERE download_status='ok'").fetchone()[0]
    prev_ok = conn.execute("SELECT COUNT(*) FROM media_items WHERE preview_status='ok'").fetchone()[0]
    emb_ok = conn.execute("SELECT COUNT(*) FROM media_items WHERE embed_status='ok'").fetchone()[0]
    ocr_ok = conn.execute("SELECT COUNT(*) FROM media_items WHERE ocr_status='ok'").fetchone()[0]
    failed = conn.execute(
        "SELECT COUNT(*) FROM media_items WHERE download_status='failed' OR preview_status='failed' OR embed_status='failed'"
    ).fetchone()[0]
    return {
        "total": total,
        "downloaded": dl_ok,
        "previewed": prev_ok,
        "embedded": emb_ok,
        "ocr": ocr_ok,
        "failed": failed,
    }


def get_app_state(key: str) -> str | None:
    row = get_conn().execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_app_state(key: str, value: str) -> None:
    get_conn().execute(
        "INSERT INTO app_state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    get_conn().commit()


if __name__ == "__main__":
    config.ensure_dirs()
    get_conn()
    print(f"Database initialized: {config.DB_PATH}")
    counts = get_status_counts()
    print(f"  media_items: {counts['total']}")
