from __future__ import annotations

import sqlite3
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
CREATE INDEX IF NOT EXISTS idx_frame_emb_model  ON frame_embeddings(model_name);
"""

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

_conn: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
        _conn.commit()
    return _conn


def close() -> None:
    global _conn
    if _conn:
        _conn.close()
        _conn = None


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


def reset_embed_status() -> int:
    """Reset embed_status to 'pending' for all embedded items. Returns count."""
    conn = get_conn()
    cur = conn.execute(
        "UPDATE media_items SET embed_status='pending', updated_at=CURRENT_TIMESTAMP "
        "WHERE embed_status='ok'"
    )
    conn.execute(
        "UPDATE media_items SET embed_status='pending', updated_at=CURRENT_TIMESTAMP "
        "WHERE embed_status='failed'"
    )
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM media_items WHERE embed_status='pending'").fetchone()[0]


def reset_preview_status() -> int:
    """Reset preview_status and embed_status to 'pending'. Used for full reindex."""
    conn = get_conn()
    conn.execute(
        "UPDATE media_items SET preview_status='pending', embed_status='pending', "
        "updated_at=CURRENT_TIMESTAMP WHERE download_status='ok'"
    )
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM media_items WHERE preview_status='pending'").fetchone()[0]


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
    failed = conn.execute(
        "SELECT COUNT(*) FROM media_items WHERE download_status='failed' OR preview_status='failed' OR embed_status='failed'"
    ).fetchone()[0]
    return {
        "total": total,
        "downloaded": dl_ok,
        "previewed": prev_ok,
        "embedded": emb_ok,
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
