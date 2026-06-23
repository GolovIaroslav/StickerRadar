from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from telethon import TelegramClient
from telethon.tl.functions.messages import (
    GetAllStickersRequest,
    GetFavedStickersRequest,
    GetRecentStickersRequest,
    GetSavedGifsRequest,
    GetStickerSetRequest,
)
from telethon.tl.types import (
    Document,
    DocumentAttributeAnimated,
    DocumentAttributeImageSize,
    DocumentAttributeSticker,
    DocumentAttributeVideo,
    InputStickerSetID,
    MessagesSavedGifs,
    MessagesSavedGifsNotModified,
    MessagesAllStickers,
    MessagesAllStickersNotModified,
    MessagesFavedStickers,
    MessagesFavedStickersNotModified,
    MessagesRecentStickers,
    MessagesRecentStickersNotModified,
)

from app.errors import DownloadError


# ---------------------------------------------------------------------------
# Internal dataclasses — never expose raw Telethon objects outside this module
# ---------------------------------------------------------------------------


@dataclass
class StickerSetMeta:
    id: int
    access_hash: int
    short_name: str
    title: str
    count: int


@dataclass
class StickerDoc:
    tg_document_id: str
    access_hash: str
    file_reference: bytes
    mime_type: str
    file_ext: str
    sticker_format: str          # static | animated | video | unknown
    emoji: str
    set_meta: StickerSetMeta | None = None


@dataclass
class GifDoc:
    tg_document_id: str
    access_hash: str
    file_reference: bytes
    mime_type: str
    file_ext: str                # almost always .mp4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIME_TO_EXT: dict[str, str] = {
    "image/webp": ".webp",
    "application/x-tgsticker": ".tgs",
    "video/webm": ".webm",
    "video/mp4": ".mp4",
    "image/gif": ".gif",
}

_MIME_TO_FORMAT: dict[str, str] = {
    "image/webp": "static",
    "application/x-tgsticker": "animated",
    "video/webm": "video",
    "video/mp4": "video",
    "image/gif": "video",
}


def _classify(doc: Document) -> tuple[str, str, str]:
    """Return (sticker_format, mime_type, file_ext)."""
    mime = (doc.mime_type or "").strip()
    fmt = _MIME_TO_FORMAT.get(mime)
    ext = _MIME_TO_EXT.get(mime)

    if fmt and ext:
        return fmt, mime, ext

    # Fallback: inspect attributes
    attr_names = {type(a).__name__ for a in (doc.attributes or [])}
    if "DocumentAttributeAnimated" in attr_names:
        return "animated", mime, ".tgs"
    if "DocumentAttributeVideo" in attr_names:
        return "video", mime, ".mp4"
    if "DocumentAttributeImageSize" in attr_names:
        return "static", mime, ".webp"

    return "unknown", mime, ".bin"


def _extract_emoji(doc: Document) -> str:
    for attr in doc.attributes or []:
        if isinstance(attr, DocumentAttributeSticker):
            return attr.alt or ""
    return ""


def _doc_to_sticker(doc: Document, set_meta: StickerSetMeta | None = None) -> StickerDoc:
    fmt, mime, ext = _classify(doc)
    return StickerDoc(
        tg_document_id=str(doc.id),
        access_hash=str(doc.access_hash),
        file_reference=bytes(doc.file_reference),
        mime_type=mime,
        file_ext=ext,
        sticker_format=fmt,
        emoji=_extract_emoji(doc),
        set_meta=set_meta,
    )


def _doc_to_gif(doc: Document) -> GifDoc:
    _, mime, ext = _classify(doc)
    return GifDoc(
        tg_document_id=str(doc.id),
        access_hash=str(doc.access_hash),
        file_reference=bytes(doc.file_reference),
        mime_type=mime,
        file_ext=ext,
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class TgUserClient:
    """Thin async wrapper around Telethon for StickerRadar scanning."""

    def __init__(self, api_id: int, api_hash: str, session_path: Path) -> None:
        self._client = TelegramClient(
            str(session_path),
            api_id,
            api_hash,
        )

    # Context manager ----------------------------------------------------------

    async def __aenter__(self) -> "TgUserClient":
        await self._client.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.disconnect()

    # Public API ---------------------------------------------------------------

    async def get_installed_sticker_sets(self) -> list[StickerSetMeta]:
        result = await self._client(GetAllStickersRequest(hash=0))
        if isinstance(result, MessagesAllStickersNotModified):
            return []
        sets = []
        for s in result.sets:
            sets.append(
                StickerSetMeta(
                    id=s.id,
                    access_hash=s.access_hash,
                    short_name=s.short_name,
                    title=s.title,
                    count=s.count,
                )
            )
        return sets

    async def get_sticker_set_documents(
        self, set_meta: StickerSetMeta
    ) -> list[StickerDoc]:
        result = await self._client(
            GetStickerSetRequest(
                stickerset=InputStickerSetID(
                    id=set_meta.id,
                    access_hash=set_meta.access_hash,
                ),
                hash=0,
            )
        )
        docs = []
        for doc in result.documents:
            docs.append(_doc_to_sticker(doc, set_meta))
        return docs

    async def get_favorite_stickers(self) -> list[StickerDoc]:
        result = await self._client(GetFavedStickersRequest(hash=0))
        if isinstance(result, MessagesFavedStickersNotModified):
            return []
        return [_doc_to_sticker(d) for d in result.stickers]

    async def get_recent_stickers(self) -> list[StickerDoc]:
        result = await self._client(GetRecentStickersRequest(attached=False, hash=0))
        if isinstance(result, MessagesRecentStickersNotModified):
            return []
        return [_doc_to_sticker(d) for d in result.stickers]

    async def get_saved_gifs(self) -> list[GifDoc]:
        result = await self._client(GetSavedGifsRequest(hash=0))
        if isinstance(result, MessagesSavedGifsNotModified):
            return []
        return [_doc_to_gif(d) for d in result.gifs]

    async def download_document(
        self,
        tg_document_id: str,
        access_hash: str,
        file_reference: bytes,
        dest_path: Path,
    ) -> Path:
        """Download a Telegram document to dest_path. Returns dest_path."""
        from telethon.tl.types import InputDocument

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await self._client.download_media(
                InputDocument(
                    id=int(tg_document_id),
                    access_hash=int(access_hash),
                    file_reference=file_reference,
                ),
                file=str(dest_path),
            )
        except Exception as e:
            raise DownloadError(f"Failed to download {tg_document_id}: {e}") from e
        return dest_path

    async def keepalive(self) -> None:
        """Run Telethon event loop until disconnected (for long-running mode)."""
        await self._client.run_until_disconnected()
