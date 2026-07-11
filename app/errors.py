from __future__ import annotations


class StickerRadarError(Exception):
    """Base exception for all application errors."""


class ConfigError(StickerRadarError):
    """Missing or invalid configuration."""


class DownloadError(StickerRadarError):
    """Failed to download a Telegram document."""


class PreviewError(StickerRadarError):
    """Failed to extract preview frames from media."""


class EmbedError(StickerRadarError):
    """Failed to compute an embedding."""


class ModelNotInstalled(StickerRadarError):
    """A selected model artifact is not available locally."""


class ModelIntegrityError(StickerRadarError):
    """A local model artifact is incomplete or failed verification."""


class SearchError(StickerRadarError):
    """Search index error."""


class BotSendError(StickerRadarError):
    """Failed to send media via bot."""
