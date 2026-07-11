from __future__ import annotations

from types import SimpleNamespace


def _message(*, animated: bool = False, video: bool = False, filename: str | None = None):
    sticker = SimpleNamespace(is_animated=animated, is_video=video) if filename is None else None
    animation = SimpleNamespace(file_name=filename) if filename is not None else None
    return SimpleNamespace(sticker=sticker, animation=animation)


def test_inbound_suffix_for_telegram_sticker_types():
    from app.bot import _inbound_suffix

    assert _inbound_suffix(_message().sticker) == ".webp"
    assert _inbound_suffix(_message(animated=True).sticker) == ".tgs"
    assert _inbound_suffix(_message(video=True).sticker) == ".webm"


def test_inbound_suffix_preserves_gif_and_video_animation_extensions():
    from app.bot import _inbound_suffix

    assert _inbound_suffix(_message(filename="meme.gif").animation) == ".gif"
    assert _inbound_suffix(_message(filename="clip.mp4").animation) == ".mp4"


def test_select_visual_target_prefers_direct_media_then_reply_target():
    from app.bot import _select_visual_target

    direct = _message().sticker
    replied_gif = SimpleNamespace(mime_type="image/gif", file_name="reply.gif")
    message = SimpleNamespace(sticker=direct, animation=None, document=None,
                              reply_to_message=SimpleNamespace(sticker=None, animation=None, document=replied_gif))
    assert _select_visual_target(message) is direct

    text_reply = SimpleNamespace(sticker=None, animation=None, document=None,
                                 reply_to_message=SimpleNamespace(sticker=None, animation=None, document=replied_gif))
    assert _select_visual_target(text_reply) is replied_gif


def test_select_visual_target_rejects_non_gif_document():
    from app.bot import _select_visual_target

    pdf = SimpleNamespace(mime_type="application/pdf", file_name="not-a-gif.pdf")
    message = SimpleNamespace(sticker=None, animation=None, document=pdf, reply_to_message=None)
    assert _select_visual_target(message) is None
