from __future__ import annotations

from pathlib import Path

from PIL import Image


def test_extract_inbound_static_image(tmp_path: Path):
    from app.inbound_media import extract_inbound_frames

    source = tmp_path / "incoming.webp"
    Image.new("RGBA", (16, 12), (255, 0, 0, 128)).save(source)
    frames = extract_inbound_frames(source, tmp_path / "frames")

    assert len(frames) == 1
    with Image.open(frames[0]) as image:
        assert image.mode == "RGB"
        assert image.size == (16, 12)


def test_extract_inbound_gif_samples_multiple_frames(tmp_path: Path, monkeypatch):
    from app import inbound_media

    monkeypatch.setattr(inbound_media.config, "FRAME_COUNT", 3)
    source = tmp_path / "incoming.gif"
    images = [Image.new("RGB", (12, 12), (i * 30, 0, 0)) for i in range(5)]
    images[0].save(source, save_all=True, append_images=images[1:], duration=50, loop=0)

    frames = inbound_media.extract_inbound_frames(source, tmp_path / "frames")
    assert len(frames) == 3
    assert all(path.exists() for path in frames)
