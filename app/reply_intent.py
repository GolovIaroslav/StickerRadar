"""Optional local VLM intent extraction for ephemeral inbound frames.

The VLM is never invoked unless explicitly enabled and its model path exists.
It returns bounded JSON intent; it never selects or persists library media.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

from app import config

_SCHEMA = {
    "scene": "",
    "visible_text": "",
    "emotion": "",
    "speaker_stance": "",
    "punchline_setup": "",
    "desired_reply_role": "",
    "safe_tone": "sarcastic",
    "retrieval_queries": [],
    "forbidden_repetition": [],
}


def _model_and_mmproj(path: Path) -> tuple[Path, Path | None]:
    if path.is_file():
        model = path
        root = path.parent
    else:
        root = path
        candidates = sorted(root.glob("*.gguf"))
        model_candidates = [p for p in candidates if not p.name.lower().startswith("mmproj")]
        if not model_candidates:
            raise FileNotFoundError(f"No main GGUF model found in {root}")
        model = next((p for p in model_candidates if "q4_k_m" in p.name.lower() or "q8_0" in p.name.lower()), model_candidates[0])
    mmproj = next((p for p in sorted(root.glob("mmproj*.gguf")) if "f16" in p.name.lower()), None)
    return model, mmproj


def _contact_sheet(frames: list[Path], workdir: Path) -> Path:
    images = [Image.open(path).convert("RGB") for path in frames[:5]]
    width = 512
    thumb_h = 512
    sheet = Image.new("RGB", (width * len(images), thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        image.thumbnail((width - 8, thumb_h - 28))
        x = index * width + (width - image.width) // 2
        sheet.paste(image, (x, 24))
        draw.text((index * width + 8, 5), f"frame {index + 1}", fill="black")
    path = workdir / "contact-sheet.png"
    sheet.save(path, format="PNG")
    return path


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, dict):
        return {}
    return value


def infer_reply_intent(frames: list[Path], workdir: Path) -> dict:
    if not config.VLM_ENABLED or config.VLM_BACKEND in {"", "none", "off"}:
        return {}
    model = Path(config.VLM_MODEL_PATH).expanduser() if config.VLM_MODEL_PATH else None
    if model is None or not model.exists():
        return {}
    sheet = _contact_sheet(frames, workdir)
    prompt = (
        "Analyze this meme contact sheet. Return ONLY JSON matching this schema: "
        + json.dumps(_SCHEMA, ensure_ascii=False)
        + ". Do not repeat the source meme; describe the conversational role of a funny reply. "
        "Never target protected groups or produce hateful content."
    )
    try:
        model_path, mmproj_path = _model_and_mmproj(model)
        command = ["llama-cli", "--model", str(model_path)]
        if mmproj_path:
            command.extend(["--mmproj", str(mmproj_path)])
        command.extend([
            "--image", str(sheet), "--prompt", prompt,
            "--single-turn", "--simple-io", "--log-disable", "--no-display-prompt",
            "--verbosity", "0", "--temp", "0.2", "--ctx-size", "4096", "--n-predict", "300",
        ])
        completed = subprocess.run(
            command,
            capture_output=True, text=True, timeout=120, check=False,
        )
        if completed.returncode != 0:
            return {}
        result = _extract_json(completed.stdout or "")
        return {key: result.get(key, default) for key, default in _SCHEMA.items()}
    except (OSError, subprocess.SubprocessError):
        return {}
    finally:
        sheet.unlink(missing_ok=True)
