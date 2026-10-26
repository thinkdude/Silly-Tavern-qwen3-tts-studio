"""Share saved voices with KoboldCpp.

KoboldCpp's Qwen3-TTS support clones voices from a folder of clips (its
"TTS Voices Dir", or --ttsdir). This keeps such a folder in step with the
voice library: one <voice-id>.wav per saved voice. KoboldCpp reads the folder
when it starts, so restart it after adding voices.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import ROOT
from .voices import VoiceLibrary

DEFAULT_URL = "http://127.0.0.1:5001"
MODEL_URL = "https://huggingface.co/koboldcpp/tts/resolve/main/qwen3-tts-0.6b-q8_0.gguf"
TOKENIZER_URL = "https://huggingface.co/koboldcpp/tts/resolve/main/qwen3-tts-tokenizer-f16.gguf"
# Lists the files this app wrote, so a shared folder's other clips are never touched.
MANIFEST = ".qwen3-tts-studio.json"


def voices_dir(settings: dict) -> Path:
    p = Path(settings.get("koboldcpp_voices_dir") or "koboldcpp_voices")
    return p if p.is_absolute() else ROOT / p


def voice_names(library: VoiceLibrary) -> list[str]:
    # KoboldCpp names each cloned voice after its file, extension included.
    return [f"{v.id}.wav" for v in library.list()]


def speech_url(settings: dict) -> str:
    return (settings.get("koboldcpp_url") or DEFAULT_URL).rstrip("/") + "/v1/audio/speech"


def sync(library: VoiceLibrary, settings: dict) -> dict:
    """Copy every saved voice into the KoboldCpp folder and drop ones that were deleted."""
    folder = voices_dir(settings)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / MANIFEST
    try:
        ours = set(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        ours = set()

    added, current = [], set()
    for v in library.list():
        name = f"{v.id}.wav"
        current.add(name)
        dst = folder / name
        src = v.reference_path
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
            added.append(name)

    removed = []
    for name in sorted(ours - current):
        try:
            (folder / name).unlink()
            removed.append(name)
        except FileNotFoundError:
            pass

    manifest_path.write_text(json.dumps(sorted(current), indent=2), encoding="utf-8")
    return {"folder": str(folder), "added": added, "removed": removed, "total": len(current)}
