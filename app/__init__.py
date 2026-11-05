"""Qwen3-TTS Studio: the Voice Studio, the voice server and Speak."""

from pathlib import Path

REPO = "thinkdude/Silly-Tavern-qwen3-tts-studio"
HOMEPAGE = "https://github.com/" + REPO
RELEASES_URL = HOMEPAGE + "/releases/latest"

try:
    __version__ = (Path(__file__).resolve().parent.parent / "VERSION").read_text(encoding="utf-8").strip() or "dev"
except OSError:
    __version__ = "dev"
