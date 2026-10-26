"""Paths and settings shared by the Studio, the voice server and the installer.

Everything (Python, packages, model weights, caches, voices) lives inside the
folder this repo was extracted to, so installing to G:\\Qwen3TTS keeps the
C: drive clean.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "settings.json"
MODELS_DIR = ROOT / "models"
VOICES_DIR = ROOT / "voices"
OUTPUTS_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / ".cache"

DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CLONE_MODEL_LARGE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CLONE_MODEL_SMALL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

LANGUAGES = [
    "Auto",
    "English",
    "Chinese",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

DEFAULTS: dict[str, Any] = {
    # Filled in by `python -m app.configure` during install.
    "device": "cuda:0",
    "dtype": "bfloat16",
    # Empty picks automatically (PyTorch SDPA). "flash_attention_2" if you installed flash-attn.
    "attn_implementation": "",
    "design_model": DESIGN_MODEL,
    "clone_model": CLONE_MODEL_LARGE,
    # When false, only one model sits in VRAM at a time (the other is unloaded).
    "keep_both_loaded": False,
    "studio_port": 7860,
    "studio_idle_unload_minutes": 10,
    "server_host": "127.0.0.1",
    # SillyTavern itself defaults to port 8000, so the voice server uses 8890.
    "server_port": 8890,
    "server_idle_unload_minutes": 0,
    # Optional. When set, API clients must send "Authorization: Bearer <key>".
    "api_key": "",
    "default_language": "Auto",
    "gpu_name": "",
    "vram_gb": 0.0,
}


def setup_environment() -> None:
    """Point Hugging Face and friends at caches inside the install folder."""
    os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("GRADIO_TEMP_DIR", str(CACHE_DIR / "gradio"))
    for d in (VOICES_DIR, OUTPUTS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict[str, Any]:
    settings = dict(DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            settings.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[config] Could not read {SETTINGS_PATH.name} ({exc}); using defaults.")
    return settings


def save_settings(settings: dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def local_model_dir(repo_id: str) -> Path:
    return MODELS_DIR / repo_id.split("/")[-1]


def resolve_model(repo_id_or_path: str) -> str:
    """Prefer the copy downloaded into models/, fall back to the Hub id."""
    p = Path(repo_id_or_path)
    if p.is_dir():
        return str(p)
    local = local_model_dir(repo_id_or_path)
    if (local / "config.json").exists():
        return str(local)
    return repo_id_or_path
