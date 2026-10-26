"""Tests run without a GPU or model weights: FakeEngine stands in for Qwen3-TTS."""

import numpy as np
import pytest

from app.config import DEFAULTS
from app.engine import TTSEngine
from app.voices import VoiceLibrary

SR = 24000


def tone(seconds: float, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class FakeModel:
    def __init__(self, kind):
        self.kind = kind
        self.calls = []

    def generate_voice_design(self, text, instruct, language):
        self.calls.append(("design", text, instruct, language))
        return [tone(1.5, 200 + 50 * i) for i in range(len(text))], SR

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        self.calls.append(("prompt", ref_text, x_vector_only_mode))
        return [object()]

    def generate_voice_clone(self, text, language, voice_clone_prompt):
        assert len(voice_clone_prompt) == len(text) == len(language)
        self.calls.append(("clone", list(text), list(language)))
        return [tone(0.5) for _ in text], SR


class FakeEngine(TTSEngine):
    def __init__(self, **overrides):
        super().__init__({**DEFAULTS, **overrides})
        self.loads = []

    def _load_model(self, kind):
        self.loads.append(kind)
        return FakeModel(kind)


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def library(tmp_path):
    return VoiceLibrary(tmp_path / "voices")
