import sys
import types

import pytest

from app import download_models
from app.config import DEFAULTS, DESIGN_MODEL
from app.engine import TTSEngine


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(download_models, "local_model_dir", lambda repo: tmp_path / repo.split("/")[-1])
    return tmp_path


def fake_download(models_dir):
    def _download(repo):
        d = models_dir / repo.split("/")[-1]
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"\0" * 1024)

    return _download


def test_size_and_remove(models_dir):
    assert not download_models.is_downloaded(DESIGN_MODEL)
    fake_download(models_dir)(DESIGN_MODEL)
    assert download_models.is_downloaded(DESIGN_MODEL)
    assert download_models.size_gb(DESIGN_MODEL) == 0.0  # 1 KB rounds to 0.0 GB
    download_models.remove(DESIGN_MODEL)
    assert not download_models.is_downloaded(DESIGN_MODEL)
    assert download_models.remove(DESIGN_MODEL) == 0.0  # removing twice is harmless


def test_engine_downloads_missing_model_on_first_use(models_dir, monkeypatch):
    calls = []
    real = fake_download(models_dir)
    monkeypatch.setattr(download_models, "download", lambda repo: (calls.append(repo), real(repo)))
    loaded_from = []
    fake_qwen = types.ModuleType("qwen_tts")
    fake_qwen.Qwen3TTSModel = types.SimpleNamespace(from_pretrained=lambda path, **kw: loaded_from.append(path) or object())
    monkeypatch.setitem(sys.modules, "qwen_tts", fake_qwen)
    monkeypatch.setattr("app.engine.resolve_model", lambda repo: str(models_dir / repo.split("/")[-1]))

    engine = TTSEngine({**DEFAULTS, "device": "cpu", "dtype": "float32"})
    monkeypatch.setattr(engine, "_torch_dtype", lambda: None)
    engine._load_model("design")
    engine._load_model("design")  # already there now, no second download
    assert calls == [DESIGN_MODEL]
    assert loaded_from[-1].endswith("Qwen3-TTS-12Hz-1.7B-VoiceDesign")
