import pytest
from fastapi.testclient import TestClient

from app.config import DEFAULTS
from app.server import create_app
from tests.conftest import SR, tone


@pytest.fixture
def client(engine, library):
    library.save("Grumpy Wizard", tone(4), SR, kind="designed", ref_text="Hello there.", language="English")
    return TestClient(create_app(engine, library, dict(DEFAULTS)))


def test_speech_mp3(client):
    r = client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Hello!", "voice": "grumpy-wizard"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/mpeg"
    assert len(r.content) > 100


def test_sillytavern_style_request(client):
    # What SillyTavern's OpenAI Compatible provider sends.
    body = {"model": "qwen3-tts", "input": "Hi there.", "voice": "grumpy-wizard", "response_format": "mp3", "speed": 1.25}
    r = client.post("/v1/audio/speech", json=body, headers={"Authorization": "Bearer "})
    assert r.status_code == 200


def test_voice_by_name_and_wav(client):
    r = client.post("/v1/audio/speech", json={"input": "Hi.", "voice": "Grumpy Wizard", "response_format": "wav"})
    assert r.status_code == 200 and r.content[:4] == b"RIFF"


def test_unknown_voice(client):
    r = client.post("/v1/audio/speech", json={"input": "Hi.", "voice": "alloy"})
    assert r.status_code == 404
    assert "grumpy-wizard" in r.json()["detail"]


def test_bad_format_and_empty_input(client):
    assert client.post("/v1/audio/speech", json={"input": "Hi.", "voice": "grumpy-wizard", "response_format": "xyz"}).status_code == 400
    assert client.post("/v1/audio/speech", json={"input": "  ", "voice": "grumpy-wizard"}).status_code == 400


def test_voices_and_models(client):
    voices = client.get("/v1/audio/voices").json()["voices"]
    assert voices[0]["id"] == "grumpy-wizard"
    assert client.get("/v1/models").json()["data"][0]["id"] == "qwen3-tts"
    assert client.get("/health").json()["status"] == "ok"


def test_api_key(engine, library):
    library.save("Bob", tone(4), SR, kind="cloned", ref_text="hi")
    c = TestClient(create_app(engine, library, {**DEFAULTS, "api_key": "secret"}))
    assert c.post("/v1/audio/speech", json={"input": "Hi.", "voice": "bob"}).status_code == 401
    ok = c.post("/v1/audio/speech", json={"input": "Hi.", "voice": "bob"}, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


def test_request_log(client):
    client.post("/v1/audio/speech", json={"input": "Hello!", "voice": "grumpy-wizard"})
    log = list(client.app.state.log)
    assert log[0][1] == "grumpy-wizard" and log[0][-1] == "ok"


class FakeKobold:
    """Stands in for gguf.KoboldTTS."""

    own_url = "http://127.0.0.1:5002"
    url = None

    def __init__(self):
        self.said, self.stopped, self.started = [], 0, 0
        self.up = False

    def state(self):
        return "ready" if self.up else "stopped"

    def ensure_running(self):
        self.started += 1
        self.up = True

    def speak(self, text, voice, language=None, speed=1.0):
        self.up = True
        self.said.append((text, voice.id, language, speed))
        return tone(1, 330), SR

    def stop(self):
        self.stopped += 1
        self.up = False


@pytest.fixture
def switchable(engine, library, monkeypatch):
    """A server that can switch models, with settings.json kept in memory."""
    from app import server

    saved = dict(DEFAULTS)
    monkeypatch.setattr(server, "load_settings", lambda: dict(saved))
    monkeypatch.setattr(server, "save_settings", lambda s: saved.update(s))
    library.save("Grumpy Wizard", tone(4), SR, kind="designed", ref_text="Hello there.", language="English")
    kobold = FakeKobold()
    app = create_app(engine, library, dict(DEFAULTS), kobold)
    return TestClient(app), app.state.backend, kobold, saved


def test_switch_to_gguf_and_back(switchable, engine):
    client, backend, kobold, saved = switchable
    body = {"input": "Hello!", "voice": "grumpy-wizard", "response_format": "wav", "speed": 1.5}
    assert client.get("/health").json()["engine"] == "full"
    assert client.post("/v1/audio/speech", json=body).status_code == 200
    assert engine.loaded() == ["clone"] and kobold.said == []

    backend.set_choice("gguf")
    assert saved["server_engine"] == "gguf"  # remembered for next time
    assert engine.loaded() == []  # the full model's GPU memory is freed
    r = client.post("/v1/audio/speech", json={**body, "language": "German"})
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    assert kobold.said == [("Hello!", "grumpy-wizard", "German", 1.5)] and engine.loaded() == []
    health = client.get("/health").json()
    assert health["engine"] == "gguf" and health["gguf"] == "ready"
    assert [row[2] for row in client.app.state.log] == ["GGUF", "full"]

    backend.set_choice("full")
    assert saved["server_engine"] == "full" and kobold.stopped == 1
    assert client.post("/v1/audio/speech", json=body).status_code == 200
    assert len(kobold.said) == 1 and engine.loaded() == ["clone"]


def test_preload_and_free(switchable, engine):
    _, backend, kobold, _ = switchable
    backend.preload()
    assert engine.loaded() == ["clone"] and kobold.started == 0
    backend.settings["server_engine"] = "gguf"
    backend.preload()
    assert kobold.started == 1 and backend.ready()
    backend.free()
    assert engine.loaded() == [] and kobold.stopped == 1


def test_gguf_needs_kobold(engine, library):
    app = create_app(engine, library, {**DEFAULTS, "server_engine": "gguf"})
    assert app.state.backend.choice == "full"
    with pytest.raises(ValueError, match="GGUF"):
        app.state.backend.speak("Hi.", None, model="gguf")


def panel_fn(panel, name):
    return next(f.fn for f in panel.fns.values() if getattr(f.fn, "__name__", "") == name)


def test_panel_compare(switchable, engine, library):
    from app.server import build_panel

    client, backend, kobold, _ = switchable
    panel = build_panel(library, backend.settings, client.app, "http://127.0.0.1:8890")
    row, full, gguf_take, note = panel_fn(panel, "compare")("grumpy-wizard", "Voice's default", "Which one sounds better?")
    assert row["visible"] and "a little different" in note
    assert full["label"].startswith("Full model: took") and "including loading it" in full["label"]
    assert gguf_take["label"].startswith("GGUF model: took")
    sr_full, wav_full = full["value"]
    sr_gguf, wav_gguf = gguf_take["value"]
    assert sr_full == sr_gguf == SR and len(wav_full) and len(wav_gguf)
    assert kobold.said == [("Which one sounds better?", "grumpy-wizard", None, 1.0)]
    # Afterwards only the selected model (full) keeps its GPU memory.
    assert engine.loaded() == ["clone"] and kobold.stopped == 1
    assert [(row[2], row[-1]) for row in client.app.state.log] == [("GGUF", "ok (compare)"), ("full", "ok (compare)")]

    # With GGUF selected, the full model is the one freed again.
    backend.set_choice("gguf")
    engine.preload("clone")
    panel_fn(panel, "compare")("grumpy-wizard", "English", "Again.")
    assert engine.loaded() == [] and kobold.up


def test_panel_switch(switchable, engine, library, monkeypatch):
    from app import gguf
    from app.server import build_panel

    client, backend, kobold, saved = switchable
    monkeypatch.setattr(gguf, "missing", lambda settings: [])
    panel = build_panel(library, backend.settings, client.app, "http://127.0.0.1:8890")
    switch = panel_fn(panel, "switch_model")
    assert "GGUF model speaks" in switch("gguf")
    assert backend.choice == "gguf" and saved["server_engine"] == "gguf"
    assert "full model speaks" in switch("full")
    assert backend.choice == "full" and kobold.stopped >= 1
