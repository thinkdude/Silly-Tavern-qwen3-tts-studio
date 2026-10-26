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
