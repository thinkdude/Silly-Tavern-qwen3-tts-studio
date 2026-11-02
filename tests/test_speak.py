import socket
import threading
import time

import gradio as gr
import pytest
import soundfile as sf
import uvicorn

from app import speak
from app.config import DEFAULTS
from app.server import create_app
from tests.conftest import SR, FakeEngine, tone


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def voice(library):
    return library.save("Grumpy Wizard", tone(4), SR, kind="designed", ref_text="Hello there.", language="English")


@pytest.fixture
def no_server():
    return {**DEFAULTS, "server_port": free_port()}


@pytest.fixture
def server(library, voice):
    """A real voice server (with a stand-in engine) on a free port."""
    settings = {**DEFAULTS, "server_port": free_port(), "api_key": "secret"}
    eng = FakeEngine()
    srv = uvicorn.Server(uvicorn.Config(create_app(eng, library, settings), host="127.0.0.1", port=settings["server_port"], log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)

    def stop():
        srv.should_exit = True
        t.join(5)

    yield eng, settings, stop
    stop()


def test_server_url():
    assert speak.server_url(DEFAULTS) == "http://127.0.0.1:8890"
    assert speak.server_url({"server_host": "0.0.0.0", "server_port": 9000}) == "http://127.0.0.1:9000"
    assert speak.server_url({"server_host": "192.168.1.5", "server_port": 9000}) == "http://192.168.1.5:9000"


def test_speaks_locally_without_server(engine, voice, no_server):
    speaker = speak.Speaker(engine, no_server)
    assert not speaker.server_running()
    wav, sr, source = speaker.say(voice, "Hello!")
    assert source == "full" and sr == SR and len(wav) > 0
    assert engine.loaded() == ["clone"]


def test_uses_running_server(engine, voice, server):
    server_engine, settings, _ = server
    speaker = speak.Speaker(engine, settings)
    engine.preload("clone")
    assert speaker.server_running()
    wav, sr, source = speaker.say(voice, "Hello!", language="French", speed=1.0)
    assert source == "server" and sr == SR and len(wav) > 0
    # Our own copy of the model is freed while the server does the speaking.
    assert engine.loaded() == []
    clone_calls = [c for c in server_engine._models["clone"].calls if c[0] == "clone"]
    assert clone_calls[0][2] == ["French"]


def test_falls_back_when_server_closes(engine, voice, server):
    _, settings, stop = server
    speaker = speak.Speaker(engine, settings)
    assert speaker.say(voice, "One.")[2] == "server"
    stop()
    wav, sr, source = speaker.say(voice, "Two.")
    assert source == "full" and len(wav) > 0 and engine.loaded() == ["clone"]


def test_server_errors_are_readable(engine, voice, server):
    _, settings, _ = server
    speaker = speak.Speaker(engine, settings)
    voice.id = "nobody"
    with pytest.raises(ValueError, match="No saved voice called 'nobody'"):
        speaker.say(voice, "Hello!")
    wrong_key = speak.Speaker(engine, {**settings, "api_key": "wrong"})
    with pytest.raises(ValueError, match="API key"):
        wrong_key.say(voice, "Hello!")


def test_save_output(tmp_path, monkeypatch):
    monkeypatch.setattr(speak, "OUTPUTS_DIR", tmp_path)
    a = speak.save_output("bob", "Hello there, how are you doing today my friend?", tone(1), SR)
    b = speak.save_output("bob", "Hello there, how are you doing today my friend?", tone(1), SR)
    assert a != b and a.exists() and b.exists()
    assert a.name.startswith("bob-") and a.name.endswith("-hello-there-how-are-you-doing.wav")
    wav, sr = sf.read(str(a))
    assert sr == SR and len(wav) == SR
    mp3 = speak.save_output("bob", "?!", tone(1), SR, "mp3")
    assert mp3.suffix == ".mp3" and mp3.stat().st_size > 100
    assert speak.save_output("bob", "你好，世界", tone(1), SR).name.endswith("-你好-世界.wav")


def test_build(engine, library, voice, no_server):
    assert isinstance(speak.build(speak.Speaker(engine, no_server), library), gr.Blocks)


class FakeKobold:
    def __init__(self):
        self.said, self.stopped = [], 0

    def speak(self, text, voice, language=None, speed=1.0):
        self.said.append((text, voice.id, language, speed))
        return tone(1), SR

    def stop(self):
        self.stopped += 1


def test_switch_to_gguf_and_back(engine, voice, no_server, tmp_path, monkeypatch):
    monkeypatch.setattr(speak, "load_settings", lambda: dict(no_server))
    saved = {}
    monkeypatch.setattr(speak, "save_settings", saved.update)
    kobold = FakeKobold()
    speaker = speak.Speaker(engine, dict(no_server), kobold)
    speaker.say(voice, "Full first.")
    assert engine.loaded() == ["clone"]

    speaker.set_choice("gguf")
    assert saved["speak_engine"] == "gguf" and engine.loaded() == []  # the full model's memory is freed
    wav, sr, source = speaker.say(voice, "Now GGUF.", language="German", speed=1.5)
    assert source == "gguf" and kobold.said == [("Now GGUF.", "grumpy-wizard", "German", 1.5)]
    assert engine.loaded() == []

    speaker.set_choice("full")
    assert saved["speak_engine"] == "full" and kobold.stopped == 1
    assert speaker.say(voice, "Full again.")[2] == "full"


def test_gguf_needs_kobold(engine, no_server):
    assert speak.Speaker(engine, {**no_server, "speak_engine": "gguf"}).choice == "full"
