import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app import gguf, koboldcpp
from app.config import DEFAULTS
from tests.conftest import SR, tone
from tests.test_speak import free_port

# A stand-in for koboldcpp.exe: serves the bits of KoboldCpp's API that Speak uses,
# with the voices it found in --ttsdir at startup, and logs every request.
FAKE_KOBOLDCPP = r'''
import argparse, io, json, math, os, struct, sys, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
for a in ("--ttsmodel", "--ttswavtokenizer", "--ttsdir", "--host", "--port"):
    ap.add_argument(a)
for f in ("--skiplauncher", "--quiet", "--singleinstance", "--usecuda", "--ttsgpu"):
    ap.add_argument(f, action="store_true")
args = ap.parse_args()
voices = sorted(f for f in os.listdir(args.ttsdir) if f.endswith(".wav"))
log = os.environ["FAKE_KCPP_LOG"]

def note(**data):
    with open(log, "a") as f:
        f.write(json.dumps({"pid": os.getpid(), **data}) + "\n")

def tone():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 10))) for i in range(12000)))
    return buf.getvalue()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/extra/version":
            self.send(json.dumps({"result": "KoboldCpp", "tts": True}).encode())
        elif self.path == "/v1/audio/voices":
            self.send(json.dumps({"status": "ok", "voices": ["random"] + voices}).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        note(request=body)
        self.send(tone() if body["voice"] in voices else b"", "audio/wav")

note(started=sys.argv[1:], temp=os.environ.get("TEMP"))
ThreadingHTTPServer((args.host, int(args.port)), H).serve_forever()
'''


def entries(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


@pytest.fixture
def voice(library):
    return library.save("Grumpy Wizard", tone(4), SR, kind="designed", ref_text="Hello there.", language="English")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    exe = tmp_path / "koboldcpp"
    exe.write_text(f"#!{sys.executable}\n{FAKE_KOBOLDCPP}")
    exe.chmod(0o755)
    for name in ("model.gguf", "tokenizer.gguf"):
        (tmp_path / name).write_text("x")
    log = tmp_path / "requests.jsonl"
    monkeypatch.setenv("FAKE_KCPP_LOG", str(log))
    monkeypatch.setattr(gguf, "LOG_PATH", tmp_path / "koboldcpp.log")
    monkeypatch.setattr(gguf, "TEMP_DIR", tmp_path / "kcpp-tmp")
    settings = {
        **DEFAULTS,
        "device": "cpu",
        "koboldcpp_exe": str(exe),
        "gguf_model": str(tmp_path / "model.gguf"),
        "gguf_tokenizer": str(tmp_path / "tokenizer.gguf"),
        "gguf_port": free_port(),
        "koboldcpp_url": f"http://127.0.0.1:{free_port()}",
        "koboldcpp_voices_dir": str(tmp_path / "kcpp_voices"),
    }
    return settings, log


@pytest.fixture
def kobold(setup, library):
    k = gguf.KoboldTTS(setup[0], library)
    yield k
    k.stop()


def test_launch_args():
    settings = {**DEFAULTS, "device": "cuda:0"}
    args = gguf.launch_args(settings, 5002)
    assert args[0].endswith("koboldcpp.exe") and "--skiplauncher" in args and args[-2:] == ["--usecuda", "--ttsgpu"]
    assert args[args.index("--ttsmodel") + 1] == str(gguf.GGUF_DIR / "qwen3-tts-0.6b-q8_0.gguf")
    assert args[args.index("--ttsdir") + 1] == str(koboldcpp.voices_dir(settings))
    assert args[args.index("--port") + 1] == "5002"
    cpu = gguf.launch_args({**DEFAULTS, "device": "cpu"}, 5002)
    assert cpu[0].endswith("koboldcpp-nocuda.exe") and "--usecuda" not in cpu


def test_install_only_fetches_what_is_missing(setup, tmp_path):
    settings, _ = setup
    assert gguf.missing(settings) == []
    gguf.install(settings)  # nothing to do, so no network
    with pytest.raises(ValueError, match="koboldcpp_exe"):
        gguf.install({**settings, "koboldcpp_exe": str(tmp_path / "nope.exe")})
    with pytest.raises(ValueError, match="gone.gguf"):
        gguf.install({**settings, "gguf_model": str(tmp_path / "gone.gguf")})


def test_speaks_with_its_own_koboldcpp(kobold, setup, voice):
    _, log = setup
    wav, sr = kobold.speak("Hello there, traveller.", voice)
    assert sr == 24000 and len(wav) == 12000 and kobold.running()
    started = [e for e in entries(log) if "started" in e]
    assert len(started) == 1 and "--usecuda" not in started[0]["started"]
    assert started[0]["temp"] == str(gguf.TEMP_DIR)  # unpacks into the install folder, not C:
    req = [e["request"] for e in entries(log) if "request" in e][0]
    assert req == {"input": "Hello there, traveller.", "voice": "grumpy-wizard.wav", "response_format": "wav", "language": "English"}
    # Speed is applied afterwards, since KoboldCpp ignores it.
    fast, _ = kobold.speak("Hello again.", voice, speed=2.0)
    assert len(fast) < len(wav) * 0.6
    assert len([e for e in entries(log) if "started" in e]) == 1  # reused


def test_long_text_and_auto_language(kobold, setup, library):
    _, log = setup
    v = library.save("Bob", tone(4), SR, kind="cloned", ref_text="hi", language="Auto")
    text = " ".join(f"Sentence number {i} goes on for a little while to take up space." for i in range(10))
    wav, _ = kobold.speak(text, v)
    reqs = [e["request"] for e in entries(log) if "request" in e]
    assert len(reqs) > 1 and all("language" not in r for r in reqs)
    assert len(wav) > 12000 * len(reqs)  # chunks joined with short gaps


def test_restarts_for_a_new_voice(kobold, setup, library, voice):
    settings, log = setup
    kobold.speak("One.", voice)
    first = kobold.proc
    dad = library.save("Dad", tone(4, 150), SR, kind="cloned", ref_text="Hi kids.")
    kobold.speak("Two.", dad)
    assert first.poll() is not None and kobold.proc is not first
    assert len([e for e in entries(log) if "started" in e]) == 2
    assert (koboldcpp.voices_dir(settings) / "dad.wav").exists()
    assert [e["request"]["voice"] for e in entries(log) if "request" in e] == ["grumpy-wizard.wav", "dad.wav"]


def test_uses_a_running_koboldcpp(kobold, setup, library, voice):
    settings, log = setup
    koboldcpp.sync(library, settings)
    port = settings["koboldcpp_url"].rsplit(":", 1)[1]
    yours = subprocess.Popen([settings["koboldcpp_exe"], "--ttsdir", str(koboldcpp.voices_dir(settings)), "--host", "127.0.0.1", "--port", port])
    try:
        for _ in range(100):
            if kobold._tts_voices(settings["koboldcpp_url"]) is not None:
                break
            threading.Event().wait(0.1)
        kobold.speak("Hello.", voice)
        assert kobold.proc is None  # didn't start a second KoboldCpp
        assert [e["pid"] for e in entries(log) if "request" in e] == [yours.pid]
    finally:
        yours.terminate()
        yours.wait(5)


def test_stop(kobold, voice):
    (gguf.TEMP_DIR / "_MEI12345").mkdir(parents=True)  # left by a KoboldCpp that was killed
    kobold.speak("Hello.", voice)
    assert not (gguf.TEMP_DIR / "_MEI12345").exists()
    proc = kobold.proc
    kobold.stop()
    assert proc.poll() is not None and not kobold.running()


def test_failed_start_shows_the_log(setup, library, voice, tmp_path):
    settings, _ = setup
    broken = tmp_path / "broken"
    broken.write_text(f"#!{sys.executable}\nprint('Error: no CUDA device found')\nraise SystemExit(1)\n")
    broken.chmod(0o755)
    k = gguf.KoboldTTS({**settings, "koboldcpp_exe": str(broken)}, library)
    with pytest.raises(ValueError, match="no CUDA device found"):
        k.speak("Hello.", voice)


class RangeHandler(BaseHTTPRequestHandler):
    data = bytes(range(256)) * 400

    def log_message(self, *a):
        pass

    def do_GET(self):
        start = int(self.headers["Range"].split("=")[1].rstrip("-")) if self.headers.get("Range") else 0
        body = self.data[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_download_resumes(tmp_path, monkeypatch):
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/koboldcpp.exe"
    try:
        fresh = tmp_path / "a" / "koboldcpp.exe"
        gguf._download(url, fresh)
        assert fresh.read_bytes() == RangeHandler.data
        resumed = tmp_path / "b" / "koboldcpp.exe"
        resumed.parent.mkdir()
        (resumed.parent / "koboldcpp.exe.part").write_bytes(RangeHandler.data[:5000])
        gguf._download(url, resumed)
        assert resumed.read_bytes() == RangeHandler.data and not (resumed.parent / "koboldcpp.exe.part").exists()
    finally:
        srv.shutdown()
