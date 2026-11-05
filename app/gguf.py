"""The GGUF voice engine: Qwen3-TTS as small GGUF files, run by KoboldCpp.

Speak and the voice server can use this instead of the full model. It needs
less GPU memory and disk space, but KoboldCpp clones from the sound of the
reference clip only (it can't use the transcript), so voices are a little less
exact.

What it needs, all inside the install folder (install.bat downloads them, and
Speak or the voice server does too the first time GGUF is picked if they're
missing):
    .tools\\koboldcpp\\koboldcpp.exe                 the KoboldCpp program
    models\\gguf\\qwen3-tts-0.6b-q8_0.gguf           the voice model
    models\\gguf\\qwen3-tts-tokenizer-f16.gguf       its audio tokenizer

The app starts KoboldCpp by itself, in the background and with no text model,
and stops it again when you switch back to the full model or close the app.
Speak and the voice server share one KoboldCpp on gguf_port: whichever needs it
first starts it, and the other uses that one instead of loading a second copy.

    python -m app.gguf      download whatever is missing
"""

from __future__ import annotations

import atexit
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio, koboldcpp
from .config import CACHE_DIR, MODELS_DIR, ROOT, load_settings, setup_environment
from .textsplit import split_text
from .voices import Voice, VoiceLibrary

GGUF_REPO = "koboldcpp/tts"
GGUF_DIR = MODELS_DIR / "gguf"
KOBOLDCPP_DIR = ROOT / ".tools" / "koboldcpp"
EXE_URL = "https://github.com/LostRuins/koboldcpp/releases/latest/download/{name}"
LOG_PATH = CACHE_DIR / "koboldcpp-gguf.log"
# KoboldCpp's one-file .exe unpacks itself (about 1 GB) into TEMP every time it starts.
# This keeps that in the install folder instead of on C:.
TEMP_DIR = CACHE_DIR / "koboldcpp-tmp"
START_TIMEOUT_S = 600
# The voice model switch in Speak and in the voice server's control panel.
MODEL_CHOICES = [
    ("Full model (best quality)", "full"),
    ("GGUF with KoboldCpp (smaller, lighter on the GPU)", "gguf"),
]

# Talking to KoboldCpp on this PC must never go through a system proxy.
_local = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class _Gone(Exception):
    pass


def _uses_cuda(settings: dict) -> bool:
    return str(settings.get("device", "")).startswith("cuda")


def exe_path(settings: dict) -> Path:
    custom = (settings.get("koboldcpp_exe") or "").strip()
    if custom:
        return Path(custom)
    # The CUDA build for NVIDIA cards; the smaller CPU/Vulkan build otherwise.
    return KOBOLDCPP_DIR / ("koboldcpp.exe" if _uses_cuda(settings) else "koboldcpp-nocuda.exe")


def model_path(name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else GGUF_DIR / name


def missing(settings: dict) -> list[Path]:
    paths = [exe_path(settings), model_path(settings["gguf_model"]), model_path(settings["gguf_tokenizer"])]
    return [p for p in paths if not p.exists()]


def _download(url: str, dest: Path) -> None:
    """Download url to dest, resuming a partial download left by an earlier try."""
    from tqdm.auto import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code != 416:  # 416: the partial file is already complete
            raise
        part.replace(dest)
        return
    with r:
        if r.status != 206:
            have = 0  # the server sent the whole file again
        total = int(r.headers.get("Content-Length") or 0) + have
        with open(part, "ab" if have else "wb") as f, tqdm(
            total=total or None, initial=have, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name
        ) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    part.replace(dest)


def install(settings: dict, log=print) -> None:
    """Download KoboldCpp and the GGUF files that aren't there yet."""
    exe = exe_path(settings)
    if not exe.exists():
        if (settings.get("koboldcpp_exe") or "").strip():
            raise ValueError(f"koboldcpp_exe in settings.json points to {exe}, which doesn't exist.")
        log(f"Downloading KoboldCpp ({exe.name}, about 600 MB) ...")
        _download(EXE_URL.format(name=exe.name), exe)
    for name in (settings["gguf_model"], settings["gguf_tokenizer"]):
        p = model_path(name)
        if p.exists():
            continue
        if Path(name).is_absolute():
            raise ValueError(f"{name} (set in settings.json) doesn't exist.")
        from huggingface_hub import hf_hub_download

        log(f"Downloading {name} ...")
        hf_hub_download(repo_id=GGUF_REPO, filename=name, local_dir=str(GGUF_DIR))


def launch_args(settings: dict, port: int) -> list[str]:
    args = [
        str(exe_path(settings)),
        "--ttsmodel", str(model_path(settings["gguf_model"])),
        "--ttswavtokenizer", str(model_path(settings["gguf_tokenizer"])),
        "--ttsdir", str(koboldcpp.voices_dir(settings)),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--skiplauncher",
        "--quiet",
        # Replaces a copy left running on this port, e.g. after Speak or the voice server was killed.
        "--singleinstance",
    ]  # fmt: skip
    if _uses_cuda(settings):
        args += ["--usecuda", "--ttsgpu"]
    return args


def _tie_to_this_process(proc: subprocess.Popen):
    """Windows: put KoboldCpp in a job that ends when this app does, so closing
    its window never leaves KoboldCpp running in the background."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", Basic),
                ("IoInfo", ctypes.c_uint64 * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))
        k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(int(proc._handle)))
        return job  # the job (and KoboldCpp with it) lasts as long as this handle
    except Exception as exc:
        print(f"[gguf] Couldn't tie KoboldCpp to this window: {exc}")
        return None


class KoboldTTS:
    """Speaks saved voices with KoboldCpp: one that's already running with the voice
    loaded (such as yours for SillyTavern, or the one Speak or the voice server
    started), or a copy it starts itself."""

    def __init__(self, settings: dict, library: VoiceLibrary):
        self.settings = settings
        self.library = library
        self.own_url = f"http://127.0.0.1:{int(settings.get('gguf_port', 5002))}"
        self.proc: subprocess.Popen | None = None
        self._job = None
        self._url: str | None = None
        self._voices: set[str] = set()
        self._ready = False
        self._lock = threading.RLock()
        atexit.register(self.stop)

    # -- finding or starting KoboldCpp ----------------------------------------

    def _get(self, url: str, path: str, timeout: float = 2):
        try:
            with _local.open(url + path, timeout=timeout) as r:
                return json.load(r)
        except (OSError, ValueError):
            return None

    def _tts_voices(self, url: str) -> set[str] | None:
        """The voices a KoboldCpp at url can speak, or None if no TTS model is running there."""
        version = self._get(url, "/api/extra/version")
        if not (isinstance(version, dict) and version.get("tts")):
            return None
        data = self._get(url, "/v1/audio/voices")
        return set(data.get("voices") or []) if isinstance(data, dict) else set()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def state(self) -> str:
        """"ready", "starting" or "stopped". Never waits, so a status page can ask any time."""
        if self.running():
            return "ready" if self._ready else "starting"
        return "ready" if self._url else "stopped"

    @property
    def url(self) -> str | None:
        """The KoboldCpp this speaks with, once one is known."""
        return self._url or (self.own_url if self.running() else None)

    def _find(self, name: str) -> str:
        urls = [self.own_url, (self.settings.get("koboldcpp_url") or koboldcpp.DEFAULT_URL).rstrip("/")]
        for url in dict.fromkeys(urls):
            voices = self._tts_voices(url)
            if voices and name in voices:
                self._url, self._voices = url, voices
                return url
        # Not loaded anywhere (KoboldCpp reads its voices folder only when it starts), so (re)start ours.
        self.start()
        voices = self._tts_voices(self.own_url) or set()
        if name not in voices:
            raise ValueError(f"KoboldCpp didn't load the voice {name}. Its log is in {LOG_PATH}.")
        self._url, self._voices = self.own_url, voices
        return self.own_url

    def start(self) -> None:
        """Start (or restart) KoboldCpp with every saved voice. Blocks until it's ready."""
        with self._lock:
            self.stop()
            self._ready = False
            if missing(self.settings):
                install(self.settings)
            koboldcpp.sync(self.library, self.settings)
            # A KoboldCpp that was killed can't tidy up its unpacked copy, so clear those first.
            for old in TEMP_DIR.glob("_MEI*"):
                shutil.rmtree(old, ignore_errors=True)
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            port = int(self.own_url.rsplit(":", 1)[1])
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            log = open(LOG_PATH, "w", encoding="utf-8", errors="replace")
            print("[gguf] Starting KoboldCpp with the GGUF voice model ...", flush=True)
            started = time.time()
            self.proc = subprocess.Popen(
                launch_args(self.settings, port),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(exe_path(self.settings).parent),
                env={**os.environ, "TMP": str(TEMP_DIR), "TEMP": str(TEMP_DIR)},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            log.close()  # the child keeps its own handle
            self._job = _tie_to_this_process(self.proc)
            while self._tts_voices(self.own_url) is None:
                if self.proc.poll() is not None:
                    tail = LOG_PATH.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-8:]
                    self.proc = None
                    raise ValueError("KoboldCpp stopped while starting:\n\n" + "\n".join(tail))
                if time.time() - started > START_TIMEOUT_S:
                    self.stop()
                    raise ValueError(f"KoboldCpp took too long to start. Its log is in {LOG_PATH}.")
                time.sleep(1)
            self._ready = True
            print(f"[gguf] KoboldCpp ready in {time.time() - started:.0f}s", flush=True)

    def ensure_running(self) -> None:
        """Make sure a KoboldCpp with the GGUF model is up on gguf_port: ours, or one
        Speak or the voice server already started there. Starts one if not."""
        with self._lock:
            if self.running() and self._ready:
                return
            voices = self._tts_voices(self.own_url)
            if voices is not None:
                self._url, self._voices = self.own_url, voices
                return
            self.start()

    def warm_up(self) -> None:
        try:
            self.ensure_running()
        except Exception as exc:
            print(f"[gguf] Could not start KoboldCpp yet: {exc}")

    def stop(self) -> None:
        with self._lock:
            proc, self.proc, self._url, self._ready = self.proc, None, None, False
            if proc is not None and proc.poll() is None:
                print("[gguf] Stopping KoboldCpp", flush=True)
                if os.name == "nt":  # its one-file .exe runs as two processes, so end the whole tree
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
                else:
                    proc.terminate()
                try:
                    proc.wait(15)
                except subprocess.TimeoutExpired:
                    proc.kill()
            self._job = None

    # -- speaking --------------------------------------------------------------

    def _speak_chunk(self, url: str, name: str, text: str, language: str | None) -> tuple[np.ndarray, int]:
        body = {"input": text, "voice": name, "response_format": "wav"}
        if language and language != "Auto":
            body["language"] = language
        req = urllib.request.Request(f"{url}/v1/audio/speech", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with _local.open(req, timeout=600) as r:
                data = r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise ValueError(f"The KoboldCpp at {url} has a password set. Remove it, or set koboldcpp_url to another one.") from None
            raise ValueError(f"KoboldCpp couldn't say that (error {exc.code}).") from None
        except OSError as exc:
            raise _Gone() from exc
        if not data.startswith(b"RIFF"):
            raise ValueError(f"KoboldCpp couldn't say that. Its log is in {LOG_PATH}.")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32")
        return audio.to_mono_float(wav), int(sr)

    def speak(self, text: str, voice: Voice, language: str | None = None, speed: float = 1.0) -> tuple[np.ndarray, int]:
        chunks = split_text(text)
        if not chunks:
            raise ValueError("There is no text to speak.")
        name = f"{voice.id}.wav"
        language = language or voice.language
        with self._lock:
            url = self._url if self._url and name in self._voices else self._find(name)
            out, sr = [], 24000
            for chunk in chunks:
                try:
                    wav, sr = self._speak_chunk(url, name, chunk, language)
                except _Gone:  # that KoboldCpp was closed; find or start another
                    self._url = None
                    url = self._find(name)
                    wav, sr = self._speak_chunk(url, name, chunk, language)
                if audio.is_silent(wav):
                    raise ValueError("KoboldCpp produced silence. Try again, or change the text a little.")
                out.append(wav)
        return audio.change_speed(audio.concat(out, sr), speed), sr


def main() -> int:
    setup_environment()
    settings = load_settings()
    todo = missing(settings)
    if not todo:
        print("KoboldCpp and the GGUF voice model are already downloaded.")
        return 0
    try:
        install(settings)
    except Exception as exc:
        print(f"!! Download failed: {exc}")
        print("Check your internet connection and run install.bat again; it picks up where it left off.")
        return 1
    print("KoboldCpp and the GGUF voice model are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
