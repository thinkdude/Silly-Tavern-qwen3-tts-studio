"""Speak: type what you want said, pick one of your saved voices, and hear it.

Start it with start_speak.bat (or `python -m app.speak`). It speaks with either
voice model:
  * full: Qwen3-TTS itself. When the voice server is already running it does the
    speaking, so the model isn't loaded onto the GPU twice. Otherwise this app
    loads its own copy.
  * gguf: the small GGUF model, run by KoboldCpp (see gguf.py).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from . import audio, gguf
from .config import LANGUAGES, OUTPUTS_DIR, load_settings, save_settings, setup_environment
from .engine import TTSEngine
from .studio import VOICE_DEFAULT_LANG, friendly
from .voices import Voice, VoiceLibrary

FORMATS = ["wav", "mp3"]
EARLIER_LINES = 10
# How long to trust the last "is the voice server running?" answer. Windows can take
# a second to refuse a connection, so it isn't asked before every line.
SERVER_CHECK_S = 10
NO_VOICE = "Pick a voice first. No voices yet? Make one in the Voice Studio (start_studio.bat), then click Refresh voices."
PREF_DEFAULTS = {"voice": None, "language": VOICE_DEFAULT_LANG, "speed": 1.0, "format": "wav"}
ENGINES = [
    ("Full model (best quality)", "full"),
    ("GGUF with KoboldCpp (smaller, lighter on the GPU)", "gguf"),
]

# Talking to the voice server on this PC must never go through a system proxy.
_local = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# Enter speaks and Shift+Enter starts a new line, like a chat box. Gradio's own
# Enter handling (Shift+Enter submits) is kept out of it.
ENTER_TO_SPEAK = """
<script>
(() => {
  const keys = (e) => {
    if (e.key !== "Enter" || e.isComposing || !e.target.closest || !e.target.closest("#speak-text")) return;
    e.stopPropagation();
    if (e.shiftKey) return;
    e.preventDefault();
    if (e.type === "keydown") document.querySelector("#speak-btn")?.click();
  };
  window.addEventListener("keydown", keys, true);
  window.addEventListener("keypress", keys, true);
})();
</script>
"""
CSS = "#speak-text textarea { font-size: 1.15rem; line-height: 1.5; }"


def server_url(settings: dict) -> str:
    host = settings.get("server_host") or "127.0.0.1"
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    elif ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{int(settings.get('server_port', 8890))}"


class _ServerGone(Exception):
    pass


class Speaker:
    """Speaks with the GGUF model or the full one. The full one runs in the voice
    server when that's running, otherwise in this app's own engine."""

    def __init__(self, engine: TTSEngine, settings: dict, kobold: gguf.KoboldTTS | None = None):
        self.engine = engine
        self.settings = settings
        self.kobold = kobold
        self.server = server_url(settings)
        self._running, self._checked = False, 0.0

    @property
    def choice(self) -> str:
        return "gguf" if self.settings.get("speak_engine") == "gguf" and self.kobold else "full"

    def set_choice(self, choice: str) -> None:
        """Switch models, free the other one's GPU memory and remember the choice in settings.json."""
        choice = "gguf" if choice == "gguf" else "full"
        self.settings["speak_engine"] = choice
        saved = load_settings()  # keep any other edits made to the file meanwhile
        saved["speak_engine"] = choice
        save_settings(saved)
        if choice == "gguf":
            self.engine.unload()
        elif self.kobold:
            self.kobold.stop()

    def server_running(self) -> bool:
        if time.time() - self._checked < SERVER_CHECK_S:
            return self._running
        try:
            with _local.open(f"{self.server}/health", timeout=1) as r:
                data = json.load(r)
            running = isinstance(data, dict) and data.get("status") == "ok" and "voices" in data
        except (OSError, ValueError):
            running = False
        self._running, self._checked = running, time.time()
        return running

    def warm_up(self) -> None:
        if self.choice == "gguf":
            if not gguf.missing(self.settings):
                self.kobold.warm_up()
            return
        if self.server_running():
            print(f"[speak] The voice server is running at {self.server}, so it does the speaking.")
            return
        try:
            self.engine.preload("clone")
        except Exception as exc:
            print(f"[speak] Could not load the voice model yet: {exc}")

    def free(self) -> None:
        self.engine.unload()
        if self.kobold:
            self.kobold.stop()

    def say(self, voice: Voice, text: str, language: str | None = None, speed: float = 1.0) -> tuple[np.ndarray, int, str]:
        """Returns (audio, sample rate, what spoke it: "gguf", "server" or "full")."""
        if self.choice == "gguf":
            wav, sr = self.kobold.speak(text, voice, language=language, speed=speed)
            return wav, sr, "gguf"
        if self.server_running():
            try:
                wav, sr = self._via_server(voice.id, text, language, speed)
            except _ServerGone:
                self._running = False  # it was closed; speak with our own copy instead
            else:
                if self.engine.loaded():
                    self.engine.unload()  # the server has its own copy, so free ours
                return wav, sr, "server"
        wav, sr = self.engine.speak(text, voice=voice, language=language, speed=speed)
        return wav, sr, "full"

    def _via_server(self, voice_id: str, text: str, language: str | None, speed: float) -> tuple[np.ndarray, int]:
        body = {"input": text, "voice": voice_id, "response_format": "wav", "speed": speed}
        if language:
            body["language"] = language
        req = urllib.request.Request(
            f"{self.server}/v1/audio/speech", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        key = (self.settings.get("api_key") or "").strip()
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with _local.open(req, timeout=1800) as r:
                data = r.read()
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("detail")
            except (OSError, ValueError, AttributeError):
                detail = None
            raise ValueError(f"The voice server couldn't say that: {detail or exc.reason}") from None
        except OSError as exc:
            raise _ServerGone() from exc
        wav, sr = sf.read(io.BytesIO(data), dtype="float32")
        return wav, int(sr)


def save_output(voice_id: str, text: str, wav: np.ndarray, sr: int, fmt: str = "wav") -> Path:
    """Save a spoken line to outputs/ as <voice>-<date-time>-<first words>.<fmt>."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    words = re.sub(r"\W+", "-", " ".join(text.lower().split()[:6])).strip("-")[:40].strip("-")
    stem = "-".join(p for p in (voice_id, time.strftime("%Y%m%d-%H%M%S"), words) if p)
    path, n = OUTPUTS_DIR / f"{stem}.{fmt}", 2
    while path.exists():
        path, n = OUTPUTS_DIR / f"{stem}-{n}.{fmt}", n + 1
    data, _ = audio.encode(wav, sr, fmt)
    path.write_bytes(data)
    return path


def _short(text: str, n: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def build(speaker: Speaker, library: VoiceLibrary) -> gr.Blocks:
    def voice_choices():
        return [(v.name, v.id) for v in library.list()]

    def voices_update(selected: str | None = None):
        choices = voice_choices()
        ids = [c[1] for c in choices]
        return gr.update(choices=choices, value=selected if selected in ids else (ids[0] if ids else None))

    with gr.Blocks(title="Qwen3-TTS Speak", theme=gr.themes.Soft(), head=ENTER_TO_SPEAK, css=CSS) as demo:
        gr.Markdown(
            "# Qwen3-TTS Speak\nType what you want to hear, pick one of your voices and press **Enter**. "
            "**Shift+Enter** starts a new line. Every line is saved in the `outputs` folder. "
            "The first line can take a minute while the voice model loads."
        )
        prefs = gr.BrowserState(dict(PREF_DEFAULTS), storage_key="qwen3-tts-speak", secret="qwen3-tts-speak")
        with gr.Row(equal_height=True):
            voice = gr.Dropdown(voice_choices(), label="Voice", scale=4)
            refresh = gr.Button("Refresh voices", scale=1)
        model = gr.Radio(
            ENGINES,
            value=speaker.choice,
            label="Voice model",
            info="The full model sounds closest to your voices. The GGUF model needs less GPU memory; "
            "it copies the sound of each voice but can't use its transcript.",
            visible=speaker.kobold is not None,
        )
        text = gr.Textbox(
            label="What should it say?",
            placeholder="Type something, then press Enter.",
            lines=4,
            max_lines=20,
            autofocus=True,
            elem_id="speak-text",
        )
        with gr.Row():
            lang = gr.Dropdown([VOICE_DEFAULT_LANG] + LANGUAGES, value=VOICE_DEFAULT_LANG, label="Language")
            speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
            fmt = gr.Radio(FORMATS, value="wav", label="Save as")
        go = gr.Button("Speak", variant="primary", elem_id="speak-btn")
        out = gr.Audio(label="Latest", type="filepath", interactive=False, autoplay=True, show_download_button=True)
        status = gr.Markdown()
        history = gr.State([])

        @gr.render(inputs=[history])
        def earlier(items):
            items = [it for it in (items or [])[1:] if Path(it["path"]).exists()]
            if items:
                gr.Markdown("### Earlier")
                for it in items:
                    gr.Audio(it["path"], label=it["label"], type="filepath", interactive=False, show_download_button=True)

        with gr.Row():
            open_btn = gr.Button("Open the outputs folder", size="sm", visible=hasattr(os, "startfile"))
            free_btn = gr.Button("Free GPU memory", size="sm")

        # ------------------------------------------------------- handlers
        @friendly
        def speak(voice_id, words, language, spd, file_type, hist, progress=gr.Progress(track_tqdm=True)):
            v = library.get(voice_id) if voice_id else None
            if v is None:
                raise ValueError(NO_VOICE)
            if not (words or "").strip():
                raise ValueError("Type something to say first.")
            started = time.time()
            wav, sr, source = speaker.say(v, words, None if language == VOICE_DEFAULT_LANG else language, float(spd))
            path = save_output(v.id, words, wav, sr, file_type if file_type in FORMATS else "wav")
            note = f"Saved as `{path.name}` in the outputs folder. Took {time.time() - started:.0f}s"
            note += {
                "gguf": " with the GGUF model.",
                "server": ", spoken by the voice server that's already running.",
            }.get(source, ".")
            hist = [{"path": str(path), "label": f"{v.name}: {_short(words)}"}, *(hist or [])][: EARLIER_LINES + 1]
            return str(path), note, hist

        go.click(speak, [voice, text, lang, speed, fmt, history], [out, status, history])
        refresh.click(voices_update, [voice], [voice])

        @friendly
        def switch_model(choice, progress=gr.Progress(track_tqdm=True)):
            speaker.set_choice(choice)
            if choice != "gguf":
                return "Using the full model. It loads with the next line (or the voice server speaks, if it's running)."
            if gguf.missing(speaker.settings):
                gguf.install(speaker.settings)  # about 2.3 GB, once
            threading.Thread(target=speaker.kobold.warm_up, daemon=True).start()
            return (
                "Using the GGUF model. KoboldCpp is starting in the background; the first line can take "
                "a minute. It also restarts by itself when you use a voice it hasn't loaded yet."
            )

        model.input(switch_model, [model], [status])

        def restore(saved):
            p = {**PREF_DEFAULTS, **(saved or {})}
            return voices_update(p["voice"]), p["language"], p["speed"], p["format"], speaker.choice

        demo.load(restore, [prefs], [voice, lang, speed, fmt, model])

        def remember(v, language, spd, file_type):
            return {"voice": v, "language": language, "speed": spd, "format": file_type}

        for event in (voice.input, lang.input, speed.release, fmt.input):
            event(remember, [voice, lang, speed, fmt], [prefs], queue=False)

        def open_outputs():
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            os.startfile(OUTPUTS_DIR)

        open_btn.click(open_outputs, None, None, queue=False)

        def free():
            speaker.free()
            return "Freed this app's GPU memory. The voice model loads again with the next line."

        free_btn.click(free, None, [status])

    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3-TTS Speak: type something and hear it in one of your voices")
    ap.add_argument("--port", type=int, help="port to serve on (default from settings.json, 7861)")
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    args = ap.parse_args()

    setup_environment()
    settings = load_settings()
    engine = TTSEngine(settings, idle_unload_minutes=settings.get("speak_idle_unload_minutes", 10))
    library = VoiceLibrary()
    speaker = Speaker(engine, settings, gguf.KoboldTTS(settings, library))
    threading.Thread(target=speaker.warm_up, daemon=True).start()
    port = args.port or int(settings.get("speak_port", 7861))
    print(f"\nSpeak: http://127.0.0.1:{port}\n(Leave this window open while you use it. Close it to stop.)\n")
    build(speaker, library).queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=not args.no_browser,
        allowed_paths=[str(OUTPUTS_DIR)],
        show_api=False,
    )


if __name__ == "__main__":
    main()
