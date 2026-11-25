"""Voice server: an OpenAI-compatible text-to-speech API for your saved voices,
with a small control panel.

    POST /v1/audio/speech   {"input": "...", "voice": "<voice id>", "response_format": "mp3"}
    GET  /v1/audio/voices   the saved voices
    GET  /ui                control panel (status, model switch, SillyTavern settings, test box, request log)

The control panel can fill in SillyTavern's TTS settings by itself (see sillytavern.py) and
keep its voice list up to date as voices are saved or deleted.

It speaks with either voice model, picked with the control panel's switch:
  * full: Qwen3-TTS itself (best quality).
  * gguf: the small GGUF model, run by KoboldCpp (see gguf.py).
SillyTavern's settings are the same for both.

Start it with start_server.bat (or `python -m app.server`, `--engine gguf` to pick GGUF).
"""

from __future__ import annotations

import argparse
import threading
import time
import traceback
import webbrowser
from collections import deque

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel

from . import audio, gguf, sillytavern
from .config import load_settings, save_settings, setup_environment
from .engine import OutOfMemory, TTSEngine
from .voices import Voice, VoiceLibrary

MODEL_NAME = "qwen3-tts"
FORMATS = ("mp3", "wav", "flac", "opus", "aac", "pcm")
# Short names for the request log.
SHORT_NAMES = {"full": "full", "gguf": "GGUF"}


class SpeechRequest(BaseModel):
    input: str
    voice: str
    model: str = MODEL_NAME
    response_format: str = "mp3"
    speed: float = 1.0
    # Not part of OpenAI's API: force a language instead of the voice's default.
    language: str | None = None
    # OpenAI's style instructions. The clone model can't follow them, so they're ignored.
    instructions: str | None = None


class Backend:
    """Speaks with whichever voice model the control panel's switch picks: the full
    Qwen3-TTS model, or the GGUF one run by KoboldCpp."""

    def __init__(self, engine: TTSEngine, settings: dict, kobold: gguf.KoboldTTS | None = None):
        self.engine = engine
        self.settings = settings
        self.kobold = kobold

    @property
    def choice(self) -> str:
        return "gguf" if self.settings.get("server_engine") == "gguf" and self.kobold else "full"

    def set_choice(self, choice: str) -> None:
        """Switch models, free the other one's GPU memory and remember the choice in settings.json."""
        choice = "gguf" if choice == "gguf" else "full"
        self.settings["server_engine"] = choice
        saved = load_settings()  # keep any other edits made to the file meanwhile
        saved["server_engine"] = choice
        save_settings(saved)
        self.free(keep=choice)

    def ready(self, model: str | None = None) -> bool:
        """Is that model (default: the chosen one) loaded and ready to speak?"""
        if (model or self.choice) == "gguf":
            return self.kobold is not None and self.kobold.state() == "ready"
        return "clone" in self.engine.loaded()

    def speak(
        self, text: str, voice: Voice, language: str | None = None, speed: float = 1.0, model: str | None = None
    ) -> tuple[np.ndarray, int]:
        """Speak with the chosen model, or with `model` ("full" or "gguf") to try the other one."""
        if (model or self.choice) == "gguf":
            if self.kobold is None:
                raise ValueError("The GGUF model isn't available here.")
            return self.kobold.speak(text, voice, language=language, speed=speed)
        return self.engine.speak(text, voice=voice, language=language, speed=speed)

    def preload(self) -> None:
        if self.choice == "gguf":
            self.kobold.ensure_running()
        else:
            self.engine.preload("clone")

    def warm_up(self) -> None:
        try:
            self.preload()
        except Exception as exc:
            print(f"[server] Could not load the voice model yet: {exc}")

    def free(self, keep: str | None = None) -> None:
        """Free the GPU memory of both models, or of every model but `keep`."""
        if keep != "full":
            self.engine.unload()
        if keep != "gguf" and self.kobold:
            self.kobold.stop()

    def status(self) -> dict:
        info = {"engine": self.choice, **self.engine.status()}
        if self.kobold:
            info["gguf"] = self.kobold.state()
            info["gguf_url"] = self.kobold.url
        return info


def create_app(engine: TTSEngine, library: VoiceLibrary, settings: dict, kobold: gguf.KoboldTTS | None = None) -> FastAPI:
    app = FastAPI(title="Qwen3-TTS voice server", docs_url="/docs", redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.log = deque(maxlen=50)
    backend = app.state.backend = Backend(engine, settings, kobold)
    app.state.st_sync = sillytavern.AutoSync(settings, library)

    def check_key(authorization: str | None = Header(default=None)):
        key = (settings.get("api_key") or "").strip()
        if key and (authorization or "").strip() != f"Bearer {key}":
            raise HTTPException(status_code=401, detail="Missing or wrong API key.")

    def record(voice: str, chars: int, started: float, status: str, model: str | None = None):
        row = [time.strftime("%H:%M:%S"), voice, SHORT_NAMES[model or backend.choice], chars, round(time.time() - started, 1), status]
        app.state.log.appendleft(row)

    app.state.record = record

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui/")

    @app.get("/health")
    def health():
        return {"status": "ok", "voices": len(library.list()), **backend.status()}

    @app.get("/v1/models", dependencies=[Depends(check_key)])
    def models():
        return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]}

    @app.get("/v1/audio/voices", dependencies=[Depends(check_key)])
    def voices():
        return {
            "voices": [
                {"id": v.id, "name": v.name, "type": v.kind, "language": v.language, "description": v.description}
                for v in library.list()
            ]
        }

    @app.post("/v1/audio/speech", dependencies=[Depends(check_key)])
    def speech(req: SpeechRequest):
        started = time.time()
        model = backend.choice
        fmt = (req.response_format or "mp3").lower()
        if fmt not in FORMATS:
            raise HTTPException(400, f"Unsupported response_format '{fmt}'. Use one of: {', '.join(FORMATS)}.")
        v = library.get(req.voice)
        if v is None:
            record(req.voice, len(req.input), started, "unknown voice", model)
            raise HTTPException(404, f"No saved voice called '{req.voice}'. Available: {', '.join(library.ids()) or 'none yet'}.")
        if not req.input.strip():
            raise HTTPException(400, "The input text is empty.")
        try:
            wav, sr = backend.speak(req.input, v, language=req.language, speed=min(max(req.speed, 0.25), 4.0), model=model)
            data, media_type = audio.encode(wav, sr, fmt)
        except OutOfMemory as exc:
            record(v.id, len(req.input), started, "out of GPU memory", model)
            raise HTTPException(503, str(exc))
        except ValueError as exc:
            record(v.id, len(req.input), started, f"error: {exc}", model)
            raise HTTPException(400, str(exc))
        except Exception as exc:
            traceback.print_exc()
            record(v.id, len(req.input), started, f"error: {exc}", model)
            raise HTTPException(500, f"Generation failed: {exc}")
        record(v.id, len(req.input), started, "ok", model)
        return Response(content=data, media_type=media_type)

    return app


def build_panel(library: VoiceLibrary, settings: dict, app: FastAPI, base_url: str):
    import gradio as gr

    from .config import LANGUAGES
    from .studio import VOICE_DEFAULT_LANG, _for_player, friendly

    backend: Backend = app.state.backend
    names = {model: label for label, model in gguf.MODEL_CHOICES}
    short = {"full": "Full model", "gguf": "GGUF model"}

    def st_voices():
        return ",".join(library.ids())

    def voice_table():
        return [[v.id, v.name, v.kind, v.language] for v in library.list()]

    def model_md(s: dict) -> str:
        if s["engine"] == "gguf":
            state = {
                "ready": "ready",
                "starting": "KoboldCpp is starting (the first time can take a few minutes)",
                "stopped": "KoboldCpp starts with the first request",
            }[s["gguf"]]
            if s["gguf"] == "ready" and s.get("gguf_url") and s["gguf_url"] != backend.kobold.own_url:
                state += f", using your KoboldCpp at {s['gguf_url']}"
            return f"GGUF with KoboldCpp, {state}"
        return "Full model, " + ("ready" if "clone" in s["loaded_models"] else "not loaded yet (loads on first request)")

    def status_md():
        s = backend.status()
        gpu = s.get("gpu", "CPU")
        mem = f" ({s['vram_used_gb']} / {s['vram_total_gb']} GB VRAM used)" if "vram_total_gb" in s else ""
        return f"**Server:** {base_url}  \n**GPU:** {gpu}{mem}  \n**Voice model:** {model_md(s)}  \n**Voices:** {len(library.list())}"

    def refresh():
        return status_md(), st_voices(), voice_table(), list(app.state.log)

    st_sync: sillytavern.AutoSync = app.state.st_sync

    def st_folder_value() -> str:
        """The saved folder, else one found in the usual places (saved once Fill in is clicked)."""
        saved = settings.get("sillytavern_dir") or ""
        if saved:
            return saved
        try:
            found = sillytavern.guess_installs()
        except Exception:
            found = []
        return str(found[0]) if found else ""

    def st_status() -> str:
        folder = settings.get("sillytavern_dir") or ""
        if not folder:
            return ""
        try:
            root = sillytavern.find_install(folder)
            files = sillytavern.settings_files(root)
        except Exception as exc:
            return f"Couldn't check SillyTavern's settings: {exc}"
        ids = library.ids()
        lists = [sillytavern.current_voices(f) for f in files]
        if lists and all(v is not None and sorted(v) == sorted(ids) for v in lists):
            msg = f"SillyTavern has your {len(ids)} voice{'s' if len(ids) != 1 else ''}."
        else:
            msg = "SillyTavern's voice list differs from yours. Click **Fill in SillyTavern's settings** to update it."
        if st_sync.note:
            msg += " " + st_sync.note
        return msg

    with gr.Blocks(title="Qwen3-TTS Voice Server", theme=gr.themes.Soft()) as panel:
        gr.Markdown("# Qwen3-TTS Voice Server\nKeep this window open while SillyTavern (or any app) uses your voices.")
        status = gr.Markdown(status_md())
        model = gr.Radio(
            gguf.MODEL_CHOICES,
            value=backend.choice,
            label="Voice model",
            info="Which model speaks for SillyTavern and your other apps. The full model sounds closest to your voices. "
            "The GGUF model needs less GPU memory; it copies the sound of each voice but can't use its transcript. "
            "SillyTavern's settings stay the same for both. To hear the difference, use Compare on the Test tab.",
            visible=backend.kobold is not None,
        )
        model_note = gr.Markdown()

        with gr.Tab("Connect SillyTavern"):
            gr.Markdown(
                "Pick SillyTavern's folder and click **Fill in SillyTavern's settings**. That turns on SillyTavern's "
                "text-to-speech, points it at this server and fills in your voices, so there's nothing to copy."
            )
            with gr.Row(equal_height=True):
                st_dir = gr.Textbox(
                    st_folder_value(),
                    label="SillyTavern folder",
                    placeholder=r"For example D:\SillyTavern (the folder with Start.bat in it)",
                    scale=4,
                )
                st_browse = gr.Button("Browse…", scale=1, min_width=120)
            st_paragraphs = gr.Checkbox(
                bool(settings.get("sillytavern_paragraphs", True)),
                label="Speak long replies a paragraph at a time",
                info="SillyTavern sends each paragraph on its own, so the first one plays while the next is being made, "
                "instead of waiting for the whole reply. Turns on its two 'Narrate by paragraphs' options.",
            )
            st_auto = gr.Checkbox(
                bool(settings.get("sillytavern_auto_sync", True)),
                label="Keep SillyTavern's voice list up to date",
                info="When you save or delete a voice in the Studio, this server updates SillyTavern's list by itself.",
            )
            st_fill = gr.Button("Fill in SillyTavern's settings", variant="primary")
            st_note = gr.Markdown(st_status())
            with gr.Accordion("Or copy the settings in by hand", open=False):
                gr.Markdown(
                    "In SillyTavern open **Extensions → TTS**, choose **OpenAI Compatible** as the provider, tick **Enabled**, "
                    "and copy these values in. Then map each character to a voice under the voice list."
                )
                gr.Textbox(f"{base_url}/v1/audio/speech", label="Provider Endpoint", show_copy_button=True, interactive=False)
                gr.Textbox(MODEL_NAME, label="Model", show_copy_button=True, interactive=False)
                st_box = gr.Textbox(st_voices(), label="Available Voices (comma separated)", show_copy_button=True, interactive=False)
                gr.Markdown("Made a new voice in the Studio? Click **Refresh** and paste the updated voice list into SillyTavern.")
            if settings.get("api_key"):
                gr.Markdown(
                    "An API key is set in settings.json, so also enter it in SillyTavern (the **API Key** button next to "
                    "Provider Endpoint). Fill in doesn't copy keys."
                )

        with gr.Tab("Voices"):
            table = gr.Dataframe(voice_table(), headers=["id (use this in apps)", "name", "type", "language"], interactive=False)

        with gr.Tab("Test"):
            with gr.Row():
                t_voice = gr.Dropdown(library.ids(), label="Voice", allow_custom_value=True)
                t_lang = gr.Dropdown([VOICE_DEFAULT_LANG] + LANGUAGES, value=VOICE_DEFAULT_LANG, label="Language")
            t_text = gr.Textbox("Hello! If you can hear this, the voice server is working.", label="Text", lines=3)
            with gr.Row():
                t_go = gr.Button("Speak", variant="primary")
                t_cmp = gr.Button("Compare the full and GGUF models", visible=backend.kobold is not None)
            t_out = gr.Audio(label="With the selected voice model", interactive=False, autoplay=True)
            with gr.Row(visible=False) as cmp_row:
                c_full = gr.Audio(label=names["full"], interactive=False)
                c_gguf = gr.Audio(label=names["gguf"], interactive=False)
            c_note = gr.Markdown()

        with gr.Tab("Request log"):
            log = gr.Dataframe(list(app.state.log), headers=["time", "voice", "model", "chars", "seconds", "result"], interactive=False)

        with gr.Row():
            refresh_btn = gr.Button("Refresh")
            load_btn = gr.Button("Load voice model now")
            unload_btn = gr.Button("Free GPU memory")

        def pick(voice_id, text):
            v = library.get(voice_id)
            if v is None:
                raise ValueError("Pick a voice first.")
            if not (text or "").strip():
                raise ValueError("Type something to say first.")
            return v

        @friendly
        def test(voice_id, lang, text):
            v = pick(voice_id, text)
            started = time.time()
            wav, sr = backend.speak(text, v, language=None if lang == VOICE_DEFAULT_LANG else lang)
            app.state.record(v.id, len(text), started, "ok (test tab)")
            return _for_player(wav, sr)

        @friendly
        def compare(voice_id, lang, text, progress=gr.Progress(track_tqdm=True)):
            """The same line from both models, one after the other."""
            v = pick(voice_id, text)
            language = None if lang == VOICE_DEFAULT_LANG else lang
            takes = {}
            for i, which in enumerate(("full", "gguf")):
                progress(i / 2, desc=f"{short[which]} speaking ...")
                was_ready = backend.ready(which)
                started = time.time()
                try:
                    wav, sr = backend.speak(text, v, language=language, model=which)
                except Exception:
                    backend.free(keep=backend.choice)
                    raise
                took = time.time() - started
                app.state.record(v.id, len(text), started, "ok (compare)", which)
                label = f"{short[which]}: took {took:.0f}s" + ("" if was_ready else ", including loading it")
                takes[which] = gr.update(value=_for_player(wav, sr), label=label)
            # Give back the GPU memory of the model that isn't selected.
            backend.free(keep=backend.choice)
            note = (
                "Same voice, same text. Every take comes out a little different, so try a few lines before you decide. "
                "Like one better? Pick it under **Voice model** at the top."
            )
            return gr.update(visible=True), takes["full"], takes["gguf"], note

        t_go.click(test, [t_voice, t_lang, t_text], t_out)
        t_cmp.click(compare, [t_voice, t_lang, t_text], [cmp_row, c_full, c_gguf, c_note])

        @friendly
        def switch_model(choice, progress=gr.Progress(track_tqdm=True)):
            if choice == "gguf" and gguf.missing(settings):
                gguf.install(settings)  # about 2.3 GB, once
            backend.set_choice(choice)
            threading.Thread(target=backend.warm_up, daemon=True).start()
            if choice == "gguf":
                return (
                    "Now the GGUF model speaks. KoboldCpp is starting in the background, which takes a minute. "
                    "It restarts by itself (another minute) the first time a new voice is used."
                )
            return "Now the full model speaks. It's loading in the background."

        @friendly
        def st_pick(current):
            picked = sillytavern.browse(current)
            return picked if picked else gr.update()

        @friendly
        def st_fill_in(folder, paragraphs, auto):
            root = sillytavern.find_install(folder)
            ids = library.ids()
            result = sillytavern.apply(root, ids, endpoint=f"{base_url}/v1/audio/speech", paragraphs=paragraphs)
            # Remember it all, keeping any other edits made to settings.json meanwhile.
            changes = {"sillytavern_dir": str(root), "sillytavern_paragraphs": bool(paragraphs), "sillytavern_auto_sync": bool(auto)}
            settings.update(changes)
            saved = load_settings()
            saved.update(changes)
            save_settings(saved)
            st_sync.note = ""
            return str(root), sillytavern.describe(result, ids)

        def st_auto_changed(auto):
            if settings.get("sillytavern_dir"):  # before the first Fill in, the button saves it
                settings["sillytavern_auto_sync"] = bool(auto)
                saved = load_settings()
                saved["sillytavern_auto_sync"] = bool(auto)
                save_settings(saved)

        # Waiting on the folder picker shouldn't hold up anything else, so it skips the one-at-a-time queue.
        st_browse.click(st_pick, [st_dir], [st_dir], concurrency_limit=None)
        st_fill.click(st_fill_in, [st_dir, st_paragraphs, st_auto], [st_dir, st_note])
        st_auto.input(st_auto_changed, [st_auto], None)

        outputs = [status, st_box, table, log]
        # Afterwards, show the model actually in use (it stays the same when switching failed).
        model.input(switch_model, [model], [model_note]).then(lambda: gr.update(value=backend.choice), None, model).then(
            refresh, None, outputs
        )
        refresh_btn.click(refresh, None, outputs).then(lambda: gr.update(choices=library.ids()), None, t_voice)
        load_btn.click(friendly(lambda: backend.preload()), None, None).then(refresh, None, outputs)
        unload_btn.click(lambda: backend.free(), None, None).then(refresh, None, outputs)
        panel.load(lambda: gr.update(value=backend.choice), None, model)
        panel.load(st_status, None, st_note)
        gr.Timer(5).tick(refresh, None, outputs)

    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3-TTS voice server")
    ap.add_argument("--host", help="address to listen on (default 127.0.0.1; use 0.0.0.0 for other PCs on your network)")
    ap.add_argument("--port", type=int, help="port (default from settings.json, 8890)")
    ap.add_argument("--no-browser", action="store_true", help="don't open the control panel")
    ap.add_argument("--no-preload", action="store_true", help="load the model on the first request instead of at startup")
    ap.add_argument(
        "--engine", choices=["full", "gguf"], help="voice model: full or gguf (default: the one last picked in the control panel)"
    )
    args = ap.parse_args()

    setup_environment()
    settings = load_settings()
    if args.engine:
        settings["server_engine"] = args.engine
    host = args.host or settings.get("server_host", "127.0.0.1")
    port = args.port or int(settings.get("server_port", 8890))
    base_url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"

    import gradio as gr
    import uvicorn

    engine = TTSEngine(settings, idle_unload_minutes=settings.get("server_idle_unload_minutes", 0))
    library = VoiceLibrary()
    api = create_app(engine, library, settings, gguf.KoboldTTS(settings, library))
    backend: Backend = api.state.backend
    app = gr.mount_gradio_app(api, build_panel(library, settings, api, base_url), path="/ui")

    print(f"\nVoice server:  {base_url}/v1/audio/speech")
    print(f"Control panel: {base_url}/ui")
    label = {m: n for n, m in gguf.MODEL_CHOICES}[backend.choice]
    print(f"Voice model:   {label} (switch it in the control panel)")
    if host in ("0.0.0.0", "::") and not settings.get("api_key"):
        print("Warning: listening on your network with no api_key set in settings.json.")
    print("(Leave this window open. Close it to stop the server.)\n")

    if not args.no_preload:
        threading.Thread(target=backend.warm_up, daemon=True).start()
    # Keeps SillyTavern's voice list up to date, once its folder is set in the control panel.
    threading.Thread(target=api.state.st_sync.run, daemon=True).start()
    if not args.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(f"{base_url}/ui")).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
