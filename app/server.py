"""Voice server: an OpenAI-compatible text-to-speech API for your saved voices,
with a small control panel.

    POST /v1/audio/speech   {"input": "...", "voice": "<voice id>", "response_format": "mp3"}
    GET  /v1/audio/voices   the saved voices
    GET  /ui                control panel (status, SillyTavern settings, test box, request log)

Start it with start_server.bat (or `python -m app.server`).
"""

from __future__ import annotations

import argparse
import threading
import time
import traceback
import webbrowser
from collections import deque

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel

from . import audio
from .config import load_settings, setup_environment
from .engine import OutOfMemory, TTSEngine
from .voices import VoiceLibrary

MODEL_NAME = "qwen3-tts"
FORMATS = ("mp3", "wav", "flac", "opus", "aac", "pcm")


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


def create_app(engine: TTSEngine, library: VoiceLibrary, settings: dict) -> FastAPI:
    app = FastAPI(title="Qwen3-TTS voice server", docs_url="/docs", redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.log = deque(maxlen=50)

    def check_key(authorization: str | None = Header(default=None)):
        key = (settings.get("api_key") or "").strip()
        if key and (authorization or "").strip() != f"Bearer {key}":
            raise HTTPException(status_code=401, detail="Missing or wrong API key.")

    def record(voice: str, chars: int, started: float, status: str):
        app.state.log.appendleft([time.strftime("%H:%M:%S"), voice, chars, round(time.time() - started, 1), status])

    app.state.record = record

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui/")

    @app.get("/health")
    def health():
        return {"status": "ok", "voices": len(library.list()), **engine.status()}

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
        fmt = (req.response_format or "mp3").lower()
        if fmt not in FORMATS:
            raise HTTPException(400, f"Unsupported response_format '{fmt}'. Use one of: {', '.join(FORMATS)}.")
        v = library.get(req.voice)
        if v is None:
            record(req.voice, len(req.input), started, "unknown voice")
            raise HTTPException(404, f"No saved voice called '{req.voice}'. Available: {', '.join(library.ids()) or 'none yet'}.")
        if not req.input.strip():
            raise HTTPException(400, "The input text is empty.")
        try:
            wav, sr = engine.speak(req.input, voice=v, language=req.language, speed=min(max(req.speed, 0.25), 4.0))
            data, media_type = audio.encode(wav, sr, fmt)
        except OutOfMemory as exc:
            record(v.id, len(req.input), started, "out of GPU memory")
            raise HTTPException(503, str(exc))
        except ValueError as exc:
            record(v.id, len(req.input), started, f"error: {exc}")
            raise HTTPException(400, str(exc))
        except Exception as exc:
            traceback.print_exc()
            record(v.id, len(req.input), started, f"error: {exc}")
            raise HTTPException(500, f"Generation failed: {exc}")
        record(v.id, len(req.input), started, "ok")
        return Response(content=data, media_type=media_type)

    return app


def build_panel(engine: TTSEngine, library: VoiceLibrary, settings: dict, app: FastAPI, base_url: str):
    import gradio as gr

    from .config import LANGUAGES
    from .studio import VOICE_DEFAULT_LANG, _for_player, friendly

    def st_voices():
        return ",".join(library.ids())

    def voice_table():
        return [[v.id, v.name, v.kind, v.language] for v in library.list()]

    def status_md():
        s = engine.status()
        gpu = s.get("gpu", "CPU")
        mem = f" ({s['vram_used_gb']} / {s['vram_total_gb']} GB VRAM used)" if "vram_total_gb" in s else ""
        loaded = "ready" if "clone" in s["loaded_models"] else "not loaded yet (loads on first request)"
        return f"**Server:** {base_url}  \n**GPU:** {gpu}{mem}  \n**Voice model:** {loaded}  \n**Voices:** {len(library.list())}"

    def refresh():
        return status_md(), st_voices(), voice_table(), list(app.state.log)

    with gr.Blocks(title="Qwen3-TTS Voice Server", theme=gr.themes.Soft()) as panel:
        gr.Markdown("# Qwen3-TTS Voice Server\nKeep this window open while SillyTavern (or any app) uses your voices.")
        status = gr.Markdown(status_md())

        with gr.Tab("Connect SillyTavern"):
            gr.Markdown(
                "In SillyTavern open **Extensions → TTS**, choose **OpenAI Compatible** as the provider, tick **Enabled**, "
                "and copy these values in. Then map each character to a voice under the voice list."
            )
            gr.Textbox(f"{base_url}/v1/audio/speech", label="Provider Endpoint", show_copy_button=True, interactive=False)
            gr.Textbox(MODEL_NAME, label="Model", show_copy_button=True, interactive=False)
            st_box = gr.Textbox(st_voices(), label="Available Voices (comma separated)", show_copy_button=True, interactive=False)
            gr.Markdown("Made a new voice in the Studio? Click **Refresh** and paste the updated voice list into SillyTavern.")
            if settings.get("api_key"):
                gr.Markdown("An API key is set in settings.json, so also enter it in SillyTavern's API key field.")

        with gr.Tab("Voices"):
            table = gr.Dataframe(voice_table(), headers=["id (use this in apps)", "name", "type", "language"], interactive=False)

        with gr.Tab("Test"):
            with gr.Row():
                t_voice = gr.Dropdown(library.ids(), label="Voice", allow_custom_value=True)
                t_lang = gr.Dropdown([VOICE_DEFAULT_LANG] + LANGUAGES, value=VOICE_DEFAULT_LANG, label="Language")
            t_text = gr.Textbox("Hello! If you can hear this, the voice server is working.", label="Text", lines=3)
            t_go = gr.Button("Speak", variant="primary")
            t_out = gr.Audio(interactive=False, autoplay=True)

        with gr.Tab("Request log"):
            log = gr.Dataframe(list(app.state.log), headers=["time", "voice", "chars", "seconds", "result"], interactive=False)

        with gr.Row():
            refresh_btn = gr.Button("Refresh")
            load_btn = gr.Button("Load voice model now")
            unload_btn = gr.Button("Free GPU memory")

        @friendly
        def test(voice_id, lang, text):
            v = library.get(voice_id)
            if v is None:
                raise ValueError("Pick a voice first.")
            started = time.time()
            wav, sr = engine.speak(text, voice=v, language=None if lang == VOICE_DEFAULT_LANG else lang)
            app.state.record(v.id, len(text), started, "ok (test tab)")
            return _for_player(wav, sr)

        t_go.click(test, [t_voice, t_lang, t_text], t_out)

        outputs = [status, st_box, table, log]
        refresh_btn.click(refresh, None, outputs).then(lambda: gr.update(choices=library.ids()), None, t_voice)
        load_btn.click(friendly(lambda: engine.preload("clone")), None, None).then(refresh, None, outputs)
        unload_btn.click(lambda: engine.unload(), None, None).then(refresh, None, outputs)
        gr.Timer(5).tick(refresh, None, outputs)

    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3-TTS voice server")
    ap.add_argument("--host", help="address to listen on (default 127.0.0.1; use 0.0.0.0 for other PCs on your network)")
    ap.add_argument("--port", type=int, help="port (default from settings.json, 8890)")
    ap.add_argument("--no-browser", action="store_true", help="don't open the control panel")
    ap.add_argument("--no-preload", action="store_true", help="load the model on the first request instead of at startup")
    args = ap.parse_args()

    setup_environment()
    settings = load_settings()
    host = args.host or settings.get("server_host", "127.0.0.1")
    port = args.port or int(settings.get("server_port", 8890))
    base_url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"

    import gradio as gr
    import uvicorn

    engine = TTSEngine(settings, idle_unload_minutes=settings.get("server_idle_unload_minutes", 0))
    library = VoiceLibrary()
    app = create_app(engine, library, settings)
    app = gr.mount_gradio_app(app, build_panel(engine, library, settings, app, base_url), path="/ui")

    print(f"\nVoice server:  {base_url}/v1/audio/speech")
    print(f"Control panel: {base_url}/ui")
    if host in ("0.0.0.0", "::") and not settings.get("api_key"):
        print("Warning: listening on your network with no api_key set in settings.json.")
    print("(Leave this window open. Close it to stop the server.)\n")

    if not args.no_preload:
        def warm():
            try:
                engine.preload("clone")
            except Exception as exc:
                print(f"[server] Could not preload the voice model: {exc}")

        threading.Thread(target=warm, daemon=True).start()
    if not args.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(f"{base_url}/ui")).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
