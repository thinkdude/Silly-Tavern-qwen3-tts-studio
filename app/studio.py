"""Voice Studio: design voices from a description, clone voices from a clip,
and try out the voices you've saved.

Start it with start_studio.bat (or `python -m app.studio`).
"""

from __future__ import annotations

import argparse
import functools
import time
import traceback

import gradio as gr
import numpy as np
import soundfile as sf

from . import audio, download_models, koboldcpp, transcribe, youtube
from .config import CODE_LANGS, LANGUAGES, OUTPUTS_DIR, VOICES_DIR, load_settings, save_settings, setup_environment
from .engine import OutOfMemory, SilentOutput, TTSEngine
from .voices import VoiceLibrary

MAX_TAKES = 4
DTYPES = ["bfloat16", "float16", "float32"]
VOICE_DEFAULT_LANG = "Voice's default"

DEFAULT_SAMPLE_LINE = (
    "Hello there! It's really good to finally meet you. I've been looking forward to this "
    "for a while, so come in, sit down, and tell me everything that's happened."
)
DEFAULT_TEST_LINE = "This is a quick test of my new voice. So, how do I sound? Not bad, right?"

DESCRIPTION_EXAMPLES = [
    "Male, around 60 years old, deep gravelly voice, slow and deliberate, a warm but weary tone.",
    "Female, mid-20s, bright and energetic, speaks quickly with a playful, teasing tone.",
    "Female, 40s, calm low alto, soothing and clear, like an audiobook narrator.",
    "Young man, about 17, tenor, nervous and a little awkward, voice cracks now and then.",
    "Elderly woman, thin and slightly shaky voice, kind and grandmotherly, speaks gently.",
    "Male, 30s, smooth baritone, confident and charming, with a slight British accent.",
    "Female, 20s, soft and breathy, shy, speaks quietly and hesitantly.",
    "Gruff middle-aged man, loud and commanding, like a drill sergeant barking orders.",
]

TIPS = """
**Tips for a good clone**
- 5 to 15 seconds of one person talking, with no music or background noise.
- Give it exactly what is said in the clip (**Transcribe** fills it in for you to check). Clones with a transcript are noticeably better.
- The clone copies the clip's mood and recording quality, so pick a clip that sounds the way you want.
"""

YOUTUBE_HELP = """
Paste a link to a video where the voice talks clearly, ideally on its own. Find a good stretch on YouTube and put
its time in **Start at**, or right-click the video at that moment, choose **Copy video URL at current time**, paste
that link and leave Start at empty. The clip lands in **Reference clip** below, with a transcript from the video's
captions, or from speech-to-text when it has none, for you to check and correct. Other video sites work too.

Only clone voices you have permission to use.
"""

FILE_HELP = """
Use a film, an episode, an interview or a podcast you already have. Click **Browse** to pick the file, or paste its
path (in File Explorer, right-click the file and choose **Copy as path**; on Windows 10, hold Shift while you
right-click). Play it in your usual player to find a stretch where the voice talks clearly, ideally on its own, and put
that time in **Start at**. The clip lands in **Reference clip** below with a speech-to-text transcript for you to check
and correct. The file stays where it is: nothing is copied or changed.

Only clone voices you have permission to use.
"""
NO_TRACKS = {"choices": [], "value": None, "visible": False}


def friendly(fn):
    """Show errors as a popup in the GUI instead of a stack trace."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except gr.Error:
            raise
        except (ValueError, KeyError, OutOfMemory, SilentOutput) as exc:
            raise gr.Error(str(exc).strip("'\""))
        except Exception as exc:
            traceback.print_exc()
            raise gr.Error(f"Something went wrong: {exc}")

    return wrapper


def _for_player(wav: np.ndarray, sr: int):
    return sr, (np.clip(np.nan_to_num(wav), -1, 1) * 32767).astype(np.int16)


def build(engine: TTSEngine, library: VoiceLibrary) -> gr.Blocks:
    def voice_choices():
        return [(f"{v.name}  ({v.kind})", v.id) for v in library.list()]

    def library_update(selected: str | None = None):
        choices = voice_choices()
        ids = [c[1] for c in choices]
        value = selected if selected in ids else (ids[0] if ids else None)
        return gr.update(choices=choices, value=value)

    def export_note() -> str:
        """Mirror the library into KoboldCpp's voices folder, if that's switched on."""
        if not engine.settings.get("koboldcpp_export", True):
            return ""
        try:
            r = koboldcpp.sync(library, engine.settings)
        except OSError as exc:
            return f" Couldn't update the KoboldCpp voices folder: {exc}"
        return " Copied to the KoboldCpp voices folder; restart KoboldCpp to pick it up." if r["added"] or r["removed"] else ""

    def kobold_voices() -> str:
        return ",".join(koboldcpp.voice_names(library))

    with gr.Blocks(title="Qwen3-TTS Voice Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Qwen3-TTS Voice Studio\nDesign a voice from a description, or clone one from a clip. Saved voices show up in **My voices** and in the voice server.")

        # ------------------------------------------------------------ Design
        with gr.Tab("Design a voice"):
            gr.Markdown("Describe the voice you want. The model reads the sample line in that voice; generate a few takes and keep your favourite.")
            d_desc = gr.Textbox(label="Voice description", lines=3, placeholder=DESCRIPTION_EXAMPLES[0])
            gr.Examples([[d] for d in DESCRIPTION_EXAMPLES], inputs=[d_desc], label="Ideas")
            d_line = gr.Textbox(label="Sample line (becomes the voice's reference clip)", value=DEFAULT_SAMPLE_LINE, lines=2)
            with gr.Row():
                d_lang = gr.Dropdown(LANGUAGES, value="English", label="Language")
                d_takes = gr.Slider(1, MAX_TAKES, value=2, step=1, label="Takes to generate")
            d_go = gr.Button("Generate", variant="primary")
            d_state = gr.State(None)
            with gr.Row():
                d_players = [gr.Audio(label=f"Take {i + 1}", interactive=False, show_download_button=True, visible=(i == 0)) for i in range(MAX_TAKES)]
            with gr.Row():
                d_pick = gr.Radio(["Take 1"], value="Take 1", label="Keep which take?")
                d_name = gr.Textbox(label="Voice name", placeholder="e.g. Grumpy Wizard")
            d_save = gr.Button("Save voice")
            d_status = gr.Markdown()

        # ------------------------------------------------------------- Clone
        with gr.Tab("Clone a voice"):
            with gr.Accordion("Get the clip from a YouTube video", open=False):
                gr.Markdown(YOUTUBE_HELP)
                with gr.Row():
                    y_url = gr.Textbox(label="Video link", placeholder="https://www.youtube.com/watch?v=...", scale=3)
                    y_start = gr.Textbox(label="Start at (minutes:seconds)", placeholder="e.g. 1:23", scale=1)
                    y_len = gr.Slider(
                        youtube.MIN_LEN, youtube.MAX_LEN, value=youtube.DEFAULT_LEN, step=1, label="Clip length (seconds)", scale=2
                    )
                with gr.Row():
                    y_clean = gr.Checkbox(value=True, label="Remove music and background noise")
                    y_transcript = gr.Dropdown(
                        [
                            ("YouTube captions, or speech-to-text if there are none", "captions"),
                            ("Speech-to-text", "speech"),
                            ("None, I'll type it", "none"),
                        ],
                        value="captions",
                        label="Transcript from",
                    )
                    y_go = gr.Button("Get clip", variant="primary")
                y_status = gr.Markdown()
                y_update = gr.Button("Update the YouTube downloader", size="sm")
            with gr.Accordion("Get the clip from a video or audio file on this PC", open=False):
                gr.Markdown(FILE_HELP)
                with gr.Row(equal_height=True):
                    f_path = gr.Textbox(label="Video or audio file", placeholder=r"e.g. D:\Videos\interview.mp4", scale=5)
                    f_browse = gr.Button("Browse…", scale=1, min_width=120)
                f_info = gr.Markdown()
                f_track_for = gr.State("")  # the file the audio track list below belongs to
                with gr.Row():
                    f_track = gr.Dropdown(label="Audio track (the language you want)", scale=2, **NO_TRACKS)
                    f_start = gr.Textbox(label="Start at (minutes:seconds)", placeholder="e.g. 1:23", scale=1)
                    f_len = gr.Slider(
                        youtube.MIN_LEN, youtube.MAX_LEN, value=youtube.DEFAULT_LEN, step=1, label="Clip length (seconds)", scale=2
                    )
                with gr.Row():
                    f_clean = gr.Checkbox(value=True, label="Remove music and background noise")
                    f_transcript = gr.Dropdown(
                        [("Speech-to-text", "speech"), ("None, I'll type it", "none")], value="speech", label="Transcript from"
                    )
                    f_go = gr.Button("Get clip", variant="primary")
                f_status = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    c_ref = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference clip")
                    c_text = gr.Textbox(
                        label="Exactly what is said in the clip",
                        info="Type it, or click Transcribe and fix any mistakes",
                        lines=3,
                    )
                    c_transcribe = gr.Button("Transcribe the clip", size="sm")
                    c_xvec = gr.Checkbox(label="I don't have a transcript (works, but the clone is less accurate)")
                    c_lang = gr.Dropdown(LANGUAGES, value="English", label="Language")
                with gr.Column():
                    gr.Markdown(TIPS)
                    c_test = gr.Textbox(label="Test line", value=DEFAULT_TEST_LINE, lines=2)
                    c_preview = gr.Button("Preview clone", variant="primary")
                    c_out = gr.Audio(label="Preview", interactive=False)
            with gr.Row():
                c_name = gr.Textbox(label="Voice name", placeholder="e.g. Dad")
                c_save = gr.Button("Save voice")
            c_status = gr.Markdown()

        # --------------------------------------------------------- My voices
        with gr.Tab("My voices") as t_lib:
            with gr.Row():
                l_voice = gr.Dropdown(voice_choices(), label="Voice", scale=4)
                l_refresh = gr.Button("Refresh", scale=1)
            l_info = gr.Markdown()
            l_refclip = gr.Audio(label="Reference clip", interactive=False, type="filepath")
            l_text = gr.Textbox(label="Text to speak", lines=5, value=DEFAULT_TEST_LINE)
            with gr.Row():
                l_lang = gr.Dropdown([VOICE_DEFAULT_LANG] + LANGUAGES, value=VOICE_DEFAULT_LANG, label="Language")
                l_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
            l_go = gr.Button("Speak", variant="primary")
            l_out = gr.Audio(label="Result (download with the button on the player)", interactive=False, type="filepath")
            with gr.Accordion("Edit or delete this voice", open=False):
                with gr.Row():
                    l_newname = gr.Textbox(label="New name")
                    l_rename = gr.Button("Rename")
                with gr.Row():
                    l_reftext = gr.Textbox(label="Transcript of the reference clip", lines=2, scale=4)
                    with gr.Column(scale=1, min_width=160):
                        l_transcribe = gr.Button("Transcribe", size="sm")
                        l_savetext = gr.Button("Save transcript")
                with gr.Row():
                    l_confirm = gr.Checkbox(label="Yes, delete it for good")
                    l_delete = gr.Button("Delete voice", variant="stop")
            l_status = gr.Markdown()

        # ---------------------------------------------------------- KoboldCpp
        with gr.Tab("KoboldCpp") as t_kcpp:
            gr.Markdown(
                "KoboldCpp can play your saved voices for SillyTavern using small GGUF files. Every voice you save here "
                "is copied into the folder below automatically.\n\n"
                "**One-time KoboldCpp setup** (in the KoboldCpp launcher's **Audio** tab):\n"
                "1. **TTS Model:** paste the model link below (KoboldCpp downloads it), or a downloaded copy of that file.\n"
                "2. **WavTokenizer Model:** paste the tokenizer link below.\n"
                "3. **TTS Voices Dir:** the voices folder below.\n"
                "4. Tick **TTS Use GPU**, then launch. Use **Save Config** so you only do this once.\n\n"
                "KoboldCpp loads voices when it starts, so **restart KoboldCpp after saving new voices**."
            )
            gr.Textbox(str(koboldcpp.voices_dir(engine.settings)), label="TTS Voices Dir", show_copy_button=True, interactive=False)
            with gr.Row():
                gr.Textbox(koboldcpp.MODEL_URL, label="TTS Model", show_copy_button=True, interactive=False)
                gr.Textbox(koboldcpp.TOKENIZER_URL, label="WavTokenizer Model", show_copy_button=True, interactive=False)
            gr.Markdown(
                "**SillyTavern:** in **Extensions → TTS** choose **OpenAI Compatible**, tick **Enabled**, and enter these. "
                "The voice names include `.wav`; that's how KoboldCpp names cloned voices."
            )
            gr.Textbox(koboldcpp.speech_url(engine.settings), label="Provider Endpoint", show_copy_button=True, interactive=False)
            k_voices = gr.Textbox(kobold_voices(), label="Available Voices (comma separated)", show_copy_button=True, interactive=False)
            with gr.Row():
                k_sync = gr.Button("Copy all voices to the folder now")
            k_status = gr.Markdown()

        # ------------------------------------------------------------ System
        with gr.Tab("System"):
            s_info = gr.JSON(label="Status")
            with gr.Row():
                s_refresh = gr.Button("Refresh")
                s_unload = gr.Button("Free GPU memory (unload models)")
            gr.Markdown("Settings live in `settings.json` in the install folder. Restart the Studio after editing it.")
            gr.Markdown("### Precision")
            with gr.Row():
                s_dtype = gr.Radio(
                    DTYPES,
                    value=engine.settings.get("dtype", "bfloat16"),
                    label="Precision",
                    info="bfloat16 suits RTX 30-series and newer, float16 older cards. float32 works on every card "
                    "but needs about twice the GPU memory; use it if takes come out silent.",
                    scale=3,
                )
                s_dtype_apply = gr.Button("Apply", scale=1)
            s_dtype_note = gr.Markdown()
            gr.Markdown("### Disk space")
            s_disk = gr.Markdown()
            with gr.Row():
                s_rm_confirm = gr.Checkbox(label="Yes, remove the voice design model")
                s_rm_design = gr.Button("Remove voice design model")

        # ------------------------------------------------------- handlers
        @friendly
        def do_design(desc, line, lang, takes):
            if not (desc or "").strip():
                raise ValueError("Describe the voice first.")
            if not (line or "").strip():
                raise ValueError("The sample line is empty.")
            wavs, sr = engine.design(line.strip(), desc.strip(), lang, int(takes))
            players = [
                gr.update(value=_for_player(wavs[i], sr), visible=True) if i < len(wavs) else gr.update(value=None, visible=False)
                for i in range(MAX_TAKES)
            ]
            labels = [f"Take {i + 1}" for i in range(len(wavs))]
            status = "Listen to the takes, pick one, name it and save."
            if len(wavs) < int(takes):
                status += f" ({int(takes) - len(wavs)} of the takes came out silent and were left out.)"
            return [(wavs, sr), *players, gr.update(choices=labels, value=labels[0]), status]

        d_go.click(do_design, [d_desc, d_line, d_lang, d_takes], [d_state, *d_players, d_pick, d_status])

        @friendly
        def save_design(state, pick, name, desc, line, lang):
            if not state:
                raise ValueError("Generate a voice first.")
            wavs, sr = state
            idx = int(str(pick).split()[-1]) - 1 if pick else 0
            v = library.save(name, wavs[idx], sr, kind="designed", ref_text=line, language=lang, description=desc)
            return f"Saved **{v.name}** (id `{v.id}`)." + export_note(), library_update(v.id)

        d_save.click(save_design, [d_state, d_pick, d_name, d_desc, d_line, d_lang], [d_status, l_voice])

        def _clip(path):
            if not path:
                raise ValueError("Upload or record a reference clip first.")
            wav, sr = audio.load_audio(path)
            wav = audio.trim_silence(wav, sr)
            secs = len(wav) / sr
            if secs < 2:
                raise ValueError("The clip is too short. Use at least 3 seconds of speech (5 to 15 is best).")
            note = f"Clip length: {secs:.1f}s."
            if secs > 20:
                note += " That's long; 5 to 15 seconds clones just as well and is faster."
            return wav, sr, note

        def _check_text(text, xvec):
            if not xvec and not (text or "").strip():
                raise ValueError("Type what is said in the clip, or tick \"I don't have a transcript\".")

        @friendly
        def preview_clone(path, text, xvec, lang, test):
            _check_text(text, xvec)
            wav, sr, note = _clip(path)
            out, out_sr = engine.speak(test, ref_audio=(wav, sr), ref_text=text, x_vector_only=xvec, language=lang)
            return _for_player(out, out_sr), note + " Happy with it? Name it and save."

        c_preview.click(preview_clone, [c_ref, c_text, c_xvec, c_lang, c_test], [c_out, c_status])

        @friendly
        def save_clone(path, text, xvec, lang, name):
            _check_text(text, xvec)
            wav, sr, _ = _clip(path)
            v = library.save(name, wav, sr, kind="cloned", ref_text="" if xvec else text, language=lang, x_vector_only=xvec)
            return f"Saved **{v.name}** (id `{v.id}`)." + export_note(), library_update(v.id)

        c_save.click(save_clone, [c_ref, c_text, c_xvec, c_lang, c_name], [c_status, l_voice])

        def check_note(language, lang):
            if language == lang:
                return ""
            if language == "Auto":
                return " Its language isn't one Qwen3-TTS lists, so Language is set to **Auto**."
            return f" Language set to **{language}** to match."

        def clip_outputs(clip, how, lang, again):
            """What the Clone tab shows for a clip from a video: the clip, its transcript and language, and a note."""
            where = f"{youtube.format_time(clip.start)} to {youtube.format_time(clip.end)}"
            msg = f"Got **{clip.seconds:.1f} seconds** of *{clip.title}* ({where}). Have a listen in **Reference clip** below."
            if clip.transcript:
                source = {
                    "written": "the video's captions",
                    "automatic": "YouTube's automatic captions",
                    "speech": "speech-to-text",
                }.get(clip.source, "the video")
                msg += (
                    f" The transcript is from {source}: **check it against the clip and fix any mistakes** "
                    "before you preview, especially the first and last words."
                )
            elif how == "none":
                msg += " Type what's said in the clip (or click **Transcribe the clip**)."
            else:
                msg += " No words were found in that part, so type what's said (or tick \"I don't have a transcript\")."
            if clip.note:
                msg += f" ({clip.note})"
            msg += check_note(clip.language, lang)
            msg += (
                f" Not quite right? Change the start time and click **Get clip** again{again}, "
                "or trim the clip with the scissors on the player and click **Transcribe the clip**."
            )
            return str(clip.path), clip.transcript, gr.update(value=False) if clip.transcript else gr.update(), clip.language, msg

        @friendly
        def youtube_clip(url, start, length, clean, how, lang, progress=gr.Progress()):
            clip = youtube.make_clip(
                url,
                start,
                length,
                language=lang,
                isolate=clean,
                transcript=how,
                device=engine.settings.get("device", "cpu"),
                settings=engine.settings,
                progress=lambda f, desc: progress(f, desc=desc),
            )
            return clip_outputs(clip, how, lang, " (quick now the video is downloaded)")

        y_go.click(
            youtube_clip, [y_url, y_start, y_len, y_clean, y_transcript, c_lang], [c_ref, c_text, c_xvec, c_lang, y_status]
        )

        def _transcribe(path, progress):
            if not path:
                raise ValueError("Upload, record or get a clip first.")
            first = "" if transcribe.is_downloaded(engine.settings) else " (downloading the model first, about 1.6 GB)"
            progress(0.2, desc="Writing down what's said" + first)
            wav, sr = audio.load_audio(path)
            text, code = transcribe.text(wav, sr, engine.settings)
            if not text:
                raise ValueError("No words found in the clip. Is it someone speaking?")
            return text, CODE_LANGS.get((code or "").split("-")[0], "Auto") if code else None

        @friendly
        def transcribe_clip(path, lang, progress=gr.Progress()):
            text, language = _transcribe(path, progress)
            language = language or lang
            note = "Transcribed. **Check it against the clip and fix any mistakes**, then click **Preview clone**."
            return text, gr.update(value=False), language, note + check_note(language, lang)

        c_transcribe.click(transcribe_clip, [c_ref, c_lang], [c_text, c_xvec, c_lang, c_status])

        @friendly
        def update_downloader(progress=gr.Progress()):
            progress(0.5, desc="Updating yt-dlp")
            return youtube.update_downloader()

        y_update.click(update_downloader, None, y_status)

        def file_info(path, loaded=""):
            """Say how long the file is, and list its audio tracks when it has more than one."""
            if not (path or "").strip():
                return "", gr.update(**NO_TRACKS), ""
            if path.strip() == (loaded or "").strip():
                return gr.update(), gr.update(), gr.update()  # shown already; keep the track that's picked
            try:
                video = youtube.open_file(path)
            except (ValueError, OSError) as exc:
                return str(exc), gr.update(**NO_TRACKS), ""
            msg = f"**{video.title}**: " + (f"{youtube.format_time(video.duration)} long" if video.duration else "length unknown")
            if len(video.tracks) > 1:
                msg += f", with {len(video.tracks)} audio tracks. Pick the one in the language you want the voice to speak."
            tracks = gr.update(
                choices=[(t.label, t.index) for t in video.tracks],
                value=youtube.default_track(video),
                visible=len(video.tracks) > 1,
            )
            return msg, tracks, path

        f_path.submit(file_info, [f_path, f_track_for], [f_info, f_track, f_track_for], show_progress="hidden")
        f_path.blur(file_info, [f_path, f_track_for], [f_info, f_track, f_track_for], show_progress="hidden")

        @friendly
        def browse_file(current, loaded):
            try:
                picked = youtube.browse(current)
            except ValueError as exc:  # no file picker: say so where the file's details go, and keep the path box usable
                return gr.update(), str(exc), gr.update(), gr.update()
            if not picked:
                return gr.update(), gr.update(), gr.update(), gr.update()
            return picked, *file_info(picked, loaded)

        # Waiting on the file picker shouldn't hold up anything else, so it skips the one-at-a-time queue.
        f_browse.click(
            browse_file,
            [f_path, f_track_for],
            [f_path, f_info, f_track, f_track_for],
            concurrency_limit=None,
            show_progress_on=[f_info],
        )

        @friendly
        def file_clip(path, track, track_for, start, length, clean, how, lang, progress=gr.Progress()):
            clip = youtube.make_file_clip(
                path,
                start,
                length,
                # A track list left over from another file doesn't count: the file's default track is used.
                track=track if track_for and (path or "").strip() == track_for.strip() else None,
                language=lang,
                isolate=clean,
                transcript=how,
                device=engine.settings.get("device", "cpu"),
                settings=engine.settings,
                progress=lambda f, desc: progress(f, desc=desc),
            )
            return clip_outputs(clip, how, lang, "")

        f_go.click(
            file_clip,
            [f_path, f_track, f_track_for, f_start, f_len, f_clean, f_transcript, c_lang],
            [c_ref, c_text, c_xvec, c_lang, f_status],
        )

        def show_voice(voice_id):
            v = library.get(voice_id) if voice_id else None
            if v is None:
                return "No voices yet. Make one in the Design or Clone tab.", None, "", ""
            lines = [f"**{v.name}**: {v.kind} voice, id `{v.id}`, language {v.language}"]
            if v.description:
                lines.append(f"*{v.description}*")
            if v.x_vector_only:
                lines.append("No transcript saved, so this clone is less accurate. Add one below to improve it.")
            return "\n\n".join(lines), str(v.reference_path), v.name, v.ref_text

        l_voice.change(show_voice, [l_voice], [l_info, l_refclip, l_newname, l_reftext])
        l_refresh.click(lambda cur: library_update(cur), [l_voice], [l_voice])
        t_lib.select(lambda cur: library_update(cur), [l_voice], [l_voice])

        @friendly
        def speak(voice_id, text, lang, speed):
            v = library.get(voice_id)
            if v is None:
                raise ValueError("Pick a voice first.")
            wav, sr = engine.speak(text, voice=v, language=None if lang == VOICE_DEFAULT_LANG else lang, speed=speed)
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            out = OUTPUTS_DIR / f"{v.id}-{time.strftime('%Y%m%d-%H%M%S')}.wav"
            sf.write(str(out), wav, sr, subtype="PCM_16")
            return str(out), f"Saved to `{out}`"

        l_go.click(speak, [l_voice, l_text, l_lang, l_speed], [l_out, l_status])

        @friendly
        def rename(voice_id, new_name):
            v = library.rename(voice_id, new_name)
            return f"Renamed to **{v.name}**. The id stays `{v.id}`, so SillyTavern keeps working.", library_update(v.id)

        l_rename.click(rename, [l_voice, l_newname], [l_status, l_voice])

        @friendly
        def save_text(voice_id, text):
            v = library.update_text(voice_id, ref_text=text)
            return f"Transcript saved for **{v.name}**."

        l_savetext.click(save_text, [l_voice, l_reftext], [l_status])

        @friendly
        def transcribe_voice(voice_id, progress=gr.Progress()):
            v = library.get(voice_id)
            if v is None:
                raise ValueError("Pick a voice first.")
            text, _ = _transcribe(str(v.reference_path), progress)
            return text, "Transcribed. Check it against the reference clip, fix any mistakes, then click **Save transcript**."

        l_transcribe.click(transcribe_voice, [l_voice], [l_reftext, l_status])

        @friendly
        def delete(voice_id, confirmed):
            if not confirmed:
                raise ValueError("Tick the box to confirm deleting.")
            v = library.get(voice_id)
            if v is None:
                raise ValueError("Pick a voice first.")
            library.delete(v.id)
            return f"Deleted **{v.name}**." + export_note(), library_update(), gr.update(value=False)

        l_delete.click(delete, [l_voice, l_confirm], [l_status, l_voice, l_confirm])

        @friendly
        def sync_now():
            r = koboldcpp.sync(library, engine.settings)
            msg = f"{r['total']} voices in `{r['folder']}`."
            if r["added"]:
                msg += f" Copied: {', '.join(r['added'])}."
            if r["removed"]:
                msg += f" Removed: {', '.join(r['removed'])}."
            if r["added"] or r["removed"]:
                msg += " Restart KoboldCpp to pick up the changes."
            return msg, kobold_voices()

        k_sync.click(sync_now, None, [k_status, k_voices])
        t_kcpp.select(kobold_voices, None, k_voices)

        s_refresh.click(engine.status, None, s_info)

        def unload():
            engine.unload()
            transcribe.unload()
            return engine.status()

        s_unload.click(unload, None, s_info)

        @friendly
        def set_dtype(dtype):
            if dtype not in DTYPES:
                raise ValueError("Pick a precision first.")
            engine.unload()
            engine.settings["dtype"] = dtype
            saved = load_settings()  # keep any other edits made to the file meanwhile
            saved["dtype"] = dtype
            save_settings(saved)
            return engine.status(), (
                f"Precision set to **{dtype}**. The model loads with it the next time you generate. "
                "If the voice server is running, restart it to use it there too."
            )

        s_dtype_apply.click(set_dtype, [s_dtype], [s_info, s_dtype_note])
        demo.load(engine.status, None, s_info)

        def disk_md():
            design, clone = engine.settings["design_model"], engine.settings["clone_model"]
            lines = [f"- Voice model ({clone.split('/')[-1]}): {download_models.size_gb(clone)} GB, needed for speaking and the voice server"]
            if download_models.is_downloaded(design):
                lines.append(
                    f"- Voice design model: {download_models.size_gb(design)} GB. Once your voices are designed you can remove it: "
                    "saved voices and the voice server don't need it, and it downloads again by itself the next time you design a voice."
                )
            else:
                lines.append("- Voice design model: not downloaded. It downloads by itself (about 4 GB) the first time you design a voice.")
            return "\n".join(lines)

        @friendly
        def remove_design(confirmed):
            if not confirmed:
                raise ValueError("Tick the box to confirm removing the voice design model.")
            engine.unload("design")
            freed = download_models.remove(engine.settings["design_model"])
            return f"Removed the voice design model and freed {freed} GB.\n\n" + disk_md(), gr.update(value=False)

        s_rm_design.click(remove_design, [s_rm_confirm], [s_disk, s_rm_confirm])
        s_refresh.click(disk_md, None, s_disk)
        demo.load(disk_md, None, s_disk)
        demo.load(show_voice, [l_voice], [l_info, l_refclip, l_newname, l_reftext])

    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen3-TTS Voice Studio")
    ap.add_argument("--port", type=int, help="port to serve on (default from settings.json, 7860)")
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    args = ap.parse_args()

    setup_environment()
    settings = load_settings()
    engine = TTSEngine(settings, idle_unload_minutes=settings.get("studio_idle_unload_minutes", 10))
    library = VoiceLibrary()
    if settings.get("koboldcpp_export", True):
        try:
            koboldcpp.sync(library, settings)
        except OSError as exc:
            print(f"[studio] Couldn't update the KoboldCpp voices folder: {exc}")
    demo = build(engine, library)
    port = args.port or int(settings.get("studio_port", 7860))
    print(f"\nVoice Studio: http://127.0.0.1:{port}\n(Leave this window open while you use it. Close it to stop.)\n")
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=not args.no_browser,
        allowed_paths=[str(VOICES_DIR), str(OUTPUTS_DIR), str(youtube.WORK_DIR)],
        show_api=False,
    )


if __name__ == "__main__":
    main()
