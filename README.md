# Qwen3-TTS Studio

Run [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) on your own Windows PC with two simple apps:

- **Voice Studio** for making voices. Describe a voice in plain English ("gravelly old man, slow and weary") or clone one from a short audio clip, then save it to your voice library.
- **Playback in SillyTavern**, either through **KoboldCpp** (small GGUF model, low GPU memory) or through the included **Voice Server** (full model, most accurate voices). Both use the OpenAI text-to-speech API that SillyTavern understands.

Everything (Python, packages, models, your voices) stays inside one folder, so it can live on your G: drive without touching C:.

## What you need

- Windows 10 or 11
- An NVIDIA graphics card. 8 GB of VRAM or more is comfortable; 6 GB works. It runs without one, but very slowly.
- About 20 GB of free space on the drive you install to
- An internet connection for the install (afterwards everything runs offline)

## Install

1. On this repo's GitHub page click **Code → Download ZIP** (or `git clone` it).
2. Extract it so the files sit in **`G:\Qwen3TTS`**. You should end up with `G:\Qwen3TTS\install.bat`, not `G:\Qwen3TTS\qwen3-tts-studio-main\install.bat`.
3. Double-click **`install.bat`**. If Windows shows "Windows protected your PC", click **More info → Run anyway**.

The installer checks your graphics card, downloads a private copy of Python, installs PyTorch with the right CUDA version, picks settings for your GPU and downloads the models (around 12 GB in total). It takes a while the first time. If anything fails, run `install.bat` again; it picks up where it left off.

### Already have Python with PyTorch?

If another AI tool already installed a GPU-enabled PyTorch (2.6 or newer, Python 3.10 to 3.13), the installer finds it and asks whether to reuse it. Saying yes skips the roughly 3 GB PyTorch download.

Your existing Python is never changed. Qwen3-TTS still gets its own `.venv`, which borrows only PyTorch from the other Python and installs everything else privately, so package versions can't clash. Before finishing, the installer checks that the borrowed PyTorch really works with your GPU, and falls back to a private copy if it doesn't.

The installer looks at the Pythons listed by `py -0p` and any `python` on your PATH. To point it at a specific one, such as a conda env or another app's venv, run `install.bat -UsePython "C:\path\to\python.exe"`. If you later uninstall or upgrade that Python, run `install.bat -Fresh` to switch to a private PyTorch.

### Installer options

Run these from a command prompt in the install folder:

| Option | What it does |
| --- | --- |
| `-Reset` | Picks settings for your GPU again |
| `-AllModels` | Also downloads the other clone model size |
| `-SkipModels` | Skips the model download |
| `-Fresh` | Always installs a private PyTorch, even if one could be reused |
| `-UsePython <path>` | Reuses PyTorch from that `python.exe` |
| `-Yes` | Asks no questions (it reuses an existing PyTorch if one is found) |
| `-KeepCache` | Keeps the download caches instead of deleting them at the end |

## Make voices: `start_studio.bat`

The Studio opens in your browser at http://127.0.0.1:7860. The first action in each tab takes a little longer while the model loads.

**Design a voice.** Write a description: age, gender, pitch, pace, mood, accent. The model reads the *sample line* in that voice. Generate 2 to 4 takes, keep your favourite, name it and save. The saved take becomes the voice's reference clip, so the voice sounds the same every time you use it.

**Clone a voice.** Upload or record 5 to 15 seconds of one person speaking clearly with no music. Type exactly what they say (clones with a transcript are much better), click **Preview clone**, and save it when you like it. Only clone voices you have permission to use.

**My voices.** Try any saved voice with your own text, change its speed, and download the result (also saved in `outputs\`). You can rename voices, fix a transcript, or delete them here.

## Play voices with KoboldCpp

[KoboldCpp](https://github.com/LostRuins/koboldcpp) can run Qwen3-TTS from a small GGUF file (the 0.6B model at 8-bit) and clone voices from a folder of clips. The Studio copies every voice you save into that folder (`koboldcpp_voices\` in the install folder) automatically, and its **KoboldCpp** tab shows everything below with copy buttons.

**One-time KoboldCpp setup.** In the KoboldCpp launcher's **Audio** tab:

1. **TTS Model:** `https://huggingface.co/koboldcpp/tts/resolve/main/qwen3-tts-0.6b-q8_0.gguf` (KoboldCpp downloads it, or point it at a downloaded copy)
2. **WavTokenizer Model:** `https://huggingface.co/koboldcpp/tts/resolve/main/qwen3-tts-tokenizer-f16.gguf`
3. **TTS Voices Dir:** the `koboldcpp_voices` folder inside your install folder
4. Tick **TTS Use GPU**, click **Save Config** so you only do this once, then **Launch**.

**SillyTavern.** In **Extensions → TTS** choose **OpenAI Compatible**, tick **Enabled**, and set:

- **Provider Endpoint:** `http://127.0.0.1:5001/v1/audio/speech` (KoboldCpp's default port)
- **Available Voices:** your voice ids with `.wav` on the end, e.g. `grumpy-wizard.wav,dad.wav`. That's how KoboldCpp names cloned voices.

Then click **Reload** and pick a voice for each character.

**Good to know:**
- KoboldCpp reads the voices folder when it starts, so **restart KoboldCpp after saving new voices**.
- KoboldCpp clones from the sound of the clip only and can't use its transcript, so its voices are a little less exact than the Studio's previews or the Voice Server. If a voice matters a lot, compare it with the Voice Server.
- Already using KoboldCpp as SillyTavern's text model? The TTS runs in the same KoboldCpp, so nothing else needs to run.
- To stop exporting voices, set `"koboldcpp_export": false` in `settings.json`. To use a different folder, change `koboldcpp_voices_dir`.

## Or use the Voice Server: `start_server.bat`

The server runs at **http://127.0.0.1:8890** and opens its control panel at http://127.0.0.1:8890/ui. Keep the window open while you use it. It loads the voice model at startup so the first line plays without a long wait.

### SillyTavern

1. Start the voice server.
2. In SillyTavern open **Extensions** (the stacked blocks icon) → **TTS**.
3. Set **Select TTS Provider** to **OpenAI Compatible** and tick **Enabled**.
4. Fill in the fields from the control panel's **Connect SillyTavern** tab:
   - **Provider Endpoint:** `http://127.0.0.1:8890/v1/audio/speech`
   - **Model:** `qwen3-tts`
   - **Available Voices:** your voice ids separated by commas, e.g. `grumpy-wizard,dad`
5. Click **Reload**, then pick a voice for each character in the voice map below.

After making new voices in the Studio, click **Refresh** in the control panel and paste the updated voice list into SillyTavern. Renaming a voice keeps its id, so SillyTavern keeps working.

SillyTavern itself uses port 8000, which is why the voice server uses 8890.

### Other apps

Any app that can use OpenAI's text-to-speech with a custom base URL works. Point it at `http://127.0.0.1:8890/v1` and use a voice id as the voice name.

```bat
curl http://127.0.0.1:8890/v1/audio/speech -H "Content-Type: application/json" ^
  -d "{\"input\": \"Hello there!\", \"voice\": \"grumpy-wizard\"}" --output hello.mp3
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8890/v1", api_key="not-needed")
client.audio.speech.create(model="qwen3-tts", voice="grumpy-wizard", input="Hello there!").write_to_file("hello.mp3")
```

| Endpoint | What it does |
| --- | --- |
| `POST /v1/audio/speech` | `input`, `voice` (id or name), optional `response_format` (`mp3`, `wav`, `flac`, `opus`, `pcm`; `aac` returns mp3), `speed` (0.25 to 4), and `language` to override the voice's language |
| `GET /v1/audio/voices` | Lists saved voices |
| `GET /v1/models` | Lists the model name (`qwen3-tts`) |
| `GET /health` | Server, GPU and voice status |
| `/ui` | Control panel |
| `/docs` | Interactive API docs |

## Saving disk space

- **Download caches** are deleted automatically at the end of the install, which frees several GB. (`install.bat -KeepCache` keeps them.)
- **Voice design model (about 4 GB):** once your voices are designed, open the Studio's **System** tab and click **Remove voice design model**. Saved voices, cloning and the voice server don't need it. It downloads again by itself the next time you design a voice.
- **Smaller voice model:** the 0.6B clone model saves about 2 GB, with slightly less accurate voices. See `clone_model` in Settings below.
- **Reuse an existing PyTorch** if you have one (see above). That skips the roughly 3 GB PyTorch download.

## Settings

`settings.json` is created by the installer. Close the Studio and server before editing it.

| Setting | Default | Meaning |
| --- | --- | --- |
| `clone_model` | 1.7B Base (0.6B under 6 GB VRAM) | Model that speaks in your saved voices |
| `dtype` | `bfloat16` (RTX 30-series and newer), else `float16` | Precision. Try `float32` if output is silent or garbled |
| `keep_both_loaded` | on with 16 GB+ VRAM | Keep the design and clone models in memory together instead of swapping |
| `studio_idle_unload_minutes` | 10 | Studio frees GPU memory after this many idle minutes |
| `server_idle_unload_minutes` | 0 | Same for the server; 0 keeps the model loaded |
| `server_port`, `studio_port` | 8890, 7860 | Ports |
| `server_host` | `127.0.0.1` | Use `0.0.0.0` to allow other PCs on your network |
| `api_key` | empty | When set, apps must send it as a Bearer token (put it in SillyTavern's API key field) |
| `koboldcpp_export` | on | Copy saved voices into the KoboldCpp voices folder |
| `koboldcpp_voices_dir` | `koboldcpp_voices` | That folder (relative to the install folder, or a full path) |
| `koboldcpp_url` | `http://127.0.0.1:5001` | Where KoboldCpp runs, for the settings shown in the Studio |

To use the 0.6B clone model on purpose, set `"clone_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"` and run `install.bat` once to download it.

## Troubleshooting

- **`'sox' is not recognized`, `SoX could not be found` or a `flash-attn` warning**: harmless, Qwen3-TTS works without either. The Studio and server hide these now; if you still see them, update to the latest version.
- **"Your GPU ran out of memory"**: the Studio and server each load a model. On cards under about 12 GB, close the server while designing voices, or click **Free GPU memory** in either app.
- **Installer says the GPU was found but PyTorch can't use it**: update your NVIDIA driver from nvidia.com, then run `install.bat` again.
- **Takes come out silent, or "The model produced silence"**: your graphics card doesn't handle the current precision well (most likely on GTX 10/16 or RTX 20 series cards). On the Studio's **System** tab set **Precision** to `float32` and click **Apply**. It needs about twice the GPU memory; if that runs out, try `bfloat16`, or switch to the 0.6B clone model.
- **SillyTavern says the TTS request failed**: make sure the server window is open, the endpoint ends in `/v1/audio/speech`, and the voice id is in the control panel's list. The **Request log** tab shows what arrived.
- **Clone sounds off**: use a cleaner 5 to 15 second clip and make sure the transcript matches word for word.
- **A port is already in use**: change `studio_port` or `server_port` in `settings.json`.

## Updating

Download the new ZIP and extract it over the existing folder (or `git pull`), then run `install.bat`. Your voices, settings and downloaded models are kept.

## Folder layout

```
G:\Qwen3TTS\
  install.bat, start_studio.bat, start_server.bat
  app\          the Studio, server and engine code
  voices\       your saved voices (one folder each: voice.json + reference.wav)
  koboldcpp_voices\  copies of your voices for KoboldCpp (one .wav per voice)
  outputs\      audio made in the Studio
  models\       downloaded model weights
  settings.json your settings
  .venv\ .python\ .tools\ .cache\   private Python, packages and caches
                                    (.python is skipped when an existing PyTorch is reused)
```

To uninstall, delete the folder. Back up `voices\` first if you want to keep your voices.

## Development

Tests run without a GPU using a stand-in engine:

```
.tools\uv\uv.exe pip install --python .venv\Scripts\python.exe pytest httpx
.venv\Scripts\python -m pytest
```

Qwen3-TTS is by the Alibaba Qwen team and is licensed under Apache-2.0.
