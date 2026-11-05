# Qwen3-TTS Studio

Run [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) on your own Windows PC with a few simple apps:

- **Voice Studio** for making voices. Describe a voice in plain English ("gravelly old man, slow and weary") or clone one from a short audio clip, a YouTube link or a video on your PC, then save it to your voice library.
- **Speak** for typing anything and hearing it in one of your saved voices, saved as a WAV or MP3. It can use the full model or the small GGUF model (through KoboldCpp, which the installer sets up).
- **Playback in SillyTavern**, either through **KoboldCpp** (small GGUF model, low GPU memory) or through the included **Voice Server** (full model, most accurate voices). Both use the OpenAI text-to-speech API that SillyTavern understands.

Everything (Python, packages, models, your voices) stays inside one folder, so it can live on your G: drive without touching C:.

## What you need

- Windows 10 or 11
- An NVIDIA graphics card. 8 GB of VRAM or more is comfortable; 6 GB works. It runs without one, but very slowly.
- About 24 GB of free space on the drive you install to
- An internet connection for the install (afterwards everything runs offline)

## Install

1. On this repo's GitHub page click **Code → Download ZIP** (or `git clone` it).
2. Extract it so the files sit in **`G:\Qwen3TTS`**. You should end up with `G:\Qwen3TTS\install.bat`, not `G:\Qwen3TTS\qwen3-tts-studio-main\install.bat`.
3. Double-click **`install.bat`**. If Windows shows "Windows protected your PC", click **More info → Run anyway**.

The installer checks your graphics card, downloads a private copy of Python, installs PyTorch with the right CUDA version, picks settings for your GPU and downloads the models (around 14 GB in total, including the speech-to-text and music remover models for cloning), plus KoboldCpp and the GGUF voice model for Speak (about 2.3 GB more). It takes a while the first time. If anything fails, run `install.bat` again; it picks up where it left off.

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
| `-SkipGguf` | Skips KoboldCpp and the GGUF model (Speak downloads them the first time you pick GGUF) |
| `-Fresh` | Always installs a private PyTorch, even if one could be reused |
| `-UsePython <path>` | Reuses PyTorch from that `python.exe` |
| `-Yes` | Asks no questions (it reuses an existing PyTorch if one is found) |
| `-KeepCache` | Keeps the download caches instead of deleting them at the end |

## Make voices: `start_studio.bat`

The Studio opens in your browser at http://127.0.0.1:7860. The first action in each tab takes a little longer while the model loads.

**Design a voice.** Write a description: age, gender, pitch, pace, mood, accent. The model reads the *sample line* in that voice. Generate 2 to 4 takes, keep your favourite, name it and save. The saved take becomes the voice's reference clip, so the voice sounds the same every time you use it.

**Clone a voice.** Upload or record 5 to 15 seconds of one person speaking clearly with no music. Click **Transcribe the clip** to have speech-to-text ([Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)) write down what's said, then read it while you listen and fix any mistakes (or type it yourself). Clones with an exact transcript are much better. Click **Preview clone**, and save it when you like it. Only clone voices you have permission to use.

**Clone from a YouTube link.** Open **Get the clip from a YouTube video** at the top of the Clone tab and paste a video link. Find a stretch where your speaker talks on their own and put its time in **Start at** (like `1:23`), or right-click the video on YouTube at that moment, choose **Copy video URL at current time** and paste that link instead. Pick a clip length (15 seconds is a good start) and click **Get clip**. The Studio then:

- downloads the video's audio (once: more clips from the same video are quick),
- cuts out that stretch and removes music and background noise with [Demucs](https://github.com/facebookresearch/demucs) (untick the box to keep the sound as it is),
- writes down what's said: from the video's captions when it has some, otherwise with speech-to-text (pick which under **Transcript from**). The cut moves to the nearest whole words so the clip and the transcript match, and the language is set to the video's.

The clip lands in **Reference clip** and the transcript in the box under it. Listen to the clip, fix anything the transcript got wrong, and carry on with **Preview clone** as usual. Not the right bit? Change the start time and click **Get clip** again, or trim it with the scissors on the player and click **Transcribe the clip** so the transcript matches. Other video sites that [yt-dlp](https://github.com/yt-dlp/yt-dlp) supports work too.

**Clone from a video or audio file on your PC.** Open **Get the clip from a video or audio file on this PC** in the Clone tab and click **Browse** to pick a film, episode, interview, podcast or any other video or audio file (MP4, MKV, AVI, MOV, MP3, WAV and the like). You can also paste its path: in File Explorer, right-click the file and choose **Copy as path** (on Windows 10, hold Shift while you right-click). Find a stretch where the voice talks clearly in your usual video player, put its time in **Start at** and click **Get clip**. It works like a YouTube clip: the stretch is cut out, music and background noise are removed if the box is ticked, and speech-to-text writes down what's said. Films often carry several audio tracks (the original and dubs); when a file has more than one, pick the language you want under **Audio track**. The file is read where it is and never copied or changed.

**My voices.** Try any saved voice with your own text, change its speed, and download the result (also saved in `outputs\`). You can rename voices, fix a transcript (or **Transcribe** one that has none), or delete them here.

## Type and listen: `start_speak.bat`

Speak opens in your browser at http://127.0.0.1:7861. Pick one of your saved voices, type what you want to hear and press **Enter** (or click **Speak**). **Shift+Enter** starts a new line, and long text is fine.

- The line plays straight away and is saved in the `outputs` folder, named after the voice and its first few words. Choose **wav** or **mp3** under **Save as**; the download button on each player saves another copy wherever you like.
- Your last ten lines stay on the page under **Earlier** so you can play them again.
- Not quite right? Press **Speak** again for a slightly different take, or change the **Speed**.
- It remembers the voice, speed and file type you picked last time.
- If the Voice Server is running, Speak lets it do the talking, so the voice model isn't loaded onto your GPU twice. Otherwise Speak loads its own copy when it starts (the first line can take a minute) and frees it after 10 idle minutes.

**Full model or GGUF.** Under **Voice model**, pick which one speaks:

- **Full model (best quality)**: Qwen3-TTS itself, the same model the Studio uses. It sounds closest to your voices because it also uses each voice's transcript.
- **GGUF with KoboldCpp**: the 0.6B model as a GGUF file, run by KoboldCpp. It needs less GPU memory, but it only copies the sound of each voice's clip, so voices are a little less exact.

You don't need to set up KoboldCpp for this. The installer downloads it into `.tools\koboldcpp` and the GGUF files into `models\gguf` (if they're missing, Speak downloads them the first time you pick GGUF). Speak starts KoboldCpp in the background with no text model, restarts it when you use a voice it hasn't loaded yet, and stops it when you switch back to the full model or close Speak. If your own KoboldCpp for SillyTavern is already running with the voice loaded, Speak uses that one instead. Switching models frees the other one's GPU memory, and Speak remembers your choice.
- New voice made in the Studio? Click **Refresh voices**.

## Play voices with KoboldCpp

[KoboldCpp](https://github.com/LostRuins/koboldcpp) can run Qwen3-TTS from a small GGUF file (the 0.6B model at 8-bit) and clone voices from a folder of clips. The Studio copies every voice you save into that folder (`koboldcpp_voices\` in the install folder) automatically, and its **KoboldCpp** tab shows everything below with copy buttons.

**One-time KoboldCpp setup.** In the KoboldCpp launcher's **Audio** tab:

1. **TTS Model:** `https://huggingface.co/koboldcpp/tts/resolve/main/qwen3-tts-0.6b-q8_0.gguf` (KoboldCpp downloads it, or point it at a downloaded copy)
2. **WavTokenizer Model:** `https://huggingface.co/koboldcpp/tts/resolve/main/qwen3-tts-tokenizer-f16.gguf`

The installer has already downloaded KoboldCpp (`.tools\koboldcpp\koboldcpp.exe` in the install folder) and both files (`models\gguf\`), so you can use those instead of downloading them again.
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
- **Speech-to-text model (about 1.6 GB):** if you always type transcripts yourself, delete `models\whisper-large-v3-turbo`. It downloads again by itself the next time you click **Transcribe** or a YouTube clip needs it.
- **GGUF model and KoboldCpp (about 2.3 GB):** if you never use GGUF in Speak, delete `models\gguf` and `.tools\koboldcpp`, and install with `install.bat -SkipGguf` from then on.

## Settings

`settings.json` is created by the installer. Close the Studio and server before editing it.

| Setting | Default | Meaning |
| --- | --- | --- |
| `clone_model` | 1.7B Base (0.6B under 6 GB VRAM) | Model that speaks in your saved voices |
| `dtype` | `bfloat16` (RTX 30-series and newer), else `float16` | Precision. Try `float32` if output is silent or garbled |
| `keep_both_loaded` | on with 16 GB+ VRAM | Keep the design and clone models in memory together instead of swapping |
| `studio_idle_unload_minutes` | 10 | Studio frees GPU memory after this many idle minutes |
| `server_idle_unload_minutes` | 0 | Same for the server; 0 keeps the model loaded |
| `speak_idle_unload_minutes` | 10 | Same for Speak |
| `server_port`, `studio_port`, `speak_port` | 8890, 7860, 7861 | Ports |
| `server_host` | `127.0.0.1` | Use `0.0.0.0` to allow other PCs on your network |
| `api_key` | empty | When set, apps must send it as a Bearer token (put it in SillyTavern's API key field) |
| `koboldcpp_export` | on | Copy saved voices into the KoboldCpp voices folder |
| `koboldcpp_voices_dir` | `koboldcpp_voices` | That folder (relative to the install folder, or a full path) |
| `koboldcpp_url` | `http://127.0.0.1:5001` | Where your KoboldCpp runs, for the settings shown in the Studio. Speak uses it too when it has the voice loaded |
| `speak_engine` | `full` | Speak's voice model, `full` or `gguf` (Speak's **Voice model** switch sets it) |
| `gguf_model`, `gguf_tokenizer` | `qwen3-tts-0.6b-q8_0.gguf`, `qwen3-tts-tokenizer-f16.gguf` | GGUF files from [koboldcpp/tts](https://huggingface.co/koboldcpp/tts), kept in `models\gguf`, or full paths to your own copies |
| `koboldcpp_exe` | empty | Path to your own `koboldcpp.exe`; empty uses the one the installer downloaded |
| `gguf_port` | 5002 | Port of the KoboldCpp that Speak starts (yours stays on 5001) |
| `stt_model` | `openai/whisper-large-v3-turbo` | Speech-to-text model for transcripts (a Hugging Face Whisper model, kept in `models\`, or a folder) |

To use the 0.6B clone model on purpose, set `"clone_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"` and run `install.bat` once to download it.

## Troubleshooting

- **`'sox' is not recognized`, `SoX could not be found` or a `flash-attn` warning**: harmless, Qwen3-TTS works without either. The Studio and server hide these now; if you still see them, update to the latest version.
- **"Your GPU ran out of memory"**: the Studio, the server and Speak can each load a model (Speak uses the server's when the server is running). On cards under about 12 GB, close the server while designing voices, or click **Free GPU memory** in either app.
- **Installer says the GPU was found but PyTorch can't use it**: update your NVIDIA driver from nvidia.com, then run `install.bat` again.
- **Takes come out silent, or "The model produced silence"**: your graphics card doesn't handle the current precision well (most likely on GTX 10/16 or RTX 20 series cards). On the Studio's **System** tab set **Precision** to `float32` and click **Apply**. It needs about twice the GPU memory; if that runs out, try `bfloat16`, or switch to the 0.6B clone model.
- **SillyTavern says the TTS request failed**: make sure the server window is open, the endpoint ends in `/v1/audio/speech`, and the voice id is in the control panel's list. The **Request log** tab shows what arrived.
- **Clone sounds off**: use a cleaner 5 to 15 second clip and make sure the transcript matches word for word.
- **Speak's GGUF model doesn't start or says KoboldCpp couldn't say that**: KoboldCpp's messages are in `.cache\koboldcpp-speak.log`. If it mentions CUDA, update your NVIDIA driver. You can always switch back to the full model.
- **A port is already in use**: change `studio_port`, `server_port`, `speak_port` or `gguf_port` in `settings.json`.
- **"Couldn't get that video"**: YouTube changes often, and the downloader has to keep up. Click **Update the YouTube downloader** under the YouTube box (running `install.bat` again updates it too) and try again. If YouTube asks you to sign in to prove you're not a bot, wait a while and try again later.
- **The transcript has mistakes**: speech-to-text and automatic captions mishear names and mumbled words, so always read the transcript while listening to the clip and fix it before you save the voice.
- **The YouTube clip is mostly music or another voice**: try a different start time, a shorter length, or a video where the speaker talks alone (interviews and podcasts work well). The same goes for clips from your own videos.
- **Clicking Browse doesn't seem to do anything**: the file window may have opened behind your browser, so check the taskbar. You can always paste the file's path into the box instead.
- **A clip from a film is in the wrong language**: the film has more than one audio track. Pick the right one under **Audio track** and click **Get clip** again.

## Updating

Download the new ZIP and extract it over the existing folder (or `git pull`), then run `install.bat`. Your voices, settings and downloaded models are kept.

## Folder layout

```
G:\Qwen3TTS\
  install.bat, start_studio.bat, start_server.bat, start_speak.bat
  app\          the Studio, server, Speak and engine code
  voices\       your saved voices (one folder each: voice.json + reference.wav)
  koboldcpp_voices\  copies of your voices for KoboldCpp (one .wav per voice)
  outputs\      audio made in the Studio and Speak
  models\       downloaded model weights (models\gguf: the GGUF files)
  settings.json your settings
  .venv\ .python\ .tools\ .cache\   private Python, packages, KoboldCpp and caches
                                    (.python is skipped when an existing PyTorch is reused)
                                    (.cache\youtube: the last few YouTube downloads, and clips cut from videos)
```

To uninstall, delete the folder. Back up `voices\` first if you want to keep your voices.

## Development

Tests run without a GPU using a stand-in engine:

```
.tools\uv\uv.exe pip install --python .venv\Scripts\python.exe pytest httpx
.venv\Scripts\python -m pytest
```

Qwen3-TTS is by the Alibaba Qwen team and is licensed under Apache-2.0.
