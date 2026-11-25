# Changelog

## 1.0.4 (2026-11-25)

- **Each person in their own voice.** When several people talk in one reply, the Voice Server reads each of them in a voice of their own, and narration in the character's voice. It works out who's talking from the name next to a line, a "Name:" in front of it, or the colour it's written in. Everyone gets a voice the first time they speak and keeps it; pick voices yourself on the new **People's voices** tab. Fill in SillyTavern's settings turns off SillyTavern's own Multi-voice option, which would split the lines up first.
- **Notes aren't read out.** Reasoning lists, `[ time | place ]` status lines and blocks such as `<internal_states>`, `<details>` and `<think>` are skipped, even when SillyTavern sends a reply a paragraph at a time. Markup like `<cyan>` is never read out either.

## 1.0.3 (2026-11-25)

- **One file starts everything.** `start_studio.bat`, `start_server.bat` and `start_speak.bat` are gone. Double-click `Qwen3-TTS.bat` and it asks which program to start. To start one directly (for a shortcut of your own), run `Qwen3-TTS.bat studio`, `Qwen3-TTS.bat server` or `Qwen3-TTS.bat speak`; `Qwen3-TTS.bat server --engine gguf` works as before. Updating removes the old start files.

## 1.0.2 (2026-11-25)

- **SillyTavern set up for you.** On the Voice Server's **Connect SillyTavern** tab, pick SillyTavern's folder (it's usually found for you) and click **Fill in SillyTavern's settings**. That turns on SillyTavern's text-to-speech, points it at the Voice Server and fills in your voices, so there's nothing to copy. A copy of SillyTavern's old settings is kept, and if SillyTavern is open you're reminded to reload its page.
- **New voices show up by themselves.** The Voice Server keeps SillyTavern's voice list up to date as you save or delete voices in the Studio (a tick box turns this off).
- **Long replies start speaking sooner.** Fill in also turns on SillyTavern's "Narrate by paragraphs" options, so each paragraph is made and played on its own and the first one plays while the next is being made, instead of waiting for the whole reply.

## 1.0.1 (2026-11-07)

- **Nothing personal in the download.** The license, the guide and the examples no longer name a particular person, drive or PC. As before, everything runs from whichever folder you install to.
- **Kept that way.** The tests now fail if a Windows user folder or a PC name ever gets into the project.

## 1.0.0 (2026-11-05)

The first public release.

- **One menu for everything.** Double-click `Qwen3-TTS.bat`. The first time, it installs everything; after that it starts the Voice Studio, the Voice Server and Speak, each in its own window, shows which ones are running, and lets you know when a new version is out.
- **Install anywhere.** The installer asks which folder to use (any drive), checks there's enough free space, warns about OneDrive and Downloads folders, and adds a desktop and Start menu shortcut. Updating is the same steps with the new ZIP, and keeps your voices, settings and models.
- **Voice Studio:** design a voice from a written description, or clone one from a recording, a YouTube link or a video or audio file on your PC. Music and background noise are removed (Demucs) and speech-to-text (Whisper) writes the transcript for you to check. Try, rename, re-transcribe and delete voices in your library.
- **Voice Server:** OpenAI-compatible text-to-speech on port 8890 for SillyTavern and other apps, with a control panel. Pick the full model (best quality) or the small GGUF model, and compare the two side by side.
- **Speak:** type something and hear it in one of your voices, saved as WAV or MP3, with the full or the GGUF model.
- **KoboldCpp:** your voices are copied to a folder KoboldCpp's text-to-speech can use, and the installer downloads KoboldCpp and the GGUF model.
- **Installer:** picks the right PyTorch for your NVIDIA card, can reuse a GPU PyTorch you already have, keeps everything in one folder, and picks up where it left off if it's interrupted.
- **Nothing is downloaded twice.** The installer checks what's already there and skips it, so running it again is quick and works offline. Models other AI apps already downloaded (in the shared Hugging Face cache) and models from another Qwen3-TTS Studio folder are linked or copied in instead of downloaded (`-ModelsFrom` points it at a folder).
