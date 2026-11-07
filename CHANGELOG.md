# Changelog

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
