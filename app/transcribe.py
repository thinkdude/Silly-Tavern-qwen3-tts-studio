"""Speech-to-text with Whisper: works out what is said in a reference clip.

Clones are much better with a transcript, and typing one out is a chore, so the
Studio fills it in (the Transcribe button, and YouTube clips without captions)
and you correct it.

It uses openai/whisper-large-v3-turbo through transformers, which Qwen3-TTS
already installs, downloaded into models/ like the other models (about 1.6 GB;
install.bat fetches it). It runs on the GPU in float16 (about 2 GB of VRAM) and
moves back to normal memory between uses. Without enough GPU memory it runs on
the CPU, which is slower but fine for a short clip.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from . import audio, reuse
from .config import local_model_dir, resolve_model

STT_MODEL = "openai/whisper-large-v3-turbo"
SR = 16000  # what Whisper listens at
PIECE_S = 28  # Whisper hears 30 seconds at a time; longer clips are split at a quiet moment

Word = tuple[float, float, str]  # start, end (seconds), text

_lock = threading.Lock()
_loaded: dict = {}  # model id -> (model, processor)


def model_id(settings: dict) -> str:
    return (settings.get("stt_model") or STT_MODEL).strip()


def is_downloaded(settings: dict) -> bool:
    return reuse.model_complete(local_model_dir(model_id(settings)))


def download(settings: dict, log=print) -> None:
    """Download the speech-to-text model into models/ if it isn't there yet."""
    repo = model_id(settings)
    if Path(repo).is_dir() or is_downloaded(settings):
        return  # a folder of your own, or downloaded already
    target = local_model_dir(repo)
    if reuse.reuse_model(repo, target, log) and is_downloaded(settings):
        return  # it was already on this PC
    from huggingface_hub import snapshot_download

    log(f"Downloading the speech-to-text model ({repo}, about 1.6 GB) to {target} ...")
    snapshot_download(repo_id=repo, local_dir=str(target), allow_patterns=["*.json", "*.safetensors", "*.txt"])


def _model(settings: dict):
    repo = model_id(settings)
    with _lock:
        if repo not in _loaded:
            try:
                download(settings)
                import torch
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

                path = resolve_model(repo)
                processor = AutoProcessor.from_pretrained(path)
                # float16 halves the memory; it's turned into float32 if it has to run on the CPU.
                model = AutoModelForSpeechSeq2Seq.from_pretrained(path, dtype=torch.float16, low_cpu_mem_usage=True)
            except Exception as exc:  # noqa: BLE001 - shown in the Studio as-is
                raise ValueError(
                    f"Couldn't load the speech-to-text model ({exc}). Check your internet connection and try again, "
                    "or type the transcript yourself."
                ) from None
            model.eval()
            _loaded[repo] = (model, processor)
        return _loaded[repo]


def unload() -> None:
    with _lock:
        _loaded.clear()


def _to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    return audio.resample(audio.to_mono_float(wav), sr, SR).astype(np.float32)


def _pieces(wav: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Split 16 kHz audio into pieces Whisper can hear in one go, cutting at the quietest moment."""
    out, start, limit = [], 0, PIECE_S * SR
    frame = SR // 10
    while len(wav) - start > limit:
        lo, hi = start + limit - 6 * SR, start + limit
        energy = [float(np.mean(np.square(wav[i : i + frame]))) for i in range(lo, hi - frame, frame)]
        cut = lo + int(np.argmin(energy)) * frame + frame // 2
        out.append((start / SR, wav[start:cut]))
        start = cut
    out.append((start / SR, wav[start:]))
    return out


def _run(settings: dict, device: str, job):
    """Run job(model, processor, device) on the GPU, or on the CPU when there's no GPU or it runs out of memory."""
    import torch

    model, processor = _model(settings)
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        device = "cpu"
    with _lock:
        try:
            try:
                model.to(device, dtype=torch.float16 if device != "cpu" else torch.float32)
                return job(model, processor, device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("[transcribe] Not enough GPU memory for speech-to-text, using the CPU (slower).")
                model.to("cpu", dtype=torch.float32)
                return job(model, processor, "cpu")
        finally:
            model.to("cpu", dtype=torch.float16)  # free the GPU for the voice model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _detect(model, processor, wav16: np.ndarray, device) -> str | None:
    """The language code Whisper hears in the first 30 seconds, like 'en'."""
    try:
        feats = processor.feature_extractor(wav16[: 30 * SR], sampling_rate=SR, return_tensors="pt").input_features
        ids = model.detect_language(input_features=feats.to(device, model.dtype))
        token = processor.tokenizer.convert_ids_to_tokens(int(ids[0]))
        return token.strip("<|>") or None
    except Exception as exc:  # noqa: BLE001 - Whisper detects it again while transcribing
        print(f"[transcribe] Couldn't detect the language: {exc}")
        return None


def _pipe(model, processor, device):
    from transformers import pipeline
    from transformers.utils import logging

    logging.set_verbosity_error()  # its deprecation notices only worry people

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
    )


def words(wav: np.ndarray, sr: int, settings: dict, language: str | None = None) -> tuple[list[Word], str | None]:
    """Transcribe with a timing for every word. Returns (words, language code).

    `language` is a code like 'en'; leave it out to let Whisper work it out.
    """
    wav16 = _to_16k(wav, sr)

    def job(model, processor, device):
        lang = language or _detect(model, processor, wav16, device)
        pipe = _pipe(model, processor, device)
        kwargs = {"task": "transcribe", **({"language": lang} if lang else {})}
        out: list[Word] = []
        for offset, piece in _pieces(wav16):
            end = offset + len(piece) / SR
            try:
                res = pipe({"raw": piece, "sampling_rate": SR}, return_timestamps="word", generate_kwargs=kwargs)
            except (ValueError, RuntimeError, IndexError) as exc:  # no word timings for this model: sentences will do
                print(f"[transcribe] No word timings ({exc}); using sentence timings.")
                res = pipe({"raw": piece, "sampling_rate": SR}, return_timestamps=True, generate_kwargs=kwargs)
            chunks = res.get("chunks") or [{"text": res.get("text", ""), "timestamp": (0.0, None)}]
            for ch in chunks:
                text = ch.get("text") or ""
                if not text.strip():
                    continue
                s, e = ch.get("timestamp") or (None, None)
                s = offset + (s or 0.0)
                if s >= end:
                    continue  # "heard" in the silence Whisper pads short clips with
                e = min(end, offset + e) if e is not None else end
                out.append((s, max(s, e), text))
        return out, lang

    return _run(settings, settings.get("device", "cpu"), job)


def detect(wav: np.ndarray, sr: int, settings: dict) -> str | None:
    """The language code Whisper hears in the clip (its first 30 seconds), like 'en'."""
    wav16 = _to_16k(wav, sr)
    return _run(settings, settings.get("device", "cpu"), lambda model, processor, device: _detect(model, processor, wav16, device))


def text(wav: np.ndarray, sr: int, settings: dict, language: str | None = None) -> tuple[str, str | None]:
    """Transcribe a whole clip. Returns (text, language code)."""
    wav16 = _to_16k(wav, sr)

    def job(model, processor, device):
        lang = language or _detect(model, processor, wav16, device)
        pipe = _pipe(model, processor, device)
        kwargs = {"task": "transcribe", **({"language": lang} if lang else {})}
        parts = [pipe({"raw": piece, "sampling_rate": SR}, generate_kwargs=kwargs)["text"] for _, piece in _pieces(wav16)]
        return " ".join(p.strip() for p in parts if p.strip()), lang

    return _run(settings, settings.get("device", "cpu"), job)
