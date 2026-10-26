"""Audio helpers: cleanup, speed change and encoding to the formats OpenAI clients ask for."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf

MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "pcm": "audio/pcm",
}


def to_mono_float(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav)
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    wav = wav.astype(np.float32, copy=False)
    if wav.ndim == 2:
        # soundfile gives (frames, channels); gradio can give either orientation.
        wav = wav.mean(axis=1) if wav.shape[0] > wav.shape[1] else wav.mean(axis=0)
    return np.clip(wav, -1.0, 1.0)


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    try:
        wav, sr = sf.read(str(path), always_2d=False)
    except Exception:
        import librosa  # handles formats libsndfile can't, via audioread

        wav, sr = librosa.load(str(path), sr=None, mono=True)
    return to_mono_float(wav), int(sr)


def trim_silence(wav: np.ndarray, sr: int, top_db: float = 40.0, pad_s: float = 0.15) -> np.ndarray:
    import librosa

    if wav.size == 0:
        return wav
    _, (start, end) = librosa.effects.trim(wav, top_db=top_db)
    pad = int(pad_s * sr)
    return wav[max(0, start - pad) : min(len(wav), end + pad)]


def normalize_peak(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(wav))) if wav.size else 0.0
    return wav if m < 1e-6 else wav * (peak / m)


def change_speed(wav: np.ndarray, speed: float) -> np.ndarray:
    if not speed or abs(speed - 1.0) < 0.01 or wav.size == 0:
        return wav
    import librosa

    return librosa.effects.time_stretch(wav, rate=float(speed))


def concat(chunks: list[np.ndarray], sr: int, gap_s: float = 0.2) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(gap_s * sr), dtype=np.float32)
    out = []
    for i, c in enumerate(chunks):
        if i:
            out.append(gap)
        out.append(to_mono_float(c))
    return np.concatenate(out)


def resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav
    import librosa

    return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)


def encode(wav: np.ndarray, sr: int, fmt: str = "mp3") -> tuple[bytes, str]:
    """Encode mono float audio. Returns (bytes, media_type).

    ``aac`` isn't available without ffmpeg, so it falls back to mp3, which every
    player that accepts aac also plays.
    """
    fmt = (fmt or "mp3").lower()
    if fmt == "aac":
        fmt = "mp3"
    wav = to_mono_float(wav)
    pcm16 = (wav * 32767.0).astype("<i2")

    if fmt == "mp3":
        import lameenc

        enc = lameenc.Encoder()
        enc.set_bit_rate(128)
        enc.set_in_sample_rate(sr)
        enc.set_channels(1)
        enc.set_quality(2)
        data = enc.encode(pcm16.tobytes()) + enc.flush()
        return bytes(data), MEDIA_TYPES["mp3"]
    if fmt == "pcm":
        # OpenAI's pcm format is raw 24 kHz 16-bit little-endian mono.
        return (resample(wav, sr, 24000) * 32767.0).astype("<i2").tobytes(), MEDIA_TYPES["pcm"]

    buf = io.BytesIO()
    if fmt == "wav":
        sf.write(buf, pcm16, sr, format="WAV", subtype="PCM_16")
    elif fmt == "flac":
        sf.write(buf, wav, sr, format="FLAC")
    elif fmt == "opus":
        sf.write(buf, resample(wav, sr, 48000), 48000, format="OGG", subtype="OPUS")
    else:
        raise ValueError(f"Unsupported response_format '{fmt}'. Use mp3, wav, flac, opus, aac or pcm.")
    return buf.getvalue(), MEDIA_TYPES[fmt]
