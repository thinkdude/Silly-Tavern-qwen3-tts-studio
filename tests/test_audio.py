import io

import numpy as np
import pytest
import soundfile as sf

from app import audio
from tests.conftest import SR, tone


@pytest.mark.parametrize("fmt", ["mp3", "wav", "flac", "opus", "aac", "pcm"])
def test_encode_formats(fmt):
    data, media = audio.encode(tone(1.0), SR, fmt)
    assert len(data) > 100
    assert media.startswith("audio/")


def test_wav_roundtrip():
    data, _ = audio.encode(tone(1.0), SR, "wav")
    wav, sr = sf.read(io.BytesIO(data))
    assert sr == SR and abs(len(wav) - SR) < 10


def test_mp3_header():
    data, media = audio.encode(tone(1.0), SR, "mp3")
    assert media == "audio/mpeg"
    assert data[:3] == b"ID3" or data[0] == 0xFF


def test_pcm_is_24khz_int16():
    data, _ = audio.encode(tone(1.0), 48000, "pcm")
    assert len(data) == 2 * 12000  # 1s of 48k-rate samples -> 0.5s at 24 kHz, 2 bytes each


def test_unknown_format():
    with pytest.raises(ValueError):
        audio.encode(tone(0.1), SR, "ogg-vorbis")


def test_speed_changes_length():
    wav = tone(2.0)
    assert len(audio.change_speed(wav, 2.0)) < len(wav) * 0.6
    assert audio.change_speed(wav, 1.0) is wav


def test_to_mono_handles_stereo_and_int():
    stereo = np.stack([tone(0.5), tone(0.5)], axis=1)
    assert audio.to_mono_float(stereo).ndim == 1
    ints = (tone(0.5) * 32767).astype(np.int16)
    out = audio.to_mono_float(ints)
    assert out.dtype == np.float32 and np.max(np.abs(out)) <= 1.0


def test_trim_silence():
    wav = np.concatenate([np.zeros(SR), tone(1.0), np.zeros(SR)]).astype(np.float32)
    trimmed = audio.trim_silence(wav, SR)
    assert len(trimmed) < 1.5 * SR


def test_is_silent():
    assert audio.is_silent(np.zeros(SR, dtype=np.float32))
    assert audio.is_silent(np.zeros(0, dtype=np.float32))
    assert audio.is_silent(np.full(SR, np.nan, dtype=np.float32))
    broken = tone(1)
    broken[100] = np.nan
    assert audio.is_silent(broken)
    assert not audio.is_silent(tone(1))
