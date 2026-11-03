"""Clone from a video or audio file on this PC: reading the file, its audio tracks, the clip, the file picker."""

import base64
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from app import studio, transcribe, youtube

needs_ffmpeg = pytest.mark.skipif(
    importlib.util.find_spec("imageio_ffmpeg") is None, reason="needs imageio-ffmpeg (install.bat adds it)"
)

# What ffmpeg prints about a film with two audio tracks (and chapters), as on Windows.
FFMPEG_SAYS = r"""Input #0, matroska,webm, from 'G:\Videos\Film.mkv':
  Metadata:
    title           : The Film
  Duration: 01:42:13.52, start: 0.000000, bitrate: 4812 kb/s
  Chapters:
    Chapter #0:0: start 0.000000, end 300.000000
      Metadata:
        title           : Opening
  Stream #0:0: Video: hevc (Main 10), yuv420p10le(tv), 1920x1080, SAR 1:1 DAR 16:9, 23.98 fps (default)
  Stream #0:1(jpn): Audio: aac (LC), 48000 Hz, stereo, fltp (default)
    Metadata:
      title           : Japanese
  Stream #0:2(eng): Audio: eac3, 48000 Hz, 5.1(side), fltp
    Metadata:
      title           : English dub
  Stream #0:3[0x4](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, mono, fltp, 69 kb/s
  Stream #0:4(eng): Subtitle: ass
    Metadata:
      title           : Signs
At least one output file must be specified
"""


def test_probe_reads_length_and_audio_tracks(monkeypatch):
    monkeypatch.setattr(youtube, "ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(
        youtube.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, b"", FFMPEG_SAYS.encode())
    )
    duration, tracks = youtube.probe(youtube.Path("Film.mkv"))
    assert duration == pytest.approx(6133.52)
    assert [t.label for t in tracks] == ["Track 1: Japanese, stereo", "Track 2: English dub, 5.1", "Track 3, mono"]
    assert [t.language for t in tracks] == ["ja", "en", ""]
    assert [t.default for t in tracks] == [True, False, False]
    assert [t.index for t in tracks] == [0, 1, 2]


def test_probe_names_the_language_when_the_title_does_not():
    youtube_lang = youtube._language
    assert (youtube_lang("ger"), youtube_lang("fra"), youtube_lang("en-US"), youtube_lang("und")) == ("de", "fr", "en", "")


# ------------------------------------------------------------ with ffmpeg


def _ffmpeg(*args):
    subprocess.run([youtube.ffmpeg_exe(), "-y", "-nostdin", "-loglevel", "error", *args], check=True)


@pytest.fixture
def film(tmp_path):
    """12 seconds of video with two audio tracks: Japanese at 220 Hz, and an English dub at 440 Hz (the default)."""
    path = tmp_path / "My Film.mkv"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=12",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=12",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
        "-map", "0", "-map", "1", "-map", "2", "-c:v", "mpeg4", "-c:a", "aac", "-ac:a:1", "6",
        "-metadata:s:a:0", "language=jpn", "-metadata:s:a:1", "language=eng", "-metadata:s:a:1", "title=English dub",
        "-disposition:a:0", "0", "-disposition:a:1", "default",
        str(path),
    )  # fmt: skip
    return path


@pytest.fixture
def clips(tmp_path, monkeypatch):
    monkeypatch.setattr(youtube, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(youtube, "CLIPS_DIR", tmp_path / "work" / "clips")
    monkeypatch.setattr(youtube, "_files", youtube.OrderedDict())
    heard = []  # the language speech-to-text was asked to listen for; it finds no words
    monkeypatch.setattr(transcribe, "words", lambda wav, sr, settings, language=None: heard.append(language) or ([], None))
    monkeypatch.setattr(transcribe, "detect", lambda wav, sr, settings: None)  # can't tell the language
    return heard


def _pitch(path) -> float:
    wav, sr = sf.read(str(path))
    return float(np.argmax(np.abs(np.fft.rfft(wav))) * sr / len(wav))


@needs_ffmpeg
def test_open_file_reads_a_film_where_it_is(film, clips):
    before = os.stat(film).st_mtime_ns
    video = youtube.open_file(f'"{film}"')  # as File Explorer's "Copy as path" gives it, with quotes
    assert video.local and video.path == film and video.title == "My Film.mkv"
    assert video.duration == pytest.approx(12, abs=0.1)
    assert [t.label for t in video.tracks] == ["Track 1: Japanese, mono", "Track 2: English dub, 5.1"]
    assert youtube.default_track(video) == 1
    assert youtube.open_file(str(film)) is video  # looked at once
    assert os.stat(film).st_mtime_ns == before  # and left alone


@needs_ffmpeg
def test_make_file_clip_from_the_track_picked(film, clips):
    steps = []
    clip = youtube.make_file_clip(str(film), "0:03", 5, isolate=False, progress=lambda f, d: steps.append(d))
    assert _pitch(clip.path) == pytest.approx(440, abs=5)  # the default track: the English dub
    assert (clip.start, clip.end) == (3, 8) and clip.title == "My Film.mkv"
    assert clip.path.parent == youtube.CLIPS_DIR and clip.path.name.startswith("My_Film-")
    assert clip.language == "English"  # speech-to-text couldn't tell, so the track's label counts
    assert clips == ["en"] and clip.transcript == "" and clip.source == ""
    assert steps[0] == "Opening the file" and steps[-1] == "Done"

    clip = youtube.make_file_clip(str(film), "3", 5, track=0, isolate=False, language="English", transcript="none")
    assert _pitch(clip.path) == pytest.approx(220, abs=5)
    assert "-t1-" not in clip.path.name and clip.language == "Japanese"
    assert clips == ["en"]  # no speech-to-text this time

    clip = youtube.make_file_clip(str(film), "", 5, track=7, isolate=False)  # no such track: the default
    assert _pitch(clip.path) == pytest.approx(440, abs=5) and "-t2-" in clip.path.name and clip.start == 0


@needs_ffmpeg
def test_make_file_clip_with_speech_to_text_and_the_music_remover(film, clips, monkeypatch):
    words = [(1.2, 1.6, " hello"), (1.7, 2.4, " there"), (2.6, 4.5, " everyone")]  # from 1 s before the start
    monkeypatch.setattr(transcribe, "words", lambda wav, sr, settings, language=None: (words, "en"))
    monkeypatch.setattr(youtube, "remove_background", lambda wav, device: wav.mean(0))
    clip = youtube.make_file_clip(str(film), "0:02", 5, track=0, isolate=True)
    assert clip.transcript == "hello there everyone" and clip.source == "speech"
    assert (clip.start, clip.end) == (pytest.approx(2.1), pytest.approx(5.8))  # whole words
    assert clip.path.name.endswith("-voice.wav")


@needs_ffmpeg
def test_audio_files_work_too(tmp_path, clips):
    path = tmp_path / "podcast.mp3"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=300:duration=20", str(path))
    clip = youtube.make_file_clip(str(path), "0:10", 60, isolate=False, transcript="none")
    assert (clip.start, clip.end) == (10, pytest.approx(20, abs=0.1))  # stops where the file ends
    with pytest.raises(ValueError, match="podcast.mp3 is only 0:20 long"):
        youtube.make_file_clip(str(path), "0:25", 5, isolate=False)


@needs_ffmpeg
def test_files_that_cannot_be_used_say_why(tmp_path, clips):
    with pytest.raises(ValueError, match="Pick a file first"):
        youtube.open_file("  ")
    with pytest.raises(ValueError, match="Couldn't find"):
        youtube.open_file(str(tmp_path / "missing.mp4"))
    with pytest.raises(ValueError, match="is a folder"):
        youtube.open_file(str(tmp_path))
    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Couldn't read notes.txt as video or audio \(Invalid data"):
        youtube.open_file(str(notes))
    silent = tmp_path / "silent.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3", "-c:v", "mpeg4", str(silent))
    with pytest.raises(ValueError, match="silent.mp4 has no sound in it"):
        youtube.open_file(str(silent))


# ------------------------------------------------------------ the Studio


def _handler(demo, name):
    return next(f.fn for f in demo.fns.values() if f.name == name)


@needs_ffmpeg
def test_studio_lists_tracks_and_ignores_a_list_left_from_another_file(film, clips, engine, library, tmp_path):
    demo = studio.build(engine, library)
    info, tracks, loaded = _handler(demo, "file_info")(str(film))
    assert "My Film.mkv" in info and "0:12 long" in info and "2 audio tracks" in info
    assert tracks["visible"] and tracks["value"] == 1 and loaded == str(film)
    assert tracks["choices"] == [("Track 1: Japanese, mono", 0), ("Track 2: English dub, 5.1", 1)]
    assert _handler(demo, "file_info")(str(film), str(film))[1] == {"__type__": "update"}  # keeps the picked track
    assert "Couldn't find" in _handler(demo, "file_info")(str(tmp_path / "nope.mkv"))[0]

    file_clip = _handler(demo, "file_clip")
    quiet = lambda f, desc=None: None  # noqa: E731
    out = file_clip(str(film), 0, str(film), "0:03", 5, False, "none", "English", progress=quiet)
    assert _pitch(out[0]) == pytest.approx(220, abs=5) and out[3] == "Japanese"
    assert "Got **5.0 seconds** of *My Film.mkv* (0:03 to 0:08)" in out[4]
    other = tmp_path / "other.mkv"
    other.write_bytes(film.read_bytes())
    out = file_clip(str(other), 0, str(film), "0:03", 5, False, "none", "English", progress=quiet)
    assert _pitch(out[0]) == pytest.approx(440, abs=5)  # track 1 was picked for the other file


# ------------------------------------------------------------ the file picker


def test_browse_on_windows_opens_the_file_dialog(monkeypatch, tmp_path):
    ran = {}

    def run(cmd, **kw):
        ran.update(cmd=cmd, **kw)
        picked = base64.b64encode("G:\\Vidéos\\Film.mkv".encode()).decode()  # base64, whatever the code page
        return subprocess.CompletedProcess(cmd, 0, f"PICKED:{picked}\r\n".encode(), b"")

    monkeypatch.setattr(youtube, "WINDOWS", True)
    monkeypatch.setattr(youtube.subprocess, "run", run)
    (tmp_path / "old.mp4").write_bytes(b"")
    assert youtube.browse(f'"{tmp_path / "old.mp4"}"') == "G:\\Vidéos\\Film.mkv"
    assert ran["cmd"][:2] == ["powershell.exe", "-NoProfile"]
    script = base64.b64decode(ran["cmd"][-1]).decode("utf-16-le")
    assert "OpenFileDialog" in script and "TopMost" in script and "ToBase64String" in script
    assert ran["env"]["QWEN_TTS_FOLDER"] == str(tmp_path)  # opens where the last file was
    assert "*.mkv" in ran["env"]["QWEN_TTS_FILTER"] and ran["timeout"] > 60

    monkeypatch.setattr(youtube.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, b"\r\n", b""))
    assert youtube.browse("") is None  # cancelled

    def broken(cmd, **kw):
        raise FileNotFoundError("powershell.exe")

    monkeypatch.setattr(youtube.subprocess, "run", broken)
    with pytest.raises(ValueError, match="Copy as path"):
        youtube.browse("")
    garbled = subprocess.CompletedProcess([], 0, b"PICKED:not base64!\r\n", b"")
    monkeypatch.setattr(youtube.subprocess, "run", lambda cmd, **kw: garbled)
    with pytest.raises(ValueError, match="Copy as path"):
        youtube.browse("")


def test_browse_elsewhere_uses_tkinter(monkeypatch):
    ran = {}

    def run(cmd, **kw):
        ran.update(cmd=cmd, **kw)
        return subprocess.CompletedProcess(cmd, 0, b"PICKED:/home/me/film.mkv\n", b"")

    monkeypatch.setattr(youtube, "WINDOWS", False)
    monkeypatch.setattr(youtube.subprocess, "run", run)
    assert youtube.browse("not a real folder/x.mp4") == "/home/me/film.mkv"
    assert ran["cmd"][0] == sys.executable and ran["cmd"][-2] == ""  # no folder to start in
    compile(youtube._PICKER_TK, "picker", "exec")
