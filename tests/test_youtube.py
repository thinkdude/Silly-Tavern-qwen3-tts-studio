"""Clone from a YouTube link, without the network: captions, timing and the clip pipeline.

The ffmpeg, yt-dlp and Demucs tests run when those packages are installed (install.bat adds them).
"""

import functools
import http.server
import importlib.util
import json
import threading

import numpy as np
import pytest
import soundfile as sf

from app import transcribe, youtube

# Automatic captions: one event per line, each word with its own offset, plus line-break events.
ASR = {
    "events": [
        {"tStartMs": 0, "dDurationMs": 60000, "id": 1, "wpWinPosId": 1},
        {
            "tStartMs": 1000,
            "dDurationMs": 4000,
            "wWinId": 1,
            "segs": [{"utf8": "hello"}, {"utf8": " there", "tOffsetMs": 500}, {"utf8": " everyone", "tOffsetMs": 1000}],
        },
        {"tStartMs": 2900, "dDurationMs": 100, "wWinId": 1, "aAppend": 1, "segs": [{"utf8": "\n"}]},
        {
            "tStartMs": 3000,
            "dDurationMs": 4000,
            "wWinId": 1,
            "segs": [{"utf8": "[Music]"}, {"utf8": " welcome", "tOffsetMs": 1000}, {"utf8": " back", "tOffsetMs": 1600}],
        },
        {"tStartMs": 6000, "dDurationMs": 2000, "wWinId": 1, "segs": [{"utf8": "today"}, {"utf8": " we", "tOffsetMs": 400}]},
    ]
}

# Captions someone wrote: one segment per line.
WRITTEN = {
    "events": [
        {"tStartMs": 500, "dDurationMs": 2000, "segs": [{"utf8": "- Hello &amp; welcome."}]},
        {"tStartMs": 2600, "dDurationMs": 2400, "segs": [{"utf8": "(laughs) It's good to be here."}]},
        {"tStartMs": 5200, "dDurationMs": 3000, "segs": [{"utf8": "♪ Let's begin ♪"}]},
    ]
}


@pytest.mark.parametrize(
    "text, secs",
    [("", 0), ("83", 83), ("83.5", 83.5), ("1:23", 83), ("01:02:03", 3723), ("1m23s", 83), ("2m", 120), ("90s", 90), ("1h2m3s", 3723), (" 1:05 ", 65)],
)
def test_parse_time(text, secs):
    assert youtube.parse_time(text) == pytest.approx(secs)


@pytest.mark.parametrize("text", ["abc", "1:2:3:4", "1:xx", "-5"])
def test_parse_time_rejects_nonsense(text):
    with pytest.raises(ValueError, match="start time"):
        youtube.parse_time(text)


def test_format_time():
    assert youtube.format_time(83) == "1:23"
    assert youtube.format_time(83.44) == "1:23.4"
    assert youtube.format_time(59.97) == "1:00"
    assert youtube.format_time(3723) == "1:02:03"


@pytest.mark.parametrize(
    "url, secs",
    [
        ("https://youtu.be/dQw4w9WgXcQ?t=83", 83),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=1m23s", 83),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ#t=1:23", 83),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ?start=90", 90),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", None),
    ],
)
def test_start_from_url(url, secs):
    assert youtube.start_from_url(url) == secs


def test_cache_key_ignores_the_start_time():
    a = youtube._cache_key("https://www.youtube.com/watch?v=abc&t=83")
    assert a == youtube._cache_key("https://WWW.youtube.com/watch?v=abc")
    assert a != youtube._cache_key("https://www.youtube.com/watch?v=xyz")


def test_clean_url():
    assert youtube._clean_url(" dQw4w9WgXcQ ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert youtube._clean_url("youtube.com/watch?v=dQw4w9WgXcQ").startswith("https://youtube.com/")
    with pytest.raises(ValueError, match="link"):
        youtube._clean_url("  ")


def test_parse_automatic_captions():
    units = youtube.parse_json3(ASR)
    assert [u.text.strip() for u in units] == ["hello", "there", "everyone", "[Music]", "welcome", "back", "today", "we"]
    hello, there, everyone = units[:3]
    assert (hello.start, hello.end) == (1.0, 1.5)
    assert (there.start, there.end) == (1.5, 2.0)
    assert everyone.end == 3.0  # ends where the next line starts
    assert units[-1].end == 8.0  # the last word lasts until its line ends


def test_caption_text_drops_noise_and_speaker_dashes():
    assert youtube.caption_text(youtube.parse_json3(ASR)) == "hello there everyone welcome back today we"
    assert youtube.caption_text(youtube.parse_json3(WRITTEN)) == "Hello & welcome. It's good to be here. Let's begin"


def test_caption_text_has_no_spaces_inside_chinese_or_japanese():
    units = [youtube.Unit(0, 1, "今日は"), youtube.Unit(1, 2, "いい天気"), youtube.Unit(2, 3, "OK"), youtube.Unit(3, 4, "です")]
    assert youtube.caption_text(units) == "今日はいい天気 OK です"


def test_snap_to_word_boundaries():
    units = youtube.parse_json3(ASR)
    # Asked for 1.3 to 4.3: starts just before "there" and ends where "back" starts.
    start, end, chosen = youtube.snap(units, 1.3, 4.3)
    assert youtube.caption_text(chosen) == "there everyone welcome"
    assert start == pytest.approx(1.4)
    assert end == pytest.approx(4.6)  # where "back" starts


def test_snap_without_captions_in_the_window_keeps_the_cut():
    units = youtube.parse_json3(ASR)
    assert youtube.snap(units, 20, 30) == (20, 30, [])
    assert youtube.snap([], 5, 10) == (5, 10, [])


def test_snap_keeps_at_least_one_long_written_line():
    units = youtube.parse_json3(WRITTEN)
    start, end, chosen = youtube.snap(units, 2.5, 3.5)
    assert youtube.caption_text(chosen) == "It's good to be here."
    assert (start, end) == (pytest.approx(2.5), pytest.approx(5.0))


def _tracks(lang, tlang=None):
    url = f"https://www.youtube.com/api/timedtext?v=x&lang={lang}" + (f"&tlang={tlang}" if tlang else "")
    return [{"ext": "vtt", "url": url + "&fmt=vtt"}, {"ext": "json3", "url": url + "&fmt=json3"}]


def test_pick_captions_prefers_written_ones():
    info = {"subtitles": {"en-US": _tracks("en")}, "automatic_captions": {"en-orig": _tracks("en"), "en": _tracks("en")}}
    url, kind = youtube.pick_captions(info, "en")
    assert kind == "written" and "fmt=json3" in url


def test_pick_captions_skips_translations():
    info = {"automatic_captions": {"ja-orig": _tracks("ja"), "ja": _tracks("ja"), "en": _tracks("ja", tlang="en")}}
    assert youtube.spoken_language(info) == "ja"
    assert youtube.pick_captions(info, "en") is None
    url, kind = youtube.pick_captions(info, "ja")
    assert kind == "automatic" and "tlang" not in url
    assert youtube.pick_captions(info, None) is None
    assert youtube.pick_captions({}, "en") is None


# ------------------------------------------------------------ with ffmpeg

needs_ffmpeg = pytest.mark.skipif(
    importlib.util.find_spec("imageio_ffmpeg") is None, reason="needs imageio-ffmpeg (install.bat adds it)"
)


def _speechy(seconds, sr=44100):
    """A stand-in for speech: a tone that switches on and off."""
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.5 * t) > -0.5)).astype(np.float32)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "talk.wav"
    sf.write(str(path), _speechy(20), 44100)
    return path


@needs_ffmpeg
def test_cut(source):
    wav = youtube.cut(source, 2.0, 5.0)
    assert wav.shape[0] == 2
    assert wav.shape[1] == pytest.approx(3 * 44100, abs=200)
    assert youtube.cut(source, 30, 35).shape[1] == 0  # past the end


@pytest.fixture
def fake_video(source, tmp_path, monkeypatch):
    monkeypatch.setattr(youtube, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(youtube, "CLIPS_DIR", tmp_path / "work" / "clips")
    video = youtube.Video(
        id="abc123", title="A talk", duration=20.0, path=source, info={"automatic_captions": {"en-orig": _tracks("en")}}
    )
    video.captions["en"] = (youtube.parse_json3(ASR), "automatic")
    calls = Calls()
    monkeypatch.setattr(youtube, "download", lambda url, progress=None: calls.append(url) or video)
    calls.video = video
    calls.heard = []  # speech-to-text calls; it finds no words unless a test says otherwise
    monkeypatch.setattr(transcribe, "words", lambda wav, sr, settings, language=None: calls.heard.append(language) or ([], None))
    calls.detected = None  # the language speech-to-text hears; None: it can't tell, so YouTube's label counts
    monkeypatch.setattr(transcribe, "detect", lambda wav, sr, settings: calls.detected)
    return calls


class Calls(list):
    pass


@needs_ffmpeg
def test_make_clip(fake_video):
    steps = []
    clip = youtube.make_clip(
        "https://youtu.be/abc123?t=1.3", None, 5, language="German", isolate=False, progress=lambda f, d: steps.append(d)
    )
    assert fake_video == ["https://youtu.be/abc123?t=1.3"]
    assert (clip.start, clip.end) == (pytest.approx(1.4), pytest.approx(6.4))
    assert clip.transcript == "there everyone welcome back today"
    assert clip.source == "automatic"
    assert clip.language == "English"  # follows the video, not the dropdown
    assert fake_video.heard == []  # captions, so no speech-to-text
    assert clip.path.exists() and clip.path.parent == youtube.CLIPS_DIR
    wav, sr = sf.read(str(clip.path))
    assert sr == 44100 and 2 < len(wav) / sr <= 5.1 and clip.seconds == pytest.approx(len(wav) / sr)
    assert steps[-1] == "Done"


@needs_ffmpeg
def test_make_clip_typed_start_wins_and_rejects_start_past_the_end(fake_video):
    clip = youtube.make_clip("https://youtu.be/abc123?t=1.3", "0:10", 5, isolate=False)
    assert (clip.start, clip.end) == (10, 15)
    assert clip.transcript == "" and clip.source == ""
    assert fake_video.heard == ["en"]  # no captions there, so speech-to-text was asked (and heard nothing)
    with pytest.raises(ValueError, match="only 0:20 long"):
        youtube.make_clip("https://youtu.be/abc123", "0:25", 5, isolate=False)


@needs_ffmpeg
def test_make_clip_uses_speech_to_text_without_captions(fake_video, monkeypatch):
    fake_video.video.info, fake_video.video.captions = {}, {}  # no captions, language unknown
    seen = {}

    def words(wav, sr, settings, language=None):
        seen.update(seconds=len(wav) / sr, settings=settings, language=language)
        # Times are from the start of the audio it gets, which starts 1 s before the asked-for start.
        return [(0.5, 0.9, "Before"), (1.2, 1.6, " hello"), (1.7, 2.2, " from"), (2.4, 3.0, " speech"),
                (5.9, 6.3, " to"), (6.4, 6.9, " text")], "de"  # fmt: skip

    monkeypatch.setattr(transcribe, "words", words)
    clip = youtube.make_clip(
        "https://youtu.be/abc123", "0:10", 5, isolate=False, device="cuda:0", settings={"stt_model": "x"}
    )
    assert seen == {"seconds": pytest.approx(7, abs=0.01), "settings": {"stt_model": "x", "device": "cuda:0"}, "language": None}
    assert clip.transcript == "hello from speech to" and clip.source == "speech"
    assert (clip.start, clip.end) == (pytest.approx(10.1), pytest.approx(15.4))  # whole words
    assert clip.language == "German"  # what Whisper heard


ARABIC = {"events": [{"tStartMs": 9000, "dDurationMs": 6000, "segs": [{"utf8": "مرحبا بكم في البرنامج"}]}]}
WORDS = [(1.2, 1.6, " hello"), (1.7, 2.2, " from"), (2.4, 3.0, " speech")]


@needs_ffmpeg
def test_make_clip_follows_the_language_it_hears_not_youtubes_label(fake_video, monkeypatch):
    video = fake_video.video
    video.info = {"automatic_captions": {"ar-orig": _tracks("ar")}}  # YouTube says Arabic...
    video.captions = {"ar": (youtube.parse_json3(ARABIC), "automatic")}
    fake_video.detected = "en"  # ...but it's English
    asked = []
    monkeypatch.setattr(transcribe, "words", lambda wav, sr, settings, language=None: asked.append(language) or (WORDS, "en"))
    clip = youtube.make_clip("https://youtu.be/abc123", "0:10", 5, language="English", isolate=False)
    assert clip.transcript == "hello from speech" and clip.source == "speech"
    assert clip.language == "English" and asked == ["en"]


@needs_ffmpeg
def test_make_clip_skips_captions_in_the_wrong_script(fake_video, monkeypatch):
    video = fake_video.video
    video.captions = {"en": (youtube.parse_json3(ARABIC), "automatic")}  # labelled English, written in Arabic
    monkeypatch.setattr(transcribe, "words", lambda wav, sr, settings, language=None: (WORDS, "en"))
    clip = youtube.make_clip("https://youtu.be/abc123", "0:10", 5, isolate=False)
    assert clip.transcript == "hello from speech" and clip.source == "speech"


def test_script_fits():
    assert youtube.script_fits("Hello, ça va? Schön!", "en")
    assert not youtube.script_fits("مرحبا بكم في البرنامج", "en")
    assert youtube.script_fits("Привет, как дела?", "ru") and not youtube.script_fits("Hello there", "ru")
    assert youtube.script_fits("今日はいい天気です", "ja") and youtube.script_fits("안녕하세요", "ko")
    assert youtube.script_fits("مرحبا", "ar")  # languages it doesn't know always fit
    assert not youtube.script_fits("", "en")


@needs_ffmpeg
def test_make_clip_transcript_choices(fake_video, monkeypatch):
    clip = youtube.make_clip("https://youtu.be/abc123", "0:01.3", 5, isolate=False, transcript="none")
    assert clip.transcript == "" and clip.source == "" and fake_video.heard == []
    clip = youtube.make_clip("https://youtu.be/abc123", "0:01.3", 5, isolate=False, transcript="speech")
    assert fake_video.heard == ["en"]  # skipped the captions, speech-to-text in the video's language

    def broken(wav, sr, settings, language=None):
        raise ValueError("Couldn't load the speech-to-text model (offline).")

    monkeypatch.setattr(transcribe, "words", broken)
    clip = youtube.make_clip("https://youtu.be/abc123", "0:10", 5, isolate=False)
    assert clip.transcript == "" and "offline" in clip.note and clip.path.exists()


@needs_ffmpeg
def test_make_clip_with_the_music_remover(fake_video, monkeypatch):
    seen = {}

    def remove_background(wav, device):
        seen["shape"], seen["device"] = wav.shape, device
        return wav.mean(0)

    monkeypatch.setattr(youtube, "remove_background", remove_background)
    clip = youtube.make_clip("https://youtu.be/abc123", "0:02", 6, isolate=True, device="cuda:0")
    assert seen["shape"][0] == 2 and seen["device"] == "cuda:0"
    assert clip.path.name.endswith("-voice.wav")


@needs_ffmpeg
def test_make_clip_says_when_there_is_no_voice(fake_video, monkeypatch):
    monkeypatch.setattr(youtube, "remove_background", lambda wav, device: wav.mean(0) * 0.001)
    with pytest.raises(ValueError, match="hardly any voice"):
        youtube.make_clip("https://youtu.be/abc123", "0:02", 6, isolate=True)


def test_speech_units_reach_into_the_pause_after_each_word():
    units = youtube._speech_units([(0.0, 0.4, "a"), (0.5, 0.8, " b"), (2.0, 2.5, " c")], 10)
    assert [(u.start, u.end) for u in units] == [(10.0, 10.5), (10.5, pytest.approx(11.1)), (12.0, 12.8)]


def test_remove_background_runs_demucs(monkeypatch):
    pytest.importorskip("demucs")
    from demucs.pretrained import get_model

    monkeypatch.setattr(youtube, "_demucs", get_model("demucs_unittest"))  # tiny and untrained, no download
    stereo = np.stack([_speechy(3), _speechy(3)])
    out = youtube.remove_background(stereo, "cuda:0")  # falls back to the CPU without a GPU
    assert out.shape == (stereo.shape[1],) and out.dtype == np.float32


@needs_ffmpeg
def test_download_with_yt_dlp(source, tmp_path, monkeypatch):
    pytest.importorskip("yt_dlp")
    monkeypatch.setattr(youtube, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(youtube, "_videos", youtube.OrderedDict())
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(source.parent))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/{source.name}"
        video = youtube.download(url)
        assert video.path.exists() and video.path.parent == youtube.WORK_DIR
        assert video.id == "talk"
        assert youtube.download(url + "?t=5") is video  # same video: no second download
        assert youtube.cut(video.path, 1, 3).shape[1] == pytest.approx(2 * 44100, abs=200)
        # Captions are fetched through yt-dlp too.
        (source.parent / "captions.json3").write_text(json.dumps(ASR), encoding="utf-8")
        base = f"http://127.0.0.1:{server.server_address[1]}"
        video.info["automatic_captions"] = {"en-orig": [{"ext": "json3", "url": f"{base}/captions.json3"}]}
        units, kind = youtube.captions(video, "en")
        assert kind == "automatic" and youtube.caption_text(units).startswith("hello there")
        video.info["automatic_captions"] = {"de-orig": [{"ext": "json3", "url": f"{base}/missing.json3"}]}
        assert youtube.captions(video, "de") == ([], "")  # a failed caption download isn't an error
        with pytest.raises(ValueError, match="Couldn't get that video"):
            youtube.download(f"http://127.0.0.1:{server.server_address[1]}/missing.wav")
    finally:
        server.shutdown()


def test_download_errors_read_well():
    msg = str(youtube._download_error(Exception("ERROR: [youtube] dQw4w9WgXcQ: Video unavailable")))
    assert msg.startswith("Couldn't get that video: Video unavailable ")
    msg = str(youtube._download_error(Exception("ERROR: [generic] Unable to download webpage: HTTP Error 404")))
    assert msg.startswith("Couldn't get that video: Unable to download webpage: HTTP Error 404 ")
    assert youtube.UPDATE_HINT in msg


def test_missing_packages_say_to_run_the_installer(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_demucs(name, *args, **kwargs):
        if name.startswith("demucs"):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(youtube, "_demucs", None)
    monkeypatch.setattr(builtins, "__import__", no_demucs)
    with pytest.raises(ValueError, match="run install.bat once"):
        youtube.remove_background(np.zeros((2, 44100), dtype=np.float32))
