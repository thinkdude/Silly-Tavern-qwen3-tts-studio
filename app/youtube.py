"""Get a reference clip for cloning from a YouTube video (or any site yt-dlp knows).

Paste a link and say where the speaker talks, and this:
  1. downloads the video's audio with yt-dlp, once: more clips from the same
     video reuse it,
  2. cuts out that stretch with ffmpeg,
  3. removes music and background noise with Demucs (optional),
  4. writes down what's said, so the clone gets a transcript: from the video's
     captions when it has some, otherwise with speech-to-text (transcribe.py).
     The cut is moved to the nearest word boundaries so the transcript matches
     the clip. The Studio shows it for checking and correcting.

install.bat installs what it needs: yt-dlp with Deno (YouTube needs a JavaScript
runtime), ffmpeg (from imageio-ffmpeg) and Demucs, and downloads the Demucs and
Whisper models. Everything stays inside the install folder.

    python -m app.youtube --setup            download the Demucs and Whisper models now
    python -m app.youtube --update           update yt-dlp, for when YouTube downloads stop working
    python -m app.youtube URL --start 1:23   make a clip from the command line
"""

from __future__ import annotations

import argparse
import html
import importlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import soundfile as sf

from . import audio, transcribe
from .config import CACHE_DIR, CODE_LANGS, LANG_CODES, ROOT, setup_environment

WORK_DIR = CACHE_DIR / "youtube"
CLIPS_DIR = WORK_DIR / "clips"
KEEP_DOWNLOADS = 5
KEEP_CLIPS = 30
MIN_LEN, MAX_LEN, DEFAULT_LEN = 5, 60, 15
DEMUCS_MODEL = "htdemucs"
SEP_SR = 44100  # what Demucs works at

NOT_INSTALLED = (
    "{what} isn't installed yet. Close the Studio, run install.bat once to add it, then start the Studio again. "
    "Until then you can untick 'Remove music and background noise'."
)
UPDATE_HINT = "If YouTube downloads stopped working, click 'Update the YouTube downloader' and try again."

Progress = Callable[[float, str], None]


class ClipError(ValueError):
    """A problem worth showing as-is in the Studio."""


# ------------------------------------------------------------------ times


def parse_time(text) -> float:
    """'83', '83.5', '1:23', '1:02:03', '1m23s', '2m', '90s' -> seconds. Empty -> 0."""
    s = str(text or "").strip().lower().replace(",", ".").replace(" ", "")
    if not s:
        return 0.0
    if ":" in s:
        parts = s.split(":")
        if len(parts) <= 3 and all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
            secs = 0.0
            for p in parts:
                secs = secs * 60 + float(p)
            return secs
    else:
        m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s?)?", s)
        if m and any(m.groups()):
            h, mins, secs = (float(g) if g else 0.0 for g in m.groups())
            return h * 3600 + mins * 60 + secs
    raise ClipError(f"Couldn't read the start time '{text}'. Use minutes:seconds, like 1:23.")


def format_time(secs: float) -> str:
    """83.4 -> '1:23.4', 3723 -> '1:02:03'."""
    tenths = round(max(0.0, float(secs)) * 10)
    whole, frac = divmod(tenths, 10)
    h, rest = divmod(whole, 3600)
    m, s = divmod(rest, 60)
    text = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    return text + (f".{frac}" if frac else "")


def start_from_url(url: str) -> float | None:
    """The start time in a 'share at current time' link (?t=83, &t=1m23s, #t=1:23)."""
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return None
    for source in (parse_qs(parts.query), parse_qs(parts.fragment)):
        for key in ("t", "start"):
            if key in source:
                try:
                    return parse_time(source[key][-1])
                except ClipError:
                    pass
    return None


def _cache_key(url: str) -> str:
    """The link without its start time, so clips from the same video share the download."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k not in ("t", "start")]
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, urlencode(query), ""))


def _clean_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise ClipError("Paste a video link first.")
    if re.fullmatch(r"[\w-]{11}", url):  # a bare YouTube video id
        return f"https://www.youtube.com/watch?v={url}"
    if not re.match(r"https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url


# --------------------------------------------------------------- captions


@dataclass
class Unit:
    """A word (automatic captions) or a line (captions someone wrote), with its timing."""

    start: float
    end: float
    text: str


def parse_json3(data: dict) -> list[Unit]:
    """Read YouTube's json3 caption format.

    Automatic captions give every word its own start time (tStartMs of the line
    plus the word's tOffsetMs); captions someone wrote have one segment per line.
    """
    units = []
    for ev in data.get("events") or []:
        segs = ev.get("segs")
        if not segs:
            continue
        t0 = (ev.get("tStartMs") or 0) / 1000
        t1 = t0 + (ev.get("dDurationMs") or 0) / 1000
        for seg in segs:
            text = seg.get("utf8") or ""
            if not text.strip():
                continue
            start = t0 + (seg.get("tOffsetMs") or 0) / 1000
            units.append(Unit(start, max(t1, start), text))
    units.sort(key=lambda u: u.start)
    # Words only come with a start time, so each one lasts until the next begins.
    for a, b in itertools.pairwise(units):
        if b.start < a.end:
            a.end = max(b.start, a.start)
    return units


_NOT_SPEECH = re.compile(r"\[[^\]]*\]|\([^)]*\)|[♪♫]+|>>")


def _is_cjk(ch: str) -> bool:
    # Chinese and Japanese are written without spaces between words (Korean uses spaces).
    return "　" <= ch <= "ヿ" or "㐀" <= ch <= "鿿" or "＀" <= ch <= "￯"


def caption_text(units: list[Unit]) -> str:
    out = ""
    for u in units:
        part = _NOT_SPEECH.sub(" ", html.unescape(u.text))
        part = re.sub(r"(^|\s)-\s+", r"\1", part)  # "- " marking a new speaker
        part = re.sub(r"\s+", " ", part).strip()
        if not part:
            continue
        if out and not (_is_cjk(out[-1]) and _is_cjk(part[0])):
            out += " "
        out += part
    return out


def snap(units: list[Unit], start: float, end: float) -> tuple[float, float, list[Unit]]:
    """Move the cut [start, end] to word (or caption line) boundaries.

    Returns the new start and end and the captions inside them, so the clip holds
    whole words and the transcript matches it. With no captions in the window
    the cut stays as asked and the list is empty.
    """
    first = next((i for i, u in enumerate(units) if u.start >= start - 0.1), None)
    if first is None:
        return start, end, []
    chosen: list[Unit] = []
    for u in units[first:]:
        if u.start >= end or (chosen and u.end > end + 0.5):
            break
        chosen.append(u)
    if not chosen:
        return start, end, []
    return max(0.0, chosen[0].start - 0.1), chosen[-1].end, chosen


_LATIN = {"en", "de", "fr", "es", "it", "pt"}


def script_fits(text: str, code: str | None) -> bool:
    """False when the text is mostly in another writing system than language `code` uses,
    like Arabic captions on an English video. Unknown languages always fit."""
    letters = [c for c in text if c.isalpha()]
    if not letters or not code:
        return bool(letters)

    def share(test) -> float:
        return sum(1 for c in letters if test(c)) / len(letters)

    if code in _LATIN:
        return share(lambda c: c < "\u0250") > 0.7  # Latin, with its accented letters
    if code == "ru":
        return share(lambda c: "\u0400" <= c <= "\u04ff") > 0.7
    if code in ("zh", "ja"):
        return share(_is_cjk) > 0.5
    if code == "ko":
        return share(lambda c: "\uac00" <= c <= "\ud7af" or "\u1100" <= c <= "\u11ff" or "\u3130" <= c <= "\u318f") > 0.5
    return True


def spoken_language(info: dict) -> str | None:
    """The language code of what's said in the video, when YouTube knows it."""
    for key in info.get("automatic_captions") or {}:
        if key.endswith("-orig"):
            return key.removesuffix("-orig")
    return info.get("language") or None


def _matches(key: str, code: str) -> bool:
    return key == code or key.startswith(code + "-")


def _json3_url(tracks) -> str | None:
    for t in tracks or []:
        if t.get("ext") == "json3" and t.get("url"):
            return t["url"]
    return None


def pick_captions(info: dict, code: str | None) -> tuple[str, str] | None:
    """(url, "written" or "automatic") of the best json3 captions in language `code`.

    Captions someone wrote win over automatic ones. Automatic captions translated
    from another language (tlang= in the link) are skipped: they don't match
    what's said.
    """
    if not code:
        return None
    subs = info.get("subtitles") or {}
    autos = info.get("automatic_captions") or {}
    for key in sorted(subs):
        if _matches(key, code) and (url := _json3_url(subs[key])):
            return url, "written"
    for key in sorted(autos, key=lambda k: not k.endswith("-orig")):
        if _matches(key.removesuffix("-orig"), code) and (url := _json3_url(autos[key])) and "tlang=" not in url:
            return url, "automatic"
    return None


# ---------------------------------------------------------------- download


@dataclass
class Video:
    id: str
    title: str
    duration: float | None
    path: Path
    info: dict
    captions: dict = field(default_factory=dict)  # language code -> (units, kind)


_videos: OrderedDict[str, Video] = OrderedDict()


class _Log:
    """Keep yt-dlp's chatter out of the console, but show its warnings and errors."""

    def debug(self, msg):
        pass

    info = debug

    def warning(self, msg):
        print(f"[youtube] {msg}")

    def error(self, msg):
        print(f"[youtube] {msg}")


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise ClipError(NOT_INSTALLED.split(" Until")[0].format(what="ffmpeg"))


def _ydl(progress: Progress | None = None):
    try:
        import yt_dlp
    except ImportError:
        raise ClipError(NOT_INSTALLED.split(" Until")[0].format(what="The YouTube downloader (yt-dlp)")) from None

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                progress(0.05 + 0.5 * min(1.0, d.get("downloaded_bytes", 0) / total), "Downloading the audio")

    return yt_dlp.YoutubeDL(
        {
            "format": "bestaudio/best",
            "outtmpl": str(WORK_DIR / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "noprogress": True,
            "logger": _Log(),
            "progress_hooks": [hook],
            "ffmpeg_location": ffmpeg_exe(),
            "fixup": "never",  # we decode it with ffmpeg anyway
            "cachedir": str(CACHE_DIR / "yt-dlp"),
        }
    )


def _download_error(exc: Exception) -> ClipError:
    msg = re.sub(r"^ERROR:\s*", "", str(exc).strip())
    msg = re.sub(r"^\[[\w:]+\]\s*(?:[\w-]+:\s+)?", "", msg)  # "[youtube] dQw4w9WgXcQ: ", "[generic] "
    return ClipError(f"Couldn't get that video: {msg} {UPDATE_HINT}")


def download(url: str, progress: Progress | None = None) -> Video:
    """Download the video's audio, or reuse it if this video was downloaded already."""
    key = _cache_key(url)
    v = _videos.get(key)
    if v and v.path.exists():
        _videos.move_to_end(key)
        os.utime(v.path)
        return v

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.02, "Looking up the video")
    ydl = _ydl(progress)
    import yt_dlp

    try:
        with ydl:
            # Look first without processing, so a playlist link isn't worked through video by video.
            info = ydl.extract_info(url, download=False, process=False)
            if not info:
                raise ClipError("Couldn't find a video at that link.")
            if info.get("_type") in ("playlist", "multi_video"):
                raise ClipError("That's a link to a playlist. Open the video you want and copy its link.")
            if info.get("is_live") or info.get("live_status") == "is_live":
                raise ClipError("That's a live stream. Wait until it has finished, then try its link again.")
            info = ydl.process_ie_result(info, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise _download_error(exc) from None

    files = [Path(d["filepath"]) for d in info.get("requested_downloads") or [] if d.get("filepath")]
    if not files or not files[0].exists():
        raise ClipError(f"The download didn't produce an audio file. {UPDATE_HINT}")
    os.utime(files[0])
    _prune(WORK_DIR, KEEP_DOWNLOADS)
    v = Video(
        id=str(info.get("id") or files[0].stem),
        title=str(info.get("title") or files[0].stem),
        duration=float(info["duration"]) if info.get("duration") else None,
        path=files[0],
        info={k: info.get(k) for k in ("subtitles", "automatic_captions", "language")},
    )
    _videos[key] = v
    while len(_videos) > KEEP_DOWNLOADS:
        _videos.popitem(last=False)
    return v


def captions(video: Video, code: str | None) -> tuple[list[Unit], str]:
    """The video's captions in language `code` as timed words or lines, plus what kind they are.

    Best effort: no captions, or a failed download of them, gives ([], "").
    """
    if code in video.captions:
        return video.captions[code]
    result: tuple[list[Unit], str] = ([], "")
    picked = pick_captions(video.info, code)
    if picked:
        url, kind = picked
        try:
            with _ydl() as ydl:
                data = json.loads(ydl.urlopen(url).read().decode("utf-8"))
            result = (parse_json3(data), kind)
        except Exception as exc:  # noqa: BLE001 - captions are a bonus, never a reason to fail
            print(f"[youtube] Couldn't read the captions: {exc}")
    video.captions[code] = result
    return result


def _prune(folder: Path, keep: int) -> None:
    files = sorted((p for p in folder.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[keep:]:
        try:
            p.unlink()
        except OSError:
            pass


# --------------------------------------------------------------- audio work


def cut(src: Path, start: float, end: float, sr: int = SEP_SR) -> np.ndarray:
    """Decode [start, end] seconds of any audio or video file as (2, samples) float32."""
    cmd = [
        ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start):.3f}", "-t", f"{max(0.0, end - start):.3f}", "-i", str(src),
        "-vn", "-ac", "2", "-ar", str(sr), "-f", "f32le", "pipe:1",
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, check=False)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ClipError(f"ffmpeg couldn't read the audio: {err[-1] if err else 'unknown error'}")
    return np.frombuffer(r.stdout, dtype="<f4").reshape(-1, 2).T.copy()


_demucs = None
_demucs_lock = threading.Lock()


def _load_demucs():
    global _demucs
    with _demucs_lock:
        if _demucs is None:
            try:
                from demucs.pretrained import get_model
            except ImportError:
                raise ClipError(NOT_INSTALLED.format(what="The music remover (Demucs)")) from None
            try:
                _demucs = get_model(DEMUCS_MODEL)
            except Exception as exc:  # noqa: BLE001
                raise ClipError(
                    f"Couldn't load the music remover ({exc}). Check your internet connection and try again, "
                    "or untick 'Remove music and background noise'."
                ) from None
        return _demucs


def remove_background(wav: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Keep only the voice in a (channels, samples) clip at SEP_SR. Returns mono."""
    import torch

    model = _load_demucs()
    from demucs.apply import apply_model

    x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    ref = x.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    def run(dev):
        with torch.no_grad():
            return apply_model(model, ((x - mean) / std)[None], device=dev, shifts=1, split=True, overlap=0.25)

    try:
        out = run(device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("[youtube] Not enough GPU memory for the music remover, using the CPU (slower).")
        out = run("cpu")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    vocals = (out * std + mean)[0, model.sources.index("vocals")]
    return vocals.mean(0).cpu().numpy()


def _rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wav, dtype=np.float64)))) if wav.size else 0.0


# --------------------------------------------------------------- the clip


TRANSCRIPTS = ("captions", "speech", "none")  # captions, else speech-to-text / only speech-to-text / none
PAD_S = 1.0  # cut this much extra on each side, so the clip can move to whole words


@dataclass
class Clip:
    path: Path
    seconds: float
    start: float  # where in the video the cut begins and ends
    end: float
    title: str
    transcript: str  # "" when there's none
    source: str  # where the transcript came from: "written" or "automatic" captions, "speech" (to text), or ""
    language: str  # the language to clone in, from config.LANGUAGES
    note: str = ""  # anything that didn't work but didn't stop the clip


def _speech_units(words: list[transcribe.Word], offset: float) -> list[Unit]:
    """Whisper's words as Units. A word's end is stretched into the pause after it (up to
    0.3 s) so cutting there doesn't clip its last sound."""
    units = [Unit(offset + s, offset + e, t) for s, e, t in words]
    for a, b in itertools.pairwise(units):
        a.end = max(a.end, min(b.start, a.end + 0.3))
    if units:
        units[-1].end += 0.3
    return units


def make_clip(
    url: str,
    start: str | float | None = None,
    length: float = DEFAULT_LEN,
    *,
    language: str = "English",
    isolate: bool = True,
    transcript: str = "captions",
    device: str = "cpu",
    settings: dict | None = None,
    progress: Progress | None = None,
) -> Clip:
    """Download (or reuse), cut, clean up, transcribe and save a reference clip. See the module docstring."""
    url = _clean_url(url)
    if start is None or str(start).strip() == "":
        start_s = start_from_url(url) or 0.0
    else:
        start_s = parse_time(start)
    length = min(MAX_LEN, max(MIN_LEN, float(length or DEFAULT_LEN)))
    if transcript not in TRANSCRIPTS:
        transcript = "captions"

    video = download(url, progress)
    if video.duration and start_s >= video.duration - 1:
        raise ClipError(f"The video is only {format_time(video.duration)} long. Pick an earlier start time.")

    stt_settings = {**(settings or {}), "device": device}
    # Cut a little more than asked, so the clip can then move to whole words.
    ws = max(0.0, start_s - PAD_S)
    we = start_s + length + PAD_S
    if video.duration:
        we = min(we, video.duration)
    if progress:
        progress(0.6, "Cutting the clip")
    wav = cut(video.path, ws, we)
    if wav.shape[1] < SEP_SR:
        raise ClipError(f"There's no audio at {format_time(start_s)}. Is the video that long?")
    we = ws + wav.shape[1] / SEP_SR  # shorter than asked when the video ends sooner

    mix = wav.mean(0)
    if isolate:
        if progress:
            progress(0.65, "Removing music and background noise" + ("" if _demucs else " (the first time takes longer)"))
        mono = remove_background(wav, device)
    else:
        mono = mix

    # The transcript has to be in the language that's spoken. Listen for it first: YouTube's own
    # labels can be wrong (or be for a dubbed track), so they only count when listening isn't possible.
    units, source, note = [], "", ""
    spoken = None
    if transcript != "none":
        if progress:
            progress(0.75, "Listening for the language")
        try:
            spoken = transcribe.detect(mono, SEP_SR, stt_settings)
        except ValueError as exc:
            note = str(exc)
    spoken = (spoken or spoken_language(video.info) or "").split("-")[0] or None
    if spoken:
        language = CODE_LANGS.get(spoken, "Auto")

    if transcript == "captions":
        if progress:
            progress(0.8, "Reading the captions")
        units, source = captions(video, spoken or LANG_CODES.get(language))
    s, e, chosen = snap(units, start_s, start_s + length)
    if chosen and not script_fits(caption_text(chosen), spoken or LANG_CODES.get(language)):
        chosen = []  # captions in another language than the one spoken
    if not chosen and transcript != "none" and not note:
        if progress:
            progress(0.85, "Writing down what's said (speech-to-text)")
        try:
            words, heard = transcribe.words(mono, SEP_SR, stt_settings, language=spoken)
        except ValueError as exc:
            words, heard, note = [], None, str(exc)
        s, e, chosen = snap(_speech_units(words, ws), start_s, start_s + length)
        source = "speech"
        if heard and not spoken:
            language = CODE_LANGS.get(heard.split("-")[0], "Auto")
    if not chosen:
        s, e, source = start_s, start_s + length, ""
    s, e = max(s, ws), min(e, we)

    a, b = round((s - ws) * SEP_SR), round((e - ws) * SEP_SR)
    trimmed = audio.trim_silence(mono[a:b], SEP_SR)
    if audio.is_silent(trimmed, 0.01) or len(trimmed) < 2 * SEP_SR or _rms(mono[a:b]) < 0.05 * _rms(mix[a:b]):
        raise ClipError(
            f"There's hardly any voice between {format_time(s)} and {format_time(e)}. Try another start time."
        )

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^\w-]", "_", video.id)[:40]
    out = CLIPS_DIR / f"{safe_id}-{int(s * 1000)}-{int(e * 1000)}{'-voice' if isolate else ''}.wav"
    sf.write(str(out), audio.normalize_peak(trimmed), SEP_SR, subtype="PCM_16")
    _prune(CLIPS_DIR, KEEP_CLIPS)
    if progress:
        progress(1.0, "Done")
    return Clip(
        path=out,
        seconds=len(trimmed) / SEP_SR,
        start=s,
        end=e,
        title=video.title,
        transcript=caption_text(chosen),
        source=source if chosen else "",
        language=language,
        note=note,
    )


# ------------------------------------------------------------- maintenance


def update_downloader() -> str:
    """Update yt-dlp, which needs to keep up with changes on YouTube's side."""
    uv = ROOT / ".tools" / "uv" / ("uv.exe" if os.name == "nt" else "uv")
    pkgs = ["--upgrade-package", "yt-dlp", "--upgrade-package", "yt-dlp-ejs", "yt-dlp[default,deno]"]
    if uv.exists():
        cmd = [str(uv), "pip", "install", "--python", sys.executable, *pkgs]
    elif shutil.which("uv"):
        cmd = ["uv", "pip", "install", "--python", sys.executable, *pkgs]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp[default,deno]"]
    env = {**os.environ, "UV_CACHE_DIR": str(CACHE_DIR / "uv"), "UV_LINK_MODE": "copy"}
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, check=False)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        last = (r.stderr or r.stdout).strip().splitlines()
        raise ClipError(f"Updating yt-dlp failed: {last[-1] if last else 'unknown error'}. Run install.bat again instead.")
    # yt-dlp is plain Python, so dropping the old modules is enough for the next download to use the new one.
    for name in [m for m in sys.modules if m == "yt_dlp" or m.startswith(("yt_dlp.", "yt_dlp_ejs"))]:
        del sys.modules[name]
    importlib.invalidate_caches()
    from importlib.metadata import version

    return f"The YouTube downloader is up to date (yt-dlp {version('yt-dlp')})."


def setup(log=print) -> None:
    """Download the Demucs and Whisper models and check the other tools, for install.bat."""
    log(f"ffmpeg: {ffmpeg_exe()}")
    from importlib.metadata import version

    try:
        from deno import find_deno_bin

        deno = find_deno_bin()
    except (ImportError, FileNotFoundError):
        deno = None
    log(f"yt-dlp {version('yt-dlp')}, Deno: {deno or 'NOT FOUND'}")
    if not deno:
        log("Without Deno, YouTube downloads may fail. Run install.bat again to add it.")
    model = _load_demucs()
    log(f"Music remover ready ({DEMUCS_MODEL}, separates: {', '.join(model.sources)})")
    from .config import load_settings

    try:
        transcribe.download(load_settings(), log)
    except Exception as exc:  # noqa: BLE001
        raise ClipError(f"Couldn't download the speech-to-text model: {exc}") from None
    log("Speech-to-text ready")


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a voice clip from a YouTube video")
    ap.add_argument("url", nargs="?", help="video link")
    ap.add_argument("--start", default=None, help="where the clip starts, e.g. 1:23 (default: the link's t=, else 0)")
    ap.add_argument("--length", type=float, default=DEFAULT_LEN, help=f"clip length in seconds ({MIN_LEN}-{MAX_LEN})")
    ap.add_argument("--language", default="English", help="language of the captions to use for the transcript")
    ap.add_argument("--keep-music", action="store_true", help="don't remove music and background noise")
    ap.add_argument("--transcript", choices=TRANSCRIPTS, default="captions", help="captions (else speech-to-text), speech or none")
    ap.add_argument("--device", default=None, help="cuda:0 or cpu (default from settings.json)")
    ap.add_argument("--setup", action="store_true", help="download the music remover model and check the tools")
    ap.add_argument("--update", action="store_true", help="update yt-dlp")
    args = ap.parse_args()

    setup_environment()
    try:
        if args.update:
            print(update_downloader())
        if args.setup:
            setup()
        if args.url:
            from .config import load_settings

            t0 = time.time()
            clip = make_clip(
                args.url,
                args.start,
                args.length,
                language=args.language,
                isolate=not args.keep_music,
                transcript=args.transcript,
                device=args.device or load_settings().get("device", "cpu"),
                settings=load_settings(),
                progress=lambda f, desc: print(f"  {int(f * 100):3d}%  {desc}"),
            )
            print(f"\n{clip.title}: {clip.seconds:.1f}s from {format_time(clip.start)} to {format_time(clip.end)}")
            print(f"Saved to {clip.path} ({time.time() - t0:.0f}s)")
            print(f"Transcript ({clip.source or 'none'}, {clip.language}): {clip.transcript}")
            if clip.note:
                print(clip.note)
        elif not (args.setup or args.update):
            ap.print_help()
    except ClipError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
