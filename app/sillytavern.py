"""Fill in SillyTavern's text-to-speech settings, so nobody has to copy the voice list by hand.

SillyTavern keeps its settings in one JSON file per user profile:

    <SillyTavern>\\data\\default-user\\settings.json   SillyTavern 1.12 and later (config.yaml's
                                                     dataRoot can move the data folder)
    <SillyTavern>\\public\\settings.json               older versions

The TTS extension's part of that file looks like this:

    "extension_settings": {"tts": {
        "enabled": true,
        "currentProvider": "OpenAI Compatible",
        "periodic_auto_generation": true,   <- "Narrate by paragraphs (when streaming)"
        "narrate_by_paragraphs": true,      <- "Narrate by paragraphs (when not streaming)"
        "OpenAI Compatible": {
            "provider_endpoint": "http://127.0.0.1:8890/v1/audio/speech",
            "model": "qwen3-tts",
            "available_voices": ["grumpy-wizard", "narrator"],
            "speed": 1,
            "voiceMap": {"Seraphina": "narrator"}
        }
    }}

SillyTavern refuses a provider section with keys it doesn't know, so only those keys are
ever written there. Its page reads the file when it loads and writes all of it back when
something changes, so after a change the page has to be reloaded (F5) to see it.

    python -m app.sillytavern "D:\\SillyTavern"      fill in the settings from the command line
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT

PROVIDER = "OpenAI Compatible"
MODEL_NAME = "qwen3-tts"
# The copy of each settings file as it was just before the last change made here.
BACKUP_SUFFIX = ".qwen3-tts-backup"
DEFAULT_PORT = 8000
WINDOWS = os.name == "nt"


class SillyTavernError(ValueError):
    pass


@dataclass
class Result:
    root: Path
    changed: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)  # no OpenAI Compatible section to update yet
    running: bool = False

    @property
    def files(self) -> list[Path]:
        return self.changed + self.unchanged + self.skipped


# ---------------------------------------------------------------- finding SillyTavern


def clean_path(text) -> str:
    """A path as typed or pasted. Windows' 'Copy as path' puts quotes around it."""
    return str(text or "").strip().strip("\"'").strip()


def is_install(folder: Path) -> bool:
    try:
        return (folder / "server.js").is_file() and (folder / "public").is_dir()
    except OSError:
        return False


def _children(folder: Path):
    """Subfolders whose name starts with SillyTavern, e.g. SillyTavern-release from the ZIP."""
    try:
        return sorted(p for p in folder.iterdir() if p.is_dir() and p.name.lower().startswith("sillytavern"))
    except OSError:
        return []


def find_install(path) -> Path:
    """SillyTavern's own folder, from the folder (or a file in it) the user picked.

    Also accepts the SillyTavern-Launcher folder, a folder holding the unzipped download,
    or a folder inside SillyTavern such as data\\default-user.
    """
    raw = clean_path(path)
    if not raw:
        raise SillyTavernError("Pick SillyTavern's folder first: click Browse, or paste the folder's path.")
    p = Path(raw).expanduser()
    if not p.exists():
        raise SillyTavernError(f"Couldn't find the folder {p}. Check the path, or click Browse to pick it.")
    if p.is_file():
        p = p.parent  # e.g. Start.bat
    candidates = [p, p / "SillyTavern", *_children(p)]
    candidates += [c / "SillyTavern" for c in _children(p)] + [g for c in _children(p) for g in _children(c)]
    candidates += list(p.parents)[:4]
    for c in candidates:
        if is_install(c):
            return c
    raise SillyTavernError(
        f"{p} doesn't look like SillyTavern's folder. Pick the folder that has Start.bat and server.js in it."
    )


def _fixed_drives() -> list[Path]:
    """Windows' hard drives (not card readers or DVD drives, which may ask for a disk)."""
    if not WINDOWS:
        return []
    try:
        import ctypes

        k32 = ctypes.windll.kernel32
        mask = k32.GetLogicalDrives()
        drives = [f"{chr(65 + i)}:\\" for i in range(26) if mask >> i & 1]
        return [Path(d) for d in drives if k32.GetDriveTypeW(d) == 3]  # DRIVE_FIXED
    except Exception:
        return []


def guess_installs() -> list[Path]:
    """SillyTavern folders in the usual places: next to this app, in your user folder
    (Documents, Desktop, Downloads) and at the top of each hard drive."""
    home = Path.home()
    bases = [ROOT.parent, home, home / "Documents", home / "Desktop", home / "Downloads", *_fixed_drives()]
    found: list[Path] = []
    for base in dict.fromkeys(bases):
        for c in [base / "SillyTavern", base / "SillyTavern-Launcher" / "SillyTavern", *_children(base)]:
            for cand in [c, *(_children(c) if c.is_dir() else [])]:
                if is_install(cand) and cand not in found:
                    found.append(cand)
    return found


# ---------------------------------------------------------------- its settings


def _config_value(root: Path, key: str) -> str | None:
    """A top-level value from SillyTavern's config.yaml (no YAML library needed for these two)."""
    try:
        text = (root / "config.yaml").read_text(encoding="utf-8-sig")
    except OSError:
        return None
    m = re.search(rf"^{key}:[ \t]*(.*?)[ \t]*(?:#.*)?$", text, re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip().strip("\"'") or None


def data_root(root: Path) -> Path:
    value = _config_value(root, "dataRoot") or "./data"
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


def port(root: Path) -> int:
    try:
        return int(_config_value(root, "port") or DEFAULT_PORT)
    except ValueError:
        return DEFAULT_PORT


def settings_files(root: Path) -> list[Path]:
    """Every user profile's settings.json (usually just default-user's)."""
    files: list[Path] = []
    data = data_root(root)
    try:
        for d in sorted(data.iterdir()):
            # data\_storage, data\_uploads and the like aren't user profiles.
            if d.is_dir() and not d.name.startswith(("_", ".")) and (d / "settings.json").is_file():
                files.append(d / "settings.json")
    except OSError:
        pass
    legacy = root / "public" / "settings.json"
    if not files and legacy.is_file():
        files.append(legacy)
    return files


def profile(path: Path) -> str:
    return "settings" if path.parent.name == "public" else path.parent.name


def is_running(root: Path) -> bool:
    """Is something answering on SillyTavern's port on this PC?"""
    for host in ("127.0.0.1", "::1"):
        try:
            with socket.create_connection((host, port(root)), timeout=0.3):
                return True
        except OSError:
            continue
    return False


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise SillyTavernError(f"Couldn't read {path}: {exc}") from None
    if not isinstance(data, dict):
        raise SillyTavernError(f"{path} isn't a SillyTavern settings file.")
    return data


def _tts(data: dict, path: Path, create: bool) -> dict | None:
    ext = data.get("extension_settings")
    if ext is None and create:
        ext = data["extension_settings"] = {}
    if ext is None:
        return None
    if not isinstance(ext, dict):
        raise SillyTavernError(f"{path} isn't a SillyTavern settings file.")
    tts = ext.get("tts")
    if not isinstance(tts, dict):
        if not create:
            return None
        tts = ext["tts"] = {}
    return tts


def _write(path: Path, data: dict) -> None:
    """Keep a copy of the old file, then swap the new one in whole (never a half-written file)."""
    shutil.copy2(path, path.with_name(path.name + BACKUP_SUFFIX))
    tmp = path.with_name(path.name + ".qwen3-tts-tmp")
    # The same layout SillyTavern writes: 4 spaces, real UTF-8, \n line ends.
    tmp.write_bytes(json.dumps(data, indent=4, ensure_ascii=False).encode("utf-8"))
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # e.g. a virus scanner looking at it right then
            if attempt == 4:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.2)


def current_voices(path: Path) -> list[str] | None:
    """The voice list SillyTavern has for the voice server, or None if it has none set up."""
    try:
        tts = _tts(_read(path), path, create=False)
    except SillyTavernError:
        return None
    section = (tts or {}).get(PROVIDER)
    voices = section.get("available_voices") if isinstance(section, dict) else None
    return [str(v).strip() for v in voices if str(v).strip()] if isinstance(voices, list) else None


def apply(
    root: Path,
    voices: list[str],
    *,
    endpoint: str | None = None,
    model: str = MODEL_NAME,
    paragraphs: bool | None = None,
) -> Result:
    """Update SillyTavern's settings files.

    With an endpoint (the Fill in button): turn TTS on, pick the OpenAI Compatible provider,
    point it at the voice server and fill in the voices, and set both "Narrate by paragraphs"
    options to `paragraphs` (when given).
    Without one (keeping the list up to date): only the voice list, and only where the
    OpenAI Compatible provider is already set up.
    A file is written (with a backup) only when something in it actually changes.
    """
    files = settings_files(root)
    if not files:
        raise SillyTavernError(
            f"SillyTavern in {root} has no settings file yet. Start SillyTavern once (so it makes one), then try again."
        )
    result = Result(root=root, running=is_running(root))
    voices = [str(v) for v in voices]
    for path in files:
        data = _read(path)
        tts = _tts(data, path, create=endpoint is not None)
        before = json.dumps(tts, sort_keys=True)
        section = tts.get(PROVIDER) if tts is not None else None
        if endpoint is not None:
            tts["enabled"] = True
            tts["currentProvider"] = PROVIDER
            if paragraphs is not None:
                tts["periodic_auto_generation"] = bool(paragraphs)
                tts["narrate_by_paragraphs"] = bool(paragraphs)
            if not isinstance(section, dict):
                section = tts[PROVIDER] = {}
            section["provider_endpoint"] = endpoint
            section["model"] = model
        if not isinstance(section, dict):
            result.skipped.append(path)
            continue
        section["available_voices"] = voices
        if json.dumps(tts, sort_keys=True) == before:
            result.unchanged.append(path)
            continue
        _write(path, data)
        result.changed.append(path)
    return result


# ---------------------------------------------------------------- folder picker

# Prints "PICKED:" and the path in base64, so no console code page can garble it.
# The hidden TopMost owner window brings the dialog up in front of the browser.
_PICKER_PS = r"""
Add-Type -AssemblyName System.Windows.Forms
$owner = New-Object System.Windows.Forms.Form -Property @{TopMost = $true; ShowInTaskbar = $false}
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Pick SillyTavern''s folder (the one with Start.bat in it)'
$dialog.ShowNewFolderButton = $false
if ($env:QWEN_TTS_FOLDER) { $dialog.SelectedPath = $env:QWEN_TTS_FOLDER }
if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
    'PICKED:' + [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($dialog.SelectedPath))
}
$owner.Dispose()
"""

_PICKER_TK = """
import sys, tkinter, tkinter.filedialog
root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
path = tkinter.filedialog.askdirectory(title="Pick SillyTavern's folder", initialdir=sys.argv[1] or None, mustexist=True)
if path:
    print("PICKED:" + path)
"""


def browse(current: str = "") -> str | None:
    """Open a folder picker on this PC and return the chosen folder, or None when it's cancelled."""
    folder = clean_path(current)
    if not (folder and Path(folder).is_dir()):
        folder = ""
    if WINDOWS:
        script = base64.b64encode(_PICKER_PS.encode("utf-16-le")).decode("ascii")
        cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-STA", "-EncodedCommand", script]
        env = {**os.environ, "QWEN_TTS_FOLDER": folder}
        extra = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    else:
        cmd = [sys.executable, "-I", "-X", "utf8", "-c", _PICKER_TK, folder]
        env, extra = None, {}
    try:
        r = subprocess.run(cmd, capture_output=True, env=env, timeout=900, check=False, **extra)
    except subprocess.TimeoutExpired:
        return None
    except OSError as exc:
        r = subprocess.CompletedProcess(cmd, 1, b"", str(exc).encode())
    lines = r.stdout.decode("utf-8", "replace").splitlines()
    picked = [ln.strip().removeprefix("PICKED:") for ln in lines if ln.strip().startswith("PICKED:")]
    try:
        if picked and WINDOWS:
            picked = [base64.b64decode(picked[-1], validate=True).decode("utf-8")]
    except ValueError:  # garbled: treat it like any other failure
        picked, r.returncode = [], r.returncode or 1
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", "replace").strip().splitlines()
        print(f"[sillytavern] The folder picker failed: {err[-1] if err else r.returncode}")
        raise SillyTavernError(
            "Couldn't open the folder picker here. Paste SillyTavern's folder into the box instead: in File Explorer, "
            "open the folder, click the address bar and copy it."
        )
    return picked[-1] if picked else None


# ---------------------------------------------------------------- keeping it up to date


class AutoSync:
    """Keeps SillyTavern's voice list in step with the saved voices, once the Fill in button
    has saved SillyTavern's folder (and unless "keep it up to date" was turned off).

    Each check costs a folder listing and a look at the files' timestamps; a settings file
    is only read when the voices or the file changed since the last check, and written only
    when its voice list differs. That also puts the list back when a SillyTavern page that
    was open from before saves its old copy over it.
    """

    def __init__(self, settings: dict, library, interval: float = 10):
        self.settings = settings
        self.library = library
        self.interval = interval
        self.note = ""
        self._seen = None
        self._error = ""

    def _stamp(self, root: Path, voices: list[str]):
        return tuple(voices), tuple((str(f), f.stat().st_mtime_ns) for f in settings_files(root))

    def tick(self) -> Result | None:
        folder = clean_path(self.settings.get("sillytavern_dir"))
        if not folder or not self.settings.get("sillytavern_auto_sync", True):
            self._seen = None
            return None
        root = find_install(folder)
        voices = self.library.ids()
        if self._stamp(root, voices) == self._seen:
            return None
        result = apply(root, voices)
        self._seen = self._stamp(root, voices)
        if result.changed:
            n = len(voices)
            self.note = f"Updated SillyTavern's voice list at {time.strftime('%H:%M')} ({n} voice{'s' if n != 1 else ''})."
            print(f"[sillytavern] {self.note}" + (" Reload SillyTavern's page (F5) to see it." if result.running else ""))
        return result

    def run(self) -> None:
        while True:
            try:
                self.tick()
                self._error = ""
            except Exception as exc:  # e.g. the folder was moved; say so once, keep checking
                if str(exc) != self._error:
                    self._error = str(exc)
                    print(f"[sillytavern] Couldn't update SillyTavern's voice list: {exc}")
            time.sleep(self.interval)


# ---------------------------------------------------------------- messages


def describe(result: Result, voices: list[str]) -> str:
    """What the Fill in button did, for people."""
    names = ", ".join(profile(p) for p in result.files)
    n = len(voices)
    count = f"{n} voice{'s' if n != 1 else ''}" if n else "no voices yet (make one in the Studio, and it's added by itself)"
    if result.changed:
        msg = f"Done. SillyTavern ({names}) now speaks through this voice server, with {count}."
        backups = ", ".join(str(p.with_name(p.name + BACKUP_SUFFIX)) for p in result.changed)
        msg += f" The old settings were saved as {backups}."
    else:
        msg = f"SillyTavern ({names}) was already set up, with {count}. Nothing needed changing."
    if result.changed and result.running:
        msg += (
            "\n\n**SillyTavern is running: reload its page in your browser now (press F5)** so it picks up the change. "
            "Until you do, a change you make there can put the old settings back (then just click Fill in again)."
        )
    elif result.changed:
        msg += " Start SillyTavern as usual and it uses them."
    msg += (
        "\n\nLast step, in SillyTavern: under **Extensions → TTS**, pick a voice for each character in the list at the bottom."
    )
    return msg


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .config import load_settings, setup_environment
    from .voices import VoiceLibrary

    ap = argparse.ArgumentParser(description="Fill in SillyTavern's TTS settings for the voice server.")
    ap.add_argument("folder", nargs="?", help="SillyTavern's folder (default: the one saved, or found)")
    ap.add_argument("--no-paragraphs", action="store_true", help="speak each reply in one go instead of a paragraph at a time")
    args = ap.parse_args(argv)

    setup_environment()
    settings = load_settings()
    folder = args.folder or settings.get("sillytavern_dir") or next(iter(map(str, guess_installs())), "")
    port_ = int(settings.get("server_port", 8890))
    try:
        root = find_install(folder)
        voices = VoiceLibrary().ids()
        result = apply(
            root, voices, endpoint=f"http://127.0.0.1:{port_}/v1/audio/speech", paragraphs=not args.no_paragraphs
        )
    except (SillyTavernError, OSError) as exc:
        print(f"!! {exc}")
        return 1
    from .config import save_settings

    saved = load_settings()
    saved["sillytavern_dir"] = str(root)
    save_settings(saved)
    print(describe(result, voices).replace("**", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
