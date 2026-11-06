"""The Qwen3-TTS Studio menu, opened by Qwen3-TTS.bat.

It starts the Voice Studio, the voice server and Speak, each in its own window,
shows which ones are running, and opens a running one's page in the browser
instead of starting a second copy. Installing and updating are left to
Qwen3-TTS.bat (this exits with INSTALL), because the installer can't replace
.venv while this Python runs from it.

Only light modules are imported here so the menu opens instantly.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import HOMEPAGE, RELEASES_URL, REPO, __version__
from .config import OUTPUTS_DIR, ROOT, load_settings

# Exit code that tells Qwen3-TTS.bat to run the installer and then open the menu again.
INSTALL = 10
LATEST_RELEASE_API = f"https://api.github.com/repos/{REPO}/releases/latest"
# A program we started counts as "starting" this long while its page isn't up yet.
STARTING_S = 600
REFRESH_S = 1.5


@dataclass(frozen=True)
class Program:
    key: str
    name: str
    about: str
    bat: str
    module: str
    port_setting: str
    default_port: int
    page: str = "/"


PROGRAMS = (
    Program("1", "Voice Studio", "make voices: describe one, or clone one from a clip",
            "start_studio.bat", "app.studio", "studio_port", 7860),
    Program("2", "Voice Server", "lets SillyTavern and other apps speak in your voices",
            "start_server.bat", "app.server", "server_port", 8890, "/ui"),
    Program("3", "Speak", "type something and hear it in one of your voices",
            "start_speak.bat", "app.speak", "speak_port", 7861),
)


def address(program: Program, settings: dict) -> tuple[str, int]:
    """Where the program listens. Only the voice server can listen beyond this PC."""
    host = "127.0.0.1"
    if program.port_setting == "server_port":
        host = settings.get("server_host") or host
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
    return host, int(settings.get(program.port_setting) or program.default_port)


def url(program: Program, settings: dict) -> str:
    host, port = address(program, settings)
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port}{program.page}"


def is_listening(host: str, port: int, timeout: float = 0.3) -> bool:
    # Windows can take a second to refuse a connection, so don't wait for that.
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def version_parts(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", version or "")[:3])


def is_newer(latest: str, current: str) -> bool:
    cur = version_parts(current)
    return bool(cur) and version_parts(latest) > cur


def latest_release(timeout: float = 5.0) -> str | None:
    """The newest released version on GitHub, or None if it can't be found out."""
    req = urllib.request.Request(
        LATEST_RELEASE_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "qwen3-tts-studio"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            tag = json.load(resp).get("tag_name") or ""
    except Exception:
        return None
    return tag.lstrip("vV") or None


def start_program(program: Program) -> subprocess.Popen:
    """Run the program in a window of its own, so this menu stays usable."""
    if os.name == "nt":
        return subprocess.Popen(
            ["cmd.exe", "/c", program.bat], cwd=ROOT, creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    return subprocess.Popen([sys.executable, "-m", program.module], cwd=ROOT, start_new_session=True)


def open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(path)  # noqa: S606 - opens File Explorer
    else:
        webbrowser.open(path.as_uri())


class Menu:
    def __init__(self, settings: dict | None = None):
        self.settings = settings if settings is not None else load_settings()
        self.started: dict[str, tuple[subprocess.Popen, float]] = {}
        self.status: dict[str, str] = {p.key: "" for p in PROGRAMS}
        self.message = ""
        self.update: str | None = None

    # ------------------------------------------------------------------ status
    def refresh(self) -> None:
        """Find out which programs are running ("running", "starting" or "")."""
        with ThreadPoolExecutor(len(PROGRAMS)) as pool:
            up = dict(zip(
                (p.key for p in PROGRAMS),
                pool.map(lambda p: is_listening(*address(p, self.settings)), PROGRAMS),
            ))
        now = time.monotonic()
        for p in PROGRAMS:
            if up[p.key]:
                self.status[p.key] = "running"
                continue
            proc, since = self.started.get(p.key, (None, 0.0))
            starting = proc is not None and proc.poll() is None and now - since < STARTING_S
            self.status[p.key] = "starting" if starting else ""

    def check_for_updates(self) -> None:
        if not self.settings.get("check_for_updates", True):
            return

        def check():
            latest = latest_release()
            if latest and is_newer(latest, __version__):
                self.update = latest

        threading.Thread(target=check, daemon=True).start()

    # ------------------------------------------------------------------ screen
    def render(self) -> str:
        lines = ["", f"  Qwen3-TTS Studio {__version__}", f"  {ROOT}", ""]
        for p in PROGRAMS:
            lines.append(f"    {p.key}  {p.name:<14} {p.about}")
            state = self.status[p.key]
            if state == "running":
                lines.append(f"       {'':<14} running at {url(p, self.settings)}")
            elif state == "starting":
                lines.append(f"       {'':<14} starting...")
        lines += [
            "",
            "    4  Open the folder with the audio you made",
            "    5  Open the guide",
            "    6  Install, update or repair",
            "    Q  Quit",
            "",
            "  Each program opens in its own window and in your browser. Close its window to stop it.",
        ]
        if self.update:
            lines += ["", f"  Version {self.update} is out. Press U to open the download page."]
        if self.message:
            lines += ["", textwrap.fill(self.message, 88, initial_indent="  ", subsequent_indent="  ")]
        lines += ["", "  Press a number: "]
        return "\n".join(lines)

    # ------------------------------------------------------------------ keys
    def handle(self, key: str) -> int | None:
        """Act on one key press. Returns an exit code when the menu should close."""
        key = (key or "").strip().lower()[:1]
        self.message = ""
        program = next((p for p in PROGRAMS if p.key == key), None)
        if program:
            self.open(program)
        elif key == "4":
            open_folder(OUTPUTS_DIR)
            self.message = f"Opened {OUTPUTS_DIR}"
        elif key == "5":
            webbrowser.open(HOMEPAGE + "#readme")
            self.message = "Opened the guide in your browser."
        elif key == "6":
            running = [p.name for p in PROGRAMS if self.status[p.key]]
            if running:
                self.message = (
                    f"Close {' and '.join(running)} first (close {'its' if len(running) == 1 else 'their'} "
                    "window): the installer can't update files that are in use."
                )
            else:
                return INSTALL
        elif key == "u" and self.update:
            webbrowser.open(RELEASES_URL)
            self.message = (
                "Opened the download page. Download the ZIP, extract it, run Qwen3-TTS.bat from it "
                "and press Enter to update this copy."
            )
        elif key in ("q", "\x1b"):
            return 0
        return None

    def open(self, program: Program) -> None:
        state = self.status[program.key]
        if state == "running":
            webbrowser.open(url(program, self.settings))
            self.message = f"{program.name} is already running, so its page was opened in your browser."
        elif state == "starting":
            self.message = f"{program.name} is still starting. Its window shows how it's going."
        else:
            try:
                self.started[program.key] = (start_program(program), time.monotonic())
            except OSError as exc:
                self.message = f"Couldn't start {program.name}: {exc}"
                return
            self.status[program.key] = "starting"
            self.message = (
                f"Starting {program.name} in its own window. "
                "Its page opens in your browser when it's ready (the first time can take a minute)."
            )


# ---------------------------------------------------------------------- console
def draw(screen: str) -> None:
    os.system("cls" if os.name == "nt" else "clear")  # noqa: S605
    print(screen, end="", flush=True)


def wait_for_key(menu: Menu, screen: str) -> str:
    """Wait for a key press, redrawing when a program starts or stops."""
    if os.name != "nt":
        return input()
    import msvcrt

    next_check = time.monotonic() + REFRESH_S
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # arrow and function keys arrive as two characters
                msvcrt.getwch()
                continue
            return ch
        time.sleep(0.05)
        if time.monotonic() >= next_check:
            menu.refresh()
            new = menu.render()
            if new != screen:
                draw(new)
                screen = new
            next_check = time.monotonic() + REFRESH_S


def main() -> int:
    menu = Menu()
    menu.check_for_updates()
    try:
        while True:
            menu.refresh()
            screen = menu.render()
            draw(screen)
            code = menu.handle(wait_for_key(menu, screen))
            if code is not None:
                if code == 0 and any(menu.status.values()):
                    print("\n\n  The programs you started keep running in their own windows.\n")
                return code
            menu.settings = load_settings()
    except (KeyboardInterrupt, EOFError):
        return 0


if __name__ == "__main__":
    sys.exit(main())
