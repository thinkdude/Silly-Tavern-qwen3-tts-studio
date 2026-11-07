"""Build the release ZIP that people download from GitHub.

    python scripts/build_release.py [--out dist] [--notes dist/notes.md]

The ZIP holds one folder, Qwen3-TTS-Studio, with only what's needed to install and
run (no tests or GitHub files), and batch and PowerShell files get Windows line
endings whatever this checkout has. --notes writes the release description: how to
get started, then this version's part of CHANGELOG.md.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = "Qwen3-TTS-Studio"


def _repo(root: Path = ROOT) -> str:
    """owner/name on GitHub: the repository the Release workflow runs in, else the app's own."""
    if os.environ.get("GITHUB_REPOSITORY"):
        return os.environ["GITHUB_REPOSITORY"]
    m = re.search(r'^REPO = "([^"]+)"', (root / "app" / "__init__.py").read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else ""


REPO = _repo()
# Files in the repo that aren't needed to install or run.
LEAVE_OUT = ("tests/", ".github/", ".gitignore", ".gitattributes", "scripts/build_release.py")
WINDOWS_LINE_ENDINGS = (".bat", ".cmd", ".ps1")
REQUIRED = (
    "Qwen3-TTS.bat",
    "install.bat",
    "start_studio.bat",
    "start_server.bat",
    "start_speak.bat",
    "scripts/install.ps1",
    "scripts/probe_python.py",
    "app/menu.py",
    "app/icon.ico",
    "requirements.txt",
    "constraints.txt",
    "VERSION",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
)


def version(root: Path = ROOT) -> str:
    return (root / "VERSION").read_text(encoding="utf-8").strip()


def repo_files(root: Path = ROOT) -> list[str]:
    """Tracked files plus new ones that aren't ignored, so a build before committing works too."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root, capture_output=True, check=True,
    ).stdout
    return sorted({f for f in out.decode("utf-8").split("\0") if f})


def release_files(files: list[str], root: Path = ROOT) -> list[str]:
    keep = [
        f for f in files
        if not any(f == x or f.startswith(x) for x in LEAVE_OUT) and (root / f).is_file()
    ]
    missing = [f for f in REQUIRED if f not in keep]
    if missing:
        raise SystemExit(f"These files are missing from the release: {', '.join(missing)}")
    return keep


def file_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() in WINDOWS_LINE_ENDINGS:
        data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return data


def build(out_dir: Path, root: Path = ROOT, files: list[str] | None = None) -> Path:
    files = release_files(repo_files(root) if files is None else files, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{TOP}-{version(root)}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            z.writestr(f"{TOP}/{f}", file_bytes(root / f))
    return zip_path


def changelog_section(ver: str, root: Path = ROOT) -> str:
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(ver)}\b[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m:
        raise SystemExit(f"CHANGELOG.md has no section for {ver} (a line starting with '## {ver}').")
    return m.group(1).strip()


def release_notes(ver: str, root: Path = ROOT) -> str:
    zip_name = f"{TOP}-{ver}.zip"
    return (
        "## Get started\n\n"
        f"1. Download **{zip_name}** under **Assets** below.\n"
        "2. Right-click it and choose **Extract All...**.\n"
        "3. Open the extracted folder and double-click **Qwen3-TTS.bat**. If Windows says "
        "\"Windows protected your PC\", click **More info**, then **Run anyway**.\n"
        "4. The first time, it asks where to install and downloads everything it needs "
        "(about 24 GB). Then a menu starts the Voice Studio, the Voice Server and Speak.\n\n"
        "Already installed? Do the same with the new ZIP and press Enter when it offers to "
        "update your existing folder. Your voices, settings and models are kept.\n\n"
        f"Needs Windows 10 or 11 and an NVIDIA graphics card. The full guide is in the "
        f"[README](https://github.com/{REPO}#readme).\n\n"
        f"## What's in {ver}\n\n" + changelog_section(ver, root) + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Qwen3-TTS Studio release ZIP")
    ap.add_argument("--out", type=Path, default=ROOT / "dist", help="folder for the ZIP (default: dist)")
    ap.add_argument("--notes", type=Path, help="also write the release description to this file")
    args = ap.parse_args()

    zip_path = build(args.out)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(zip_path) as z:
        count = len(z.namelist())
    print(f"{zip_path} ({zip_path.stat().st_size / 1024:.0f} KB, {count} files)\nSHA-256 {digest}")
    if args.notes:
        args.notes.parent.mkdir(parents=True, exist_ok=True)
        args.notes.write_text(release_notes(version()), encoding="utf-8")
        print(f"Release notes: {args.notes}")


if __name__ == "__main__":
    main()
