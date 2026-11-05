import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_release", ROOT / "scripts" / "build_release.py")
build_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_release)


def test_zip_holds_what_people_need(tmp_path):
    zip_path = build_release.build(tmp_path)
    version = (ROOT / "VERSION").read_text().strip()
    assert zip_path.name == f"Qwen3-TTS-Studio-{version}.zip"
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        assert all(n.startswith("Qwen3-TTS-Studio/") for n in names)
        rel = {n.split("/", 1)[1] for n in names}
        assert set(build_release.REQUIRED) <= rel
        assert not any(n.startswith(("tests/", ".github/")) or "__pycache__" in n for n in rel)
        assert "scripts/build_release.py" not in rel
        # cmd.exe and Windows PowerShell need Windows line endings.
        for n in rel:
            if n.endswith((".bat", ".ps1")):
                data = z.read(f"Qwen3-TTS-Studio/{n}")
                assert data.count(b"\n") == data.count(b"\r\n") > 0, n
        assert z.read("Qwen3-TTS-Studio/app/icon.ico")[:4] == b"\x00\x00\x01\x00"


def test_launchers_are_ascii():
    # Batch files are read in the console's code page and the installer by Windows
    # PowerShell 5.1 as ANSI, so they must stay plain ASCII.
    for path in [*ROOT.glob("*.bat"), ROOT / "scripts" / "install.ps1"]:
        path.read_bytes().decode("ascii")


def test_release_notes_use_this_versions_changelog():
    version = (ROOT / "VERSION").read_text().strip()
    notes = build_release.release_notes(version)
    assert f"Qwen3-TTS-Studio-{version}.zip" in notes
    assert f"## What's in {version}" in notes
    assert build_release.changelog_section(version) in notes


def test_changelog_starts_with_the_current_version():
    version = (ROOT / "VERSION").read_text().strip()
    first = next(l for l in (ROOT / "CHANGELOG.md").read_text().splitlines() if l.startswith("## "))
    assert first.split()[1] == version
