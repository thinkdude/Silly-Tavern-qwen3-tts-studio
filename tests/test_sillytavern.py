import base64
import json
import socket
import subprocess
import sys

import pytest

from app import sillytavern as st
from app.config import DEFAULTS
from tests.conftest import SR, tone

ENDPOINT = "http://127.0.0.1:8890/v1/audio/speech"
# The keys SillyTavern's OpenAI Compatible provider accepts; any other key makes it throw.
PROVIDER_KEYS = {"voiceMap", "model", "speed", "available_voices", "provider_endpoint"}


def make_install(folder, settings=None, config="dataRoot: ./data\nport: 8000\n", users=("default-user",)):
    """A SillyTavern folder as far as these tests care: server.js, public\\, config.yaml and user settings."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "server.js").write_text("// SillyTavern\n")
    (folder / "public").mkdir(exist_ok=True)
    (folder / "Start.bat").write_text("@echo off\n")
    if config is not None:
        (folder / "config.yaml").write_text(config)
    (folder / "data" / "_storage").mkdir(parents=True, exist_ok=True)
    for user in users:
        d = folder / "data" / user
        d.mkdir(parents=True, exist_ok=True)
        data = settings if settings is not None else sample_settings()
        (d / "settings.json").write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    return folder


def sample_settings():
    return {
        "firstRun": False,
        "username": "Zoë",
        "power_user": {"theme": "Dark V 1.0"},
        "extension_settings": {
            "tts": {
                "voiceMap": "",
                "currentProvider": "ElevenLabs",
                "auto_generation": True,
                "enabled": False,
                "ElevenLabs": {"voiceMap": {"Seraphina": "Rachel"}},
            },
            "memory": {"source": "extras"},
        },
    }


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------ finding SillyTavern


def test_find_install_from_any_folder_near_it(tmp_path):
    root = make_install(tmp_path / "SillyTavern")
    assert st.find_install(str(root)) == root
    assert st.find_install(f'"{root}"') == root  # Windows' Copy as path adds quotes
    assert st.find_install(str(root / "Start.bat")) == root
    assert st.find_install(str(root / "data" / "default-user")) == root
    assert st.find_install(str(tmp_path)) == root  # the folder it's in

    launcher = make_install(tmp_path / "launcher" / "SillyTavern-Launcher" / "SillyTavern")
    assert st.find_install(str(launcher.parent)) == launcher

    zipped = make_install(tmp_path / "dl" / "SillyTavern-release" / "SillyTavern-release")
    assert st.find_install(str(tmp_path / "dl")) == zipped


def test_find_install_explains_what_is_wrong(tmp_path):
    with pytest.raises(st.SillyTavernError, match="Pick SillyTavern's folder first"):
        st.find_install("  ")
    with pytest.raises(st.SillyTavernError, match="Couldn't find the folder"):
        st.find_install(str(tmp_path / "nope"))
    (tmp_path / "Games").mkdir()
    with pytest.raises(st.SillyTavernError, match="Start.bat and server.js"):
        st.find_install(str(tmp_path / "Games"))


def test_guess_installs_looks_in_the_usual_places(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(st.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(st, "ROOT", tmp_path / "apps" / "Qwen3TTS")
    monkeypatch.setattr(st, "_fixed_drives", lambda: [tmp_path / "drive"])
    a = make_install(home / "Documents" / "SillyTavern")
    b = make_install(tmp_path / "apps" / "SillyTavern-staging")
    c = make_install(tmp_path / "drive" / "SillyTavern-Launcher" / "SillyTavern")
    assert set(st.guess_installs()) == {a, b, c}


def test_settings_files_and_config(tmp_path):
    root = make_install(tmp_path / "ST", users=("default-user", "alice"))
    assert [st.profile(p) for p in st.settings_files(root)] == ["alice", "default-user"]  # not _storage
    assert st.port(root) == 8000

    # A data folder moved elsewhere in config.yaml, and another port.
    moved = tmp_path / "elsewhere"
    (moved / "default-user").mkdir(parents=True)
    (moved / "default-user" / "settings.json").write_text("{}")
    (root / "config.yaml").write_text(f"# comment\ndataRoot: '{moved}'  # moved\nport: 8123\nlisten: false\n")
    assert st.settings_files(root) == [moved / "default-user" / "settings.json"]
    assert st.port(root) == 8123

    # Older SillyTavern (before 1.12) kept one settings file in public\.
    old = make_install(tmp_path / "old", users=(), config=None)
    (old / "public" / "settings.json").write_text("{}")
    assert st.settings_files(old) == [old / "public" / "settings.json"]
    assert st.port(old) == 8000


def test_is_running(tmp_path):
    root = make_install(tmp_path / "ST")
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        free_port = server.getsockname()[1]
        (root / "config.yaml").write_text(f"port: {free_port}\n")
        assert st.is_running(root)
    assert not st.is_running(root)


# ------------------------------------------------------------ filling in the settings


def test_fill_in_sets_up_the_tts_extension(tmp_path):
    root = make_install(tmp_path / "ST")
    path = root / "data" / "default-user" / "settings.json"
    original = path.read_bytes()

    result = st.apply(root, ["grumpy-wizard", "narrator"], endpoint=ENDPOINT, paragraphs=True)
    assert result.changed == [path] and not result.running
    data = read(path)
    tts = data["extension_settings"]["tts"]
    assert tts["enabled"] is True and tts["currentProvider"] == "OpenAI Compatible"
    assert tts["periodic_auto_generation"] is True and tts["narrate_by_paragraphs"] is True
    assert tts["OpenAI Compatible"] == {
        "provider_endpoint": ENDPOINT,
        "model": "qwen3-tts",
        "available_voices": ["grumpy-wizard", "narrator"],
    }
    # Everything else is left as it was.
    assert tts["ElevenLabs"] == {"voiceMap": {"Seraphina": "Rachel"}} and tts["auto_generation"] is True
    assert data["extension_settings"]["memory"] == {"source": "extras"} and data["username"] == "Zoë"
    # Same layout SillyTavern writes, and a backup of the old file.
    raw = path.read_bytes()
    assert b"\r\n" not in raw and "Zoë".encode() in raw and raw.startswith(b'{\n    "firstRun"')
    assert path.with_name("settings.json.qwen3-tts-backup").read_bytes() == original
    assert not list(path.parent.glob("*.qwen3-tts-tmp"))

    # Again: nothing to change, so the file isn't touched.
    mtime = path.stat().st_mtime_ns
    again = st.apply(root, ["grumpy-wizard", "narrator"], endpoint=ENDPOINT, paragraphs=True)
    assert again.unchanged == [path] and not again.changed and path.stat().st_mtime_ns == mtime


def test_fill_in_keeps_the_users_own_provider_settings(tmp_path):
    settings = sample_settings()
    settings["extension_settings"]["tts"]["OpenAI Compatible"] = {
        "voiceMap": {"Seraphina": "narrator", "[Default Voice]": "narrator"},
        "speed": 1.25,
        "model": "tts-1",
        "available_voices": ["alloy"],
        "provider_endpoint": "http://127.0.0.1:8000/v1/audio/speech",
    }
    root = make_install(tmp_path / "ST", settings=settings)
    st.apply(root, ["narrator"], endpoint=ENDPOINT, paragraphs=False)
    tts = read(root / "data" / "default-user" / "settings.json")["extension_settings"]["tts"]
    section = tts["OpenAI Compatible"]
    assert set(section) <= PROVIDER_KEYS
    assert section["voiceMap"] == {"Seraphina": "narrator", "[Default Voice]": "narrator"} and section["speed"] == 1.25
    assert section["available_voices"] == ["narrator"] and section["provider_endpoint"] == ENDPOINT
    assert tts["periodic_auto_generation"] is False and tts["narrate_by_paragraphs"] is False


def test_fill_in_a_brand_new_settings_file(tmp_path):
    root = make_install(tmp_path / "ST", settings={"firstRun": True})
    st.apply(root, [], endpoint=ENDPOINT)
    tts = read(root / "data" / "default-user" / "settings.json")["extension_settings"]["tts"]
    assert tts["OpenAI Compatible"]["available_voices"] == [] and "narrate_by_paragraphs" not in tts


def test_fill_in_every_profile_and_old_versions(tmp_path):
    root = make_install(tmp_path / "ST", users=("default-user", "alice"))
    assert len(st.apply(root, ["a"], endpoint=ENDPOINT).changed) == 2
    old = make_install(tmp_path / "old", users=(), config=None)
    (old / "public" / "settings.json").write_text(json.dumps(sample_settings()))
    st.apply(old, ["a"], endpoint=ENDPOINT)
    assert read(old / "public" / "settings.json")["extension_settings"]["tts"]["currentProvider"] == "OpenAI Compatible"


def test_fill_in_explains_problems(tmp_path):
    root = make_install(tmp_path / "ST", users=())
    with pytest.raises(st.SillyTavernError, match="Start SillyTavern once"):
        st.apply(root, ["a"], endpoint=ENDPOINT)
    make_install(root, users=("default-user",))
    (root / "data" / "default-user" / "settings.json").write_text("{ not json")
    with pytest.raises(st.SillyTavernError, match="Couldn't read"):
        st.apply(root, ["a"], endpoint=ENDPOINT)
    (root / "data" / "default-user" / "settings.json").write_text('{"extension_settings": []}')
    with pytest.raises(st.SillyTavernError, match="isn't a SillyTavern settings file"):
        st.apply(root, ["a"], endpoint=ENDPOINT)


def test_voice_list_only_update(tmp_path):
    root = make_install(tmp_path / "ST")
    path = root / "data" / "default-user" / "settings.json"
    # Not set up for the voice server yet: left alone.
    assert st.apply(root, ["a"]).skipped == [path]
    assert "OpenAI Compatible" not in read(path)["extension_settings"]["tts"]

    st.apply(root, ["a"], endpoint=ENDPOINT, paragraphs=True)
    tts = read(path)["extension_settings"]["tts"]
    tts["currentProvider"] = "ElevenLabs"  # the user picked another provider since
    tts["narrate_by_paragraphs"] = False
    path.write_text(json.dumps({"extension_settings": {"tts": tts}}, indent=4))
    assert st.apply(root, ["a", "b"]).changed == [path]
    tts = read(path)["extension_settings"]["tts"]
    assert tts["OpenAI Compatible"]["available_voices"] == ["a", "b"]
    assert tts["currentProvider"] == "ElevenLabs" and tts["narrate_by_paragraphs"] is False
    assert st.current_voices(path) == ["a", "b"]


def test_describe(tmp_path):
    root = make_install(tmp_path / "ST")
    result = st.apply(root, ["a", "b"], endpoint=ENDPOINT)
    msg = st.describe(result, ["a", "b"])
    assert "Done" in msg and "default-user" in msg and "2 voices" in msg and "qwen3-tts-backup" in msg
    assert "reload" not in msg
    result.running = True
    assert "reload its page" in st.describe(result, ["a", "b"])
    same = st.apply(root, ["a", "b"], endpoint=ENDPOINT)
    assert "already set up" in st.describe(same, ["a", "b"])
    assert "no voices yet" in st.describe(same, [])


# ------------------------------------------------------------ keeping it up to date


def test_auto_sync(tmp_path, library):
    root = make_install(tmp_path / "ST")
    path = root / "data" / "default-user" / "settings.json"
    settings = dict(DEFAULTS)
    sync = st.AutoSync(settings, library)
    assert sync.tick() is None  # no SillyTavern folder set yet

    library.save("Grumpy Wizard", tone(4), SR, kind="designed", ref_text="Hello there.")
    st.apply(root, [], endpoint=ENDPOINT)
    settings["sillytavern_dir"] = str(root)
    assert sync.tick().changed == [path]
    assert st.current_voices(path) == ["grumpy-wizard"] and "1 voice)" in sync.note
    assert sync.tick() is None  # nothing new

    library.save("Narrator", tone(4), SR, kind="cloned", ref_text="Once upon a time.")
    assert sync.tick().changed == [path]
    assert st.current_voices(path) == ["grumpy-wizard", "narrator"]

    # A SillyTavern page left open from before saves its old copy: the list is put back.
    data = read(path)
    data["extension_settings"]["tts"]["OpenAI Compatible"]["available_voices"] = ["grumpy-wizard"]
    data["active_character"] = "Seraphina.png"
    path.write_text(json.dumps(data, indent=4))
    assert sync.tick().changed == [path]
    assert st.current_voices(path) == ["grumpy-wizard", "narrator"]
    assert read(path)["active_character"] == "Seraphina.png"

    settings["sillytavern_auto_sync"] = False
    library.delete("narrator")
    assert sync.tick() is None and st.current_voices(path) == ["grumpy-wizard", "narrator"]


# ------------------------------------------------------------ the folder picker


def test_browse_on_windows(monkeypatch, tmp_path):
    ran = {}

    def run(cmd, **kw):
        ran.update(cmd=cmd, **kw)
        picked = base64.b64encode("D:\\Jeux\\SillyTavern".encode()).decode()
        return subprocess.CompletedProcess(cmd, 0, f"PICKED:{picked}\r\n".encode(), b"")

    monkeypatch.setattr(st, "WINDOWS", True)
    monkeypatch.setattr(st.subprocess, "run", run)
    assert st.browse(f'"{tmp_path}"') == "D:\\Jeux\\SillyTavern"
    script = base64.b64decode(ran["cmd"][-1]).decode("utf-16-le")
    assert ran["cmd"][:2] == ["powershell.exe", "-NoProfile"] and "-STA" in ran["cmd"]
    assert "FolderBrowserDialog" in script and "TopMost" in script and "ToBase64String" in script
    assert ran["env"]["QWEN_TTS_FOLDER"] == str(tmp_path)

    monkeypatch.setattr(st.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, b"\r\n", b""))
    assert st.browse("") is None  # cancelled

    def broken(cmd, **kw):
        raise FileNotFoundError("powershell.exe")

    monkeypatch.setattr(st.subprocess, "run", broken)
    with pytest.raises(ValueError, match="address bar"):
        st.browse("")


def test_browse_elsewhere_uses_tkinter(monkeypatch):
    ran = {}

    def run(cmd, **kw):
        ran.update(cmd=cmd, **kw)
        return subprocess.CompletedProcess(cmd, 0, b"PICKED:/home/me/SillyTavern\n", b"")

    monkeypatch.setattr(st, "WINDOWS", False)
    monkeypatch.setattr(st.subprocess, "run", run)
    assert st.browse("not a folder") == "/home/me/SillyTavern"
    assert ran["cmd"][0] == sys.executable and ran["cmd"][-1] == ""
    compile(st._PICKER_TK, "picker", "exec")


# ------------------------------------------------------------ the control panel and command line


def test_panel_fill_in(tmp_path, engine, library, monkeypatch):
    from fastapi.testclient import TestClient

    from app import server
    from app.server import build_panel, create_app
    from tests.test_server import panel_fn

    saved = dict(DEFAULTS)
    monkeypatch.setattr(server, "load_settings", lambda: dict(saved))
    monkeypatch.setattr(server, "save_settings", lambda s: saved.update(s))
    found = make_install(tmp_path / "SillyTavern")
    monkeypatch.setattr(st, "guess_installs", lambda: [found])
    library.save("Grumpy Wizard", tone(4), SR, kind="designed", ref_text="Hello there.")
    settings = dict(DEFAULTS)
    app = create_app(engine, library, settings)
    client = TestClient(app)
    panel = build_panel(library, settings, client.app, "http://127.0.0.1:8890")

    # The folder found is filled in, but nothing is saved until Fill in is clicked.
    st_dir = next(c for c in panel.blocks.values() if getattr(c, "label", None) == "SillyTavern folder")
    assert st_dir.value == str(found) and saved["sillytavern_dir"] == ""
    assert panel_fn(panel, "st_status")() == ""

    folder, note = panel_fn(panel, "st_fill_in")(str(tmp_path), True, False)
    assert folder == str(found) and "Done" in note and "1 voice" in note
    assert saved["sillytavern_dir"] == str(found) and saved["sillytavern_auto_sync"] is False
    assert settings["sillytavern_dir"] == str(found)  # the running server sees it too
    tts = read(found / "data" / "default-user" / "settings.json")["extension_settings"]["tts"]
    assert tts["OpenAI Compatible"]["available_voices"] == ["grumpy-wizard"]
    assert tts["OpenAI Compatible"]["provider_endpoint"] == ENDPOINT
    assert "has your 1 voice" in panel_fn(panel, "st_status")()

    panel_fn(panel, "st_auto_changed")(True)
    assert saved["sillytavern_auto_sync"] is True and settings["sillytavern_auto_sync"] is True
    library.save("Narrator", tone(4), SR, kind="cloned", ref_text="Once upon a time.")
    assert "differs" in panel_fn(panel, "st_status")()
    app.state.st_sync.tick()
    assert "has your 2 voices" in panel_fn(panel, "st_status")()

    import gradio as gr

    (tmp_path / "Games").mkdir()
    with pytest.raises(gr.Error, match="Start.bat"):
        panel_fn(panel, "st_fill_in")(str(tmp_path / "Games"), True, True)


def test_command_line(tmp_path, library, monkeypatch, capsys):
    from app import config, voices

    root = make_install(tmp_path / "ST")
    saved = dict(DEFAULTS)
    monkeypatch.setattr(config, "load_settings", lambda: dict(saved))
    monkeypatch.setattr(config, "save_settings", lambda s: saved.update(s))
    monkeypatch.setattr(config, "setup_environment", lambda: None)
    monkeypatch.setattr(voices, "VoiceLibrary", lambda: library)
    library.save("Narrator", tone(4), SR, kind="cloned", ref_text="Once upon a time.")
    assert st.main([str(root)]) == 0
    assert "Done" in capsys.readouterr().out and saved["sillytavern_dir"] == str(root)
    assert st.current_voices(root / "data" / "default-user" / "settings.json") == ["narrator"]
    assert st.main([str(tmp_path / "nope")]) == 1
