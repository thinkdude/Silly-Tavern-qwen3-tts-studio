import socket

import pytest

from app import menu
from app.config import DEFAULTS


class FakeProc:
    def __init__(self, alive=True):
        self.alive = alive

    def poll(self):
        return None if self.alive else 0


@pytest.fixture
def m(monkeypatch):
    """A menu where nothing is running, nothing is really started and no browser opens."""
    listening = set()
    started, opened = [], []
    monkeypatch.setattr(menu, "is_listening", lambda host, port, timeout=0.3: port in listening)
    monkeypatch.setattr(menu, "start_program", lambda p: started.append(p.key) or FakeProc())
    monkeypatch.setattr(menu.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(menu, "open_folder", lambda path: opened.append(str(path)))
    mm = menu.Menu(dict(DEFAULTS))
    mm.listening, mm.started_keys, mm.opened = listening, started, opened
    return mm


def test_lists_the_three_programs(m):
    m.refresh()
    screen = m.render()
    for name in ("Voice Studio", "Voice Server", "Speak", "Install, update or repair", "Quit"):
        assert name in screen
    assert "running" not in screen


def test_starts_a_program_and_shows_it_starting(m):
    m.refresh()
    assert m.handle("1") is None
    assert m.started_keys == ["1"]
    assert "Starting Voice Studio" in m.message
    m.refresh()
    assert m.status["1"] == "starting"
    assert "starting..." in m.render()
    # Pressing it again while it starts doesn't start a second copy.
    m.handle("1")
    assert m.started_keys == ["1"]
    assert "still starting" in m.message


def test_running_program_opens_in_the_browser(m):
    m.listening.add(7860)
    m.refresh()
    assert "running at http://127.0.0.1:7860/" in m.render()
    m.handle("1")
    assert m.started_keys == []
    assert m.opened == ["http://127.0.0.1:7860/"]


def test_voice_server_page_and_host(m):
    m.settings.update(server_host="0.0.0.0", server_port=9000)
    m.listening.add(9000)
    m.refresh()
    m.handle("2")
    assert m.opened == ["http://127.0.0.1:9000/ui"]
    m.settings["server_host"] = "192.168.1.5"
    assert menu.url(menu.PROGRAMS[1], m.settings) == "http://192.168.1.5:9000/ui"
    m.settings["server_host"] = "::1"
    assert menu.url(menu.PROGRAMS[1], m.settings) == "http://[::1]:9000/ui"


def test_a_program_that_stopped_is_no_longer_starting(m):
    m.handle("3")
    m.started["3"][0].alive = False
    m.refresh()
    assert m.status["3"] == ""


def test_install_waits_until_programs_are_closed(m):
    m.listening.add(7861)
    m.refresh()
    assert m.handle("6") is None
    assert "Close Speak first" in m.message
    m.listening.clear()
    m.refresh()
    assert m.handle("6") == menu.INSTALL


def test_other_keys(m):
    m.refresh()
    assert m.handle("4") is None and m.opened == [str(menu.OUTPUTS_DIR)]
    assert m.handle("5") is None and m.opened[-1].startswith(menu.HOMEPAGE)
    assert m.handle("u") is None and len(m.opened) == 2  # no update known: nothing happens
    assert m.handle("x") is None
    assert m.handle("Q") == 0
    assert m.handle("\x1b") == 0


def test_update_notice(m):
    m.update = "9.9.9"
    assert "Version 9.9.9 is out" in m.render()
    m.handle("U")
    assert m.opened == [menu.RELEASES_URL]


def test_version_comparison():
    assert menu.is_newer("1.0.1", "1.0.0")
    assert menu.is_newer("v1.10.0", "1.9.3")
    assert not menu.is_newer("1.0.0", "1.0.0")
    assert not menu.is_newer("0.9", "1.0.0")
    assert not menu.is_newer("1.0.0", "dev")


def test_update_check_can_be_turned_off(monkeypatch):
    calls = []
    monkeypatch.setattr(menu, "latest_release", lambda: calls.append(1) or "99.0.0")
    mm = menu.Menu({"check_for_updates": False})
    mm.check_for_updates()
    assert calls == [] and mm.update is None


def test_is_listening():
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen()
        port = srv.getsockname()[1]
        assert menu.is_listening("127.0.0.1", port)
    assert not menu.is_listening("127.0.0.1", port)
