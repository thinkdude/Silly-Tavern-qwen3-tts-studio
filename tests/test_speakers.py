import json

import pytest
from fastapi.testclient import TestClient

from app import sillytavern as st
from app import speakers
from app.config import DEFAULTS
from app.server import create_app
from app.speakers import NoteFilter, PeopleVoices, Reader, split
from tests.conftest import SR, tone

# The shape of a role-play reply with colour-coded dialogue and a block of game state at the end.
REPLY = """# Reasoning Process
- Game State: the crew is in the office.
- Formatting: Colored dialogue for Jake (cyan) and Tucker (orange).
[ Time 08:11 AM | Day 1 | Office ]
The radio crackles and Jake leans over the desk.
<cyan>"Everyone, meeting in five,"</cyan> Jake says, letting go of the button.
Tucker laughs from the doorway.
<orange>"You hear that, boss?"</orange> Tucker grunts. <orange>"Big day."</orange>
<!-- GFX_START -->
<internal_states>
<details>
<summary>INTERNAL STATES (Turn: 4)</summary>
<details>
<summary>NPC AGENDAS</summary>
- <b>Tucker</b> | Agenda: finish the shift
</details>
<details><summary>BONDS</summary><br>- <b>Tucker</b> | -19</details>
1.[Report: a line with no markers at all
ERROR text that must never be read
</details>
</internal_states>
<!-- GFX_END -->"""

STORY = [
    "The radio crackles and Jake leans over the desk.",
    '<cyan>"Everyone, meeting in five,"</cyan> Jake says, letting go of the button.',
    "Tucker laughs from the doorway.",
    '<orange>"You hear that, boss?"</orange> Tucker grunts. <orange>"Big day."</orange>',
]


# ---------------------------------------------------------------- notes


def test_whole_reply_keeps_only_the_story():
    assert NoteFilter()(REPLY).split("\n") == STORY


def test_a_paragraph_at_a_time_like_sillytavern():
    f = NoteFilter()
    read = [out for out in (f(line) for line in REPLY.split("\n")) if out]
    assert read == STORY
    assert not f.inside_block


def test_a_block_left_open_is_forgotten():
    now = [0.0]
    f = NoteFilter(clock=lambda: now[0])
    assert f("<internal_states>") == ""
    assert f("Agenda: nothing to see") == ""
    now[0] += speakers.FORGET_OPEN_BLOCK_S + 1  # the reply was cut off; the next one starts later
    assert f("A new reply begins.") == "A new reply begins."
    # A whole reply never leaves a block open for the next request.
    f2 = NoteFilter()
    assert f2("Story.\n<details>\nhidden") == "Story."
    assert f2("Next reply.") == "Next reply."


def test_other_kinds_of_notes():
    f = NoteFilter()
    text = "\n".join([
        "Kept.",
        "<think>planning the scene</think> Also kept.",
        "```",
        "code that isn't read",
        "```",
        "| a | table |",
        "---",
        "<!-- a comment --> Kept too.",
        "1. a numbered list",
        "* a bullet",
        "*She smiles.* Kept, with its asterisks for now.",
    ])  # fmt: skip
    assert f(text).split("\n") == ["Kept.", " Also kept.", " Kept too.", "*She smiles.* Kept, with its asterisks for now."]


def test_markup_only_lines_say_nothing():
    f = NoteFilter()
    assert f("<br>") == "" and f("<cyan></cyan>") == "" and f("<b>Hi</b>") == "<b>Hi</b>"


# ---------------------------------------------------------------- who says what


def quotes(line, colours=None):
    return [(p.text, p.speaker) for p in split(line, {} if colours is None else colours) if p.quote]


def test_names_next_to_quotes():
    assert quotes('"Hold the line," Mace growls.') == [("Hold the line,", "Mace")]
    assert quotes('"Yes," says Anna.') == [("Yes,", "Anna")]
    assert quotes('Sergeant Callahan turns to you: "Move it!"') == [("Move it!", "Sergeant Callahan")]
    assert quotes('Then Jake says, "Okay."') == [("Okay.", "Jake")]
    assert quotes('"Over here." Jake\'s voice booms.') == [("Over here.", "Jake")]
    assert quotes('Kira: "Ready."') == [("Ready.", "Kira")]
    # Not names: pronouns and sentence starters.
    assert quotes('"Fine," you say.') == [("Fine,", None)]
    assert quotes("He shrugs. “Whatever.”") == [("Whatever.", None)]
    assert quotes('"No," he says.') == [("No,", None)]


def test_a_second_line_from_the_same_person():
    assert quotes('"Hi," Jake says. "Come in."') == [("Hi,", "Jake"), ("Come in.", "Jake")]


def test_colours_are_learned_and_used():
    colours = {}
    assert quotes('<cyan>"Meeting in five,"</cyan> Jake says.', colours) == [("Meeting in five,", "Jake")]
    assert colours == {"cyan": "Jake"}
    # Later, a cyan line with no name next to it is still Jake's.
    assert quotes('<cyan>"I call next,"</cyan> he says.', colours) == [("I call next,", "Jake")]
    # A colour nobody has been seen with yet still gets told apart.
    assert quotes('<pink>"Room for one more?"</pink>', colours) == [("Room for one more?", "colour:pink")]
    assert quotes('<font color="#ff8800">"Hm."</font> Rex mutters.', colours) == [("Hm.", "Rex")]
    assert quotes('<span style="color: teal">"Yo."</span>', {"teal": "Bo"}) == [("Yo.", "Bo")]


def test_narration_is_cleaned_for_speaking():
    parts = split('<cyan>"Hi,"</cyan> Jake *grins*, <b>wide</b>.', {})
    assert [(p.text, p.quote) for p in parts] == [("Hi,", True), ("Jake grins, wide.", False)]


# ---------------------------------------------------------------- who gets which voice


@pytest.fixture
def voices(library):
    for name in ("Narrator", "Deep", "Bright", "Rough"):
        library.save(name, tone(4), SR, kind="designed", ref_text="Hello there.", language="English")
    return library


def test_each_person_gets_their_own_voice(voices):
    saved = []
    settings = dict(DEFAULTS)
    people = PeopleVoices(settings, voices.ids, saved.append)
    jake = people.voice_for("Jake", narrator="narrator")
    tucker = people.voice_for("Tucker", narrator="narrator")
    assert jake != tucker and "narrator" not in (jake, tucker)
    assert people.voice_for("jake", narrator="narrator") == jake  # the same voice every time
    assert saved[-1]["people_voices"] == {"Jake": jake, "Tucker": tucker}
    # Picked in the control panel.
    people.set("Tucker", "rough")
    assert people.voice_for("Tucker", narrator="narrator") == "rough"
    # Someone first heard by colour keeps that voice once their name is known.
    pink = people.voice_for("colour:pink", narrator="narrator")
    people.learn({"pink": "Marlon"})
    assert people.voice_for("Marlon", narrator="narrator") == pink
    assert "colour:pink" not in settings["people_voices"]
    assert ["Marlon", pink, "pink"] in people.table()
    # A voice that was deleted is replaced.
    people.set("Jake", "gone")
    assert people.voice_for("Jake", narrator="narrator") in voices.ids()
    people.forget()
    assert settings["people_voices"] == {} and settings["people_colours"] == {}


def test_reader_plans_who_reads_what(voices):
    settings = dict(DEFAULTS)
    people = PeopleVoices(settings, voices.ids)
    people.set("Jake", "deep")
    people.set("Tucker", "rough")
    reader = Reader(settings, voices, people)
    narrator = voices.get("narrator")
    plan = [(text, v.id) for text, v in reader.plan(REPLY, narrator)]
    assert plan == [
        ("The radio crackles and Jake leans over the desk.", "narrator"),
        ("Everyone, meeting in five,", "deep"),
        ("Jake says, letting go of the button. Tucker laughs from the doorway.", "narrator"),
        ("You hear that, boss?", "rough"),
        ("Tucker grunts.", "narrator"),
        ("Big day.", "rough"),
    ]
    assert settings["people_colours"] == {"cyan": "Jake", "orange": "Tucker"}

    settings["voices_per_person"] = False
    assert [v.id for _, v in reader.plan(REPLY, narrator)] == ["narrator"]
    settings["skip_notes"] = False
    assert "ERROR text" in reader.plan(REPLY, narrator)[0][0]


def test_someone_heard_by_colour_first_keeps_their_voice(voices):
    # Their first line has only a colour; their name turns up later in the same reply.
    settings = dict(DEFAULTS)
    reader = Reader(settings, voices, PeopleVoices(settings, voices.ids))
    narrator = voices.get("narrator")
    reply = '<cyan>"Attention, everyone."</cyan>\nThe speaker clicks off.\n<cyan>"I call next,"</cyan> Jake says.'
    first, second = [v.id for text, v in reader.plan(reply, narrator) if text in ("Attention, everyone.", "I call next,")]
    assert first == second != "narrator"
    assert settings["people_voices"] == {"Jake": first}


# ---------------------------------------------------------------- the voice server


@pytest.fixture
def server(engine, voices):
    app = create_app(engine, voices, dict(DEFAULTS))
    said = []

    def speak(text, voice, language=None, speed=1.0, model=None):
        said.append((text, voice.id))
        return tone(0.3), SR

    app.state.backend.speak = speak
    return TestClient(app), said, app


def test_server_reads_people_in_their_own_voices(server):
    client, said, app = server
    app.state.people.set("Tucker", "rough")
    body = {"input": STORY[3], "voice": "narrator", "response_format": "wav"}
    r = client.post("/v1/audio/speech", json=body)
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    assert said == [("You hear that, boss?", "rough"), ("Tucker grunts.", "narrator"), ("Big day.", "rough")]


def test_server_skips_notes_sent_a_paragraph_at_a_time(server):
    client, said, app = server
    for line in REPLY.split("\n"):
        r = client.post("/v1/audio/speech", json={"input": line, "voice": "narrator", "response_format": "wav"})
        assert r.status_code == 200 and r.content[:4] == b"RIFF"  # notes get a moment of silence
    spoken = " ".join(text for text, _ in said)
    assert "Game State" not in spoken and "Agenda" not in spoken and "ERROR" not in spoken
    assert "Time 08:11" not in spoken and "orange" not in spoken and "<" not in spoken
    assert "The radio crackles" in spoken and "Big day." in spoken
    assert list(app.state.log)[0][-1] == "skipped (notes)"


# ---------------------------------------------------------------- SillyTavern


def test_fill_in_hands_who_speaks_to_the_voice_server(tmp_path):
    settings = {
        "extension_settings": {
            "tts": {
                "multi_voice_enabled": True,
                "OpenAI Compatible": {
                    "voiceMap": {
                        "Seraphina": "narrator",
                        'Seraphina ("Quotes")': "bright",
                        'Glory Wall ("Quotes")': "bright",
                        "Glory Wall (*Text inside asterisks*)": "rough",
                        "Glory Wall (Other text)": "deep",
                        'Mace ("Quotes")': "rough",
                    },
                    "available_voices": [],
                    "provider_endpoint": "",
                    "model": "",
                },
            }
        }
    }
    root = tmp_path / "ST"
    (root / "public").mkdir(parents=True)
    (root / "server.js").write_text("")
    path = root / "data" / "default-user" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(settings), encoding="utf-8")

    st.apply(root, ["narrator"], endpoint="http://127.0.0.1:8890/v1/audio/speech", per_person=True)
    tts = json.loads(path.read_text(encoding="utf-8"))["extension_settings"]["tts"]
    assert tts["multi_voice_enabled"] is False
    vmap = tts["OpenAI Compatible"]["voiceMap"]
    # Characters that only had multi-voice voices get one of their own (narration first).
    assert vmap["Glory Wall"] == "deep" and vmap["Mace"] == "rough"
    assert vmap["Seraphina"] == "narrator"  # one that was already set stays
    assert vmap['Glory Wall ("Quotes")'] == "bright"  # the multi-voice ones are kept for later

    # Without per_person, SillyTavern's own multi-voice is left as it is.
    tts["multi_voice_enabled"] = True
    path.write_text(json.dumps({"extension_settings": {"tts": tts}}), encoding="utf-8")
    st.apply(root, ["narrator"], endpoint="http://127.0.0.1:8890/v1/audio/speech")
    assert json.loads(path.read_text(encoding="utf-8"))["extension_settings"]["tts"]["multi_voice_enabled"] is True


def test_control_panel_people_tab(server, voices, monkeypatch):
    from app import server as srv

    client, said, app = server
    saved = {}
    monkeypatch.setattr(srv, "load_settings", lambda: dict(saved))
    monkeypatch.setattr(srv, "save_settings", lambda s: saved.update(s))
    panel = srv.build_panel(voices, app.state.backend.settings, app, "http://127.0.0.1:8890")

    def fn(name):
        return next(f.fn for f in panel.fns.values() if getattr(f.fn, "__name__", "") == name)

    table, _, note = fn("pv_use")("Tucker", "rough")
    assert ["Tucker", "rough", ""] in table and "Tucker now speaks with rough" in note
    app.state.people.voice_for("colour:pink", narrator="narrator")
    table, _, _ = fn("pv_use")("(someone in pink)", "deep")
    assert ["(someone in pink)", "deep", "pink"] in table

    # The Test tab reads a pasted reply the same way SillyTavern's requests are read.
    sr, wav = fn("test")("narrator", "Voice's default", STORY[3])
    assert sr == SR and len(wav)
    assert said == [("You hear that, boss?", "rough"), ("Tucker grunts.", "narrator"), ("Big day.", "rough")]

    table, _, note = fn("pv_forget_all")()
    assert table == [] and "Forgotten" in note
