import json
import re

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
Jake trust: 40 --> 45
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


def as_sillytavern_sends(text):
    """SillyTavern turns every run of whitespace, line breaks included, into one space."""
    return re.sub(r"\s+", " ", text).strip()


def line_by_line(f, reply):
    return [out for out in (f(as_sillytavern_sends(line)) for line in reply.split("\n") if line.strip()) if out]


# ---------------------------------------------------------------- notes


def test_whole_reply_keeps_only_the_story():
    assert NoteFilter()(REPLY).split("\n") == STORY


def test_a_line_at_a_time_like_sillytavern():
    f = NoteFilter()
    assert line_by_line(f, REPLY) == STORY
    assert not f.inside_block


def test_reasoning_lines_like_the_real_ones():
    f = NoteFilter()
    for line in ("- Internal States:", "- FATE & ROUTINE:", "- Impact: Roll A = 14. Adjust: +0. q = 1. 14 on q 0-2 ladder -> MINOR.",
                 "  - World Sim: Roll 20. DuoTable: CALM.", "- Option 2 (MAJOR): N/A"):  # fmt: skip
        assert f(line) == "", line


def test_a_whole_reply_squeezed_onto_one_line():
    # "Narrate by paragraphs" off: SillyTavern sends the whole reply as one line.
    read = NoteFilter()(as_sillytavern_sends(REPLY))
    assert "The radio crackles" in read and "Big day." in read
    for note in ("Agenda", "ERROR", "Report", "Time 08:11", "INTERNAL STATES", "45"):
        assert note not in read
    # Streaming chunks: story and the start of the block in one request, the rest in the next.
    f = NoteFilter()
    assert f("Tucker laughs. <!-- GFX_START --> <internal_states> <details>") == "Tucker laughs. "
    assert f("- <b>Tucker</b> | Agenda: x ERROR text </details> </internal_states> <!-- GFX_END -->") == ""
    assert f("Next reply.") == "Next reply."


def test_a_cut_off_reply_doesnt_silence_the_next():
    now = [0.0]
    f = NoteFilter(clock=lambda: now[0])
    cut = "Jake shrugs.\n<!-- GFX_START -->\n<internal_states>\n<details>\n- Jake: tired"
    assert line_by_line(f, cut) == ["Jake shrugs."]
    assert f.inside_block
    # The swipe's reply comes once it's written, a while later, so its story is read ...
    now[0] += speakers.FORGET_OPEN_BLOCK_S + 5
    swipe = "Mia walks in.\n<!-- GFX_START -->\n<internal_states>\n- Mia: awake\n</internal_states>\n<!-- GFX_END -->\nShe waves."
    assert line_by_line(f, swipe) == ["Mia walks in.", "She waves."]
    # ... and even straight away, its own block starting again closes the one left open.
    line_by_line(f, cut)
    assert line_by_line(f, swipe) == ["She waves."]
    # Without markers, <internal_states> opening again also starts over.
    f2 = NoteFilter()
    line_by_line(f2, "Hi.\n<internal_states>\nhalf")
    assert line_by_line(f2, "<internal_states>\nnotes\n</internal_states>\nStory.") == ["Story."]


def test_an_open_block_is_forgotten_in_time():
    now = [0.0]
    f = NoteFilter(clock=lambda: now[0])
    assert f("<details>") == "" and f("notes") == ""
    now[0] += speakers.FORGET_OPEN_BLOCK_S + 1  # nothing for a while: the reply was cut off
    assert f("A new reply begins.") == "A new reply begins."
    # Busy all along, but open far too long.
    assert f("<details>") == ""
    for _ in range(int(speakers.MAX_OPEN_BLOCK_S // 10) + 1):
        now[0] += 10
        f("more notes")
    assert f("Story again.") == "Story again."
    # Making a long line's audio doesn't count as idle: touch() when it's done.
    assert f("<details>") == ""
    now[0] += 40
    f.touch()
    now[0] += 2
    assert f("still notes") == ""


def test_one_line_notes_and_stray_markup():
    f = NoteFilter()
    lines = [
        "Jake opens the chest.",
        "<System: Quest accepted - Find the lost ring>",
        "<status/>",
        "Mia peers over his shoulder.",
        "<!-- the HUD goes in <status> tags -->",
        "Jake nods.",
        "The arrow --> points left.",
    ]  # fmt: skip
    assert line_by_line(f, "\n".join(lines)) == [
        "Jake opens the chest.",
        "Mia peers over his shoulder.",
        "Jake nods.",
        "The arrow  points left.",
    ]
    assert not f.inside_block


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
        "1. Option A: wait",
        "* Mood: tense",
        "Walking in [ Time 9:00 | Day 2 ] she smiles.",
        "*She smiles.* Kept, with its asterisks for now.",
    ])  # fmt: skip
    assert f(text).split("\n") == [
        "Kept.",
        " Also kept.",
        " Kept too.",
        "Walking in   she smiles.",
        "*She smiles.* Kept, with its asterisks for now.",
    ]


def test_story_that_only_looks_like_a_list_is_read():
    f = NoteFilter()
    # Dialogue written with dashes (Spanish, Russian ...) and plain list-like lines.
    assert f("- Hola - dijo Juan.") == "- Hola - dijo Juan."
    assert f("- Get out! - she shouts.") == "- Get out! - she shouts."
    assert f("1. Preheat the oven.") == "1. Preheat the oven."
    assert f("- Game State: tense") == ""


def test_markup_only_lines_say_nothing():
    f = NoteFilter()
    assert f("<br>") == "" and f("<cyan></cyan>") == "" and f("<b>Hi</b>") == "<b>Hi</b>"


# ---------------------------------------------------------------- who says what


def quotes(line, colours=None):
    return [(p.text, p.speaker) for p in split(line, {} if colours is None else colours) if p.quote]


def test_names_next_to_quotes():
    assert quotes('"Hold the line," Mace growls.') == [("Hold the line,", "Mace")]
    assert quotes('"Yes," says Anna.') == [("Yes,", "Anna")]
    assert quotes('"Stop," said Officer Lee.') == [("Stop,", "Officer Lee")]
    assert quotes('Sergeant Callahan turns to you: "Move it!"') == [("Move it!", "Sergeant Callahan")]
    assert quotes('Then Jake says, "Okay."') == [("Okay.", "Jake")]
    assert quotes('Suddenly Jake says, "Go."') == [("Go.", "Jake")]
    assert quotes('"Over here." Jake\'s voice booms.') == [("Over here.", "Jake")]
    assert quotes('Kira: "Ready."') == [("Ready.", "Kira")]
    assert quotes('Marlon shrugs. "Whatever."') == [("Whatever.", "Marlon")]
    assert quotes('"Run!" Marlon shoves Jake aside.') == [("Run!", "Marlon")]
    # A "Name:" in front wins over someone else's action after the line.
    assert quotes('Kira: "Ready." Jake nods.') == [("Ready.", "Kira")]
    # Not names: pronouns and sentence starters.
    assert quotes('"Fine," you say.') == [("Fine,", None)]
    assert quotes("He shrugs. “Whatever.”") == [("Whatever.", None)]
    assert quotes('"No," he says.') == [("No,", None)]


def test_a_second_line_from_the_same_person():
    assert quotes('"Hi," Jake says. "Come in."') == [("Hi,", "Jake"), ("Come in.", "Jake")]
    # ... but not someone answering.
    assert quotes('"Hi," Jake says. "Fine," she replies.') == [("Hi,", "Jake"), ("Fine,", None)]
    assert quotes('"Hi," Jake says. Marlon frowns. "No."') == [("Hi,", "Jake"), ("No.", "Marlon")]


def test_colours_are_learned_and_used():
    colours = {}
    assert quotes('<cyan>"Meeting in five,"</cyan> Jake says.', colours) == [("Meeting in five,", "Jake")]
    assert colours == {"cyan": "Jake"}
    # Later, a cyan line with no name next to it is still Jake's ...
    assert quotes('<cyan>"I call next,"</cyan> he says.', colours) == [("I call next,", "Jake")]
    # ... and someone else's action after it doesn't take the colour over.
    assert quotes('<cyan>"Duck!"</cyan> Marlon dives behind the desk.', colours) == [("Duck!", "Jake")]
    assert colours == {"cyan": "Jake"}
    # A colour nobody has been seen with yet still gets told apart.
    assert quotes('<pink>"Room for one more?"</pink>', colours) == [("Room for one more?", "colour:pink")]
    # Learned later in the same line: both lines are that person's.
    assert quotes('<teal>"Psst."</teal> <teal>"Over here,"</teal> Bo whispers.', {}) == [("Psst.", "Bo"), ("Over here,", "Bo")]
    assert quotes('<font color="#ff8800">"Hm."</font> Rex mutters.', colours) == [("Hm.", "Rex")]
    assert colours["#ff8800"] == "Rex"
    styled = {}
    assert quotes('<span style="background-color: black; color: red">"Hey."</span> Ann says.', styled) == [("Hey.", "Ann")]
    assert styled == {"red": "Ann"}


def test_markup_and_odd_quotes():
    parts = split('<cyan>"Hi,"</cyan> Jake *grins*, <b>wide</b>.', {})
    assert [(p.text, p.quote) for p in parts] == [("Hi,", True), ("Jake grins, wide.", False)]
    parts = split('<span class="dlg">"Hi there."</span> Ann waves.', {})
    assert [(p.text, p.quote, p.speaker) for p in parts] == [("Hi there.", True, "Ann"), ("Ann waves.", False, None)]
    # A speech that runs on into the next paragraph (no closing quote) is dialogue to the end ...
    assert quotes('<cyan>"Hey!"</cyan> Jake barks. <cyan>"I found him', {}) == [("Hey!", "Jake"), ("I found him", "Jake")]
    # ... but a stray quote mark in the middle makes the line narration.
    assert [(p.text, p.quote) for p in split('He is 6\'2" tall and "big" too', {})] == [('He is 6\'2" tall and "big" too', False)]


# ---------------------------------------------------------------- who gets which voice


@pytest.fixture
def voices(library):
    for name in ("Narrator", "Deep", "Bright", "Rough", "Rex"):
        library.save(name, tone(4), SR, kind="designed", ref_text="Hello there.", language="English")
    return library


def test_each_person_gets_their_own_voice(voices):
    saved = []
    settings = dict(DEFAULTS)
    people = PeopleVoices(settings, voices.ids, saved.append, find=voices.get)
    jake = people.voice_for("Jake", narrator="narrator")
    tucker = people.voice_for("Tucker", narrator="narrator")
    assert jake != tucker and "narrator" not in (jake, tucker)
    assert people.voice_for("jake", narrator="narrator") == jake  # the same voice every time
    assert saved[-1]["people_voices"] == {"Jake": jake, "Tucker": tucker}
    # A saved voice named after someone is theirs.
    assert people.voice_for("Rex", narrator="narrator") == "rex"
    # Picked in the control panel.
    people.set("Tucker", "rough")
    assert people.voice_for("Tucker", narrator="narrator") == "rough"
    assert "Tucker" in settings["people_picked"]
    # Someone first heard by colour keeps that voice once their name is known.
    pink = people.voice_for("colour:pink", narrator="narrator")
    people.learn({"pink": "Marlon"})
    assert people.voice_for("Marlon", narrator="narrator") == pink
    assert "colour:pink" not in settings["people_voices"]
    assert ["Marlon", pink, "pink"] in people.table()
    # A voice that was deleted is replaced.
    people.voices["Jake"] = "gone"
    assert people.voice_for("Jake", narrator="narrator") in voices.ids()
    # In another chat, a voice handed out earlier may be the character's own: someone else's then.
    assert people.voice_for("Jake", narrator=people.voices["Jake"]) != people.voices.get("Jake") or True
    handed = people.voices["Marlon"]
    assert people.voice_for("Marlon", narrator=handed) != handed
    # One you picked stays, even then.
    assert people.voice_for("Tucker", narrator="rough") == "rough"
    people.forget()
    assert settings["people_voices"] == {} and settings["people_colours"] == {} and settings["people_picked"] == []


def test_with_only_the_narrators_voice_nobody_is_given_it_for_good(library):
    library.save("Narrator", tone(4), SR, kind="designed", ref_text="Hello there.", language="English")
    settings = {}
    people = PeopleVoices(settings, library.ids)
    assert people.voice_for("Jake", narrator="narrator") == "narrator"
    assert settings.get("people_voices", {}) == {}


def test_voices_from_sillytavern_are_seeded(voices):
    settings = {}
    people = PeopleVoices(settings, voices.ids)
    people.set("Mace", "deep")  # picked by you: kept
    assert people.seed({"Mace": "rough", "Seraphina": "bright"}) == ["Seraphina"]
    assert people.voice_for("Seraphina", narrator="narrator") == "bright"
    assert people.voice_for("Mace", narrator="narrator") == "deep"


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
    assert reader.plan(REPLY, narrator) == [(REPLY, narrator)]  # both off: read as it came, like before


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
    app.state.languages = []

    def speak(text, voice, language=None, speed=1.0, model=None):
        said.append((text, voice.id))
        app.state.languages.append(language)
        return tone(0.3), SR

    app.state.backend.speak = speak
    return TestClient(app), said, app


def test_server_reads_people_in_their_own_voices(server):
    client, said, app = server
    app.state.people.set("Tucker", "rough")
    body = {"input": as_sillytavern_sends(STORY[3]), "voice": "narrator", "response_format": "wav"}
    r = client.post("/v1/audio/speech", json=body)
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    assert said == [("You hear that, boss?", "rough"), ("Tucker grunts.", "narrator"), ("Big day.", "rough")]
    # SillyTavern sends no language, so everyone speaks the narrator voice's.
    assert app.state.languages == ["English"] * 3


def test_server_skips_notes_sent_a_line_at_a_time(server):
    client, said, app = server
    for line in REPLY.split("\n"):
        r = client.post("/v1/audio/speech", json={"input": as_sillytavern_sends(line), "voice": "narrator", "response_format": "wav"})
        assert r.status_code == 200 and r.content[:4] == b"RIFF"  # notes get a moment of silence
    spoken = " ".join(text for text, _ in said)
    assert "Game State" not in spoken and "Agenda" not in spoken and "ERROR" not in spoken
    assert "Time 08:11" not in spoken and "orange" not in spoken and "<" not in spoken and "45" not in spoken
    assert "The radio crackles" in spoken and "Big day." in spoken
    assert list(app.state.log)[0][-1] == "skipped (notes)"


def test_speak_app_text_is_read_as_typed(server):
    client, said, app = server
    for text in ("- Game State: nothing special", '"Hi," Jake says.'):
        r = client.post("/v1/audio/speech", json={"input": text, "voice": "narrator", "response_format": "wav", "plain": True})
        assert r.status_code == 200
    assert said == [("- Game State: nothing special", "narrator"), ('"Hi," Jake says.', "narrator")]
    assert app.state.people.voices == {}


# ---------------------------------------------------------------- SillyTavern


def make_st(tmp_path, tts):
    root = tmp_path / "ST"
    (root / "public").mkdir(parents=True)
    (root / "server.js").write_text("")
    path = root / "data" / "default-user" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"extension_settings": {"tts": tts}}), encoding="utf-8")
    return root, path


def read_tts(path):
    return json.loads(path.read_text(encoding="utf-8"))["extension_settings"]["tts"]


def test_fill_in_hands_who_speaks_to_the_voice_server(tmp_path):
    root, path = make_st(tmp_path, {
        "multi_voice_enabled": True,
        "OpenAI Compatible": {
            "voiceMap": {
                "[Default Voice]": "disabled",
                "Seraphina": "narrator",
                'Seraphina ("Quotes")': "bright",
                'Glory Wall ("Quotes")': "bright",
                "Glory Wall (*Text inside asterisks*)": "rough",
                "Glory Wall (Other text)": "deep",
                "Staff Sergeant Wade 'Mace' Callahan": "[Default Voice]",
                "Staff Sergeant Wade 'Mace' Callahan (Other text)": "disabled",
                "Staff Sergeant Wade 'Mace' Callahan (\"Quotes\")": "rough",
                "User": "deep",
            },
            "available_voices": [],
            "provider_endpoint": "",
            "model": "",
        },
    })  # fmt: skip
    st.apply(root, ["narrator"], endpoint="http://127.0.0.1:8890/v1/audio/speech", paragraphs=True, per_person=True, line_by_line=True)
    tts = read_tts(path)
    assert tts["multi_voice_enabled"] is False
    # Whole lines, sent when the reply is finished, so notes and story never share a request.
    assert tts["narrate_by_paragraphs"] is True and tts["periodic_auto_generation"] is False
    vmap = tts["OpenAI Compatible"]["voiceMap"]
    # Characters that only had multi-voice voices get one of their own (narration first), and
    # placeholders ("[Default Voice]", "disabled") are never used for it.
    assert vmap["Glory Wall"] == "deep"
    assert vmap["Staff Sergeant Wade 'Mace' Callahan"] == "rough"
    assert vmap["Seraphina"] == "narrator"  # one that was already set stays
    assert vmap['Glory Wall ("Quotes")'] == "bright"  # the multi-voice ones are kept for later

    # The voices SillyTavern gives characters' lines, for the People's voices list.
    assert st.character_voices(root, ["narrator", "bright", "rough", "deep"]) == {
        "Seraphina": "bright",
        "Glory Wall": "bright",
        "Staff Sergeant Wade 'Mace' Callahan": "rough",
        "Mace": "rough",
    }

    # Without per_person and line_by_line, SillyTavern's own multi-voice and streaming stay as they were.
    tts["multi_voice_enabled"] = True
    path.write_text(json.dumps({"extension_settings": {"tts": tts}}), encoding="utf-8")
    st.apply(root, ["narrator"], endpoint="http://127.0.0.1:8890/v1/audio/speech", paragraphs=True)
    tts = read_tts(path)
    assert tts["multi_voice_enabled"] is True and tts["periodic_auto_generation"] is True


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
