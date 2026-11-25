"""Read a reply the way a person would: skip the notes, and give each person their own voice.

Role-play replies often carry more than the story: a "# Reasoning Process" list at the top,
a status line in [ brackets | like this ], and whole blocks of game state in <internal_states>,
<details> or <!-- --> markup. None of that should be read out. And when several people talk
in one reply, each of them should sound like themselves.

    NoteFilter      drops those notes. SillyTavern sends long replies a paragraph at a time,
                    so a block of notes can span many requests; the filter remembers that it's
                    inside one until the block ends.
    split           cuts a paragraph into narration and quotes, and works out who says each
                    quote: from the name next to it ("Hi," Jake says / Jake says, "Hi"), from
                    a "Name:" in front, or from its colour (<cyan>"Hi"</cyan>), once a name has
                    been seen with that colour.
    PeopleVoices    which saved voice each person speaks with: one you picked, or one handed
                    out the first time they speak and remembered in settings.json.
    Reader          puts it together for the voice server: narration in the voice the app asked
                    for, each person's lines in their own voice, all in one clip.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import audio

# ---------------------------------------------------------------- notes to skip

# Blocks that are never read: their tags, everything inside them, however many lines.
SKIP_TAGS = (
    "internal_states", "details", "summary", "think", "thinking", "reasoning", "thoughts",
    "analysis", "notes", "status", "stats", "hud", "ooc", "system", "memory", "scratchpad",
)  # fmt: skip
_BLOCK = re.compile(
    r"<!--\s*(?P<start>[\w-]+?)_START\s*-->"  # <!-- GFX_START --> ... <!-- GFX_END -->
    r"|<!--\s*(?P<end>[\w-]+?)_END\s*-->"
    r"|(?P<copen><!--)|(?P<cclose>-->)"
    r"|(?P<fence>```|~~~)"
    r"|<(?P<slash>/?)(?P<tag>" + "|".join(SKIP_TAGS) + r")\b[^>]*>",
    re.IGNORECASE,
)
# Whole lines that are notes rather than story.
_NOTE_LINES = (
    re.compile(r"^\s*#{1,6}\s"),  # # Reasoning Process
    re.compile(r"^\s*(?:[-•+]|\*(?=\s)|\d{1,3}[.)])\s"),  # - Game State: ...  /  1. ...
    re.compile(r"^\s*\|.*\|\s*$"),  # | table | row |
    re.compile(r"^\s*\[[^\]]*\|[^\]]*\]\s*$"),  # [ Time 08:11 AM | Day 1 | Office ]
    re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,}|={3,})\s*$"),  # ---
)
_TAG = re.compile(r"<[^<>]{1,200}>")
# A block left open this long (the reply was cut off, say) doesn't hide the next reply.
FORGET_OPEN_BLOCK_S = 45.0


class NoteFilter:
    """Takes out notes and hidden blocks. Called with one paragraph at a time it remembers,
    between calls, that a block is still open; text with several lines is a whole reply,
    so a block left open at its end is forgotten."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._depth = 0  # open <details>, <!-- -->, <internal_states> ... blocks
        self._fence = False  # inside a ``` code block
        self._last = 0.0

    @property
    def inside_block(self) -> bool:
        return bool(self._depth or self._fence)

    def reset(self) -> None:
        with self._lock:
            self._depth, self._fence = 0, False

    def __call__(self, text: str) -> str:
        with self._lock:
            now = self._clock()
            if self.inside_block and now - self._last > FORGET_OPEN_BLOCK_S:
                self._depth, self._fence = 0, False
            self._last = now
            lines = (text or "").replace("\r\n", "\n").split("\n")
            kept = [line for line in (self._line(raw) for raw in lines) if line]
            if len(lines) > 1:  # a whole reply: what's left open ends with it
                self._depth, self._fence = 0, False
            return "\n".join(kept)

    def _line(self, line: str) -> str:
        out, pos = [], 0
        for m in _BLOCK.finditer(line):
            if self._depth == 0 and not self._fence:
                out.append(line[pos : m.start()])
            pos = m.end()
            if m.group("fence"):
                if not self._depth:
                    self._fence = not self._fence
            elif self._fence:
                continue
            elif m.group("start") or m.group("copen") or (m.group("tag") and not m.group("slash")):
                self._depth += 1
            elif m.group("end") or m.group("cclose") or (m.group("tag") and m.group("slash")):
                self._depth = max(0, self._depth - 1)
        if self._depth == 0 and not self._fence:
            out.append(line[pos:])
        kept = "".join(out)
        if any(p.match(kept) for p in _NOTE_LINES):
            return ""
        return kept if _TAG.sub("", kept).strip(" \t*_~") else ""


# ---------------------------------------------------------------- who says what

# CSS colour names, as used for colour-coded dialogue: <cyan>"Hi"</cyan>.
COLOURS = frozenset(
    """aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue blueviolet
    brown burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk crimson cyan darkblue
    darkcyan darkgoldenrod darkgray darkgreen darkgrey darkkhaki darkmagenta darkolivegreen darkorange
    darkorchid darkred darksalmon darkseagreen darkslateblue darkslategray darkslategrey darkturquoise
    darkviolet deeppink deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite forestgreen fuchsia
    gainsboro ghostwhite gold goldenrod gray green greenyellow grey honeydew hotpink indianred indigo
    ivory khaki lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan
    lightgoldenrodyellow lightgray lightgreen lightgrey lightpink lightsalmon lightseagreen lightskyblue
    lightslategray lightslategrey lightsteelblue lightyellow lime limegreen linen magenta maroon
    mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen mediumslateblue
    mediumspringgreen mediumturquoise mediumvioletred midnightblue mintcream mistyrose moccasin
    navajowhite navy oldlace olive olivedrab orange orangered orchid palegoldenrod palegreen
    paleturquoise palevioletred papayawhip peachpuff peru pink plum powderblue purple rebeccapurple red
    rosybrown royalblue saddlebrown salmon sandybrown seagreen seashell sienna silver skyblue slateblue
    slategray slategrey snow springgreen steelblue tan teal thistle tomato turquoise violet wheat white
    whitesmoke yellow yellowgreen""".split()
)
_TOKEN = re.compile(
    r"<font\b[^>]*?\bcolor\s*=\s*[\"']?(?P<font>[#\w]+)[^>]*>"
    r"|<span\b[^>]*?\bcolor\s*:\s*(?P<span>[#\w]+)[^>]*>"
    r"|<(?P<open>[a-z]+)>"
    r"|</(?P<close>[a-z]+)\s*>"
    r"|(?P<quote>\"[^\"]*\"|“[^”]*”|«[^»]*»|「[^」]*」)",
    re.IGNORECASE,
)
# Capitalised words that start sentences but aren't anyone's name.
NOT_NAMES = frozenset(
    """a an and as at but by for from he her hers him his how i if in into it its me my no not now of
    oh on once or our she so some still that the their them then there these they this those though
    to too up us we what when where which while who why with yet you your yours okay ok yes yeah
    well hey god fuck shit damn sir maam mister miss mrs mr ms dr""".split()
)
_WORD = r"[A-Z][\w'’-]*"
_NAME_AFTER = re.compile(rf"^[\s,.;:!?—–-]*(?P<name>{_WORD}(?:\s+{_WORD})?)\s+[a-z]")
_SAYS_NAME = re.compile(
    rf"^[\s,]*(?:says|said|asks|asked|replies|replied|calls|called|shouts|shouted|whispers|whispered)\s+(?P<name>{_WORD})"
)
_SENTENCE_END = re.compile(r"[.!?…]\s+")


@dataclass
class Part:
    text: str
    quote: bool = False
    speaker: str | None = None  # a person's name, or "colour:<name>" when only the colour is known


def clean(text: str) -> str:
    """Text as it should be spoken: no markup, no *asterisks*, single spaces."""
    text = _TAG.sub("", text or "")
    text = re.sub(r"[*_~]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _name(candidate: str | None) -> str | None:
    """'Jake', 'Sergeant Callahan', or 'Jake' from 'Then Jake' and "Jake's"; None for 'He' or 'The'."""
    words = (candidate or "").split()
    while words and words[0].lower().strip("'’") in NOT_NAMES:
        words.pop(0)
    if not words or any(w.lower().strip("'’") in NOT_NAMES for w in words):
        return None
    name = " ".join(words)
    for suffix in ("'s", "’s"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name or None


def _name_after(narration: str) -> str | None:
    """Jake in: ' Jake says, grinning.' or ', says Jake.'"""
    text = clean(narration)
    m = _SAYS_NAME.match(text)
    if m:
        return _name(m.group("name"))
    m = _NAME_AFTER.match(text)
    return _name(m.group("name")) if m else None


def _name_before(narration: str) -> str | None:
    """Jake in: 'Jake leans in, ' or 'Jake:' just before a quote."""
    text = clean(narration)
    if not text.endswith((",", ":", "—", "-")):
        return None
    sentence = _SENTENCE_END.split(text)[-1]
    m = re.match(rf"\s*(?P<name>{_WORD}(?:\s+{_WORD})?)\s*(?:[:,]|\s+[a-z])", sentence)
    return _name(m.group("name")) if m else None


def split(paragraph: str, colours: dict[str, str] | None = None) -> list[Part]:
    """Cut a paragraph into narration and quotes, and work out who says each quote.

    `colours` maps a dialogue colour to the person it belongs to. It's read, and also updated
    when a quote has both a colour and a name next to it, so later colour-only quotes are known.
    """
    colours = colours if colours is not None else {}
    parts: list[Part] = []
    quote_colour: list[str | None] = []
    stack: list[str] = []  # open colour tags
    narration_start = 0
    for m in _TOKEN.finditer(paragraph):
        colour = m.group("font") or m.group("span")
        if colour is None and m.group("open") and m.group("open").lower() in COLOURS:
            colour = m.group("open")
        if colour:
            stack.append(colour.lower())
        elif m.group("close") and stack and (
            m.group("close").lower() in ("font", "span") or m.group("close").lower() == stack[-1]
        ):
            stack.pop()
        elif m.group("quote"):
            before = paragraph[narration_start : m.start()]
            if clean(before):
                parts.append(Part(before))
            parts.append(Part(m.group("quote")[1:-1], quote=True))
            quote_colour.append(stack[-1] if stack else None)
            narration_start = m.end()
    rest = paragraph[narration_start:]
    if clean(rest) or not parts:
        parts.append(Part(rest))

    quotes = [i for i, p in enumerate(parts) if p.quote]
    for n, i in enumerate(quotes):
        colour = quote_colour[n]
        after = parts[i + 1].text if i + 1 < len(parts) and not parts[i + 1].quote else ""
        before = parts[i - 1].text if i > 0 and not parts[i - 1].quote else ""
        name = _name_after(after) or _name_before(before)
        if name is None and n and parts[quotes[n - 1]].speaker and len(clean(before).split()) <= 8:
            # "Hi," Jake says. "Come in." The second line is Jake's too.
            if quote_colour[n - 1] == colour:
                name = parts[quotes[n - 1]].speaker
        if colour and name and not name.startswith("colour:"):
            colours[colour] = name
        if name is None and colour:
            name = colours.get(colour) or f"colour:{colour}"
        parts[i].speaker = name
    for p in parts:
        p.text = clean(p.text)
    return [p for p in parts if p.text]


# ---------------------------------------------------------------- who speaks with which voice


class PeopleVoices:
    """Remembers each person's voice, and the colour each person's lines are written in.

    settings["people_voices"]   {"Jake": "deep-voice", ...}, picked in the control panel or
                                handed out the first time someone speaks
    settings["people_colours"]  {"cyan": "Jake", ...}, learned from the replies
    """

    def __init__(self, settings: dict, voice_ids: Callable[[], list[str]], save: Callable[[dict], None] | None = None):
        self.settings = settings
        self.voice_ids = voice_ids
        self._save = save
        self._lock = threading.RLock()

    @property
    def voices(self) -> dict[str, str]:
        return self.settings.setdefault("people_voices", {})

    @property
    def colours(self) -> dict[str, str]:
        return self.settings.setdefault("people_colours", {})

    def _key(self, person: str) -> str | None:
        return next((k for k in self.voices if k.casefold() == person.casefold()), None)

    def voice_for(self, person: str, narrator: str) -> str:
        """The person's voice, handing one out the first time they speak: a saved voice the
        narrator and the other people don't use yet, or the least used one."""
        with self._lock:
            ids = self.voice_ids()
            key = self._key(person)
            if key and self.voices[key] in ids:
                return self.voices[key]
            if not person.startswith("colour:"):
                # Someone first heard only by colour keeps that voice once their name is known.
                colour = next((c for c, p in self.colours.items() if p.casefold() == person.casefold()), None)
                ckey = self._key(f"colour:{colour}") if colour else None
                if ckey and self.voices[ckey] in ids:
                    voice = self.voices.pop(ckey)
                    self.voices[person] = voice
                    self._store()
                    return voice
            pool = [v for v in ids if v != narrator] or ids
            if not pool:
                return narrator
            used = list(self.voices.values())
            voice = min(pool, key=lambda v: (used.count(v), pool.index(v)))
            self.voices[key or person] = voice
            self._store()
            return voice

    def learn(self, colours: dict[str, str]) -> None:
        with self._lock:
            if any(self.colours.get(c) != p for c, p in colours.items()):
                self.colours.update(colours)
                self._store()

    def set(self, person: str, voice: str) -> None:
        with self._lock:
            key = self._key(person) or person
            self.voices[key] = voice
            self._store()

    def forget(self) -> None:
        with self._lock:
            self.voices.clear()
            self.colours.clear()
            self._store()

    def table(self) -> list[list[str]]:
        with self._lock:
            by_colour = {p.casefold(): c for c, p in self.colours.items()}
            rows = []
            for person, voice in self.voices.items():
                if person.startswith("colour:"):
                    rows.append([f"(someone in {person.split(':', 1)[1]})", voice, person.split(":", 1)[1]])
                else:
                    rows.append([person, voice, by_colour.get(person.casefold(), "")])
            return rows

    def _store(self) -> None:
        if self._save:
            self._save({"people_voices": dict(self.voices), "people_colours": dict(self.colours)})


# ---------------------------------------------------------------- reading a reply


class Reader:
    """Reads text for the voice server: notes skipped, narration in the voice asked for,
    each person's lines in their own voice, joined into one clip."""

    def __init__(self, settings: dict, library, people: PeopleVoices, notes: NoteFilter | None = None):
        self.settings = settings
        self.library = library
        self.people = people
        self.notes = notes or NoteFilter()

    def plan(self, text: str, voice) -> list[tuple[str, object]]:
        """[(text, voice), ...] in reading order; empty when there's nothing to read."""
        if self.settings.get("skip_notes", True):
            text = self.notes(text)
        lines = [ln for ln in text.split("\n") if clean(ln)]
        if not self.settings.get("voices_per_person", True):
            spoken = " ".join(clean(ln) for ln in lines)
            return [(spoken, voice)] if spoken else []
        colours = dict(self.people.colours)
        plan: list[tuple[str, object]] = []
        for line in lines:
            parts = split(line, colours)
            # Learn colours before handing out voices, so someone first heard only by their
            # colour keeps that voice once their name turns up, even later in the same reply.
            learned = {c: p for c, p in colours.items() if self.people.colours.get(c) != p}
            if learned:
                self.people.learn(learned)
            for part in parts:
                who = voice
                if part.quote and part.speaker:
                    found = self.library.get(self.people.voice_for(part.speaker, voice.id))
                    who = found or voice
                if plan and plan[-1][1].id == who.id:
                    plan[-1] = (plan[-1][0] + " " + part.text, plan[-1][1])
                else:
                    plan.append((part.text, who))
        return plan

    def read(self, text: str, voice, speak) -> tuple[np.ndarray, int] | None:
        """The clip, or None when everything in it was notes. speak(text, voice) -> (wav, sr)."""
        plan = self.plan(text, voice)
        if not plan:
            return None
        clips, sr = [], 24000
        for words, who in plan:
            wav, sr = speak(words, who)
            clips.append(wav)
        return (clips[0] if len(clips) == 1 else audio.concat(clips, sr, gap_s=0.15)), sr


def silence(sr: int = 24000, seconds: float = 0.05) -> tuple[np.ndarray, int]:
    return np.zeros(int(sr * seconds), dtype=np.float32), sr
