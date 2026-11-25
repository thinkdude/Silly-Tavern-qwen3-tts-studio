"""Read a reply the way a person would: skip the notes, and give each person their own voice.

Role-play replies often carry more than the story: a "# Reasoning Process" list at the top,
a status line in [ brackets | like this ], and whole blocks of game state in <internal_states>,
<details> or <!-- --> markup. None of that should be read out. And when several people talk
in one reply, each of them should sound like themselves.

How SillyTavern sends a reply matters here. With "Narrate by paragraphs" it sends the finished
reply one line at a time, so a block of notes spans many requests; and it always turns line
breaks into spaces before sending, so a request never shows where its lines were.

    NoteFilter      drops notes. It remembers, between requests, that it's inside a block
                    until the block ends (or a new reply starts one again, or it's been open
                    too long), so the lines of an <internal_states> block are never read.
    split           cuts a paragraph into narration and quotes, and works out who says each
                    quote: a "Name:" in front of it, a speech tag next to it ("Hi," Jake says),
                    an action beat beside it (Jake grins. "Hi."), or its colour
                    (<cyan>"Hi"</cyan>) once that colour has been seen with a speech tag.
    PeopleVoices    which saved voice each person speaks with: one you picked, one named after
                    them, or one handed out the first time they speak (remembered in settings.json).
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

# Blocks that are never read: their tags, everything inside them, however many requests they span.
SKIP_TAGS = (
    "internal_states", "details", "summary", "think", "thinking", "reasoning", "thoughts",
    "analysis", "notes", "status", "stats", "hud", "ooc", "system", "memory", "scratchpad",
)  # fmt: skip
# Blocks a reply has only one of at a time: seeing one open again means a new reply has started
# (say the last one was cut off before its block closed), so the old block is forgotten.
ONE_PER_REPLY = frozenset({"internal_states", "think", "thinking", "reasoning", "thoughts", "analysis", "hud", "scratchpad"})
_TAGS = "|".join(SKIP_TAGS)
_ATTRS = r"(?:\s+[\w:.-]+(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'<>]+))?)*\s*"
_BLOCK = re.compile(
    r"<!--\s*(?P<start>[\w-]+?)_START\s*-->"  # <!-- GFX_START --> ... <!-- GFX_END -->
    r"|<!--\s*(?P<end>[\w-]+?)_END\s*-->"
    r"|(?P<copen><!--)|(?P<cclose>-->)"
    r"|(?P<fence>```|~~~)"
    rf"|<(?P<open>{_TAGS})(?:{_ATTRS})>"
    rf"|</(?P<close>{_TAGS})\s*>"
    rf"|(?P<inline><(?:{_TAGS})\b[^<>]*>)",  # <status/>, <System: Quest accepted>: a note on its own
    re.IGNORECASE,
)
_STATUS = re.compile(r"\[[^\[\]]*\|[^\[\]]*\]")  # [ Time 08:11 AM | Day 1 | Office ]
# Lines that are notes, not story: dropped whole.
_ITEM = r"(?:[-•+*]|\d{1,3}[.)])"
_KEY = r"(?:<[^<>]+>\s*)*[^\s\"“«:][^\"“«:]{0,59}:(?:\s|$)"  # Game State:
_NOTE_LINES = (
    re.compile(r"^\s*#{1,6}\s+[A-Z]"),  # # Reasoning Process
    re.compile(rf"^\s*{_ITEM}\s*{_KEY}"),  # - Game State: ...   1. Option A: ...
    re.compile(rf"^\s*{_ITEM}\s*<b>"),  # - <b>Tucker</b> | Agenda: ...
    re.compile(r"^\s*\|.*\|\s*$"),  # | table | row |
    re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,}|={3,})\s*$"),  # ---
)
# SillyTavern joins a request's lines with spaces. Where a note line would have started (a
# heading, or a "- Key:" bullet), cut there; numbers aren't used, since "Roll = 14. Adjust:" isn't a list.
_LINE_START = re.compile(rf"(?<=\S)\s+(?=#{{1,6}}\s+[A-Z]|[-•+*]\s*{_KEY}|[-•+*]\s*<b>)")
_FIRST_SENTENCE = re.compile(r"[.!?…]\s+(?=[A-Z\"“<*])")
_REAL_TAG = re.compile(r"</?[A-Za-z][\w-]*(?:\s[^<>]*)?/?>")
# An open block is forgotten after this long without requests (the reply was cut off: with
# "Narrate by paragraphs" SillyTavern asks for each line as soon as the last one's audio is back) ...
FORGET_OPEN_BLOCK_S = 15.0
# ... or after this long in all, however busy it's been.
MAX_OPEN_BLOCK_S = 300.0


class NoteFilter:
    """Takes out notes and hidden blocks, remembering between requests that a block is open."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._open: list[str] = []  # open blocks, innermost last: "details", "comment", "gfx:GFX", ...
        self._opened = 0.0
        self._last = 0.0

    @property
    def inside_block(self) -> bool:
        return bool(self._open)

    def reset(self) -> None:
        with self._lock:
            self._open.clear()

    def touch(self) -> None:
        """Call when a request is done (its audio made), so slow speaking doesn't count as idle."""
        with self._lock:
            self._last = self._clock()

    def __call__(self, text: str) -> str:
        with self._lock:
            now = self._clock()
            if self._open and (now - self._last > FORGET_OPEN_BLOCK_S or now - self._opened > MAX_OPEN_BLOCK_S):
                self._open.clear()
            lines = (text or "").replace("\r\n", "\n").split("\n")
            kept = [line for line in (self._line(raw, now) for raw in lines) if line]
            if len(lines) > 1:  # a whole reply with its line breaks: what's left open ends with it
                self._open.clear()
            self._last = now
            return "\n".join(kept)

    def _push(self, kind: str, now: float) -> None:
        if not self._open:
            self._opened = now
        self._open.append(kind)

    def _close(self, kind: str) -> None:
        """Close the innermost block of this kind, and anything left open inside it."""
        if kind in self._open:
            del self._open[len(self._open) - 1 - self._open[::-1].index(kind) :]

    def _line(self, line: str, now: float) -> str:
        out, pos = [], 0
        for m in _BLOCK.finditer(line):
            if not self._open:
                out.append(line[pos : m.start()])
            pos = m.end()
            top = self._open[-1] if self._open else None
            if top == "comment":  # nothing counts inside a comment but its end
                if m.group("cclose"):
                    self._open.pop()
                continue
            if top == "fence":
                if m.group("fence"):
                    self._open.pop()
                continue
            if m.group("start"):
                kind = "gfx:" + m.group("start").lower()
                if kind in self._open:  # a new reply: forget the one left open
                    self._open.clear()
                self._push(kind, now)
            elif m.group("end"):
                self._close("gfx:" + m.group("end").lower())
            elif m.group("copen"):
                self._push("comment", now)
            elif m.group("fence"):
                self._push("fence", now)
            elif m.group("open"):
                kind = m.group("open").lower()
                if kind in ONE_PER_REPLY and kind in self._open:
                    self._open.clear()
                self._push(kind, now)
            elif m.group("close"):
                self._close(m.group("close").lower())
            # A lone "-->" or a one-line note like <status/> is just left out.
        if not self._open:
            out.append(line[pos:])
        kept = _STATUS.sub(" ", "".join(out))
        # Note lines are dropped whole.
        pieces = _LINE_START.split(kept)
        story = []
        for n, piece in enumerate(pieces):
            if not any(rule.match(piece) for rule in _NOTE_LINES):
                story.append(piece)
            elif len(pieces) > 1 and n == len(pieces) - 1:
                # Several lines squeezed into one request (SillyTavern sending a whole reply, or a
                # streamed chunk): the last note line has no end of its own, so the story is taken
                # to start after its first sentence, rather than losing it all.
                m = _FIRST_SENTENCE.search(piece)
                if m:
                    story.append(piece[m.end() :])
        kept = " ".join(story)
        return kept if _REAL_TAG.sub("", kept).strip(" \t*_~") else ""


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
    r"<(?P<open>[a-z][\w-]*)(?P<attrs>(?:\s[^<>]*)?)>"  # any opening tag, whole (attributes and all)
    r"|</(?P<close>[a-z][\w-]*)\s*>"
    r"|(?P<quote>\"[^\"]*\"|“[^”]*”|«[^»]*»|「[^」]*」)",
    re.IGNORECASE,
)
_COLOR_ATTR = re.compile(r"(?<![\w-])color\s*[:=]\s*[\"']?\s*(?P<c>#?[\w]+(?:\([^)]*\))?)", re.IGNORECASE)
# Capitalised words that start sentences but aren't anyone's name.
NOT_NAMES = frozenset(
    """a an and as at but by for from he her hers him his how i if in into it its me my no not now of
    oh on once or our she so some still that the their them then there these they this those though
    to too up us we what when where which while who why with yet you your yours okay ok yes yeah
    well hey god fuck shit damn sir maam mister miss mrs mr ms dr everyone someone something nothing
    somebody nobody everybody another each both one two three after before behind beside around
    outside inside across over under above below between through again just even only maybe""".split()
)
SPEECH_VERBS = (
    "says said asks asked replies replied answers answered shouts shouted yells yelled whispers whispered "
    "mutters muttered murmurs murmured growls growled grunts grunted barks barked snaps snapped calls called "
    "adds added continues continued sighs sighed groans groaned laughs laughed chuckles chuckled hisses hissed "
    "snarls snarled purrs purred drawls drawled admits admitted insists insisted demands demanded orders ordered "
    "begs begged cries cried screams screamed breathes breathed rasps rasped coos cooed teases teased pants "
    "panted wheezes wheezed gasps gasped exclaims exclaimed declares declared announces announced counters "
    "countered agrees agreed explains explained warns warned interrupts interrupted mumbles mumbled stammers "
    "stammered booms boomed roars roared jokes joked offers offered notes noted responds responded retorts "
    "retorted remarks remarked repeats repeated tells told calls hollers hollered pleads pleaded"
).split()
_SPEECH = "|".join(sorted(set(SPEECH_VERBS), key=len, reverse=True))
_NAME = r"[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*){0,2}"
_LEAD = r"^[\s,.;:!?—–-]*"
_SPEECH_AFTER = re.compile(rf"{_LEAD}(?P<name>{_NAME})\s+(?:\w+ly\s+)?(?:{_SPEECH})\b")
_SAYS_NAME = re.compile(rf"{_LEAD}(?:{_SPEECH})\s+(?P<name>{_NAME})")
_BEAT = re.compile(rf"(?P<name>{_NAME})(?:(?:'s|’s)\s+[a-z]|\s+(?:\w+ly\s+)?[a-z]+(?:s|ed)\b)")
_LABEL = re.compile(rf"(?P<name>{_NAME})\s*:\s*$")
_OWN_SUBJECT = re.compile(rf"{_LEAD}(?:he|she|they|you|i|we|it|his|her|the\s+\w+|a\s+\w+)\b", re.IGNORECASE)
_PRONOUN_SAYS = re.compile(rf"{_LEAD}(?:he|she|they|you|i|we)\s+(?:\w+ly\s+)?(?:{_SPEECH})\b", re.IGNORECASE)
_SENTENCES = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class Part:
    text: str
    quote: bool = False
    speaker: str | None = None  # a person's name, or "colour:<name>" when only the colour is known


def clean(text: str) -> str:
    """Text as it should be spoken: no markup, no *asterisks*, single spaces."""
    text = _REAL_TAG.sub(" ", text or "")
    text = text.replace("_", " ")
    text = re.sub(r"\*+|~{2,}|~(?=\s|$)", "", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s+(?=[,.;:!?…])", "", text).strip()


def _name(candidate: str | None) -> str | None:
    """'Jake', 'Officer Lee', or 'Jake' from 'Suddenly Jake' and "Jake's"; None for 'He' or 'The'."""
    words = (candidate or "").split()
    while words and (words[0].lower().strip("'’") in NOT_NAMES or words[0].lower().endswith("ly")):
        words.pop(0)
    if not words or any(w.lower().strip("'’") in NOT_NAMES for w in words):
        return None
    name = " ".join(words)
    for suffix in ("'s", "’s"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name or None


def _last_sentence(text: str) -> str:
    sentences = [s for s in _SENTENCES.split(clean(text)) if s.strip()]
    return sentences[-1] if sentences else ""


def _label_before(before: str) -> str | None:
    """Kira in: 'Kira: ' right before a quote."""
    m = _LABEL.search(clean(before))
    return _name(m.group("name")) if m else None


def _speech_after(after: str) -> str | None:
    """Jake in: ' Jake says, grinning.' or ', says Jake.'"""
    text = clean(after)
    m = _SPEECH_AFTER.match(text) or _SAYS_NAME.match(text)
    return _name(m.group("name")) if m else None


def _beat_before(before: str) -> str | None:
    """Marlon in: 'Marlon shrugs. ' or 'Jake leans in, ' just before a quote (the name nearest it)."""
    names = [_name(m.group("name")) for m in _BEAT.finditer(_last_sentence(before))]
    names = [n for n in names if n]
    return names[-1] if names else None


def _beat_after(after: str) -> str | None:
    """Marlon in: ' Marlon shoves Jake aside.' right after a quote."""
    m = re.match(_LEAD + _BEAT.pattern, clean(after))
    return _name(m.group("name")) if m else None


def _colour_of(tag: str, attrs: str) -> str | None:
    m = _COLOR_ATTR.search(attrs or "")
    if m:
        return re.sub(r"\s+", "", m.group("c")).lower()
    return tag.lower() if tag.lower() in COLOURS else None


def split(paragraph: str, colours: dict[str, str] | None = None) -> list[Part]:
    """Cut a paragraph into narration and quotes, and work out who says each quote.

    `colours` maps a dialogue colour to the person it belongs to. It's read, and updated when a
    quote in a colour nobody has yet comes with a speech tag or a "Name:", so later quotes in
    that colour are known even without a name next to them.
    """
    colours = colours if colours is not None else {}
    if _REAL_TAG.sub("", paragraph).count('"') % 2:
        last = paragraph.rfind('"')
        before, after = paragraph[last - 1 : last] or " ", paragraph[last + 1 : last + 2]
        if (before.isspace() or before in ">([") and after and not after.isspace():
            paragraph += '"'  # a speech running on into the next paragraph: dialogue to the end
        else:
            # A stray quote mark somewhere: dialogue and narration can't be told apart, so it's narration.
            text = clean(paragraph)
            return [Part(text)] if text else []
    parts: list[Part] = []
    quote_colour: list[str | None] = []
    stack: list[tuple[str, str]] = []  # (tag, colour) for open tags that set a colour
    narration_start = 0
    for m in _TOKEN.finditer(paragraph):
        if m.group("open"):
            colour = _colour_of(m.group("open"), m.group("attrs"))
            if colour and not (m.group("attrs") or "").rstrip().endswith("/"):
                stack.append((m.group("open").lower(), colour))
        elif m.group("close"):
            if stack and stack[-1][0] == m.group("close").lower():
                stack.pop()
        else:
            before = paragraph[narration_start : m.start()]
            if clean(before):
                parts.append(Part(before))
            parts.append(Part(m.group("quote")[1:-1], quote=True))
            quote_colour.append(stack[-1][1] if stack else None)
            narration_start = m.end()
    rest = paragraph[narration_start:]
    if clean(rest) or not parts:
        parts.append(Part(rest))

    quotes = [i for i, p in enumerate(parts) if p.quote]
    for n, i in enumerate(quotes):
        colour = quote_colour[n]
        before = parts[i - 1].text if i > 0 and not parts[i - 1].quote else ""
        after = parts[i + 1].text if i + 1 < len(parts) and not parts[i + 1].quote else ""
        if colour and colour in colours:
            parts[i].speaker = colours[colour]  # a colour, once known, outranks a guess
            continue
        tagged = _label_before(before) or _speech_after(after)
        prev = parts[quotes[n - 1]].speaker if n else None
        name = tagged
        if name is None:
            beat = _beat_before(before)
            # '"Hi," Jake says. "Fine," she replies.': Jake's tag sits before the reply, but it isn't his.
            if beat and not (beat == prev and _PRONOUN_SAYS.match(clean(after))):
                name = beat
        name = name or _beat_after(after)
        if name is None and n:
            between = clean(before)
            others = {x for x in (_beat_before(before), _speech_after(before)) if x} - {prev}
            if (
                prev
                and quote_colour[n - 1] == colour
                and len(between.split()) <= 8
                and not others
                and not _OWN_SUBJECT.match(clean(after))
            ):
                name = prev  # "Hi," Jake says. "Come in." The second line is Jake's too.
        if colour and tagged and not tagged.startswith("colour:"):
            colours[colour] = tagged  # learned only from a clear speech tag or label
        parts[i].speaker = name or (f"colour:{colour}" if colour else None)
    for p in parts:
        if p.speaker and p.speaker.startswith("colour:") and p.speaker[7:] in colours:
            p.speaker = colours[p.speaker[7:]]
        p.text = clean(p.text)
    return [p for p in parts if p.text]


# ---------------------------------------------------------------- who speaks with which voice


class PeopleVoices:
    """Remembers each person's voice, and the colour each person's lines are written in.

    settings["people_voices"]   {"Jake": "deep-voice", ...}: picked in the control panel, taken
                                from SillyTavern's voice map, or handed out the first time
                                someone speaks
    settings["people_picked"]   the names whose voice you (or SillyTavern's voice map) chose
    settings["people_colours"]  {"cyan": "Jake", ...}, learned from the replies
    """

    def __init__(
        self,
        settings: dict,
        voice_ids: Callable[[], list[str]],
        save: Callable[[dict], None] | None = None,
        find: Callable[[str], object] | None = None,
    ):
        self.settings = settings
        self.voice_ids = voice_ids
        self.find = find  # a saved voice by id or name, e.g. VoiceLibrary.get
        self._save = save
        self._lock = threading.RLock()

    @property
    def voices(self) -> dict[str, str]:
        return self.settings.setdefault("people_voices", {})

    @property
    def colours(self) -> dict[str, str]:
        return self.settings.setdefault("people_colours", {})

    @property
    def picked(self) -> list[str]:
        return self.settings.setdefault("people_picked", [])

    def _key(self, person: str) -> str | None:
        return next((k for k in self.voices if k.casefold() == person.casefold()), None)

    def _is_picked(self, person: str) -> bool:
        return any(p.casefold() == person.casefold() for p in self.picked)

    def voice_for(self, person: str, narrator: str) -> str:
        """The person's voice. The first time they speak: a saved voice with their name, else one
        the narrator and the other people don't use yet, else the least used one."""
        with self._lock:
            ids = self.voice_ids()
            if person.startswith("colour:") and person[7:] in self.colours:
                person = self.colours[person[7:]]
            key = self._key(person)
            if key and self.voices[key] in ids:
                voice = self.voices[key]
                if voice != narrator or self._is_picked(key) or not [v for v in ids if v != narrator]:
                    return voice
                # Handed out earlier, but it's this character's own voice now: pick another.
            if not person.startswith("colour:") and not key:
                # Someone first heard only by colour keeps that voice once their name is known.
                colour = next((c for c, p in self.colours.items() if p.casefold() == person.casefold()), None)
                ckey = self._key(f"colour:{colour}") if colour else None
                if ckey and self.voices[ckey] in ids:
                    voice = self.voices.pop(ckey)
                    self.voices[person] = voice
                    self._store()
                    return voice
            named = self.find(person) if self.find and not person.startswith("colour:") else None
            if named is not None and getattr(named, "id", None) in ids:
                self.voices[key or person] = named.id
                self._store()
                return named.id
            pool = [v for v in ids if v != narrator]
            if not pool:
                return narrator  # nothing else to hand out yet; not remembered
            used = [v for k, v in self.voices.items() if k != key]
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
        """You picked this voice for them."""
        with self._lock:
            key = self._key(person) or person
            self.voices[key] = voice
            if not self._is_picked(key):
                self.picked.append(key)
            self._store()

    def seed(self, voices: dict[str, str]) -> list[str]:
        """Voices SillyTavern's voice map gives characters, for anyone you haven't picked yourself."""
        with self._lock:
            added = []
            for person, voice in voices.items():
                key = self._key(person)
                if key and self._is_picked(key):
                    continue
                self.voices[key or person] = voice
                if not self._is_picked(key or person):
                    self.picked.append(key or person)
                added.append(person)
            if added:
                self._store()
            return added

    def forget(self) -> None:
        with self._lock:
            self.voices.clear()
            self.colours.clear()
            self.picked.clear()
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
            self._save(
                {"people_voices": dict(self.voices), "people_colours": dict(self.colours), "people_picked": list(self.picked)}
            )


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
        skip_notes = self.settings.get("skip_notes", True)
        per_person = self.settings.get("voices_per_person", True)
        if not skip_notes and not per_person:
            return [(text, voice)] if text.strip() else []  # as it came, like before
        if skip_notes:
            text = self.notes(text)
        lines = [ln for ln in text.split("\n") if clean(ln)]
        if not per_person:
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
                    who = self.library.get(self.people.voice_for(part.speaker, voice.id)) or voice
                if plan and plan[-1][1].id == who.id:
                    plan[-1] = (plan[-1][0] + " " + part.text, plan[-1][1])
                else:
                    plan.append((part.text, who))
        return plan

    def read(self, text: str, voice, speak) -> tuple[np.ndarray, int] | None:
        """The clip, or None when everything in it was notes. speak(text, voice) -> (wav, sr)."""
        plan = self.plan(text, voice)
        try:
            if not plan:
                return None
            clips, sr = [], 24000
            for words, who in plan:
                wav, sr = speak(words, who)
                clips.append(wav)
            return (clips[0] if len(clips) == 1 else audio.concat(clips, sr, gap_s=0.15)), sr
        finally:
            self.notes.touch()


def silence(sr: int = 24000, seconds: float = 0.05) -> tuple[np.ndarray, int]:
    return np.zeros(int(sr * seconds), dtype=np.float32), sr
