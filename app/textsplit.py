"""Split long text into sentence-sized chunks so each generation stays short."""

from __future__ import annotations

import re

# Sentence enders for the languages Qwen3-TTS supports (Latin + CJK punctuation).
_SENTENCE_END = re.compile(r"(?<=[.!?。！？…])[\"'”’)\]]*\s+|(?<=[。！？])")
_CLAUSE_END = re.compile(r"(?<=[,;:，；：、])\s*")


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    words = text.split(" ")
    out, cur = [], ""
    for w in words:
        if len(w) > max_chars:  # e.g. CJK text with no spaces
            if cur:
                out.append(cur)
                cur = ""
            out.extend(w[i : i + max_chars] for i in range(0, len(w), max_chars))
            continue
        cand = f"{cur} {w}" if cur else w
        if len(cand) > max_chars:
            out.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    pieces = []
    for clause in (c.strip() for c in _CLAUSE_END.split(sentence) if c.strip()):
        pieces.extend(_hard_wrap(clause, max_chars) if len(clause) > max_chars else [clause])
    return pieces


def split_text(text: str, max_chars: int = 250) -> list[str]:
    """Return chunks of at most ``max_chars`` that break on sentence boundaries."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    pieces: list[str] = []
    for s in (s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()):
        pieces.extend(_split_long(s, max_chars))

    chunks, cur = [], ""
    for p in pieces:
        sep = "" if cur and _is_cjk(cur[-1]) else " "
        cand = f"{cur}{sep}{p}" if cur else p
        if len(cand) > max_chars and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks


def _is_cjk(ch: str) -> bool:
    return "　" <= ch <= "鿿" or "가" <= ch <= "힯" or "＀" <= ch <= "￯"
