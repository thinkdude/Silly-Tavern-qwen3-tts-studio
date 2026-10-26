"""The voice library: one folder per saved voice under voices/.

voices/<voice-id>/
    voice.json      name, type, language, transcript of the reference clip, ...
    reference.wav   the clip the clone model copies the voice from

Designed voices are stored the same way as cloned ones (a reference clip made
by the VoiceDesign model), so every voice sounds the same each time it is used.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio
from .config import VOICES_DIR


@dataclass
class Voice:
    id: str
    name: str
    kind: str  # "designed" or "cloned"
    ref_text: str = ""
    language: str = "Auto"
    description: str = ""
    x_vector_only: bool = False
    created: float = field(default_factory=time.time)
    folder: Path | None = field(default=None, repr=False, compare=False)

    @property
    def reference_path(self) -> Path:
        return self.folder / "reference.wav"

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop("folder")
        return data


def slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug[:48] or "voice"


class VoiceLibrary:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else VOICES_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[Voice]:
        voices = []
        for meta in sorted(self.root.glob("*/voice.json")):
            v = self._read(meta)
            if v and v.reference_path.exists():
                voices.append(v)
        return sorted(voices, key=lambda v: v.name.lower())

    def ids(self) -> list[str]:
        return [v.id for v in self.list()]

    def get(self, key: str) -> Voice | None:
        """Look a voice up by id or (case-insensitive) display name."""
        if not key:
            return None
        key = key.strip()
        meta = self.root / key / "voice.json"
        if meta.exists():
            return self._read(meta)
        low = key.lower()
        for v in self.list():
            if v.name.lower() == low or v.id == slugify(key):
                return v
        return None

    def save(
        self,
        name: str,
        wav: np.ndarray,
        sr: int,
        *,
        kind: str,
        ref_text: str = "",
        language: str = "Auto",
        description: str = "",
        x_vector_only: bool = False,
        trim: bool = True,
    ) -> Voice:
        name = (name or "").strip()
        if not name:
            raise ValueError("Give the voice a name first.")
        wav = audio.to_mono_float(wav)
        if trim:
            wav = audio.trim_silence(wav, sr)
        if wav.size < sr:  # under one second
            raise ValueError("The reference clip is too short. Use at least 3 seconds of speech.")
        wav = audio.normalize_peak(wav)

        voice_id = self._unique_id(slugify(name))
        v = Voice(
            id=voice_id,
            name=name,
            kind=kind,
            ref_text=(ref_text or "").strip(),
            language=language or "Auto",
            description=(description or "").strip(),
            x_vector_only=bool(x_vector_only),
            folder=self.root / voice_id,
        )
        v.folder.mkdir(parents=True, exist_ok=False)
        sf.write(str(v.reference_path), wav, sr, subtype="PCM_16")
        self._write(v)
        return v

    def rename(self, key: str, new_name: str) -> Voice:
        v = self._require(key)
        new_name = (new_name or "").strip()
        if not new_name:
            raise ValueError("The new name is empty.")
        v.name = new_name
        self._write(v)
        return v

    def update_text(self, key: str, *, ref_text: str | None = None, language: str | None = None) -> Voice:
        v = self._require(key)
        if ref_text is not None:
            v.ref_text = ref_text.strip()
            v.x_vector_only = not v.ref_text
        if language is not None:
            v.language = language
        self._write(v)
        return v

    def delete(self, key: str) -> None:
        v = self._require(key)
        shutil.rmtree(v.folder)

    # -- internals ---------------------------------------------------------

    def _require(self, key: str) -> Voice:
        v = self.get(key)
        if v is None:
            raise KeyError(f"No saved voice called '{key}'.")
        return v

    def _unique_id(self, base: str) -> str:
        cand, n = base, 2
        while (self.root / cand).exists():
            cand = f"{base}-{n}"
            n += 1
        return cand

    def _read(self, meta: Path) -> Voice | None:
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            data["id"] = meta.parent.name  # the folder name is the source of truth
            known = {k: data[k] for k in Voice.__dataclass_fields__ if k in data and k != "folder"}
            return Voice(**known, folder=meta.parent)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            print(f"[voices] Skipping {meta.parent.name}: {exc}")
            return None

    def _write(self, v: Voice) -> None:
        (v.folder / "voice.json").write_text(json.dumps(v.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
