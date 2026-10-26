"""Loads the Qwen3-TTS models on demand and runs generation.

Two models are used:
  * design: Qwen3-TTS-12Hz-1.7B-VoiceDesign, turns a text description into a voice
  * clone:  Qwen3-TTS-12Hz-1.7B-Base (or 0.6B), speaks new text in a saved voice

Unless ``keep_both_loaded`` is on, loading one unloads the other so a single
mid-range GPU is enough.
"""

from __future__ import annotations

import gc
import threading
import time
from typing import Any

import numpy as np

from . import audio
from .config import resolve_model
from .textsplit import split_text
from .voices import Voice

# How many text chunks to generate in one batch for long inputs.
BATCH_SIZE = 4


class OutOfMemory(RuntimeError):
    pass


class TTSEngine:
    def __init__(self, settings: dict[str, Any], idle_unload_minutes: float = 0):
        self.settings = settings
        self.idle_unload_s = float(idle_unload_minutes or 0) * 60
        self._models: dict[str, Any] = {}
        self._last_used: dict[str, float] = {}
        self._prompt_cache: dict[tuple, Any] = {}
        self._lock = threading.RLock()
        if self.idle_unload_s > 0:
            threading.Thread(target=self._idle_watch, daemon=True).start()

    # -- model management --------------------------------------------------

    def _torch_dtype(self):
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(str(self.settings.get("dtype", "bfloat16")).lower(), torch.bfloat16)

    def _get(self, kind: str):
        with self._lock:
            if kind in self._models:
                self._last_used[kind] = time.time()
                return self._models[kind]
            if not self.settings.get("keep_both_loaded"):
                for other in [k for k in self._models if k != kind]:
                    self.unload(other)

            print(f"[engine] Loading {kind} model ...", flush=True)
            t0 = time.time()
            try:
                model = self._load_model(kind)
            except Exception as exc:
                self._raise_friendly(exc)
            print(f"[engine] {kind} model ready in {time.time() - t0:.0f}s", flush=True)
            self._models[kind] = model
            self._last_used[kind] = time.time()
            return model

    def _load_model(self, kind: str):
        from qwen_tts import Qwen3TTSModel

        repo = self.settings["design_model" if kind == "design" else "clone_model"]
        path = resolve_model(repo)
        print(f"[engine]   from {path}", flush=True)
        return Qwen3TTSModel.from_pretrained(
            path,
            device_map=self.settings.get("device", "cuda:0"),
            dtype=self._torch_dtype(),
            attn_implementation=self.settings.get("attn_implementation") or None,
        )

    def unload(self, kind: str | None = None) -> None:
        with self._lock:
            kinds = [kind] if kind else list(self._models)
            for k in kinds:
                if self._models.pop(k, None) is not None:
                    print(f"[engine] Unloaded {k} model", flush=True)
                if k == "clone":
                    self._prompt_cache.clear()
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def loaded(self) -> list[str]:
        return sorted(self._models)

    def preload(self, kind: str = "clone") -> None:
        self._get(kind)

    def _idle_watch(self) -> None:
        while True:
            time.sleep(30)
            now = time.time()
            for k, t in list(self._last_used.items()):
                if k in self._models and now - t > self.idle_unload_s and self._lock.acquire(blocking=False):
                    try:
                        self.unload(k)
                    finally:
                        self._lock.release()

    @staticmethod
    def _raise_friendly(exc: Exception):
        msg = str(exc)
        if "out of memory" in msg.lower() or exc.__class__.__name__ == "OutOfMemoryError":
            raise OutOfMemory(
                "Your GPU ran out of memory. If the voice server or another GPU app is running, "
                "close it and try again, or switch to the 0.6B clone model in settings.json."
            ) from exc
        raise exc

    # -- generation --------------------------------------------------------

    def _run(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            self._raise_friendly(exc)

    def design(self, text: str, instruct: str, language: str = "Auto", takes: int = 1) -> tuple[list[np.ndarray], int]:
        """Generate ``takes`` different readings of ``text`` in a voice matching ``instruct``."""
        takes = max(1, int(takes))
        with self._lock:
            model = self._get("design")
            wavs, sr = self._run(
                model.generate_voice_design,
                text=[text] * takes,
                instruct=[instruct] * takes,
                language=[language or "Auto"] * takes,
            )
            self._last_used["design"] = time.time()
        return [audio.to_mono_float(w) for w in wavs], int(sr)

    def _prompt(self, ref_audio, ref_text: str, x_vector_only: bool, cache_key: tuple | None):
        if cache_key is not None and cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]
        model = self._get("clone")
        x_vector_only = x_vector_only or not (ref_text or "").strip()
        items = self._run(
            model.create_voice_clone_prompt,
            ref_audio=ref_audio,
            ref_text=None if x_vector_only else ref_text,
            x_vector_only_mode=x_vector_only,
        )
        if cache_key is not None:
            self._prompt_cache[cache_key] = items
        return items

    def speak(
        self,
        text: str,
        *,
        voice: Voice | None = None,
        ref_audio=None,
        ref_text: str = "",
        x_vector_only: bool = False,
        language: str | None = None,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """Speak ``text`` in a saved voice, or in the voice of an unsaved reference clip."""
        chunks = split_text(text)
        if not chunks:
            raise ValueError("There is no text to speak.")
        if voice is not None:
            ref_path = voice.reference_path
            key = (voice.id, ref_path.stat().st_mtime, voice.ref_text, voice.x_vector_only)
            ref_audio, ref_text, x_vector_only = str(ref_path), voice.ref_text, voice.x_vector_only
            language = language or voice.language
        else:
            if ref_audio is None:
                raise ValueError("No voice or reference clip was given.")
            key = None
        language = language or "Auto"

        with self._lock:
            model = self._get("clone")
            items = self._prompt(ref_audio, ref_text, x_vector_only, key)
            out: list[np.ndarray] = []
            sr = 24000
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                wavs, sr = self._run(
                    model.generate_voice_clone,
                    text=batch,
                    language=[language] * len(batch),
                    voice_clone_prompt=items * len(batch),
                )
                out.extend(wavs)
            self._last_used["clone"] = time.time()
        wav = audio.concat(out, int(sr))
        return audio.change_speed(wav, speed), int(sr)

    # -- info --------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "loaded_models": self.loaded(),
            "device": self.settings.get("device"),
            "dtype": self.settings.get("dtype"),
            "clone_model": self.settings.get("clone_model"),
            "design_model": self.settings.get("design_model"),
        }
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                info["gpu"] = torch.cuda.get_device_name(0)
                info["vram_used_gb"] = round((total - free) / 1024**3, 1)
                info["vram_total_gb"] = round(total / 1024**3, 1)
        except Exception:
            pass
        return info
