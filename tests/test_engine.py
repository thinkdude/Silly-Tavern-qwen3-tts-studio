import numpy as np
import pytest

from tests.conftest import SR, FakeEngine, FakeModel, tone


def test_design_returns_takes(engine):
    wavs, sr = engine.design("Hello.", "old man", "English", takes=3)
    assert len(wavs) == 3 and sr == SR
    assert engine.loaded() == ["design"]


def test_models_swap_unless_keep_both(engine):
    engine.design("Hello.", "old man")
    engine.speak("Hi.", ref_audio=(tone(4), SR), ref_text="hello")
    assert engine.loaded() == ["clone"]
    both = FakeEngine(keep_both_loaded=True)
    both.design("Hello.", "old man")
    both.speak("Hi.", ref_audio=(tone(4), SR), ref_text="hello")
    assert both.loaded() == ["clone", "design"]


def test_speak_saved_voice_caches_prompt(engine, library):
    v = library.save("Bob", tone(4), SR, kind="cloned", ref_text="hello there", language="English")
    engine.speak("One.", voice=v)
    engine.speak("Two.", voice=v)
    model = engine._models["clone"]
    prompts = [c for c in model.calls if c[0] == "prompt"]
    assert len(prompts) == 1 and prompts[0][1] == "hello there"
    clone_calls = [c for c in model.calls if c[0] == "clone"]
    assert clone_calls[0][2] == ["English"]


def test_long_text_is_batched(engine, library):
    v = library.save("Bob", tone(4), SR, kind="cloned", ref_text="hello")
    text = " ".join(f"Sentence {i} goes on for a little while to take up space." for i in range(40))
    wav, sr = engine.speak(text, voice=v)
    calls = [c for c in engine._models["clone"].calls if c[0] == "clone"]
    assert len(calls) > 1 and all(len(c[1]) <= 4 for c in calls)
    assert len(wav) > SR  # many 0.5s chunks joined


def test_no_transcript_uses_xvector(engine):
    engine.speak("Hi.", ref_audio=(tone(4), SR), ref_text="")
    prompt = [c for c in engine._models["clone"].calls if c[0] == "prompt"][0]
    assert prompt == ("prompt", None, True)


def test_unload_clears_prompt_cache(engine, library):
    v = library.save("Bob", tone(4), SR, kind="cloned", ref_text="hello")
    engine.speak("One.", voice=v)
    engine.unload()
    assert engine.loaded() == [] and engine._prompt_cache == {}


def test_empty_text(engine):
    with pytest.raises(ValueError):
        engine.speak("   ", ref_audio=(tone(4), SR), ref_text="x")


def test_oom_message():
    class OOMEngine(FakeEngine):
        def _load_model(self, kind):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    from app.engine import OutOfMemory

    with pytest.raises(OutOfMemory, match="ran out of memory"):
        OOMEngine().design("Hi.", "x")


def test_qwen_tts_import_hides_sox_and_flash_attn_warnings(capfd, caplog):
    import logging
    import os

    from app.engine import _quiet_startup_checks

    with _quiet_startup_checks():
        os.popen("sox -h").close()  # what the sox package runs on import
        logging.getLogger("sox").warning("SoX could not be found!")
        print("\n********\nWarning: flash-attn is not installed. Will only run the manual PyTorch version.\n********\n ")
        print("something else")
    out, err = capfd.readouterr()
    assert out.strip() == "something else"
    assert "sox" not in err.lower()
    assert "SoX" not in caplog.text
    logging.getLogger("sox").warning("SoX could not be found!")  # only hidden during the import
    assert "SoX" in caplog.text


class SilentModel(FakeModel):
    """Like a model running at a precision the GPU can't handle."""

    def __init__(self, kind, bad_takes):
        super().__init__(kind)
        self.bad_takes = bad_takes

    def generate_voice_design(self, text, instruct, language):
        wavs, sr = super().generate_voice_design(text, instruct, language)
        return [np.full_like(w, np.nan) if i in self.bad_takes else w for i, w in enumerate(wavs)], sr

    def generate_voice_clone(self, text, language, voice_clone_prompt):
        wavs, sr = super().generate_voice_clone(text, language, voice_clone_prompt)
        return [np.zeros_like(w) for w in wavs], sr


def silent_engine(bad_takes, **settings):
    eng = FakeEngine(**settings)
    eng._load_model = lambda kind: SilentModel(kind, bad_takes)
    return eng


def test_silent_takes_are_dropped():
    wavs, _ = silent_engine({1}).design("Hello.", "old man", takes=3)
    assert len(wavs) == 2


def test_all_silent_takes_explain_the_precision_fix():
    from app.engine import SilentOutput

    with pytest.raises(SilentOutput, match="float16.*float32"):
        silent_engine({0, 1}, dtype="float16").design("Hello.", "old man", takes=2)
    with pytest.raises(SilentOutput, match="Try generating again"):
        silent_engine({0}, dtype="float32").design("Hello.", "old man", takes=1)


def test_silent_speech_raises():
    from app.engine import SilentOutput

    with pytest.raises(SilentOutput):
        silent_engine(set()).speak("Hi there.", ref_audio=(tone(4), SR), ref_text="hello")
