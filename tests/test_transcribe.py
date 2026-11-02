"""Speech-to-text. The real model is 1.6 GB, so these use a tiny untrained Whisper built on the spot:
its words are nonsense, but every step around it (loading, language detection, word timings,
long clips, GPU/CPU placement) runs for real."""

import json

import numpy as np
import pytest

from app import transcribe

SR = 16000


def test_pieces_split_long_clips_at_a_quiet_moment():
    t = np.arange(70 * SR) / SR
    wav = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    wav[25 * SR : 25 * SR + SR // 5] = 0  # a pause at 25 s
    pieces = transcribe._pieces(wav)
    assert pieces[0][0] == 0 and 25 <= pieces[1][0] <= 25.2
    assert all(len(p) <= transcribe.PIECE_S * SR for _, p in pieces)
    assert sum(len(p) for _, p in pieces) == len(wav)
    assert [o for o, _ in transcribe._pieces(wav[: 10 * SR])] == [0]


@pytest.fixture(scope="module")
def tiny_whisper(tmp_path_factory):
    pytest.importorskip("transformers")
    from transformers import (
        GenerationConfig,
        WhisperConfig,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )
    from transformers.models.whisper.tokenization_whisper import LANGUAGES, bytes_to_unicode

    out = tmp_path_factory.mktemp("tiny-whisper")
    (out / "vocab.json").write_text(json.dumps({c: i for i, c in enumerate(bytes_to_unicode().values())}))
    (out / "merges.txt").write_text("#version: 0.2\n")
    eot = "<|endoftext|>"
    tok = WhisperTokenizer(str(out / "vocab.json"), str(out / "merges.txt"), unk_token=eot, bos_token=eot, eos_token=eot, pad_token=eot)
    langs = [f"<|{c}|>" for c in LANGUAGES]
    tasks = ["<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>"]
    tok.add_special_tokens({"additional_special_tokens": [eot, "<|startoftranscript|>", *langs, *tasks]})
    tok.add_tokens([f"<|{i * 0.02:.2f}|>" for i in range(1501)])  # timestamps follow <|notimestamps|>
    WhisperProcessor(feature_extractor=WhisperFeatureExtractor(feature_size=128), tokenizer=tok).save_pretrained(out)
    ids = tok.convert_tokens_to_ids
    special = {"pad_token_id": ids(eot), "bos_token_id": ids(eot), "eos_token_id": ids(eot), "decoder_start_token_id": ids("<|startoftranscript|>")}
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(
        WhisperConfig(
            vocab_size=len(tok), num_mel_bins=128, d_model=32, encoder_layers=2, decoder_layers=2,
            encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, **special,
        )
    )  # fmt: skip
    model.generation_config = GenerationConfig(
        **special,
        max_length=40,
        is_multilingual=True,
        lang_to_id={t: ids(t) for t in langs},
        task_to_id={"transcribe": ids("<|transcribe|>"), "translate": ids("<|translate|>")},
        no_timestamps_token_id=ids("<|notimestamps|>"),
        prev_sot_token_id=ids("<|startofprev|>"),
        alignment_heads=[[1, 0], [1, 1]],
        begin_suppress_tokens=[ids(eot)],
        suppress_tokens=[],
    )
    model.save_pretrained(out)
    yield {"stt_model": str(out), "device": "cuda:0"}  # falls back to the CPU without a GPU
    transcribe.unload()


def _voice(seconds, sr=44100):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 0.4 * t) > 0)).astype(np.float32), sr


def test_words_with_a_real_whisper_pipeline(tiny_whisper):
    wav, sr = _voice(6)
    words, lang = transcribe.words(wav, sr, tiny_whisper)
    assert lang and lang.isalpha()  # detected (nonsense, but a language code)
    assert all(0 <= s <= e <= 6.0 + 1e-6 and isinstance(w, str) for s, e, w in words)
    model, _ = transcribe._loaded[tiny_whisper["stt_model"]]
    assert next(model.parameters()).device.type == "cpu"  # the GPU is left free between uses


def test_long_clips_and_plain_text(tiny_whisper):
    wav, sr = _voice(35)
    words, lang = transcribe.words(wav, sr, tiny_whisper, language="en")
    assert lang == "en" and all(e <= 35.0 + 1e-6 for _, e, _ in words)
    text, lang = transcribe.text(wav[: 5 * sr], sr, tiny_whisper, language="ja")
    assert isinstance(text, str) and lang == "ja"
    lang = transcribe.detect(wav[: 5 * sr], sr, tiny_whisper)
    assert lang and lang.replace("-", "").isalpha()


def test_a_missing_model_reads_well(tmp_path):
    with pytest.raises(ValueError, match="Couldn't load the speech-to-text model"):
        transcribe.words(*_voice(2), {"stt_model": str(tmp_path)})  # an empty folder
