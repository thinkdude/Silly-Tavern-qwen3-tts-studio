from app.textsplit import split_text


def test_short_text_is_one_chunk():
    assert split_text("  Hello   there.  ") == ["Hello there."]


def test_empty():
    assert split_text("") == []
    assert split_text("   ") == []


def test_long_text_breaks_on_sentences():
    text = " ".join(f"This is sentence number {i}, and it rambles on a bit." for i in range(30))
    chunks = split_text(text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert all(c.endswith(".") for c in chunks)
    assert " ".join(chunks) == text


def test_very_long_sentence_is_wrapped():
    text = "word " * 200
    chunks = split_text(text, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks) == text.strip()


def test_cjk_text_without_spaces():
    text = "今天天气很好。我们去公园散步吧！" * 20
    chunks = split_text(text, max_chars=50)
    assert all(len(c) <= 50 for c in chunks)
    assert "".join(chunks) == text
