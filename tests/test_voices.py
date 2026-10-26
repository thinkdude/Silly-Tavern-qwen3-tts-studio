import pytest

from tests.conftest import SR, tone


def test_save_list_get(library):
    v = library.save("Grumpy Wizard", tone(4), SR, kind="designed", ref_text="Hello.", description="old man")
    assert v.id == "grumpy-wizard"
    assert v.reference_path.exists()
    assert [x.id for x in library.list()] == ["grumpy-wizard"]
    assert library.get("grumpy-wizard").name == "Grumpy Wizard"
    assert library.get("GRUMPY WIZARD").id == "grumpy-wizard"
    assert library.get("nope") is None


def test_unique_ids(library):
    a = library.save("Bob", tone(4), SR, kind="cloned", ref_text="hi")
    b = library.save("Bob", tone(4), SR, kind="cloned", ref_text="hi")
    assert (a.id, b.id) == ("bob", "bob-2")


def test_rename_keeps_id(library):
    library.save("Bob", tone(4), SR, kind="cloned", ref_text="hi")
    v = library.rename("bob", "Robert")
    assert v.id == "bob" and library.get("bob").name == "Robert"


def test_update_transcript(library):
    library.save("Bob", tone(4), SR, kind="cloned", x_vector_only=True)
    v = library.update_text("bob", ref_text="Now with words.")
    assert v.ref_text == "Now with words." and not v.x_vector_only


def test_delete(library):
    library.save("Bob", tone(4), SR, kind="cloned", ref_text="hi")
    library.delete("bob")
    assert library.list() == []
    with pytest.raises(KeyError):
        library.delete("bob")


def test_rejects_bad_input(library):
    with pytest.raises(ValueError):
        library.save("", tone(4), SR, kind="cloned")
    with pytest.raises(ValueError):
        library.save("Short", tone(0.3), SR, kind="cloned")


def test_unicode_name(library):
    v = library.save("Ünïcødé Voice!", tone(4), SR, kind="cloned", ref_text="hi")
    assert v.id == "unicde-voice"
    v2 = library.save("日本語", tone(4), SR, kind="cloned", ref_text="hi")
    assert v2.id == "voice" and library.get("日本語").id == "voice"
