from app import koboldcpp
from tests.conftest import SR, tone


def settings(tmp_path):
    return {"koboldcpp_voices_dir": str(tmp_path / "kcpp")}


def test_sync_copies_and_removes(library, tmp_path):
    s = settings(tmp_path)
    library.save("Grumpy Wizard", tone(4), SR, kind="designed", ref_text="Hello.")
    library.save("Dad", tone(4), SR, kind="cloned", ref_text="Hi.")
    r = koboldcpp.sync(library, s)
    folder = tmp_path / "kcpp"
    assert sorted(r["added"]) == ["dad.wav", "grumpy-wizard.wav"] and r["total"] == 2
    assert (folder / "dad.wav").read_bytes() == library.get("dad").reference_path.read_bytes()

    assert koboldcpp.sync(library, s)["added"] == []  # nothing changed

    library.delete("dad")
    r = koboldcpp.sync(library, s)
    assert r["removed"] == ["dad.wav"] and not (folder / "dad.wav").exists()


def test_sync_leaves_other_files_alone(library, tmp_path):
    s = settings(tmp_path)
    folder = tmp_path / "kcpp"
    folder.mkdir()
    (folder / "my-own-voice.wav").write_bytes(b"RIFF")
    library.save("Bob", tone(4), SR, kind="cloned", ref_text="hi")
    koboldcpp.sync(library, s)
    library.delete("bob")
    koboldcpp.sync(library, s)
    assert (folder / "my-own-voice.wav").exists()
    assert not (folder / "bob.wav").exists()


def test_names_and_urls(library, tmp_path):
    library.save("Bob", tone(4), SR, kind="cloned", ref_text="hi")
    assert koboldcpp.voice_names(library) == ["bob.wav"]
    assert koboldcpp.speech_url({}) == "http://127.0.0.1:5001/v1/audio/speech"
    assert koboldcpp.speech_url({"koboldcpp_url": "http://localhost:5002/"}) == "http://localhost:5002/v1/audio/speech"
    assert koboldcpp.voices_dir({}).name == "koboldcpp_voices"
