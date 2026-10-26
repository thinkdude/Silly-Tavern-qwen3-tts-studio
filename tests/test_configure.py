from app.config import CLONE_MODEL_LARGE, CLONE_MODEL_SMALL
from app.configure import recommend


def test_rtx_3060_12gb():
    r = recommend({"cuda": True, "name": "RTX 3060", "vram_gb": 12.0, "cc": (8, 6)})
    assert r["dtype"] == "bfloat16" and r["clone_model"] == CLONE_MODEL_LARGE and not r["keep_both_loaded"]


def test_rtx_4090():
    assert recommend({"cuda": True, "name": "RTX 4090", "vram_gb": 24.0, "cc": (8, 9)})["keep_both_loaded"]


def test_small_old_gpu():
    r = recommend({"cuda": True, "name": "GTX 1650", "vram_gb": 4.0, "cc": (7, 5)})
    assert r["dtype"] == "float16" and r["clone_model"] == CLONE_MODEL_SMALL


def test_cpu():
    r = recommend({"cuda": False})
    assert r["device"] == "cpu" and r["dtype"] == "float32"
