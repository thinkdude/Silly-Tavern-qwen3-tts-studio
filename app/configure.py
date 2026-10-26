"""Pick sensible settings for this PC's GPU and write settings.json.

Run by the installer. Safe to re-run: settings you changed by hand are kept
unless you pass --reset.
"""

from __future__ import annotations

import argparse
import sys

from .config import CLONE_MODEL_LARGE, CLONE_MODEL_SMALL, DEFAULTS, SETTINGS_PATH, load_settings, save_settings


def detect() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": props.name,
        "vram_gb": round(props.total_memory / 1024**3, 1),
        "cc": (props.major, props.minor),
    }


def recommend(hw: dict) -> dict:
    if not hw.get("cuda"):
        return {
            "device": "cpu",
            "dtype": "float32",
            "clone_model": CLONE_MODEL_SMALL,
            "keep_both_loaded": False,
            "gpu_name": "",
            "vram_gb": 0.0,
        }
    vram = hw["vram_gb"]
    return {
        "device": "cuda:0",
        # bfloat16 needs an RTX 30-series (Ampere) or newer; older cards use float16.
        "dtype": "bfloat16" if tuple(hw["cc"]) >= (8, 0) else "float16",
        "clone_model": CLONE_MODEL_LARGE if vram >= 6 else CLONE_MODEL_SMALL,
        "keep_both_loaded": vram >= 16,
        "gpu_name": hw["name"],
        "vram_gb": vram,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reset", action="store_true", help="overwrite settings.json with fresh defaults")
    ap.add_argument("--expect-gpu", action="store_true", help="warn loudly if an NVIDIA GPU was found but CUDA is unusable")
    args = ap.parse_args()

    hw = detect()
    rec = recommend(hw)
    first_run = args.reset or not SETTINGS_PATH.exists()
    settings = dict(DEFAULTS) if args.reset else load_settings()
    if first_run:
        settings.update(rec)
    else:
        # Keep hand-edited settings on a re-install; only refresh what the GPU is.
        settings["gpu_name"], settings["vram_gb"] = rec["gpu_name"], rec["vram_gb"]
        changed = [k for k in ("device", "dtype", "clone_model") if settings.get(k) != rec[k]]
        if changed:
            print("Keeping your existing settings.json. For this GPU I'd suggest:")
            for k in changed:
                print(f"  {k}: {rec[k]}  (currently {settings.get(k)})")
            print("Run 'install.bat -Reset' to apply these suggestions.")
            print()
    save_settings(settings)

    if hw.get("cuda"):
        print(f"GPU: {hw['name']} ({hw['vram_gb']} GB VRAM)")
        print(f"  precision:   {settings['dtype']}")
        print(f"  clone model: {settings['clone_model']}")
        print(f"  both models loaded at once: {'yes' if settings['keep_both_loaded'] else 'no (swaps as needed)'}")
        if hw["vram_gb"] < 6:
            print("  Note: under 6 GB of VRAM. Voice design may be slow or run out of memory.")
    else:
        print("No usable NVIDIA GPU found. Running on the CPU, which works but is very slow.")
        if args.expect_gpu:
            print()
            print("!! An NVIDIA GPU was detected but PyTorch can't use it.")
            print("!! Update your NVIDIA driver (https://www.nvidia.com/Download/index.aspx), then run install.bat again.")
            return 2
    print(f"Saved {SETTINGS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
