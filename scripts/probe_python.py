"""Report what an existing Python install has, so the installer can reuse its PyTorch.

Run by install.ps1 with each Python it finds. Must not import anything from
this repo: it runs under someone else's interpreter. Prints one line:
PROBE:{json}
"""

import json
import os
import sys
import sysconfig

info = {
    "exe": sys.executable,
    "version": "%d.%d" % sys.version_info[:2],
    "torch": None,
    "cuda": False,
    "cuda_version": None,
    "torchaudio": None,
    "site_dirs": [],
}

for key in ("purelib", "platlib"):
    p = sysconfig.get_paths().get(key)
    if p and os.path.isdir(p) and p not in info["site_dirs"]:
        info["site_dirs"].append(p)

try:
    import torch

    info["torch"] = torch.__version__
    info["cuda_version"] = torch.version.cuda
    info["cuda"] = bool(torch.cuda.is_available())
    # torch may live in the user site-packages rather than the main one.
    torch_site = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
    if torch_site not in info["site_dirs"]:
        info["site_dirs"].insert(0, torch_site)
except Exception as exc:  # not installed, or broken
    info["error"] = str(exc)[:200]

try:
    import torchaudio

    info["torchaudio"] = torchaudio.__version__
except Exception:
    pass

print("PROBE:" + json.dumps(info))
