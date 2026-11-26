"""Put Liberate and this directory on ``sys.path``. Model code lives here."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_PLAIN = os.path.join(_REPO, "Plain")
_LIBERATE_SRC = os.path.join(_HERE, "liberate", "src")


def ensure_thor_path() -> None:
    """Liberate CKKS runtime, ``HE/``, and plaintext modules in ``Plain/``."""
    for p in (_REPO, _LIBERATE_SRC, _HERE, _PLAIN):
        ap = os.path.abspath(p)
        if ap in sys.path:
            sys.path.remove(ap)
        sys.path.insert(0, ap)


def liberate_resources_dir() -> str:
    return os.path.join(
        _LIBERATE_SRC, "liberate", "fhe", "cache", "resources"
    )


def diagnose() -> dict:
    """Return import / GPU / resources status for smoke scripts."""
    ensure_thor_path()
    info: dict = {
        "liberate_src": _LIBERATE_SRC,
        "resources_dir": liberate_resources_dir(),
    }
    try:
        import torch

        info["torch"] = getattr(torch, "__version__", "?")
        info["cuda"] = bool(torch.cuda.is_available())
        info["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception as e:
        info["torch_error"] = f"{type(e).__name__}: {e}"

    try:
        import liberate  # noqa: F401
        from liberate.csprng import chacha20_cuda  # noqa: F401

        info["liberate"] = "ok"
        info["chacha20_cuda"] = "ok"
    except Exception as e:
        info["liberate"] = f"{type(e).__name__}: {e}"

    try:
        from ckks import CkksEngine  # noqa: F401

        info["ckks"] = "ok"
    except Exception as e:
        info["ckks"] = f"{type(e).__name__}: {e}"

    res = liberate_resources_dir()
    info["resources_files"] = sorted(os.listdir(res)) if os.path.isdir(res) else []
    info["has_scale_primes"] = "scale_primes.pkl" in info["resources_files"]
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(diagnose(), indent=2, default=str))
