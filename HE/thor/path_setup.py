"""Ensure Liberate + upstream THOR are importable (CUDA extensions already built in-tree)."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_LIBERATE_SRC = os.path.join(_REPO, "thirdparty", "THOR-main", "liberate", "src")
_THOR_SRC = os.path.join(_REPO, "thirdparty", "THOR-main", "src")


def ensure_thor_path() -> None:
    """
    Prefer upstream ``thirdparty/THOR-main/src/thor`` over ``HE/thor``.

    ``HE/`` must not sit ahead of the upstream src on ``sys.path``, or
    ``import thor`` resolves to this adapter package.
    """
    # Remove accidental HE-shadowing entries first.
    he_dir = os.path.join(_REPO, "HE")
    for p in list(sys.path):
        if os.path.abspath(p) == he_dir:
            sys.path.remove(p)

    # Highest priority first via successive insert(0) in reverse.
    for p in (_REPO, _LIBERATE_SRC, _THOR_SRC):
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
        "thor_src": _THOR_SRC,
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
        from thor.ckks import CkksEngine  # noqa: F401

        info["thor.ckks"] = "ok"
    except Exception as e:
        info["thor.ckks"] = f"{type(e).__name__}: {e}"

    res = liberate_resources_dir()
    info["resources_files"] = sorted(os.listdir(res)) if os.path.isdir(res) else []
    info["has_scale_primes"] = "scale_primes.pkl" in info["resources_files"]
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(diagnose(), indent=2, default=str))
