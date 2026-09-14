"""Build a real Liberate ``CkksEngine`` for the THOR complex path."""
from __future__ import annotations

import os
from typing import Any

try:
    from .path_setup import diagnose, ensure_thor_path, liberate_resources_dir
except ImportError:
    from path_setup import diagnose, ensure_thor_path, liberate_resources_dir


# Matches thirdparty/THOR-main/encode.py and forward.ipynb
THOR_DEFAULT_PARAMS: dict[str, Any] = {
    "logN": 16,
    "scale_bits": 41,
    "num_special_primes": 4,
    "devices": [0],
    "quantum": "pre_quantum",
}

# Smaller engine for primitive smoke (same security model as THOR)
SMOKE_PARAMS: dict[str, Any] = {
    "logN": 15,
    "scale_bits": 40,
    "num_special_primes": 2,
    "devices": [0],
    "quantum": "pre_quantum",
    "num_scales": 8,
}

# Bootstrap rotation bundle (→ add_bs_key); from forward.ipynb
ROTK_DICT_KEYS: list[int] = [
    -32768,
    -16384,
    -1024,
    -512,
    -32,
    -16,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    32,
    64,
    96,
    128,
    160,
    192,
    224,
    256,
    288,
    320,
    352,
    384,
    416,
    448,
    480,
    512,
    1024,
    2048,
    3072,
    4096,
    5120,
    6144,
    7168,
    8192,
    9216,
    10240,
    11264,
    12288,
    13312,
    14336,
    15360,
    16384,
]

# Ordinary left-rotation deltas generated from sk; from forward.ipynb
ROT_DELTAS: list[int] = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    16,
    2048,
    4096,
    6144,
    8192,
    10240,
    12288,
    14336,
    18432,
    20480,
    22528,
    24576,
    26624,
    28672,
    30720,
    256,
    2288,
    4320,
    6352,
    8384,
    10416,
    12448,
    14480,
    16512,
    18544,
    20576,
    22608,
    24640,
    26672,
    28704,
    30736,
    512,
    768,
    1280,
    2544,
    2800,
    3312,
    4576,
    4832,
    5344,
    6608,
    6864,
    8640,
    8896,
    10672,
]

DEFAULT_KEYS0 = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "thirdparty",
        "THOR-main",
        "keys",
        "keys0",
    )
)


def create_engine(
    params: dict[str, Any] | None = None,
    *,
    verbose: bool = False,
):
    """
    Construct upstream ``thor.ckks.CkksEngine``.

    First call without ``scale_primes.pkl`` triggers Liberate prime generation
    (can take many minutes). Subsequent calls read the cache.
    """
    ensure_thor_path()
    from thor.ckks import CkksEngine

    p = dict(THOR_DEFAULT_PARAMS if params is None else params)
    return CkksEngine(p, verbose=verbose)


def create_keys(engine, *, with_evk: bool = True):
    """Secret / public / (optional) evaluation keys; wire onto engine."""
    sk = engine.create_secret_key()
    pk = engine.create_public_key(sk)
    engine.add_pk(pk)
    evk = None
    if with_evk:
        evk = engine.create_evk(sk)
        engine.add_evk(evk)
    return {"sk": sk, "pk": pk, "evk": evk}


def load_thor_keys(
    engine,
    keys0: str | None = None,
    *,
    with_bootstrap: bool = True,
    with_rot_deltas: bool = True,
    rot_deltas: list[int] | None = None,
) -> dict[str, Any]:
    """
    Load official THOR ``keys/keys0`` the same way as ``forward.ipynb``.

    - ``rotk_dict`` → ``add_bs_key`` (bootstrap bundle; NOT ordinary rotate keys)
    - ordinary rotates → ``add_rot_keys_from_sk(deltas, sk)``
    - calls ``bs.create_cts_stc_const(engine)`` when ``with_bootstrap``
    """
    ensure_thor_path()
    keys0 = os.path.abspath(keys0 or DEFAULT_KEYS0)
    if not os.path.isdir(keys0):
        raise FileNotFoundError(f"keys0 not found: {keys0}")

    sk = engine.load(os.path.join(keys0, "sk"))
    pk = engine.load(os.path.join(keys0, "pk"))
    evk = engine.load(os.path.join(keys0, "evk"))
    gk = engine.load(os.path.join(keys0, "gk"))
    conjk = engine.load(os.path.join(keys0, "conjk"))
    engine.add_pk(pk)
    engine.add_evk(evk)
    engine.add_gk(gk)
    engine.add_conj_key(conjk)

    bs_key = None
    if with_bootstrap:
        import gc

        import torch
        from liberate.fhe.bootstrapping import ckks_bootstrapping as bs

        rotk_dict = {}
        n = len(ROTK_DICT_KEYS)
        for i, key in enumerate(ROTK_DICT_KEYS, 1):
            path = os.path.join(keys0, "rotk_dict", str(key))
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing bs rotk: {path}")
            rotk_dict[key] = engine.load(path)
            # Fragmentation peaks while uploading ~55×307MB rotks onto 24GB.
            if i % 5 == 0 or i == n:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if i == 1 or i == n or i % 10 == 0:
                used = (
                    torch.cuda.memory_reserved(0) / (1024**3)
                    if torch.cuda.is_available()
                    else float("nan")
                )
                print(
                    f"  loaded rotk_dict {i}/{n} (key={key}) "
                    f"cuda_reserved≈{used:.1f}GiB",
                    flush=True,
                )
        print("  create_cts_stc_const + add_bs_key ...", flush=True)
        bs.create_cts_stc_const(engine)
        engine.add_bs_key(rotk_dict)
        bs_key = rotk_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if with_rot_deltas:
        engine.add_rot_keys_from_sk(
            list(ROT_DELTAS if rot_deltas is None else rot_deltas), sk
        )

    return {
        "sk": sk,
        "pk": pk,
        "evk": evk,
        "gk": gk,
        "conjk": conjk,
        "bs_key": bs_key,
        "keys0": keys0,
    }


def resources_ready() -> bool:
    return os.path.isfile(
        os.path.join(liberate_resources_dir(), "scale_primes.pkl")
    )


def status() -> dict:
    info = diagnose()
    info["resources_ready"] = resources_ready()
    info["keys0_ready"] = os.path.isfile(os.path.join(DEFAULT_KEYS0, "sk"))
    return info
