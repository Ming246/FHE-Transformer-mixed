#!/usr/bin/env python3
"""Generate minimal Liberate ``scale_primes.pkl`` for THOR / smoke engines."""
from __future__ import annotations

import os
import pickle
import sys
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "thirdparty", "THOR-main", "liberate", "src"))

from liberate.fhe.cache import cache  # noqa: E402
from liberate.fhe.context.generate_primes import (  # noqa: E402
    generate_alternating_prime_sequence,
)

# (scale_bits, N=2**logN, how_many)
# Prefer fixed-direction search — optimize/alternate can overflow or take hours.
NEEDED = [
    (40, 2**15, 64),
    (59, 2**15, 16),
    (40, 2**16, 128),
    (41, 2**16, 128),
    (59, 2**16, 16),
]


def main() -> int:
    out = os.path.join(cache.path_cache, "scale_primes.pkl")
    existing: dict = {}
    if os.path.isfile(out):
        with open(out, "rb") as f:
            existing = pickle.load(f)
        print(f"loaded existing keys={sorted(existing)}")

    for sb, N, how in NEEDED:
        key = (sb, N)
        if key in existing and isinstance(existing[key], list) and len(existing[key]) >= how:
            print(f"skip {key}")
            continue
        print(f"generating sb={sb} N={N} how={how} ...", flush=True)
        t0 = time.time()
        primes = generate_alternating_prime_sequence(
            sb=sb,
            N=N,
            how_many=how,
            optimize=False,
            alternate_directions=False,
            fixed_direction=True,
        )
        existing[key] = primes
        print(f"  -> {len(primes)} in {time.time()-t0:.1f}s", flush=True)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(existing, f)

    print(f"wrote {out} keys={sorted(existing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
