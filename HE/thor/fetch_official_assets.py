#!/usr/bin/env python3
"""Resume / download official THOR Drive tarballs into ``_official_assets/``.

File IDs from the public THOR Google Drive folder
(https://drive.google.com/drive/folders/1mWBkNdsu3JCQPrSuedyeN_3WJD7h-6RO).

Drive often rate-limits large files (resources.tar ≈ 14.8GB). Use ``--resume``.
If gdown fails with quota errors, download in a browser on the host and copy
into ``thirdparty/THOR-main/_official_assets/``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import gdown

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_OUT = os.path.join(_REPO, "thirdparty", "THOR-main", "_official_assets")

FILES = {
    # name: (file_id, min_complete_bytes)
    "resources.tar": ("1LPQex129MuFclJp5F4sJN9fIrMVYLgXH", 10_000_000_000),  # ~14.8GB
    "keys.tar": ("1Sgfu0jt6HIyrou6XndZQRiK75L-7R5zJ", 50_000_000),
    "datasets.tar": ("1pQaHu5ifWQpliOIseoZ3JbNwo4G_xNOe", 100_000_000),
    "encoded_models_new.tar": ("1jEMjXRbN7qTgP75QZBsQTiSnLdiRPSOq", 1_000_000_000),
    "finetuned_models.tar": ("1gC6-C2bCf-bEALhQtzk2s3MGSuHpcGAh", 100_000_000),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only",
        nargs="+",
        choices=list(FILES),
        default=["resources.tar", "keys.tar"],
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    args = ap.parse_args()

    os.makedirs(_OUT, exist_ok=True)
    # Adopt any gdown *.part left under /tmp
    for name in args.only:
        part_glob = f"/tmp/thor_drive/folder/{name}*.part"
        import glob

        for p in glob.glob(part_glob):
            dest = os.path.join(_OUT, name)
            if not os.path.isfile(dest):
                print(f"adopting partial {p} -> {dest}")
                os.rename(p, dest)

    for name in args.only:
        fid, min_bytes = FILES[name]
        out = os.path.join(_OUT, name)
        if os.path.isfile(out) and os.path.getsize(out) >= min_bytes:
            print(f"complete {name} size={os.path.getsize(out)}")
            continue
        if os.path.isfile(out):
            print(
                f"incomplete {name} size={os.path.getsize(out)} "
                f"(need >= {min_bytes}); resuming...",
                flush=True,
            )
        print(f"download {name} -> {out}", flush=True)
        for attempt in range(5):
            try:
                gdown.download(
                    id=fid,
                    output=out,
                    quiet=False,
                    resume=args.resume,
                    use_cookies=False,
                )
                sz = os.path.getsize(out) if os.path.isfile(out) else 0
                if sz < min_bytes:
                    raise RuntimeError(
                        f"downloaded size {sz} < expected min {min_bytes} "
                        f"(Drive may have returned an HTML error page)"
                    )
                print(f"OK {name} size={sz}")
                break
            except Exception as e:
                print(f"attempt {attempt+1} FAIL: {type(e).__name__}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            print(f"GIVEUP {name}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
