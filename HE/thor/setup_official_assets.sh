#!/usr/bin/env bash
# Install official THOR Drive assets into the upstream layout (1:1 with README).
#
# Expected inputs (any one location is fine):
#   thirdparty/THOR-main/_official_assets/{resources.tar,keys.tar,...}
#   or paths passed as env:
#     RESOURCES_TAR=... KEYS_TAR=...
#
# Layout after install (matches upstream THOR README):
#   thirdparty/THOR-main/liberate/src/liberate/fhe/cache/resources/   ← extract resources.tar here
#   thirdparty/THOR-main/keys/                                        ← extract keys.tar here
#   thirdparty/THOR-main/datasets/          (optional)
#   thirdparty/THOR-main/encoded_models_new/ (optional)
#   thirdparty/THOR-main/finetuned_models/   (optional)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
THOR="$ROOT/thirdparty/THOR-main"
ASSETS="${THOR}/_official_assets"
RES_DIR="$THOR/liberate/src/liberate/fhe/cache/resources"

RESOURCES_TAR="${RESOURCES_TAR:-$ASSETS/resources.tar}"
KEYS_TAR="${KEYS_TAR:-$ASSETS/keys.tar}"

echo "THOR root: $THOR"

install_resources() {
  local tar="$1"
  if [[ ! -f "$tar" ]]; then
    echo "MISSING resources tar: $tar"
    return 1
  fi
  mkdir -p "$RES_DIR"
  echo "Extracting resources from $tar -> $RES_DIR (and parent if archived as resources/) ..."
  # Official archive members look like: resources/*.pkl
  # Extract into liberate/.../cache/ so members land in .../cache/resources/
  local cache_parent
  cache_parent="$(dirname "$RES_DIR")"
  # Backup homemade scale_primes if present and official will overwrite
  if [[ -f "$RES_DIR/scale_primes.pkl" && ! -f "$RES_DIR/scale_primes.pkl.homemade.bak" ]]; then
    cp -a "$RES_DIR/scale_primes.pkl" "$RES_DIR/scale_primes.pkl.homemade.bak"
    echo "backed up homemade scale_primes.pkl"
  fi
  tar -xf "$tar" -C "$cache_parent"
  echo "resources installed. sample:"
  ls -lh "$RES_DIR" | head -20
}

install_keys() {
  local tar="$1"
  if [[ ! -f "$tar" ]]; then
    echo "MISSING keys tar: $tar"
    return 1
  fi
  mkdir -p "$THOR"
  echo "Extracting keys from $tar -> $THOR/ ..."
  # Expect members like keys/...
  tar -xf "$tar" -C "$THOR"
  echo "keys installed. sample:"
  ls -lh "$THOR/keys" | head -20
}

ok=0
if [[ -f "$RESOURCES_TAR" ]]; then
  install_resources "$RESOURCES_TAR" && ok=1
else
  echo "skip resources (not found at $RESOURCES_TAR)"
fi

if [[ -f "$KEYS_TAR" ]]; then
  install_keys "$KEYS_TAR" && ok=1
else
  echo "skip keys (not found at $KEYS_TAR)"
fi

# Optional siblings
for name in datasets encoded_models_new finetuned_models; do
  t="$ASSETS/${name}.tar"
  if [[ -f "$t" ]]; then
    echo "Extracting optional $t -> $THOR/"
    tar -xf "$t" -C "$THOR"
  fi
done

if [[ "$ok" -eq 0 ]]; then
  echo
  echo "No official tarballs found. Download from:"
  echo "  https://drive.google.com/drive/folders/1mWBkNdsu3JCQPrSuedyeN_3WJD7h-6RO"
  echo "Place resources.tar / keys.tar under:"
  echo "  $ASSETS/"
  echo "Then re-run: bash HE/thor/setup_official_assets.sh"
  exit 2
fi

echo "DONE"
