#!/usr/bin/env python3
"""
Encode DualRail THOR weights, **one pickle per layer**.

Writes (under ``encoded_models_new/{dataset}/``)::

  ff_L{0..11}.pkl   ~6 GiB each (FF / LN2 tiles for that layer)
  att_L{0..11}.pkl  ~3 GiB each (QKV / W_O / LN1 tiles)
  pooler.pkl        small
  cls.pkl           classifier (after last att encode)

Does **not** write the old monolithic ``ff.pkl`` / ``att.pkl``.
Each layer is dumped and dropped before the next, so RSS stays ~one layer.

Usage::

  python3 HE/thor/encode_thor_weights.py --dataset mrpc
  python3 HE/thor/encode_thor_weights.py --dataset mrpc --layers 3 4 5
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_THOR_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "thirdparty", "THOR-main"))
sys.path.insert(0, _HERE)
from path_setup import ensure_thor_path

ensure_thor_path()

import torch


def encoded_layer_pkl(
    dataset: str,
    kind: str,
    layer_idx: int,
    *,
    encoded_dir: str | None = None,
    thor_root: str = _THOR_ROOT,
) -> str:
    """``…/encoded_models_new/{dataset}/{ff|att}_L{n}.pkl``."""
    if kind not in ("ff", "att"):
        raise ValueError(f"kind must be ff|att, got {kind!r}")
    root = encoded_dir or os.path.join(thor_root, "encoded_models_new", dataset)
    return os.path.join(root, f"{kind}_L{int(layer_idx)}.pkl")


def _remap_cuda_device() -> None:
    """Map ``cuda:1`` (THOR encode) → ``cuda:0`` when only one GPU exists."""
    _Old = torch.cuda.device

    class RemapDevice(_Old):  # type: ignore[misc,valid-type]
        def __init__(self, device, *args, **kwargs):
            if isinstance(device, int) and device >= torch.cuda.device_count():
                device = 0
            super().__init__(device, *args, **kwargs)

    torch.cuda.device = RemapDevice  # type: ignore[misc,assignment]


def _normalize_layernorm_keys(weights_m: dict) -> None:
    """Map TF-style ``LayerNorm.{gamma,beta}`` → HF ``.{weight,bias}`` in-place."""
    remap = []
    for k in list(weights_m.keys()):
        if k.endswith(".LayerNorm.gamma"):
            remap.append((k, k[: -len("gamma")] + "weight"))
        elif k.endswith(".LayerNorm.beta"):
            remap.append((k, k[: -len("beta")] + "bias"))
    for old, new in remap:
        if new not in weights_m:
            weights_m[new] = weights_m[old]


def _dump_layer_pkl(
    weights_pt: dict, out_path: str, layer_idx: int, *, label: str
) -> None:
    needle = f".layer.{layer_idx}."
    layer_w = {
        k: v for k, v in weights_pt.items() if needle in k and v is not None
    }
    if not layer_w:
        raise RuntimeError(f"no non-None {label} keys for layer {layer_idx}")
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(layer_w, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_path)
    n = len(layer_w)
    for k in list(weights_pt):
        if needle in k:
            weights_pt[k] = None
    del layer_w
    gc.collect()
    print(
        f"  wrote {out_path} ({n} keys, "
        f"{os.path.getsize(out_path) / (1024**3):.2f} GiB)",
        flush=True,
    )


def _dump_named_pkl(weights_pt: dict, out_path: str, pred, *, label: str) -> None:
    subset = {k: v for k, v in weights_pt.items() if v is not None and pred(k)}
    if not subset:
        raise RuntimeError(f"no non-None keys for {label}")
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(subset, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_path)
    print(
        f"  wrote {out_path} ({len(subset)} keys, "
        f"{os.path.getsize(out_path) / 1e6:.1f} MB)",
        flush=True,
    )


_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
from softmax_poly import k_key_encode_scale  # noqa: E402


def _attach_key_encode_scale(encoder, dataset: str) -> None:
    encoder._k_key_encode_scale_fn = lambda layer: k_key_encode_scale(dataset, layer)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mrpc", choices=("mrpc", "rte", "sst2"))
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="encode only these layer indices (default: all 0..11)",
    )
    ap.add_argument(
        "--kind",
        choices=("ff", "att", "both"),
        default="both",
        help="which DualRail family to encode (default: both)",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="path to model.safetensors (default: finetuned_models/{dataset}/model.safetensors)",
    )
    ap.add_argument(
        "--thor-root",
        default=_THOR_ROOT,
        help="thirdparty/THOR-main root (finetuned_models + encoded_models_new)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: encoded_models_new/{dataset} under --thor-root)",
    )
    ap.add_argument(
        "--skip-pooler",
        action="store_true",
        help="do not re-encode pooler.pkl",
    )
    ap.add_argument(
        "--skip-cls",
        action="store_true",
        help="do not re-encode cls.pkl",
    )
    args = ap.parse_args()
    os.chdir(args.thor_root)
    out_dir = args.out_dir or os.path.join("encoded_models_new", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    model_dir = args.model or f"./finetuned_models/{args.dataset}/model.safetensors"
    if not os.path.isfile(model_dir):
        raise FileNotFoundError(model_dir)

    layers = list(range(12) if args.layers is None else args.layers)
    do_ff = args.kind in ("ff", "both")
    do_att = args.kind in ("att", "both")
    _remap_cuda_device()
    from thor import CkksEngine, ThorModelEncoder

    params = {
        "logN": 16,
        "scale_bits": 41,
        "num_special_primes": 4,
        "devices": [0],
        "quantum": "pre_quantum",
    }
    print("create engine ...", flush=True)
    engine = CkksEngine(params)

    print("-" * 50)
    print(
        f"Encoding start for {args.dataset} layers={layers} kind={args.kind}",
        flush=True,
    )
    print(f"  model={model_dir}", flush=True)
    print(f"  out_dir={os.path.abspath(out_dir)}", flush=True)

    encoder = ThorModelEncoder(engine, model_dir, dataset=args.dataset)
    _attach_key_encode_scale(encoder, args.dataset)
    encoder.weights_m = dict(encoder.weights_m)
    _normalize_layernorm_keys(encoder.weights_m)
    assert (
        f"bert.encoder.layer.{layers[0]}.output.LayerNorm.weight"
        in encoder.weights_m
    ), "LayerNorm.weight missing after remap — check safetensors key names"
    keys = list(encoder.weights_m.keys())
    encoder.weights_pt = {key: None for key in keys}
    print(f"  LN remap ok; n_keys={len(encoder.weights_m)}", flush=True)

    if not args.skip_pooler:
        encoder.encode_pooler()
        _dump_named_pkl(
            encoder.weights_pt,
            os.path.join(out_dir, "pooler.pkl"),
            lambda k: "pooler" in k.lower(),
            label="pooler",
        )
        for k in list(encoder.weights_pt):
            if "pooler" in k.lower():
                encoder.weights_pt[k] = None
        gc.collect()

    if do_ff:
        for layer in layers:
            print(f"FF layer {layer}", flush=True)
            encoder.encode_ff(layer)
            _dump_layer_pkl(
                encoder.weights_pt,
                encoded_layer_pkl(
                    args.dataset, "ff", layer, encoded_dir=os.path.abspath(out_dir)
                ),
                layer,
                label="ff",
            )

    if do_att:
        print("-" * 50)
        print(f"Encoding start for {args.dataset} Attention", flush=True)
        encoder = ThorModelEncoder(engine, model_dir, dataset=args.dataset)
        _attach_key_encode_scale(encoder, args.dataset)
        encoder.weights_m = dict(encoder.weights_m)
        _normalize_layernorm_keys(encoder.weights_m)
        keys = list(encoder.weights_m.keys())
        encoder.weights_pt = {key: None for key in keys}
        for layer in layers:
            print(f"Att layer {layer}", flush=True)
            encoder.encode_att(layer)
            _dump_layer_pkl(
                encoder.weights_pt,
                encoded_layer_pkl(
                    args.dataset, "att", layer, encoded_dir=os.path.abspath(out_dir)
                ),
                layer,
                label="att",
            )
        if not args.skip_cls:
            encoder.encode_cls()
            _dump_named_pkl(
                encoder.weights_pt,
                os.path.join(out_dir, "cls.pkl"),
                lambda k: True,
                label="cls",
            )

    print(f"Encoding complete for {args.dataset}", flush=True)
    print("-" * 50)
    for layer in layers:
        for kind in (("ff",) if do_ff else ()) + (("att",) if do_att else ()):
            p = encoded_layer_pkl(
                args.dataset, kind, layer, encoded_dir=os.path.abspath(out_dir)
            )
            if os.path.isfile(p):
                print(
                    f"  {p}  {os.path.getsize(p) / (1024**3):.2f} GiB",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
