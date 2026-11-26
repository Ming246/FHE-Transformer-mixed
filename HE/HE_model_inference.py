#!/usr/bin/env python3
"""
Plan-following DualRail ciphertext BERT.

Encrypts the embedding, runs all 12 encoder layers under the bootstrap
plan, then the ciphertext pooler and classifier. Remaining-level checks
stay on. The head decrypts only the logits.

Usage::

  python3 HE/HE_model_inference.py --dataset mrpc --sample 0
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_PLAIN = os.path.join(_REPO, "Plain")
if _PLAIN not in sys.path:
    sys.path.insert(0, _PLAIN)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from path_setup import ensure_thor_path

ensure_thor_path()

from ckks import (  # noqa: E402
    THOR_DEFAULT_PARAMS,
    create_engine,
    load_thor_keys,
)
from bert import (  # noqa: E402
    HeBertHead,
    ThorBertAttention,
    ThorBertFF,
    _forward_layer,
    _release_he_module_weights,
    _require_assets,
    _thor_paths,
    decrypt_logits_slot0,
)
from data import ThorDataEncryptor, encrypt_embedding_real8  # noqa: E402
from encode_weights import encoded_layer_pkl  # noqa: E402
from linear import ThorLinearEvaluator  # noqa: E402


def _gc_cuda() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _slim_encoded_pkl_to_temp(
    src_path: str, layers: list[int], *, label: str
) -> dict[int, str]:
    """Load full DualRail pkl in a **subprocess**; write one temp pkl per layer.

    Child exits after dump so the 9–19 GiB deserialize RSS returns to the OS
    before the next pkl / engine load (in-process GC is not enough on ~46 GiB).
    """
    import json
    import subprocess
    import tempfile

    print(
        f"  slim {label}: load {src_path} → per-layer temps {layers} "
        f"(subprocess) ...",
        flush=True,
    )
    t0 = time.time()
    out_dir = tempfile.mkdtemp(prefix=f"thor_{label}_slim_")
    manifest = os.path.join(out_dir, "manifest.json")
    worker = r"""
import json, os, pickle, sys, tempfile
src, out_dir, layers_json, label = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
layers = json.loads(layers_json)
with open(src, "rb") as f:
    weights = pickle.load(f)
paths = {}
for li in layers:
    needle = f".layer.{li}."
    layer_w = {k: weights.pop(k) for k in list(weights) if needle in k}
    if not layer_w:
        raise SystemExit(f"no keys for layer {li}")
    fd, dst = tempfile.mkstemp(prefix=f"thor_{label}_L{li}_", suffix=".pkl", dir=out_dir)
    os.close(fd)
    with open(dst, "wb") as f:
        pickle.dump(layer_w, f, protocol=pickle.HIGHEST_PROTOCOL)
    paths[str(li)] = {"path": dst, "n_keys": len(layer_w), "bytes": os.path.getsize(dst)}
    del layer_w
n_left = len(weights)
weights.clear()
del weights
with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump({"paths": paths, "n_left": n_left}, f)
"""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            worker,
            src_path,
            out_dir,
            json.dumps(list(layers)),
            label,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"slim {label} subprocess failed (rc={proc.returncode}): {err[:2000]}"
        )
    with open(manifest, encoding="utf-8") as f:
        meta = json.load(f)
    out: dict[int, str] = {}
    for li_s, info in meta["paths"].items():
        li = int(li_s)
        dst = info["path"]
        # Move out of mkdtemp dir so atexit can unlink individually.
        final = os.path.join(
            tempfile.gettempdir(),
            f"thor_{label}_L{li}_{os.getpid()}_{li}.pkl",
        )
        os.replace(dst, final)
        out[li] = final
        print(
            f"  slim {label} L{li}: {info['n_keys']} keys → {final} "
            f"({info['bytes'] / (1024**3):.2f} GiB)",
            flush=True,
        )
    try:
        os.rmdir(out_dir)
    except OSError:
        pass
    _gc_cuda()
    print(
        f"  slim {label}: discarded {meta.get('n_left', '?')} leftover keys "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    return out


def _extract_and_take_layer_weights(weights: dict, layer_idx: int) -> dict:
    """Move ``.layer.{idx}.`` keys out of ``weights`` (caller owns the rest)."""
    needle = f".layer.{layer_idx}."
    out = {}
    for k in list(weights.keys()):
        if needle in k:
            out[k] = weights.pop(k)
    if not out:
        raise KeyError(f"no encoded keys for layer {layer_idx}")
    return out


def _bind_poly(
    thor_attention,
    thor_ff,
    *,
    dataset: str,
    layer_idx: int,
    gelu_level: int,
    ln1_level: int,
    ln2_level: int,
    softmax_level: int,
    bts_hook,
) -> None:
    from gelu_poly import gelu_level_allowed
    from nonlinear.gelu import bind_ff_gelu_cheb
    from nonlinear.layernorm import bind_layer_layernorm
    from nonlinear.softmax import bind_attention_softmax

    ge_lv = int(gelu_level)
    if not gelu_level_allowed(layer_idx, ge_lv):
        raise ValueError(f"L{layer_idx} GeLU level={ge_lv} is not allowed")
    gp = bind_ff_gelu_cheb(
        thor_ff,
        layer_idx=layer_idx,
        level=ge_lv,
        suppress_post_bts=True,
        bts_hook=bts_hook,
    )
    print(
        f"  L{layer_idx} GeLU → Cheb ({gp.scheme_name}, depth_he={gp.depth_he})",
        flush=True,
    )
    lv1 = int(ln1_level)
    lv2 = int(ln2_level)
    p_ln1, p_ln2 = bind_layer_layernorm(
        thor_attention,
        thor_ff,
        task=dataset,
        layer_idx=layer_idx,
        level=lv1,
        level_ln2=lv2,
        bts_hook=bts_hook,
    )
    print(
        f"  L{layer_idx} LN → poly "
        f"ln1(iters={p_ln1.invsqrt_max_iters},lv={lv1}) "
        f"ln2(iters={p_ln2.invsqrt_max_iters},lv={lv2})",
        flush=True,
    )
    sp = bind_attention_softmax(
        thor_attention,
        task=dataset,
        layer_idx=layer_idx,
        level=int(softmax_level),
        bts_hook=bts_hook,
    )
    print(
        f"  L{layer_idx} Softmax → poly "
        f"(δ2={sp.delta2:g}, shift={sp.shift:g}, "
        f"iters_σ={sp.asor_max_iters_sigma}, "
        f"iters_Σy²={sp.asor_max_iters_sum_sq})",
        flush=True,
    )


def _assert_layer_weights(weights: dict, layer_idx: int, *, kind: str) -> None:
    if kind == "att":
        keys = [
            f"bert.encoder.layer.{layer_idx}.attention.self.query.weight",
            f"bert.encoder.layer.{layer_idx}.attention.output.dense.weight",
            f"bert.encoder.layer.{layer_idx}.attention.output.LayerNorm.weight",
        ]
    else:
        keys = [
            f"bert.encoder.layer.{layer_idx}.intermediate.dense.weight",
            f"bert.encoder.layer.{layer_idx}.output.dense.weight",
            f"bert.encoder.layer.{layer_idx}.output.LayerNorm.weight",
        ]
    missing = [k for k in keys if weights.get(k) is None]
    if missing:
        raise RuntimeError(
            f"{kind} layer {layer_idx} weights missing/None: {missing}. "
            f"Encode with: python3 HE/encode_weights.py --layers {layer_idx}"
        )


def _poly_scheme_from_args(args) -> list[int]:
    from gelu_poly import gelu_level_allowed
    from cost import NUM_LAYERS, SCHEME_LEN, validate_scheme

    if args.scheme:
        import ast
        import json

        raw = str(args.scheme).strip()
        try:
            scheme = json.loads(raw)
        except json.JSONDecodeError:
            scheme = ast.literal_eval(raw)
        scheme = [int(x) for x in scheme]
        if len(scheme) != SCHEME_LEN:
            raise ValueError(
                f"--scheme length {len(scheme)} != {SCHEME_LEN}"
            )
        validate_scheme(scheme, args.dataset)
        for li in range(NUM_LAYERS):
            g = scheme[li * 4 + 2]
            if not gelu_level_allowed(li, g):
                raise ValueError(
                    f"--scheme L{li} gelu level={g} not allowed"
                )
        print(f"  scheme from --scheme ({SCHEME_LEN}-d)", flush=True)
        return scheme
    sm_lv = int(args.softmax_level)
    ln_lv = int(args.ln_level)
    ge_lv = int(args.gelu_level)
    scheme = []
    for li in range(NUM_LAYERS):
        if not gelu_level_allowed(li, ge_lv):
            raise ValueError(f"L{li} GeLU level={ge_lv} is not allowed")
        scheme.extend([sm_lv, ln_lv, ge_lv, ln_lv])
    print(
        f"  scheme from CLI levels sm/ln/gelu="
        f"{sm_lv}/{ln_lv}/{ge_lv}",
        flush=True,
    )
    return scheme


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mrpc")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument(
        "--he-head-c",
        type=float,
        default=5.0,
        help="pooler tanh scale C for p2(p1(z/C)); default 5",
    )
    ap.add_argument("--gelu-level", type=int, default=1)
    ap.add_argument("--ln-level", type=int, default=1)
    ap.add_argument("--softmax-level", type=int, default=1)
    ap.add_argument(
        "--scheme",
        type=str,
        default=None,
        help="48-int JSON/list (sm,ln1,gelu,ln2)×12; overrides --*-level",
    )
    ap.add_argument("--plan-discard", action="store_true")
    ap.add_argument("--encrypt-level", type=int, default=None)
    ap.add_argument(
        "--rem-tol",
        type=int,
        default=0,
        help="allowed |HE rem − DP rem| at probe/boot sites (default 0)",
    )
    ap.add_argument(
        "--overlay-dir",
        action="append",
        default=[],
        help="dir named …/L{n} with ff.pkl+att.pkl for that layer",
    )
    args = ap.parse_args()

    from cost import NUM_LAYERS

    layers = list(range(NUM_LAYERS))

    scheme = _poly_scheme_from_args(args)

    paths = _thor_paths(args.dataset)
    overlay_by_layer: dict[int, str] = {}
    for raw in args.overlay_dir:
        d = os.path.abspath(raw)
        m = re.search(r"L(\d+)$", os.path.basename(d.rstrip(os.sep)))
        if not m:
            raise SystemExit(
                f"--overlay-dir must end with L{{n}} (got {raw!r})"
            )
        overlay_by_layer[int(m.group(1))] = d

    layer_ff_paths: dict[int, str] = {}
    layer_att_paths: dict[int, str] = {}
    for li in layers:
        if li in overlay_by_layer:
            d = overlay_by_layer[li]
            layer_ff_paths[li] = os.path.join(d, "ff.pkl")
            layer_att_paths[li] = os.path.join(d, "att.pkl")
        else:
            layer_ff_paths[li] = encoded_layer_pkl(
                args.dataset, "ff", li, gelu_level=scheme[li * 4 + 2]
            )
            layer_att_paths[li] = encoded_layer_pkl(args.dataset, "att", li)
    _require_assets(
        paths,
        layers=[li for li in layers if li not in overlay_by_layer],
        gelu_levels={li: scheme[li * 4 + 2] for li in layers},
    )
    for li in layers:
        for kind, p in (
            ("ff", layer_ff_paths[li]),
            ("att", layer_att_paths[li]),
        ):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"missing {kind} L{li}: {p}. "
                    "Encode with: python3 HE/encode_weights.py "
                    f"--layers {li}"
                )
    print(
        f"per-layer DualRail weights {layers}: "
        + ", ".join(
            f"L{li}=ff:{os.path.basename(layer_ff_paths[li])}"
            f"/att:{os.path.basename(layer_att_paths[li])}"
            for li in layers
        ),
        flush=True,
    )

    devices = [0]
    print("create engine + load keys0 (bootstrap) ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(
        engine,
        with_bootstrap=True,
        with_rot_deltas=True,
    )
    sk, pk = keys["sk"], keys["pk"]
    print(f"  ready ({time.time() - t0:.1f}s) slots={engine.num_slots}", flush=True)

    print(
        "load BERT for the embedding (unload before HE weights) ...",
        flush=True,
    )
    _repo = _REPO
    local_model = os.path.join(_repo, "finetuned_weight", args.dataset)
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    _tok_orig = AutoTokenizer.from_pretrained

    def _tok_local(pretrained_model_name_or_path, *a, **k):
        if pretrained_model_name_or_path == "bert-base-uncased":
            pretrained_model_name_or_path = local_model
            k.setdefault("local_files_only", True)
        return _tok_orig(pretrained_model_name_or_path, *a, **k)

    AutoTokenizer.from_pretrained = _tok_local  # type: ignore[method-assign]
    import data as _data

    _data.AutoTokenizer.from_pretrained = _tok_local  # type: ignore[method-assign]
    model_plain = AutoModelForSequenceClassification.from_pretrained(
        local_model, local_files_only=True, output_hidden_states=True
    )
    model_plain.eval()
    device = torch.device("cpu")
    model_plain.to(device)

    evaluator = ThorLinearEvaluator(engine)

    from bootstrap import BootstrapHook
    from bts_ops import (
        BOOTSTRAP_DEPTH_BUDGET,
        DEFAULT_L0_ENTRY_REMAINING,
        optimize_bootstrap,
        remaining_to_level_calc,
    )
    bts_hook = BootstrapHook(engine, mode="always")
    # L0 入口：HE rem = num_levels − level_calc = initial_rem（8real 加密后再 pack）。
    plan_initial_rem = DEFAULT_L0_ENTRY_REMAINING
    thor_enc_level = remaining_to_level_calc(
        plan_initial_rem, int(engine.num_levels)
    )
    plan_result = optimize_bootstrap(
        args.dataset,
        scheme,
        budget=BOOTSTRAP_DEPTH_BUDGET,
        initial_level=plan_initial_rem,
        phase="C",
    )
    bts_hook.load_plan(
        plan_result, apply_discards=bool(args.plan_discard)
    )
    print(
        f"  bootstrap plan: {plan_result.summary()} "
        f"(chain={layers}; initial_rem={plan_initial_rem}; "
        f"enc level_calc={thor_enc_level}; "
        f"discard={'on' if args.plan_discard else 'off'})",
        flush=True,
    )
    from rem_guard import RemGuard, RemMismatchError, write_rem_trace

    print("encrypt validation sample ...", flush=True)
    emb_model = model_plain.bert.embeddings
    data_encryptor = ThorDataEncryptor(
        args.dataset,
        paths["dataset"],
        embedding_model=emb_model,
        ckks_engine=engine,
        test=False,
    )
    batch = None
    for idx, b in enumerate(data_encryptor.eval_dataloader):
        if idx == args.sample:
            batch = b
            break
    if batch is None:
        raise RuntimeError(f"sample {args.sample} not in dataloader")

    last_layer = layers[-1]
    embed_ids = {
        k: v for k, v in batch.items() if k in ("input_ids", "token_type_ids")
    }
    attention_mask = batch["attention_mask"]
    del model_plain, emb_model, batch
    _gc_cuda()
    print("  HF BERT unloaded after embedding encrypt setup", flush=True)

    thor_attention_mask = data_encryptor.encode_attention_mask(
        attention_mask.cpu().numpy().squeeze(), level=15
    )
    embedding = data_encryptor.embed_data(embed_ids)
    if args.encrypt_level is not None:
        enc_level = int(args.encrypt_level)
    else:
        enc_level = thor_enc_level
        print(
            f"  plan initial rem={plan_initial_rem} "
            f"encrypt level_calc={enc_level} → HE rem={plan_initial_rem}",
            flush=True,
        )
    enc_arr = np.asarray(embedding, dtype=np.float64)
    if enc_arr.ndim == 3:
        enc_arr = enc_arr.squeeze(0)
    print(
        f"  encrypt 8-real from embedding {tuple(enc_arr.shape)} "
        f"level_calc={enc_level}",
        flush=True,
    )
    x = encrypt_embedding_real8(engine, enc_arr, pk, level=enc_level)
    del data_encryptor, embedding, enc_arr, embed_ids, attention_mask
    _gc_cuda()
    print("  plaintext embedding / encryptor dropped", flush=True)

    he_vars_last = None
    t_chain = time.time()
    rem_guard_fail = False
    for li in layers:
        # Per-layer: load slim pkl → build → forward → release.
        print(f"  L{li} load slim weight pickles ...", flush=True)
        with open(layer_att_paths[li], "rb") as f:
            att_w = pickle.load(f)
        with open(layer_ff_paths[li], "rb") as f:
            ff_w = pickle.load(f)
        _assert_layer_weights(att_w, li, kind="att")
        _assert_layer_weights(ff_w, li, kind="ff")
        print(
            f"  L{li} weights loaded (att={len(att_w)} ff={len(ff_w)} keys)",
            flush=True,
        )
        att = ThorBertAttention(evaluator, att_w, li)
        ff = ThorBertFF(evaluator, ff_w, li)
        att_w.clear()
        ff_w.clear()
        del att_w, ff_w
        _gc_cuda()

        _bind_poly(
            att,
            ff,
            dataset=args.dataset,
            layer_idx=li,
            gelu_level=scheme[li * 4 + 2],
            ln1_level=scheme[li * 4 + 1],
            ln2_level=scheme[li * 4 + 3],
            softmax_level=scheme[li * 4 + 0],
            bts_hook=bts_hook,
        )

        rem_guard = RemGuard.from_bootstrap_result(
            plan_result,
            layer_idx=li,
            tol=int(args.rem_tol),
        )
        bts_hook.rem_guard = rem_guard
        trace_path = f"/tmp/l{li}_rem_trace.tsv"
        write_rem_trace(plan_result, trace_path, layer_idx=li)
        print(rem_guard.dump_layer_trace(), flush=True)
        print(f"  rem_trace written → {trace_path}", flush=True)
        t1 = time.time()
        try:
            x, he_vars = _forward_layer(
                engine=engine,
                evaluator=evaluator,
                thor_attention=att,
                thor_ff=ff,
                layer_idx=li,
                x=x,
                thor_attention_mask=thor_attention_mask,
                devices=devices,
                bts_hook=bts_hook,
                rem_guard=rem_guard,
            )
        except RemMismatchError as exc:
            print(
                f"  REM GUARD STOP L{li} ({time.time() - t1:.1f}s): {exc}",
                flush=True,
            )
            print(rem_guard.dump_layer_trace(), flush=True)
            rem_guard_fail = True
            _release_he_module_weights(att)
            _release_he_module_weights(ff)
            del att, ff
            _gc_cuda()
            break
        print(f"  HE L{li} done ({time.time() - t1:.1f}s)", flush=True)
        print(
            f"  rem_guard L{li}: {len(rem_guard.checked)} checks passed",
            flush=True,
        )
        if li == last_layer:
            he_vars_last = he_vars
        else:
            del he_vars
        _release_he_module_weights(att)
        _release_he_module_weights(ff)
        del att, ff
        _gc_cuda()
        print(f"  L{li} modules released", flush=True)

    _gc_cuda()
    print(f"  HE chain {layers} total ({time.time() - t_chain:.1f}s)", flush=True)

    if rem_guard_fail:
        print("FAIL (rem_guard)", flush=True)
        return 3

    summ = bts_hook.summary()
    print(
        f"  BootstrapHook: mode={summ['mode']} calls={summ['calls']} "
        f"ct_boots={summ['total_ct_bootstraps']}",
        flush=True,
    )
    for ev, n in sorted(summ["by_event"].items()):
        if n:
            print(f"    {ev}: {n}", flush=True)

    if he_vars_last is None:
        print("FAIL (no HE output)", flush=True)
        return 2
    print(f"HE chain finished through L{last_layer}", flush=True)
    print("PASS")
    print(
        f"  [he-head] pooler + classifier on L{last_layer} DualRail ln2 ...",
        flush=True,
    )
    t_head = time.time()
    he_ln2 = he_vars_last["ln2_out"]
    if isinstance(he_ln2, np.ndarray) and he_ln2.ndim > 1:
        he_ln2 = he_ln2[0]
    packs8 = [he_ln2[i] for i in range(8)]
    head = HeBertHead(
        evaluator,
        dataset=args.dataset,
        encoded_dir=paths["encoded_dir"],
        device=int(devices[0]),
        tanh_C=float(args.he_head_c),
    )
    logits_ct = head.forward(packs8)
    he_logits = decrypt_logits_slot0(engine, sk, logits_ct)
    he_pred = int(np.argmax(he_logits))
    print(
        f"  [he-head] HE logits={he_logits.tolist()} pred={he_pred} "
        f"({time.time() - t_head:.1f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
