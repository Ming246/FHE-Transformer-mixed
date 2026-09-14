#!/usr/bin/env python3
"""Smoke: localized ``bts_ops`` (budget=14) + DualRail-as-option plan."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_setup import ensure_thor_path

ensure_thor_path()

from bootstrap_hook import BootstrapHook  # noqa: E402
from bts_ops import (  # noqa: E402
    DISCARD_MAX_SPAN,
    active_geometry,
    optimize_bootstrap,
    run_phase_d_audit,
    scheme_all,
)


def main() -> int:
    print(f"budget=14")
    g = active_geometry()
    print(
        f"geom DualRail: Hc={g.ct_hidden_cplx} Hr={g.ct_hidden_real} "
        f"R={g.ct_pc_rot} Rctx={g.ct_pc_rot_ctx} "
        f"Q={g.ct_qkv_out} V/C={g.ct_v_cplx}/{g.ct_context_cplx} "
        f"Qc={g.ct_q_copies} S={g.ct_att_score} A={g.ct_alpha_copies} F={g.ct_ffn}"
    )
    errs = run_phase_d_audit(tasks=("mrpc",), levels=(1,))
    print(f"phase_d_audit errors={len(errs)}")
    if errs:
        for e in errs[:20]:
            print(" ", e)
        return 2

    scheme = scheme_all(1)
    result = optimize_bootstrap("mrpc", scheme, phase="C")
    print(result)
    print(
        "level_discards (first 8): show keep/discard → implied level_calc "
        "@ num_levels=29"
    )
    for d in result.level_discards[:8]:
        lc = 29 - d.remaining_keep
        print(
            f"  {d.reason:12} site={d.site:40} "
            f"after={d.remaining_after_bts} keep={d.remaining_keep} "
            f"discard={d.discard} → level_calc={lc}"
        )

    # 事件图须含关键微事件（bts 只作为其间 before 放置）
    ev_names = {e.name for e in result.events}
    for need in (
        "L0.softmax_exp_stockmeyer",
        "L0.gelu_f1",
        "L0.gelu_f2",
        "L0.gelu_reconstruct",
        "L0.ln2_scale",
        "L0.bridge_rot_attn_dense",
    ):
        if need not in ev_names:
            print("FAIL missing micro-event:", need)
            return 3
    dr_evs = [e for e in result.events if e.dualrail_ct is not None]
    if not dr_evs:
        # DP 已不再选 DualRail pack；layout=dualrail 计价即可
        print("WARN: no dualrail_ct fields (expected; pack choice retired)")

    n_direct = sum(1 for p in result.placements if p.pack_mode == "direct")
    n_rp = sum(1 for p in result.placements if p.pack_mode == "real_pack")
    print(f"placements pack_mode: direct={n_direct} real_pack={n_rp}")
    if n_dr == 0:
        print("WARN: DP chose zero DualRail boots (possible but unusual)")

    hook = BootstrapHook(engine=None, mode="record")
    hook.load_plan(result)
    assert hook.mode == "plan"
    if result.placements:
        sample = result.placements[0].event_name
        assert hook.should_refresh(sample)
        assert hook.pack_mode(sample) in ("direct", "real_pack")
    assert not hook.should_refresh("__never__")

    if DISCARD_MAX_SPAN is not None:
        bad_disc = [
            d for d in result.level_discards if d.discard > DISCARD_MAX_SPAN
        ]
        if bad_disc:
            print(
                "FAIL discard span > "
                f"{DISCARD_MAX_SPAN}: "
                f"{[(d.site, d.discard) for d in bad_disc[:5]]}"
            )
            return 4
        disc_msg = f"discard spans ≤{DISCARD_MAX_SPAN}"
    else:
        disc_msg = "discard span uncapped"
    print(
        "hook plan wiring OK; DualRail is optional (DP-chosen); "
        f"{disc_msg}"
    )
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
