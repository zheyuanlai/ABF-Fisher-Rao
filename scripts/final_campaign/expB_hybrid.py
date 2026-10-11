#!/usr/bin/env python
"""Experiment B, Amendment 3 (A3.3 b): law-insensitive hybrid projection of W_mid.

    python scripts/final_campaign/expB_hybrid.py --system gateway      # the frozen robustness analysis (A3.3 b)
    python scripts/final_campaign/expB_hybrid.py --system lta150       # same computation, descriptive only

For every production arm (method, N, implementation, t_arm):

    W_hyb(s) = tau_mid,u(s) x B x w_arm

tau_mid,u(s): the VALIDATED engine's per-seed persistent mid-threshold budget fraction from the equal-budget production
(`tau_bgrid_e_F_mid_u`, A2.3; censored = inf), same N and method; w_arm: the arm's median measured wall time per
walker-step (wall_total_s / B) over its production records that are uncontended for the whole run (first such
replicate per seed).  The accuracy-vs-walker-steps law therefore comes from the validated engine only and the
backend contributes only its speed.  The section 2.7 decision rule (expB_stats.compare_arms: IUT, rho_best, budget-
artefact labels with completion = B x w_arm) is applied to W_hyb, bootstrapping the validated seeds (ABF 512 / FR 512
share one index draw: the equal-budget arms are paired).  An arm without a validated ladder point or without an
uncontended production record is omitted and listed.  The measured decision is read from
results/final_campaign/expB/analysis/<system>/summary.json; the exploratory decision is "robust to the backend's law"
iff the two labels agree, otherwise "law-sensitive".

Output: results/final_campaign/expB/analysis/<system>/hybrid_projection.json
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import expB_analyze as AN  # noqa: E402
import expB_bench as BN  # noqa: E402
import expB_common as C  # noqa: E402
import expB_run as R  # noqa: E402
import expB_stats as S  # noqa: E402

HYBRID_VERSION = "expB_hybrid/1"


def hybrid_values(tau_by_arm, w_by_arm, B):
    """{arm: {seed: W_hyb}} and {arm: {seed: completion}} for arms with both a validated tau and a measured w."""
    vals, comp = {}, {}
    for a, tau in tau_by_arm.items():
        w = w_by_arm.get(a)
        if not tau or w is None or not math.isfinite(w):
            continue
        vals[a] = {int(s): (math.inf if math.isinf(u) else float(u) * B * w) for s, u in tau.items()}
        comp[a] = {int(s): B * w for s in tau}
    return vals, comp


DECISION_LABELS = ("FR advantage", "ABF advantage", "unresolved")


def agreement(measured_label, hybrid_label):
    """A3.3 (b): robust iff the two DECISION labels agree; anything else (an INCOMPLETE stage, no data) is not
    evaluable."""
    if measured_label not in DECISION_LABELS or hybrid_label not in DECISION_LABELS:
        return "not evaluable"
    return "robust to the backend's law" if measured_label == hybrid_label else "law-sensitive"


def measured_w(spec, arm_rec, out_root):
    """Median wall per walker-step over whole-run-uncontended production records (first such replicate per seed)."""
    ws = []
    for s in spec["seeds"]:
        rr = R.replicate_records(R.prod_job(spec, arm_rec, s, 0), out_root)
        r, prim = AN.select(rr, "whole_run")
        if r is not None and prim:
            ws.append(float(r["wall_total_s"]) / float(spec["B"]))
    return (float(statistics.median(ws)) if ws else None), len(ws)


def run(system, out_root=C.OUT_ROOT, write=True, key=BN.TAU_KEY):
    spec = C.system_spec(system)
    decision = R.load_decision(system, out_root)
    arms, F, A_ = AN.arm_sets(decision)
    tau, w, omitted, n_w = {}, {}, [], {}
    for a in arms:
        t = BN.eqb_tau_mid(spec, a["method"], int(a["N"]), key=key)
        wa, n = measured_w(spec, a, out_root)
        n_w[a["arm"]] = n
        if not t:
            omitted.append(dict(arm=a["arm"], reason="no validated equal-budget ladder point"))
            continue
        if wa is None:
            omitted.append(dict(arm=a["arm"], reason="no whole-run-uncontended production record"))
            continue
        tau[a["arm"]], w[a["arm"]] = t, wa
    vals, comp = hybrid_values(tau, w, float(spec["B"]))
    Fh, Ah = [f for f in F if f in vals], [x for x in A_ if x in vals]
    pairs = [("abf512", "fr512")] if ("abf512" in vals and "fr512" in vals) else []
    res = S.compare_arms(vals, Fh, Ah, pairs=pairs, completion=comp) if (Fh and Ah) else dict(status="no data")
    hyb_label = res.get("decision", {}).get("decision") if isinstance(res, dict) else None
    p = os.path.join(out_root, "analysis", system, "summary.json")
    meas = C.read_json(p) if os.path.exists(p) else {}
    meas_label = meas.get("decision_primary")
    out = dict(version=HYBRID_VERSION, system=system, amendment="PREREGISTRATION.md Amendment 3, A3.3 (b)",
               role=("frozen robustness analysis of the exploratory gateway decision" if system == "gateway"
                     else "descriptive (A3.3 b is frozen for the gateway only)"),
               generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), tau_key=key,
               tau_source=spec["eqb_summary"], B=spec["B"], w_arm=w, n_uncontended_records=n_w, omitted=omitted,
               fr_arms=Fh, abf_arms=Ah, paired=pairs, compare=AN._strip(res),
               hybrid_decision=hyb_label, measured_decision=meas_label,
               measured_stage_status=meas.get("stage_status"),
               measured_decision_on_available_seeds=meas.get("decision_on_available_seeds"),
               agreement=agreement(meas_label, hyb_label),
               agreement_on_available_seeds_descriptive=agreement(meas.get("decision_on_available_seeds"), hyb_label))
    if write:
        C.write_json(os.path.join(out_root, "analysis", system, "hybrid_projection.json"), out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--system", required=True, choices=("gateway", "lta150"))
    ap.add_argument("--out-root", default=C.OUT_ROOT)
    a = ap.parse_args(argv)
    out = run(a.system, a.out_root)
    print(f"{a.system}: hybrid decision {out['hybrid_decision']}; measured {out['measured_decision']} -> "
          f"{out['agreement']}; omitted {out['omitted']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
