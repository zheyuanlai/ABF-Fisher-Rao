#!/usr/bin/env python
"""Mechanism extras for a pentane R15 OT stage: marginal flatness, support deficit, occupancy
shift, torsional kinetics, region-resolved mean-force error, compact deposit share.

    python scripts/pentane_r15_mechanism_extras.py --stage pilot|confirmatory
-> results/ot_repair_campaign/pentane_r15/<stage>/mechanism_extras.{json,md}
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CAMP = os.path.join(ROOT, "results", "ot_repair_campaign", "pentane_r15")
REF_V2 = os.path.join(ROOT, "cache", "alkanes_cv", "ref_pentane_b2_R15_v2_meanforce.npz")
LAB = {"abf": "A", "fr": "F", "ot": "T", "abf_r": "R", "fr_r": "F+R", "ot_r": "T+R"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--root", default=CAMP)
    a = ap.parse_args()
    v2 = np.load(REF_V2, allow_pickle=True); grid = v2["grid"]; dz = float(v2["dz"]); win = v2["window_mask"]; Fp2 = v2["Fprime"]
    rows = {}
    for f in sorted(glob.glob(os.path.join(a.root, a.stage, "raw", "*.npz"))):
        d = np.load(f, allow_pickle=True); lab = LAB[str(d["arm"])]
        lo, hi = d["ot_domain"]; inside = (grid >= lo) & (grid <= hi); U = 1 / (hi - lo)
        ph = d["p_hat"]; m_in = ph[..., inside].sum(-1) * dz; p_in = ph[..., inside] / m_in[..., None]
        kl = (p_in * np.log(np.clip(p_in, 1e-12, None) / U)).sum(-1) * dz
        steps = d["steps"]; sel = steps >= 12000
        ec = d["final_eff_counts"]
        low = [np.mean(ec[r][win] < 0.25 * np.median(ec[r][win])) for r in range(ec.shape[0])]
        comp = float("nan")
        if "csum_prod" in d.files:
            cs = d["csum_prod"]; comp = float(np.median(cs[:, win & (grid < 2.65)].sum(1) / cs[:, win].sum(1)))
        mf = d["mean_force"][-1]; mc = win & (grid < 2.75); me = win & (grid >= 2.75)
        ess = d["ancestor_ess"][-1]
        rows[lab] = dict(kl_final=float(np.median(kl[-1])), kl_int=float(np.median(np.trapezoid(kl[sel], steps[sel], axis=0))),
                         outside_domain_mass=float(np.median(1 - m_in[-1])),
                         n_transitions=float(np.median(d["n_transitions"])), round_trips=float(np.median(d["n_round_trips"])),
                         frac_compact_end=float(np.median(d["frac_compact"][-1])), frac_extended_end=float(np.median(d["frac_extended"][-1])),
                         low_support_fraction=float(np.median(low)), compact_deposit_share=comp,
                         mf_rms_err_R_lt_275=float(np.median(np.sqrt(np.mean((mf[:, mc] - Fp2[mc]) ** 2, 1)))),
                         mf_rms_err_R_ge_275=float(np.median(np.sqrt(np.mean((mf[:, me] - Fp2[me]) ** 2, 1)))),
                         ancestor_ess_end=(float(np.nanmedian(ess)) if np.isfinite(ess).any() else float("nan")),
                         replacements=float(np.median(d["total_replacement_events"])), n_seeds=int(d["n_seeds"]))
    hdr = "| arm | KL(p‖U) end | ∫KL dt | outside domain | transitions | round trips | walkers R<2.17 | walkers R>3.17 | low-support frac | compact deposit share | mf RMS err R<2.75 | mf RMS err R≥2.75 | ESS | replacements |"
    lines = [f"# {a.stage}: mechanism extras (medians over seeds)", "", hdr, "|" + "---|" * 14]
    for lab in ("A", "F", "T", "R", "F+R", "T+R"):
        if lab in rows:
            r = rows[lab]
            lines.append(f"| {lab} | {r['kl_final']:.3f} | {r['kl_int']:.0f} | {r['outside_domain_mass']:.3f} | {r['n_transitions'] / 1e6:.2f} M | {r['round_trips']:.0f} | "
                         f"{100 * r['frac_compact_end']:.1f} % | {100 * r['frac_extended_end']:.1f} % | {r['low_support_fraction']:.3f} | {100 * r['compact_deposit_share']:.1f} % | "
                         f"{r["mf_rms_err_R_lt_275"]:.2f} | {r["mf_rms_err_R_ge_275"]:.2f} | {r['ancestor_ess_end']:.0f} | {r['replacements']:.0f} |")
    out = os.path.join(a.root, a.stage)
    json.dump(rows, open(os.path.join(out, "mechanism_extras.json"), "w"), indent=1)
    open(os.path.join(out, "mechanism_extras.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
