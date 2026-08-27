"""Cross-system tables for the IO-ABF campaign.

Emits, verbatim from the frozen criteria and with no re-derivation of any
threshold: the headline verdict table, the difficulty-decomposition table, and
the per-system Gamma statistics the preregistration lists in section 13.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "io_abf_overnight")
ORDER = ["eb_beta4", "eb_beta8", "gateway"]
LABEL = {"eb_beta4": "Bottleneck beta=4", "eb_beta8": "Bottleneck beta=8",
         "gateway": "Entropic gateway"}


def load(system, phase="confirmatory"):
    p = os.path.join(OUT, system, "analysis", f"{phase}_endpoint.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def fmt(v, n=3):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v:.{n}f}"


def headline():
    rows = ["| System | role | R_Γ | A6b S(ε₂) | 95% CI | hit A6b/A0 | A6c S(ε₂) | mass ESS | final A6b/A0 | full-domain A6b/A0 | verdict |",
            "|---|---|---:|---:|:--:|:--:|---:|---:|---:|---:|:--|"]
    for s in ORDER:
        d = load(s)
        if d is None:
            rows.append(f"| {LABEL[s]} | — | — | — | — | — | — | — | — | — | not run |")
            continue
        e = d["endpoint"]
        b = e["arms"].get("A6b", {})
        c = e["arms"].get("A6c", {})
        a6c = e.get("a6c", {})
        rows.append(
            f"| {LABEL[s]} | {e['role']} | {e['R_gamma']:.1f} | "
            f"**{fmt(b.get('S_eps2'))}** | "
            f"[{fmt(b.get('S_eps2_ci',[np.nan,np.nan])[0])}, "
            f"{fmt(b.get('S_eps2_ci',[np.nan,np.nan])[1])}] | "
            f"{fmt(b.get('hit_eps2'),2)}/{fmt(e['arms']['A0'].get('hit_eps2'),2)} | "
            f"{fmt(c.get('S_eps2'))} | {fmt(a6c.get('mass_ess_median'))} | "
            f"{fmt(b.get('final_ratio_to_A0'))} | "
            f"{fmt(b.get('final_full_ratio_to_A0'))} | "
            f"**{e['verdict']}** |")
    return "\n".join(rows)


def checks_table():
    rows = ["| System | S ≥ 1.15 | CI lower > 1 | censoring ok | final ≤ 1.10× | full ≤ 1.10× |",
            "|---|:--:|:--:|:--:|:--:|:--:|"]
    tick = {True: "PASS", False: "**FAIL**"}
    for s in ORDER:
        d = load(s)
        if d is None:
            continue
        v = d["endpoint"].get("verdict_checks", {})
        rows.append(f"| {LABEL[s]} | " + " | ".join(
            tick[bool(v.get(k))] for k in
            ("speedup_at_least_1_15", "ci_lower_above_1", "censoring_not_worse",
             "final_within_10pct", "final_full_within_10pct")) + " |")
    return "\n".join(rows)


def decomposition_table():
    rows = ["| System | Q₁₀(σ²) | Q₉₀(σ²) | R_σ | Q₁₀(τ) | Q₉₀(τ) | R_τ | Q₁₀(Γ) | Q₉₀(Γ) | R_Γ | valid-τ | ρ_s(Γ early, late) | dominant |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|"]
    for s in ORDER:
        d = load(s)
        if d is None:
            continue
        c = d["decomposition"]
        rows.append(
            f"| {LABEL[s]} | {c['sigma2']['q10']:.3g} | {c['sigma2']['q90']:.3g} | "
            f"{c['sigma2']['ratio']:.1f} | {c['tau']['q10']:.3g} | {c['tau']['q90']:.3g} | "
            f"{c['tau']['ratio']:.1f} | {c['gamma']['q10']:.3g} | {c['gamma']['q90']:.3g} | "
            f"{c['gamma']['ratio']:.1f} | {c['valid_tau_fraction']:.3f} | "
            f"{c.get('spearman_gamma_early_late', float('nan')):.3f} | "
            f"{c.get('dominant_source','—')} |")
    return "\n".join(rows)


def wca_screening_table():
    files = sorted(glob.glob(os.path.join(OUT, "wca", "screening", "*.npz")))
    if not files:
        return None, 0
    sig, tau, gam, a = [], [], [], None
    for p in files:
        with np.load(p, allow_pickle=True) as d:
            sig.append(d["io_sigma2"]); tau.append(d["io_tau"])
            gam.append(d["io_gamma"]); a = d["io_a_cell"]
    scored = a > 0

    def spread(v):
        w = np.stack(v)[:, scored]
        w = w[np.isfinite(w) & (w > 0)]
        if w.size < 4:
            return (float("nan"),) * 3
        q10, q90 = np.quantile(w, 0.1), np.quantile(w, 0.9)
        return float(q10), float(q90), float(q90 / max(q10, 1e-300))

    t = np.stack(tau)[:, scored]
    vt = float(np.mean(np.isfinite(t) & (t > 0)))
    s10, s90, sr = spread(sig); t10, t90, tr = spread(tau); g10, g90, gr = spread(gam)
    dom = "sigma2" if sr > 3 * tr else "tau" if tr > 3 * sr else "both"
    row = ("| WCA dimer (A0 only) | "
           f"{s10:.3g} | {s90:.3g} | {sr:.1f} | {t10:.3g} | {t90:.3g} | {tr:.1f} | "
           f"{g10:.3g} | {g90:.3g} | {gr:.1f} | {vt:.3f} | — | {dom} |")
    return row, len(files)


def main():
    print("## Headline\n"); print(headline())
    print("\n## Preregistered checks (A6b vs A0)\n"); print(checks_table())
    print("\n## Difficulty decomposition\n")
    dec = decomposition_table()
    wca_row, n_wca = wca_screening_table()
    if wca_row:
        dec = dec + "\n" + wca_row
    print(dec)
    if n_wca:
        print(f"\nWCA row is a **diagnostic only** ({n_wca} A0 seeds): its reference "
              f"gate failed, so no speedup is reported for it.")


if __name__ == "__main__":
    main()
