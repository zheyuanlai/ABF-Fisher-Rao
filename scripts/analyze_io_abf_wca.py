"""Score the WCA IO-ABF runs post hoc against the current reference, and report.

The sampler stores ``pmf`` at every save, so the free-energy error curve is
recovered without re-running dynamics -- which is the whole point of that
storage decision, and why a future reference change costs nothing here.

**The reference is `cache/phase_hp_v3`, never the default cache.** The default
`cache/wca_ti_reference.npz` is the build the Stage-A audit found wrong by 24.8
sigma at z = 0.255; loading it here would silently score every arm against a
known-defective curve.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import wca_abffr_core as core                                    # noqa: E402
from scipy.stats import spearmanr                                # noqa: E402

OUT = os.path.join(ROOT, "results", "io_abf_overnight", "wca")
REF = os.path.join(ROOT, "cache", "phase_hp_v3",
                   "wca_ti_b1_h2_w2_n10_a1.5_g160.npz")


def reference():
    with np.load(REF, allow_pickle=True) as d:
        return {"grid": d["grid"], "free_energy": d["free_energy"],
                "mean_force": d["mean_force"], "label": str(d["label"])}


def score(phase="screening"):
    sim = core.SimConfig()
    ref = reference()
    grid = ref["grid"]
    emask = core.eval_window_mask_np(grid, sim)
    full = np.ones_like(emask, dtype=bool)
    files = sorted(glob.glob(os.path.join(OUT, phase, "*.npz")))
    if not files:
        raise SystemExit(f"no records in {os.path.join(OUT, phase)}")
    recs = []
    for p in files:
        with np.load(p, allow_pickle=True) as d:
            r = {k: d[k] for k in d.files}
        pmf = np.asarray(r["pmf"], dtype=float)            # (n_saves, G)
        e = np.array([core.profile_l2_error_np(
            core.align_additive_constant_np(pmf[i], ref["free_energy"], grid, emask),
            ref["free_energy"], grid, emask) for i in range(pmf.shape[0])])
        ef = np.array([core.profile_l2_error_np(
            core.align_additive_constant_np(pmf[i], ref["free_energy"], grid, full),
            ref["free_energy"], grid, full) for i in range(pmf.shape[0])])
        r["l2_f_t"], r["l2_f_full_t"] = e, ef
        r["t"] = np.asarray(r["times"], dtype=float)
        recs.append(r)
    return recs, emask, grid


def report(phase="screening"):
    recs, emask, grid = score(phase)
    arm = str(recs[0].get("io_arm", "A0"))
    t = recs[0]["t"]
    E = np.stack([r["l2_f_t"] for r in recs])
    EF = np.stack([r["l2_f_full_t"] for r in recs])
    T = float(t[-1])

    a_cell = recs[0]["io_a_cell"]
    scored = a_cell > 0

    def spread(name):
        v = np.stack([r[f"io_{name}"] for r in recs])[:, scored]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size < 4:
            return dict(q10=float("nan"), q90=float("nan"), ratio=float("nan"))
        q10, q90 = float(np.quantile(v, .1)), float(np.quantile(v, .9))
        return dict(q10=q10, q90=q90, ratio=q90 / max(q10, 1e-300))

    tau = np.stack([r["io_tau"] for r in recs])[:, scored]
    valid = float(np.mean(np.isfinite(tau) & (tau > 0)))

    g_t = np.stack([r["io_gamma_t"] for r in recs])
    sp = float("nan")
    if g_t.ndim == 3 and g_t.shape[1] >= 4:
        n = g_t.shape[1]
        a_ = np.log(np.maximum(g_t[:, :n // 3].mean(1)[:, scored], 1e-300))
        b_ = np.log(np.maximum(g_t[:, -n // 3:].mean(1)[:, scored], 1e-300))
        sp = float(np.median([spearmanr(a_[i], b_[i]).statistic
                              for i in range(a_.shape[0])]))

    s2, tt, gg = spread("sigma2"), spread("tau"), spread("gamma")
    out = dict(
        phase=phase, arm=arm, n_seeds=len(recs), T_total=T,
        reference=os.path.relpath(REF, ROOT), reference_label=reference()["label"],
        n_cells=int(a_cell.size), n_scored_cells=int(scored.sum()),
        obs_every=int(recs[0].get("io_obs_every", -1)),
        eps1=float(np.median(E[:, int(np.argmin(abs(t - 0.4 * T)))])),
        eps2=float(np.median(E[:, int(np.argmin(abs(t - 0.6 * T)))])),
        final_l2_f_median=float(np.median(E[:, -1])),
        final_l2_f_full_median=float(np.median(EF[:, -1])),
        valid_tau_fraction=valid, gamma_unresolved=bool(valid < 0.80),
        R_sigma2=s2, R_tau=tt, R_gamma=gg,
        spearman_gamma_early_late=sp,
        dominant_source=("sigma2" if s2["ratio"] > 3 * tt["ratio"]
                         else "tau" if tt["ratio"] > 3 * s2["ratio"] else "both"))
    os.makedirs(os.path.join(OUT, "analysis"), exist_ok=True)
    with open(os.path.join(OUT, "analysis", f"{phase}_report.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    np.savez_compressed(os.path.join(OUT, "analysis", f"{phase}_curves.npz"),
                        t=t, l2_f=E, l2_f_full=EF,
                        sigma2=np.stack([r["io_sigma2"] for r in recs]),
                        tau=np.stack([r["io_tau"] for r in recs]),
                        gamma=np.stack([r["io_gamma"] for r in recs]),
                        a_cell=a_cell, cell_edges=recs[0]["io_cell_edges"])
    print(json.dumps(out, indent=2, default=str))
    return out


# --------------------------------------------------------------------------- #
# paired arms: the same endpoint as the batched systems, on WCA's own records
# --------------------------------------------------------------------------- #
def arms_report(phase="pilot"):
    """A0 / A6b / A6c on the frozen thresholds, scored against hp_v3.

    The thresholds come from the A0 calibration report and are **read**, never
    recomputed: the whole point of freezing them in a separate phase is that a
    later phase cannot see a candidate result while choosing them.
    """
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from analyze_io_abf import (hitting_time, restricted, speedup,
                                paired_bootstrap)                # noqa: E402

    cal = os.path.join(OUT, "analysis", "screening_report.json")
    if not os.path.exists(cal):
        raise SystemExit("[wca] no frozen thresholds; run the A0 calibration first")
    with open(cal) as fh:
        thr = json.load(fh)

    sim = core.SimConfig()
    ref = reference()
    grid, emask = ref["grid"], core.eval_window_mask_np(ref["grid"], sim)
    full = np.ones_like(emask, dtype=bool)

    by = {}
    for path in sorted(glob.glob(os.path.join(OUT, phase, "*.npz"))):
        with np.load(path, allow_pickle=True) as d:
            r = {k: d[k] for k in d.files}
        pmf = np.asarray(r["pmf"], dtype=float)
        e = np.array([core.profile_l2_error_np(
            core.align_additive_constant_np(pmf[i], ref["free_energy"], grid, emask),
            ref["free_energy"], grid, emask) for i in range(pmf.shape[0])])
        ef = np.array([core.profile_l2_error_np(
            core.align_additive_constant_np(pmf[i], ref["free_energy"], grid, full),
            ref["free_energy"], grid, full) for i in range(pmf.shape[0])])
        by.setdefault(str(r["io_arm"]), {})[int(r["seed"])] = dict(
            t=np.asarray(r["times"], float), e=e, ef=ef, rec=r)

    if "A0" not in by:
        raise SystemExit(f"[wca/{phase}] no A0 rows")
    seeds = sorted(set.intersection(*[set(v) for v in by.values()]))
    if not seeds:
        raise SystemExit(f"[wca/{phase}] no seed is present for every arm")
    t = by["A0"][seeds[0]]["t"]
    T = float(t[-1])
    eps1, eps2 = float(thr["eps1"]), float(thr["eps2"])

    tab = {}
    for arm, d in by.items():
        E = np.stack([d[s]["e"] for s in seeds])
        EF = np.stack([d[s]["ef"] for s in seeds])
        tab[arm] = dict(
            tau1=np.array([hitting_time(t, E[i], eps1) for i in range(len(seeds))]),
            tau2=np.array([hitting_time(t, E[i], eps2) for i in range(len(seeds))]),
            final=E[:, -1], final_full=EF[:, -1], E=E, EF=EF)

    ref_row = tab["A0"]
    out = dict(system="wca", phase=phase, n_seeds=len(seeds), seeds=seeds, T=T,
               eps1=eps1, eps2=eps2,
               gamma_unresolved=bool(thr["gamma_unresolved"]),
               R_gamma=thr["R_gamma"]["ratio"],
               reference=os.path.relpath(REF, ROOT), arms={})
    for arm, r in tab.items():
        rec = dict(arm=arm)
        for lab, key in (("eps1", "tau1"), ("eps2", "tau2")):
            S = speedup(ref_row[key], r[key], T)
            lo, hi = paired_bootstrap(ref_row[key], r[key], T)
            rec[f"S_{lab}"] = S
            rec[f"S_{lab}_ci"] = [lo, hi]
            rec[f"hit_{lab}"] = float(np.mean(np.isfinite(r[key])))
        rec["final_median"] = float(np.median(r["final"]))
        rec["final_full_median"] = float(np.median(r["final_full"]))
        rec["final_ratio_to_A0"] = float(np.median(r["final"]) / np.median(ref_row["final"]))
        rec["final_full_ratio_to_A0"] = float(
            np.median(r["final_full"]) / np.median(ref_row["final_full"]))
        # threshold-free view, the statistic section 3b of the summary prefers
        frac = [0.2, 0.4, 0.6, 0.8, 1.0]
        idx = [int(np.argmin(np.abs(t - f * T))) for f in frac]
        rec["ratio_at_fractions"] = {
            f"{f:.1f}": float(np.median(r["E"][:, k]) / np.median(ref_row["E"][:, k]))
            for f, k in zip(frac, idx)}
        out["arms"][arm] = rec

    if "A6b" in out["arms"]:
        a = out["arms"]["A6b"]
        checks = dict(
            speedup_at_least_1_15=bool(a["S_eps2"] >= 1.15),
            ci_lower_above_1=bool(a["S_eps2_ci"][0] > 1.0),
            censoring_not_worse=bool(a["hit_eps2"] >= out["arms"]["A0"]["hit_eps2"] - 0.05),
            final_within_10pct=bool(a["final_ratio_to_A0"] <= 1.10),
            final_full_within_10pct=bool(a["final_full_ratio_to_A0"] <= 1.10))
        out["verdict_checks"] = checks
        out["verdict"] = "POSITIVE" if all(checks.values()) else "NOT POSITIVE"
        # the registered pre-candidate prediction, checked mechanically
        out["prediction_check"] = dict(
            registered="ratio at horizon >= 0.92 (leverage-only, not difficulty)",
            observed=a["ratio_at_fractions"]["1.0"],
            holds=bool(a["ratio_at_fractions"]["1.0"] >= 0.92),
            falsifier="ratio < 0.65 would mean the gain does not follow Gamma",
            falsified=bool(a["ratio_at_fractions"]["1.0"] < 0.65))

    os.makedirs(os.path.join(OUT, "analysis"), exist_ok=True)
    with open(os.path.join(OUT, "analysis", f"{phase}_endpoint.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="screening")
    a = ap.parse_args()
    (report if a.phase == "screening" else arms_report)(a.phase)
