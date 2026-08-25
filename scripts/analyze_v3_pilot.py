#!/usr/bin/env python
"""Frozen gate analysis for the v3 pilot.

Frozen protocol: docs/V3_PREREGISTRATION.md (v3.1) with Amendments 1-6.

Nothing here is a choice.  Thresholds come from the immutable
``V3_THRESHOLDS.json``; censoring follows Amendment 6a; the Track-C reporting
order follows 6c (R_shape, R_FR, then R_total); the gates are the frozen
mechanism-positive and advancement-positive conditions.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

PERSIST_FRAMES = 5          # Amendment 6a
TAU_CENSOR = 96             # one index past the latest valid start (95)
N_SEEDS = 8
FAVORABLE_REQUIRED = 6      # >= 6/8


def tau_eps(err: np.ndarray, eps: float) -> tuple[int, bool]:
    """Restricted hitting time and hit indicator (Amendment 6a)."""
    n = len(err)
    for start in range(0, n - PERSIST_FRAMES):
        if np.all(err[start:start + PERSIST_FRAMES + 1] <= eps):
            return start, True
    return TAU_CENSOR, False


def _arm_frame(root: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(root / "production_gpu" / "production_gpu_runs_long.csv")


def per_seed_taus(df: pd.DataFrame, col: str, eps: float) -> pd.DataFrame:
    out = []
    for seed, g in df.groupby("seed"):
        g = g.sort_values("step")
        t, hit = tau_eps(g[col].to_numpy(), eps)
        out.append(dict(seed=int(seed), tau=t, hit=bool(hit)))
    return pd.DataFrame(out)


def speedup(ref: pd.DataFrame, arm: pd.DataFrame) -> dict:
    m = ref.merge(arm, on="seed", suffixes=("_ref", "_arm"))
    if len(m) != N_SEEDS:
        raise ValueError(f"expected {N_SEEDS} matched seeds, got {len(m)}")
    s = m.tau_ref / m.tau_arm
    return dict(median_S=float(s.median()), favorable=int((s > 1.0).sum()),
                censored_ref=int((~m.hit_ref).sum()),
                censored_arm=int((~m.hit_arm).sum()),
                both_censored=int(((~m.hit_ref) & (~m.hit_arm)).sum()))


def dose_decay(df: pd.DataFrame) -> float:
    """Replacements per opportunity, first quarter of the window vs last.

    fr_stride == eval_every, so exactly one FR opportunity falls between saved
    frames and differencing the cumulative counter recovers the per-opportunity
    dose exactly.
    """
    ratios = []
    for _, g in df.groupby("seed"):
        g = g.sort_values("step")
        d = np.diff(g["cumulative_replacements"].to_numpy())
        active = np.flatnonzero(d > 0)
        if active.size < 8:
            return float("nan")
        lo, hi = active[0], active[-1] + 1
        window = d[lo:hi]
        q = max(len(window) // 4, 1)
        first, last = window[:q].mean(), window[-q:].mean()
        ratios.append(first / max(last, 1e-9))
    return float(np.median(ratios))


def retention_trajectory(df: pd.DataFrame, burnin=0.2, stop=0.8) -> list[float]:
    """G_t = ESS_anc(after)/ESS_anc(before) per FR opportunity, median over seeds.

    Restricted to the FR window.  Intervals outside it contain no opportunity and
    contribute a trivial G = 1; padding the trajectory with those would dilute
    exactly the early-versus-late distinction the diagnostic exists to make.
    """
    n_steps = int(df["step"].max())
    lo, hi = int(round(burnin * n_steps)), int(round(stop * n_steps))
    per = []
    for _, g in df.groupby("seed"):
        g = g.sort_values("step")
        steps = g["step"].to_numpy()
        e = g["ancestor_ess"].to_numpy()
        ratio = np.where(e[:-1] > 0, e[1:] / np.maximum(e[:-1], 1e-9), np.nan)
        in_window = (steps[1:] >= lo) & (steps[1:] <= hi)
        per.append(np.where(in_window, ratio, np.nan))
    m = np.nanmedian(np.vstack(per), axis=0)
    return [float(v) for v in m if np.isfinite(v)]


def apply_gates(rec: dict, track: str, role: str) -> dict:
    """The frozen mechanism-positive and advancement-positive conditions.

    Every condition is conjunctive and every inequality direction is pinned by a
    boundary test.  NaN fails closed: a missing component can never read as a
    pass, which is the no-data-reads-as-PASS class from this project's history.
    """
    def ok(v, cmp, bound):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
        return (v >= bound) if cmp == "ge" else (v <= bound)

    gen = (ok(rec.get("seeds_ess_ok"), "ge", FAVORABLE_REQUIRED)
           and ok(rec.get("median_ess_frac"), "ge", 0.5)
           and ok(rec.get("seeds_wmax_ok"), "ge", FAVORABLE_REQUIRED)
           and ok(rec.get("median_wmax"), "le", 0.10))
    noninf = (ok(rec.get("final_ratio_F_R12"), "le", 1.05)
              and ok(rec.get("final_ratio_Fp_R12"), "le", 1.05))
    barrier = ok(rec.get("final_ratio_Fp_barrier"), "le", 1.10)

    # mechanism-positive: Track C candidates only, judged against the control
    mech = None
    if track == "C" and role == "candidate":
        mech = (ok(rec.get("dose_decay"), "ge", 5.0)
                and gen and noninf and barrier
                and ok(rec.get("SvsCtrl_F_2"), "ge", 1.05)
                and ok(rec.get("SvsCtrl_Fprime_2"), "ge", 1.05)
                and ok(rec.get("favCtrl_F_2"), "ge", FAVORABLE_REQUIRED)
                and ok(rec.get("favCtrl_Fprime_2"), "ge", FAVORABLE_REQUIRED))

    adv = (all(ok(rec.get(f"S_{o}_{i}"), "ge", 1.10)
               for o in ("F", "Fprime") for i in ("1", "2"))
           and ok(rec.get("fav_F_2"), "ge", FAVORABLE_REQUIRED)
           and ok(rec.get("fav_Fprime_2"), "ge", FAVORABLE_REQUIRED)
           and noninf
           and ok(rec.get("seeds_noninferior_F_R12"), "ge", FAVORABLE_REQUIRED)
           and ok(rec.get("seeds_noninferior_Fp_R12"), "ge", FAVORABLE_REQUIRED)
           and barrier and gen
           and ok(rec.get("final_ratio_F_full"), "le", 1.25))
    if track == "C" and role == "candidate":
        adv = adv and bool(mech)

    return dict(passes_genealogy=gen, passes_noninferiority=noninf,
                passes_barrier=barrier, mechanism_positive=mech,
                advancement_positive=bool(adv))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="results/v3/arms/arm_manifest.json")
    ap.add_argument("--baseline", default="results/v3/plain_abf")
    ap.add_argument("--thresholds", default="results/v3/V3_THRESHOLDS.json")
    ap.add_argument("--out", default="results/v3/pilot_analysis")
    args = ap.parse_args(argv)

    th = json.loads(pathlib.Path(args.thresholds).read_text())
    if not th.get("frozen_before_fr"):
        raise ValueError("threshold artifact is not marked frozen_before_fr")
    if th.get("online_fr_results_viewed_before_freeze") is not False:
        raise ValueError("threshold artifact admits online FR was seen first")
    eps = {k: v["value"] for k, v in th["thresholds"].items()}
    print(f"thresholds (scope {th['scope']}): " +
          "  ".join(f"{k}={v:.6f}" for k, v in eps.items()))

    arms = json.loads(pathlib.Path(args.manifest).read_text())["arms"]
    base = _arm_frame(pathlib.Path(args.baseline))
    frames = {"plain_abf": base}
    for a in arms:
        root = pathlib.Path(a["output_root"])
        if (root / "production_gpu" / "production_gpu_runs_long.csv").exists():
            frames[a["arm"]] = _arm_frame(root)
    print(f"arms present: {len(frames)}/21")

    # same-bias control for each Track-C candidate (Amendment 6c)
    control_of = {
        "C_capped8_FT_rho0.70": "C_capped8_noFR",
        "C_capped8_FT_rho0.85": "C_capped8_noFR",
        "C_capped12_FT_rho0.70": "C_capped12_noFR",
        "C_capped12_FT_rho0.85": "C_capped12_noFR",
        "C_capped12_holdout": "C_capped12_noFR",
        "C_capped12_oracle_refresh": "C_capped12_noFR",
        "C_capped12_oracle_target": "C_capped12_noFR",
        "C_capped12_K1024": "C_capped12_noFR",
        "C_tempered8_FT_rho0.85": "C_tempered8_noFR",
        "C_flat_FT_rho0.85": "plain_abf",       # the flat family IS plain ABF
    }

    rows = []
    specs = [("F", "l2_F_R12", "eps_F_1", "eps_F_2"),
             ("Fprime", "l2_Fprime_R12", "eps_Fprime_1", "eps_Fprime_2")]
    for name, df in frames.items():
        if name == "plain_abf":
            continue
        rec = dict(arm=name)
        for obs, col, k1, k2 in specs:
            for lab, key in (("1", k1), ("2", k2)):
                ref = per_seed_taus(base, col, eps[key])
                arm = per_seed_taus(df, col, eps[key])
                s = speedup(ref, arm)
                rec[f"S_{obs}_{lab}"] = s["median_S"]
                rec[f"fav_{obs}_{lab}"] = s["favorable"]
                rec[f"cens_{obs}_{lab}"] = s["censored_arm"]
                rec[f"censref_{obs}_{lab}"] = s["censored_ref"]
                ctrl = control_of.get(name)
                if ctrl and ctrl in frames:
                    cref = per_seed_taus(frames[ctrl], col, eps[key])
                    sc = speedup(cref, arm)
                    rec[f"SvsCtrl_{obs}_{lab}"] = sc["median_S"]
                    rec[f"favCtrl_{obs}_{lab}"] = sc["favorable"]
                    sh = speedup(ref, cref)
                    rec[f"Rshape_{obs}_{lab}"] = sh["median_S"]
        f_arm = df[df.step == df.step.max()]
        f_base = base[base.step == base.step.max()]
        # Paired per seed, then take the median.  A ratio of medians is not the
        # same statistic and discards the matched-noise design that every other
        # comparison in this campaign relies on.
        for col, tag in (("l2_F_R12", "F_R12"), ("l2_Fprime_R12", "Fp_R12"),
                         ("l2_Fprime_barrier", "Fp_barrier"), ("l2_F_full", "F_full")):
            m = f_arm[["seed", col]].merge(
                f_base[["seed", col]], on="seed", suffixes=("_arm", "_base"))
            if len(m) != N_SEEDS:
                raise ValueError(f"{name}: {len(m)} matched seeds for {col}")
            ratios = m[f"{col}_arm"] / m[f"{col}_base"]
            rec[f"final_ratio_{tag}"] = float(ratios.median())
            rec[f"seeds_noninferior_{tag}"] = int((ratios <= 1.05).sum())
        n_part = int(df["n_unique_ancestors"].max())
        rec["median_ess_frac"] = float((f_arm["ancestor_ess"] / n_part).median())
        rec["seeds_ess_ok"] = int((f_arm["ancestor_ess"] >= 0.5 * n_part).sum())
        rec["median_wmax"] = float(f_arm["max_clone_weight"].median())
        rec["seeds_wmax_ok"] = int((f_arm["max_clone_weight"] <= 0.10).sum())
        rec["dose_decay"] = dose_decay(df)
        rec["replacements"] = float(f_arm["cumulative_replacements"].median())
        meta = next((a for a in arms if a["arm"] == name), {})
        rec["track"], rec["role"] = meta.get("track", "?"), meta.get("role", "?")
        rec.update(apply_gates(rec, rec["track"], rec["role"]))
        rows.append(rec)

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "v3_pilot_gates.csv", index=False)
    print(f"\nwrote {out/'v3_pilot_gates.csv'} ({len(res)} arms)")
    cand = res[res.role == "candidate"]
    print(f"\ncandidates: {len(cand)}   "
          f"mechanism-positive: {int(cand.mechanism_positive.fillna(False).sum())}   "
          f"advancement-positive: {int(cand.advancement_positive.sum())}")

    ret = {n: retention_trajectory(d) for n, d in frames.items()
           if n != "plain_abf" and d["cumulative_replacements"].max() > 0}
    (out / "retention_trajectories.json").write_text(json.dumps(ret, indent=2))
    print(f"wrote {out/'retention_trajectories.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
