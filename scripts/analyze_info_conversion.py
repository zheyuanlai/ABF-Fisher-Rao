#!/usr/bin/env python
"""Information-conversion audit: gates, dose selection, verdicts.

Frozen protocol: ``docs/INFORMATION_CONVERSION_AUDIT_PREREGISTRATION.md``.

Reads ONLY the saved raw CSV outputs of ``run_info_conversion.py`` (never the
in-memory state), so the analysis can be rerun independently of the runs.

Dose selection is structurally blind to FEC outcomes: :func:`select_dose`
refuses any table that carries an error/time-to-accuracy column.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

#: Columns that must never reach the dose-selection table.
FEC_MARKERS = ("e_f", "efprime", "tta", "tau_eps", "speedup", "time_to")

GATES = dict(kl_ratio_max=0.90, ancestor_ess_min=0.90,
             tv_future_ratio_max=0.90, risk_ratio_max=0.90)
BOOT_N = 10_000


def paired_ratio_of_means(a: np.ndarray, b: np.ndarray, n_boot=BOOT_N,
                          seed=0):
    """mean(a)/mean(b) with a paired bootstrap over seeds."""
    n = a.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = a[idx].mean(axis=1) / np.maximum(b[idx].mean(axis=1), 1e-300)
    return (float(a.mean() / b.mean()),
            float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def gate_table(runs_by_cell: dict) -> pd.DataFrame:
    """One row per dose; per-cell gate statistics.  No FEC columns exist here."""
    rows = []
    cells = sorted(runs_by_cell)
    doses = sorted(set(
        runs_by_cell[cells[0]].loc[
            runs_by_cell[cells[0]].arm != "abf", "p90"].dropna()))
    for d in doses:
        row = dict(p90=float(d))
        for cell in cells:
            df = runs_by_cell[cell]
            abf = df[df.arm == "abf"].set_index("seed").sort_index()
            fr = df[np.isclose(df.p90.astype(float), d, atol=1e-12)
                    ].set_index("seed").sort_index()
            if not (abf.index == fr.index).all():
                raise ValueError(f"{cell}: unmatched seeds for dose {d}")
            kl_ratio = (fr.kl_post / fr.kl_pre).values
            tvf_ratio = (fr.tv_future / abf.tv_future).values
            R_fr = fr.R_s.values
            R_abf = abf.R_s.values
            ratio, lo, hi = paired_ratio_of_means(R_fr, R_abf)
            row[f"kl_ratio_median_{cell}"] = float(np.median(kl_ratio))
            row[f"ess_anc_median_{cell}"] = float(np.median(fr.ess_anc.values))
            row[f"tv_future_ratio_median_{cell}"] = float(np.median(tvf_ratio))
            row[f"risk_ratio_{cell}"] = ratio
            row[f"risk_ratio_lo95_{cell}"] = lo
            row[f"risk_ratio_hi95_{cell}"] = hi
            row[f"n_events_median_{cell}"] = float(np.median(fr.n_events.values))
        rows.append(row)
    return pd.DataFrame(rows)


def gate_flags(t: pd.DataFrame, cells) -> pd.DataFrame:
    out = t.copy()
    for cell in cells:
        out[f"g_kl_{cell}"] = t[f"kl_ratio_median_{cell}"] <= GATES["kl_ratio_max"]
        out[f"g_ess_{cell}"] = t[f"ess_anc_median_{cell}"] >= GATES["ancestor_ess_min"]
        out[f"g_tvf_{cell}"] = (t[f"tv_future_ratio_median_{cell}"]
                                <= GATES["tv_future_ratio_max"])
        out[f"g_risk_{cell}"] = ((t[f"risk_ratio_{cell}"] <= GATES["risk_ratio_max"])
                                 & (t[f"risk_ratio_hi95_{cell}"] < 1.0))
    out["safe_both"] = np.logical_and.reduce(
        [out[f"g_kl_{c}"] & out[f"g_ess_{c}"] for c in cells])
    out["pass_both"] = np.logical_and.reduce(
        [out[f"g_kl_{c}"] & out[f"g_ess_{c}"] & out[f"g_tvf_{c}"]
         & out[f"g_risk_{c}"] for c in cells])
    return out


def select_dose(flags: pd.DataFrame):
    """Smallest p90 passing every gate in both cells.  FEC-blind by test."""
    bad = [c for c in flags.columns
           if any(m in c.lower() for m in FEC_MARKERS)]
    if bad:
        raise ValueError(
            f"dose selection saw FEC columns {bad}: the selection must be "
            f"blind to time-to-accuracy outcomes")
    passing = flags[flags.pass_both]
    if passing.empty:
        return None
    return float(passing.p90.min())


def pilot_verdict(flags: pd.DataFrame, cells, stage0_medians) -> dict:
    sel = select_dose(flags)
    weak = flags[flags.safe_both & ~flags.pass_both
                 & np.logical_and.reduce(
                     [flags[f"risk_ratio_{c}"] < 1.0 for c in cells])]
    if sel is not None:
        v = "PILOT_DOSE_SELECTED"
    elif not flags.safe_both.any():
        v = "FR_STRENGTH_GENEALOGY_CONFLICT"
    else:
        v = "FR_DOES_NOT_CONVERT_REPRESENTATION_TO_INFORMATION"
    return dict(verdict=v, selected_p90=sel,
                weak_signal_doses=[float(x) for x in weak.p90.tolist()],
                median_G_ideal=stage0_medians,
                gates=GATES)


# --------------------------------------------------------------------------- #
# Continuation endpoints: frozen q-r conventions, verbatim
# --------------------------------------------------------------------------- #
def tau(df_seed, eps, T, n_consec=3):
    t = df_seed.t.values
    e = df_seed.e_F.values
    for k in range(len(t) - n_consec + 1):
        if np.all(e[k:k + n_consec] <= eps):
            return float(t[k])
    return float(T)


def restricted_speedup(base, meth, eps, T, n_boot=BOOT_N, seed=0):
    b = np.array([tau(base[base.seed == s], eps, T)
                  for s in sorted(base.seed.unique())])
    m = np.array([tau(meth[meth.seed == s], eps, T)
                  for s in sorted(meth.seed.unique())])
    n = min(b.size, m.size)
    b, m = b[:n], m[:n]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        boot.append(b[i].mean() / max(m[i].mean(), 1e-12))
    return (float(b.mean() / m.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
            float(np.mean(b < T)), float(np.mean(m < T)), n)


def stitch_profiles(root, cell):
    """Full e_F(t) per seed/arm: shared burn-in + cooldown + continuation.

    The frozen tau rule scans the whole run, so the burn-in segment (identical
    across arms by construction -- the fork happens at its end) is replicated
    under each arm label before concatenation.
    """
    burn = pd.read_csv(os.path.join(root, "confirm", "burnin_profiles.csv"))
    burn = burn[burn.cell == cell]
    cool = pd.read_csv(os.path.join(root, "confirm",
                                    f"{cell}_cooldown_profiles.csv"))
    cont = pd.read_csv(os.path.join(root, "continuation",
                                    f"{cell}_profiles.csv"))
    arms = sorted(cont.arm.unique())
    parts = []
    for a in arms:
        b = burn.copy()
        b["arm"] = a
        parts += [b, cool[cool.arm == a], cont[cont.arm == a]]
    full = pd.concat(parts, ignore_index=True)
    return full.sort_values(["arm", "seed", "t"]).reset_index(drop=True)


def analyze_continuation(root, cells, thresholds_path):
    thr = json.load(open(thresholds_path))
    out = {}
    for cell in cells:
        prof = stitch_profiles(root, cell)
        base = prof[prof.arm == "abf"]
        meth = prof[prof.arm != "abf"]
        T = thr[cell]["T"]
        res = {}
        for name in ("eps_2", "eps_1"):
            S, lo, hi, pb, pm, n = restricted_speedup(base, meth,
                                                      thr[cell][name], T)
            res[name] = dict(speedup=S, lo95=lo, hi95=hi,
                             p_hit_base=pb, p_hit_meth=pm, n=n,
                             censored_more=bool(pm < pb - 1e-9))
        fb = base.groupby("seed").apply(lambda g: g.e_F.values[-1],
                                        include_groups=False)
        fm = meth.groupby("seed").apply(lambda g: g.e_F.values[-1],
                                        include_groups=False)
        res["final_eF_ratio"] = float(fm.mean() / fb.mean())
        out[cell] = res
    s2 = {c: out[c]["eps_2"] for c in cells}
    fec_pass = all(
        s2[c]["speedup"] >= 1.15 and s2[c]["lo95"] > 1.0
        and not s2[c]["censored_more"]
        and out[c]["final_eF_ratio"] <= 1.05 for c in cells)
    out["practical_pass"] = bool(fec_pass)
    return out


# --------------------------------------------------------------------------- #
def load_runs(root, stage, cells):
    out = {}
    for cell in cells:
        p = os.path.join(root, stage, f"{cell}_runs.csv")
        if not os.path.exists(p):
            return None
        out[cell] = pd.read_csv(p)
    return out


def stage0_medians(root, stage, cells):
    med = {}
    for cell in cells:
        df = pd.read_csv(os.path.join(root, stage, f"{cell}_stage0.csv"))
        med[cell] = float(df.G_ideal.median())
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "confirm", "continuation"])
    ap.add_argument("--root", default="results/information_conversion")
    ap.add_argument("--cells", nargs="+", default=["K2", "K3"])
    ap.add_argument("--thresholds",
                    default="results/qr_decoupling/thresholds.json")
    args = ap.parse_args()

    if args.stage == "continuation":
        res = analyze_continuation(args.root, args.cells, args.thresholds)
        out = os.path.join(args.root, "continuation", "continuation_result.json")
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(json.dumps(res, indent=1))
        return

    runs = load_runs(args.root, args.stage, args.cells)
    if runs is None:
        print(f"{args.stage}: no runs CSVs (Stage-0D stop, or runs missing); "
              f"nothing to analyze beyond the runner's verdict")
        return
    t = gate_table(runs)
    flags = gate_flags(t, args.cells)
    med = stage0_medians(args.root, args.stage, args.cells)
    flags.to_csv(os.path.join(args.root, args.stage, "gates.csv"), index=False)

    if args.stage == "pilot":
        v = pilot_verdict(flags, args.cells, med)
        with open(os.path.join(args.root, "pilot", "pilot_verdict.json"),
                  "w") as fh:
            json.dump(v, fh, indent=1)
        print(flags.to_string(index=False))
        print(json.dumps({k: v[k] for k in
                          ("verdict", "selected_p90", "weak_signal_doses")},
                         indent=1))
    else:
        assert len(flags) == 1, "confirmation carries exactly one dose"
        row = flags.iloc[0]
        mech = bool(row.pass_both)
        v = dict(mechanism_pass_both_cells=mech,
                 p90=float(row.p90), median_G_ideal=med,
                 gates=GATES,
                 detail=json.loads(flags.to_json(orient="records"))[0])
        with open(os.path.join(args.root, "confirm", "confirm_verdict.json"),
                  "w") as fh:
            json.dump(v, fh, indent=1)
        print(flags.to_string(index=False))
        print("mechanism PASS" if mech else "mechanism FAIL")


if __name__ == "__main__":
    main()
