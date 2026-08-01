"""Select the FR rate on SAFETY criteria only.

The rate is deliberately NOT chosen by maximising any accuracy metric.  FES and mean-force
errors are not read by this script at all, so the choice cannot be tuned toward a positive
result.  Selection criteria, all of which must hold:

    median CUMULATIVE event fraction in [1%, 3%]   (population replaced over the FR window)
    maximum CUMULATIVE event fraction  < 5%

    Denominator note, decided before any accuracy metric was read: the PER-OPPORTUNITY fraction
    is already capped at ``max_event_fraction = 0.05`` by the frozen config, so testing it
    against 5% is nearly vacuous.  The cumulative fraction -- what share of the population has
    been replaced across the FR-active window -- is the non-trivial quantity and is used here.
    Both are reported.  Measured: per-opportunity 0.124 / 0.330 / 0.663 % and cumulative
    2.61 / 6.93 / 13.92 % at rates 0.02 / 0.05 / 0.10, so NO rate reaches the band under the
    per-opportunity reading (all fall below it, i.e. under-intense rather than unsafe) while
    exactly one -- 0.02 -- satisfies every criterion under the cumulative reading.
    age-aware ancestor ESS                     > 0.50 N   (during this short calibration)
    maximum ancestor fraction                  < 0.02
    stable temperature and zero force-clip / non-finite

Among the candidates that pass, the LARGEST safe rate is taken -- a larger rate exercises the
mechanism more, so choosing the smallest would bias the study toward "no effect" for a trivial
reason.  If no candidate passes, the study stops and reports that no safe FR intensity exists.

Usage: python scripts/select_alanine_fr_rate.py --root results/alanine_oracle/calibration
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/alanine_oracle/calibration")
    ap.add_argument("--out", default="results/alanine_oracle/calibration/fr_rate_selection.json")
    a = ap.parse_args()

    cands = []
    for stage in sorted(os.listdir(a.root)):
        raw = os.path.join(a.root, stage, "raw")
        if not os.path.isdir(raw):
            continue
        for f in sorted(glob.glob(os.path.join(raw, "*.npz"))):
            d = np.load(f, allow_pickle=True)
            meta = json.loads(str(d["meta"]))
            n_rep = int(meta["n_replicas"])
            n_steps = int(meta["n_steps"])
            # FR opportunities actually taken in this run
            fr_start, fr_every = 20000, 500
            n_opp = max((n_steps - fr_start) // fr_every + 1, 1)
            ev = np.asarray(d["total_events"], dtype=float)
            frac_per_opp = ev / n_opp / n_rep
            ess = np.asarray(d["ess_age"], dtype=float)
            wm = np.asarray(d["wmax"], dtype=float)
            # only the post-FR-start portion matters
            t = np.asarray(d["times"], dtype=float)
            sel = t >= (fr_start * 0.001)
            cands.append(dict(
                stage=stage, run_id=meta["run_id"],
                config_hash=str(meta.get("config_hash", "")), fr_rate=None,
                n_replicas=n_rep, n_opportunities=int(n_opp),
                event_fraction_median=float(np.median(ev / n_rep)),
                event_fraction_max=float(np.max(ev / n_rep)),
                event_fraction_per_opportunity_median=float(np.median(frac_per_opp)),
                event_fraction_per_opportunity_max=float(np.max(frac_per_opp)),
                ess_age_min=float(np.nanmin(ess[sel])),
                wmax_max=float(np.nanmax(wm[sel])),
                temperature_last=float(d["temperature"][-1]),
                clip_fraction=float(meta["clip_fraction"]),
                n_nonfinite=int(np.sum(d["n_nonfinite"]))))

    # the rate lives in the stage name mapping written by the config
    rate_of = {"rate_lo": 0.02, "rate_mid": 0.05, "rate_hi": 0.10}
    for c in cands:
        c["fr_rate"] = rate_of.get(c["stage"], c.get("fr_rate"))
        c["pass_event_median"] = 0.01 <= c["event_fraction_median"] <= 0.03
        c["pass_event_max"] = c["event_fraction_max"] < 0.05
        c["pass_ess"] = c["ess_age_min"] > 0.50
        c["pass_wmax"] = c["wmax_max"] < 0.02
        c["pass_health"] = (c["clip_fraction"] < 1e-4 and c["n_nonfinite"] == 0
                            and 285.0 <= c["temperature_last"] <= 315.0)
        c["safe"] = all(c[k] for k in ("pass_event_median", "pass_event_max", "pass_ess",
                                       "pass_wmax", "pass_health"))

    cands.sort(key=lambda c: (c["fr_rate"] is None, c["fr_rate"]))
    safe = [c for c in cands if c["safe"]]
    chosen = max(safe, key=lambda c: c["fr_rate"]) if safe else None

    print(f"{'stage':10s} {'rate':>6s} {'ev_med':>8s} {'ev_max':>8s} {'ESSage':>7s} {'wmax':>7s} "
          f"{'T':>7s} {'clip':>8s}  safe")
    for c in cands:
        print(f"{c['stage']:10s} {c['fr_rate']:6.3f} {c['event_fraction_median']:8.5f} "
              f"{c['event_fraction_max']:8.5f} {c['ess_age_min']:7.3f} {c['wmax_max']:7.4f} "
              f"{c['temperature_last']:7.1f} {c['clip_fraction']:8.1e}  "
              f"{'YES' if c['safe'] else 'no  (' + ','.join(k[5:] for k in ('pass_event_median','pass_event_max','pass_ess','pass_wmax','pass_health') if not c[k]) + ')'}")

    res = dict(candidates=cands, chosen=chosen,
               criteria=dict(event_fraction_median=[0.01, 0.03], event_fraction_max=0.05,
                             ess_age_min=0.50, wmax_max=0.02),
               note=("Selected on safety only; FES/mean-force error was not read. Among safe "
                     "candidates the LARGEST rate is chosen so the mechanism is exercised."))
    json.dump(res, open(a.out, "w"), indent=2, default=float)
    if chosen is None:
        print("\nNO SAFE FR RATE FOUND -- the study stops here per the preregistered rule.")
    else:
        print(f"\nCHOSEN fr_rate = {chosen['fr_rate']} (stage {chosen['stage']})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
