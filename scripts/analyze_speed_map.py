"""Retrospective SPEED map over every stored campaign run (no new simulations).

I_F answers "how much error, integrated over the budget"; it mixes how fast the error
fell with how low it ended.  The question this script answers is the other one, in the
campaign's own frozen convention (metrics.time_to_accuracy, 0.2 T persistence,
right-censored, never dropped):

    tau_eps = first t whose trailing window stays at or below eps
    S_eps   = tau_eps^baseline / tau_eps^arm         (> 1 means the arm got there first)

on a ladder of thresholds expressed in units of each cell's own analytic mollifier
floor e*, so the rungs mean the same thing across systems.  Censoring is reported, and
a speedup is computed only over seeds where BOTH arms attained the rung -- with the
count of such seeds printed, because a ratio over three seeds is not a measurement.

Groups: gateway (untuned and tuned baselines), 2D torus on a tuned baseline, Phase F
(equal-weight conditional reallocation + the augmented-CV arm it was measured against),
Phase I (weighted vs equal weight, incl. the oracle target) and Phase J (the
variance-limited regime).
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abpfr.metrics import paired_bootstrap_ci, time_to_accuracy

LADDER = (64.0, 32.0, 16.0, 8.0, 4.0, 2.0)      # thresholds in units of e*
PERSIST = 0.2                                    # frozen convention


# -----------------------------------------------------------------------------
# per-system analytic floor and cell key
# -----------------------------------------------------------------------------
def floor_gateway(cfg):
    from abpfr.systems.gateway import GatewayConfig, mollified_fixed_point
    keep = {k: v for k, v in cfg.items()
            if k in GatewayConfig.__dataclass_fields__}
    return mollified_fixed_point(GatewayConfig(**keep))["e_star"]


def floor_torus(cfg):
    from abpfr.systems.torus2d import Torus2DConfig, analytic_floors
    keep = {k: v for k, v in cfg.items()
            if k in Torus2DConfig.__dataclass_fields__}
    return analytic_floors(Torus2DConfig(**keep))["e_star"]


def floor_bichannel(cfg):
    from abpfr.systems.bichannel import BiChannelConfig, analytic_floors
    keep = {k: v for k, v in cfg.items()
            if k in BiChannelConfig.__dataclass_fields__}
    return analytic_floors(BiChannelConfig(**keep))["e_star"]


GROUPS = [
    dict(name="gateway anchor_D, UNTUNED baseline (stage 3, seeds 100-131)",
         dirs=["stage3_confirmatory"], base="shus", floor=floor_gateway,
         arms=["fr_temp", "count", "fr_persistent", "sham"],
         key=lambda c: "anchor_D"),
    dict(name="gateway anchor_D, TUNED baseline (A2, seeds 300-315)",
         dirs=["appmap_phaseA2_compare"], base="shus_gbest", floor=floor_gateway,
         arms=["fr_temp", "count", "shus", "sham"], key=lambda c: "anchor_D"),
    dict(name="2D torus t_mid, TUNED baseline g* = 8 (D3b, seeds 500-515)",
         dirs=["appmap_phaseD3b_fr_on_tuned"], base="shus_gstar", floor=floor_torus,
         arms=["gstar_fr", "gstar_count9", "gstar_sham", "shus_g1"],
         key=lambda c: "t_mid"),
    dict(name="Phase F Type-C: equal-weight conditional FR vs the augmented CV "
              "(F2 + F3a, seeds 600-615)",
         dirs=["appmap_phaseF2_realloc", "appmap_phaseF3a_augcv"], base="shus",
         floor=floor_bichannel,
         arms=["fr_cond", "cnt_cond", "fr_marg", "sham_cond", "aug_g1"],
         key=lambda c: f"hp{c['Hperp']:g}_d{c['Delta']:g}"),
    dict(name="Phase I: weighted vs equal weight, uniform and ORACLE targets "
              "(seeds 800-815)",
         dirs=["appmap_phaseI_weighted"], base="shus", floor=floor_bichannel,
         arms=["fr_cond", "wfr_cond", "fr_cond_oracle", "wfr_cond_oracle",
               "wfr_cond_hot", "sham_cond", "wsham_cond"],
         key=lambda c: f"hp{c['Hperp']:g}_d{c['Delta']:g}"),
    dict(name="Phase J: the VARIANCE-limited regime (stationary start, seeds 920-935)",
         dirs=["appmap_phaseJ_variance"], base="shus", floor=floor_bichannel,
         arms=["fr_cond", "wfr_cond", "wfr_cond_hot", "wstate_hot", "sham_cond",
               "wsham_cond"],
         key=lambda c: f"hp{c['Hperp']:g}_d{c['Delta']:g}"),
]


def load_group(g):
    """-> {cell: {arm: {seed: (times, e_F)}}}, {cell: config}"""
    out, cfgs = {}, {}
    for d in g["dirs"]:
        for p in sorted(glob.glob(f"results/{d}/*.json")):
            j = json.load(open(p))
            if "method" not in j or "config" not in j:
                continue
            arm = j["method"]["name"]
            if arm != g["base"] and arm not in g["arms"]:
                continue
            cell = g["key"](j["config"])
            with np.load(p.replace(".json", ".npz")) as z:
                if "l2_f_t" not in z.files:
                    continue
                series = (np.asarray(z["time"]), np.asarray(z["l2_f_t"]))
            out.setdefault(cell, {}).setdefault(arm, {})[int(j["seed"])] = series
            cfgs.setdefault(cell, j["config"])
    return out, cfgs


def report(g):
    data, cfgs = load_group(g)
    if not data:
        print(f"\n### {g['name']}\n    (no stored records found)")
        return []
    rows = []
    print(f"\n{'=' * 100}\n### {g['name']}\n{'=' * 100}")
    for cell in sorted(data):
        arms = data[cell]
        if g["base"] not in arms:
            continue
        e_star = g["floor"](cfgs[cell])
        seeds = sorted(arms[g["base"]])
        if cfgs[cell].get("warm_start"):
            # a warm-started run has NO convergence phase: it starts at the fixed
            # point, is driven up to its own sampling-noise level and relaxes back.
            # tau_eps is undefined there, so the comparable quantity is the level of
            # that noise -- the mean e_F over the last quarter of the run.
            print(f"\n-- cell {cell} | e* = {e_star:.5f} | WARM START: no "
                  f"convergence phase, so tau_eps is not defined.  Late-run noise "
                  f"level instead (mean e_F over the last 25% of T):")
            def late(a):
                out = []
                for sd in seeds:
                    if sd not in arms[a]:
                        continue
                    t, e = arms[a][sd]
                    out.append(e[t >= 0.75 * t[-1]].mean())
                return np.array(out)
            lb = late(g["base"])
            print(f"{'arm':>18} {'level / e*':>11} {'ratio vs base [95% CI]':>30}")
            print(f"{g['base']:>18} {np.median(lb) / e_star:11.2f} {'--':>30}")
            for arm in g["arms"]:
                if arm not in arms:
                    continue
                la = late(arm)
                m, lo, hi = paired_bootstrap_ci(la / lb)
                print(f"{arm:>18} {np.median(la) / e_star:11.2f} "
                      f"{m:9.3f} [{lo:6.3f},{hi:6.3f}]{'':>4}")
            continue
        print(f"\n-- cell {cell} | e* = {e_star:.5f} | "
              f"T = {arms[g['base']][seeds[0]][0][-1]:.0f} | {len(seeds)} seeds")
        print(f"{'eps/e*':>7} {'arm':>18} {'tau (median)':>13} {'censored':>9} "
              f"{'S_eps [95% CI]':>26} {'n paired':>9}")
        for mult in LADDER:
            eps = mult * e_star
            tb = {s: time_to_accuracy(*arms[g["base"]][s], eps, PERSIST)
                  for s in seeds}
            nb = sum(np.isnan(v) for v in tb.values())
            if nb == len(seeds):
                continue                       # the baseline never gets here: no rung
            if np.nanmedian(list(tb.values())) <= 0.0:
                continue                       # rung sits above the initial error
            print(f"{mult:7.0f} {g['base']:>18} "
                  f"{np.nanmedian(list(tb.values())):13.1f} {nb:9d} "
                  f"{'--':>26} {'':>9}")
            for arm in g["arms"]:
                if arm not in arms:
                    continue
                ta = {s: time_to_accuracy(*arms[arm][s], eps, PERSIST)
                      for s in seeds if s in arms[arm]}
                na = sum(np.isnan(v) for v in ta.values())
                both = [s for s in ta if ta[s] > 0 and tb[s] > 0]
                if len(both) >= 3:
                    S = np.array([tb[s] / ta[s] for s in both])
                    m, lo, hi = paired_bootstrap_ci(S)
                    txt = f"{m:8.2f} [{lo:5.2f},{hi:5.2f}]"
                else:
                    m, txt = float("nan"), " " * 8 + "(too few paired)"
                med = (np.nanmedian(list(ta.values())) if na < len(ta)
                       else float("nan"))
                print(f"{'':7} {arm:>18} {med:13.1f} {na:9d} {txt:>26} "
                      f"{len(both):9d}")
                rows.append((g["name"], cell, mult, arm, m, len(both), na))
    return rows


def main():
    summary = []
    for g in GROUPS:
        summary += report(g)
    print(f"\n{'=' * 100}\n### SPEED MAP — S_eps at the TIGHTEST rung each pair "
          f"resolves (>= 8 paired seeds)\n{'=' * 100}")
    print(f"{'experiment':>62} {'cell':>12} {'arm':>18} {'eps/e*':>7} {'S_eps':>7}")
    seen = {}
    for name, cell, mult, arm, S, n, _ in summary:
        if n >= 8 and not math.isnan(S):
            seen[(name, cell, arm)] = (mult, S, n)      # ladder descends: last wins
    for (name, cell, arm), (mult, S, n) in seen.items():
        print(f"{name[:62]:>62} {cell:>12} {arm:>18} {mult:7.0f} {S:7.2f}")


if __name__ == "__main__":
    main()
