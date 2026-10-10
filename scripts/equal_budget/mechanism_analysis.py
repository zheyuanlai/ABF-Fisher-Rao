#!/usr/bin/env python
"""Finite-N mechanism tests of the equal-budget ladders (FINAL_RESULTS.md section 8).

    python scripts/equal_budget/mechanism_analysis.py [--systems lta300 lta150 gateway] [--out results/equal_budget_v2/mechanism.json]

Reads ONLY results/equal_budget_v2/<dir>/analysis/summary.json (analyze_ladder.py); no metric is recomputed.
Prints, per system, one row per N with the quantities each candidate mechanism makes a prediction about, and the
seed-level tests below.  Every N-monotone covariate (T_N, KDE noise, deaths per walker, ABF establishment time)
is rank-collinear across the ladder, so a correlation ACROSS N cannot tell the mechanisms apart.  The
discriminating tests are:

  W  within-N, seed level: Spearman(per-seed G(Ibar_F), per-seed ABF tau(TV_half) in u) at each N, combined
     as the mean rho over a group of N with a stratified permutation test (seeds re-paired within each N,
     20000 permutations).  Establishment starvation predicts rho < 0 (the seeds whose ABF arm establishes
     the marginal late gain most from FR); KDE noise, birth-death excess and genealogy collapse act on
     every seed of an N alike and predict rho ~ 0.
  T  across temperature at equal N (LTA only): identical FR machinery, KDE noise and nearly identical
     deaths per walker, different landscape.  Starvation predicts the larger gain at the temperature whose
     ABF arm establishes later.
  M  marginal vs free energy: KDE score noise predicts that FR stops improving TV_half once the score is
     noise (sigma_KDE >~ 1); if TV_half still improves where F does not, the marginal is being repaired but
     ABF does not need the repair (outcome C).

sigma_KDE(N) = sqrt(L / (2 sqrt(pi) N eta)) is the relative standard deviation of a Gaussian KDE (bandwidth
eta) of N walkers under the uniform target on a domain of length L (LTA: L = 2 pi, eta = 0.10 rad;
gateway: KDE grid [-1.8, 1.8], L = 3.6, eta = 0.1).
"""
import argparse
import json
import math
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DIRS = {"lta300": "lta_T300", "lta150": "lta_T150", "gateway": "gateway"}
KDE = {"lta300": (2 * math.pi, 0.10), "lta150": (2 * math.pi, 0.10), "gateway": (3.6, 0.1)}
GROUPS = {"all N >= 2": lambda N, N0: N >= 2, "N >= N0/8": lambda N, N0: N >= N0 // 8,
          "N < N0/8": lambda N, N0: 2 <= N < N0 // 8}


def load(sysk):
    p = os.path.join(ROOT, "results", "equal_budget_v2", DIRS[sysk], "analysis", "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _f(x):
    if x is None or x == "inf":
        return math.inf
    return float(x)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def within_N_test(S, Ns, n_perm=20000, seed=12345):
    """Mean over Ns of Spearman(G Ibar_F, ABF tau_TV_half_u) per seed; stratified permutation p (two-sided)."""
    pairs = []
    for N in Ns:
        g = S["contrasts"][str(N)]["primary"]["Ibar_F"]["per_seed"]
        t = S["per_N"][str(N)]["abf"]["tau_TV_half_u"]["per_seed"]
        ks = [k for k in g if k in t and g[k] is not None and t[k] is not None]
        x = np.array([_f(t[k]) for k in ks]); y = np.array([float(g[k]) for k in ks])
        x = np.where(np.isfinite(x), x, 2.0)  # censored establishment ranks last
        if len(ks) >= 4 and np.ptp(x) > 0:
            pairs.append((x, y))
    if not pairs:
        return None
    obs = float(np.mean([spearman(x, y) for x, y in pairs]))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.mean([spearman(rng.permutation(x), y) for x, y in pairs])
    p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    return dict(mean_rho=obs, p_perm=p, n_N=len(pairs), per_N=[round(spearman(x, y), 3) for x, y in pairs])


def rows(sysk, S):
    L, eta = KDE[sysk]
    Ns = [int(n) for n in S["plan"]["N_ladder"]]
    ibar_abf = {N: S["per_N"][str(N)]["abf"]["Ibar_F"]["median"] for N in Ns if str(N) in S["per_N"]}
    best = min(v for v in ibar_abf.values() if v is not None)
    out = []
    for N in Ns:
        P = S["per_N"].get(str(N))
        if P is None:
            continue
        a, f = P["abf"], P.get("fr")
        c = S["contrasts"].get(str(N))
        r = dict(N=N, T=P["T"], sigma_kde=math.sqrt(L / (2 * math.sqrt(math.pi) * N * eta)),
                 abf_tau_tv_u=a["tau_TV_half_u"]["median"], abf_est_cum_u=a["est_cum_u"]["median"],
                 abf_ibar_over_best=ibar_abf[N] / best)
        if f is not None and c is not None:
            r.update(fr_deaths_per_walker=f["fr_deaths_per_walker"]["median"],
                     fr_final_unique=f["final_gen_nuniq_run"]["median"],
                     fr_min_ess_win=f["min_gen_ess_win"]["median"],
                     G_tv=(c["marginal"]["Ibar_TV_half"]["G_median"], *c["marginal"]["Ibar_TV_half"]["G_ci95"]),
                     G_F=(c["primary"]["Ibar_F"]["G_median"], *c["primary"]["Ibar_F"]["G_ci95"]),
                     G_Fp=(c["primary"]["Ibar_Fp"]["G_median"], *c["primary"]["Ibar_Fp"]["G_ci95"]))
        out.append(r)
    return out


def fmt_g(t):
    return f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=["lta300", "lta150", "gateway"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "equal_budget_v2", "mechanism.json"))
    a = ap.parse_args()
    res = {}
    for sysk in a.systems:
        S = load(sysk)
        if S is None:
            print(f"## {sysk}: no summary.json\n"); continue
        R = rows(sysk, S)
        N0 = max(r["N"] for r in R)
        print(f"## {sysk}\n")
        print("| N | T_N | sigma_KDE | ABF tau(TV_half) u | ABF cum. est. u | ABF Ibar_F / best ABF | FR deaths/walker | "
              "FR final unique anc. | FR min ESS win | G Ibar_TV_half % | G Ibar_F % | G Ibar_F' % |")
        print("|" + "---|" * 12)
        for r in R:
            if "G_F" in r:
                print(f"| {r['N']} | {r['T']:g} | {r['sigma_kde']:.2f} | {r['abf_tau_tv_u']:.3g} | {r['abf_est_cum_u']:.3g} | "
                      f"{r['abf_ibar_over_best']:.2f} | {r['fr_deaths_per_walker']:.3g} | {r['fr_final_unique']:.0f} | "
                      f"{(r['fr_min_ess_win'] if r['fr_min_ess_win'] is not None else float('nan')):.2f} | {fmt_g(r['G_tv'])} | "
                      f"{fmt_g(r['G_F'])} | {fmt_g(r['G_Fp'])} |")
            else:
                print(f"| {r['N']} | {r['T']:g} | {r['sigma_kde']:.2f} | {r['abf_tau_tv_u']:.3g} | {r['abf_est_cum_u']:.3g} | "
                      f"{r['abf_ibar_over_best']:.2f} | -- | -- | -- | -- | -- | -- |")
        tests = {}
        for name, sel in GROUPS.items():
            Ns = [r["N"] for r in R if "G_F" in r and sel(r["N"], N0)]
            tests[name] = within_N_test(S, Ns)
            t = tests[name]
            if t:
                print(f"\nW [{name}, N = {Ns}]: mean within-N Spearman(G Ibar_F, ABF tau TV_half) = {t['mean_rho']:+.3f}, "
                      f"stratified permutation p = {t['p_perm']:.4f}; per N {t['per_N']}")
        print()
        res[sysk] = dict(rows=R, within_N=tests)
    if "lta300" in res and "lta150" in res:
        print("## T: LTA 150 K vs 300 K at equal N\n")
        print("| N | ABF tau(TV_half) u 300 / 150 | FR deaths/walker 300 / 150 | G Ibar_F 300 K | G Ibar_F 150 K | larger gain at later-establishing T? |")
        print("|---|---|---|---|---|---|")
        r3 = {r["N"]: r for r in res["lta300"]["rows"] if "G_F" in r}
        r1 = {r["N"]: r for r in res["lta150"]["rows"] if "G_F" in r}
        agree = 0; n = 0; decided = 0
        for N in sorted(set(r3) & set(r1), reverse=True):
            x, y = r3[N], r1[N]
            later = "150" if y["abf_tau_tv_u"] > x["abf_tau_tv_u"] else "300" if x["abf_tau_tv_u"] > y["abf_tau_tv_u"] else "tie"
            bigger = "150" if y["G_F"][0] < x["G_F"][0] else "300"
            ok = (later == bigger) if later != "tie" else None
            n += 1; decided += ok is not None; agree += bool(ok)
            print(f"| {N} | {x['abf_tau_tv_u']:.3g} / {y['abf_tau_tv_u']:.3g} | {x['fr_deaths_per_walker']:.3g} / {y['fr_deaths_per_walker']:.3g} | "
                  f"{fmt_g(x['G_F'])} | {fmt_g(y['G_F'])} | {'--' if ok is None else ('yes' if ok else 'no')} |")
        print(f"\nT: agreement {agree}/{decided} decided N (of {n})")
        res["T_test"] = dict(agree=agree, decided=decided, n=n)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
