#!/usr/bin/env python
"""Analysis of configs/numerical_validation/lta_validation_prereg.json.

    python scripts/numerical_validation/analyze_lta_validation.py --T 300

Profiles on the 180 production bins (kJ/mol, centred on the circle):
  F_density  fine-bin (1800) binned WHAM over the 40 umbrella windows, summed to 180 bins
  F_mid180   180-bin WHAM with the bias at bin midpoints (the existing reference's convention)
  F_MF       pooled conditional mean of the local mean force, histogram_pmf convention
Noise: leave-one-group-out jackknife (16 groups); D = debiased RMS; 95 % upper bound from a group
bootstrap debiased by 2x the jackknife noise.  Then the published 300 K production runs are rescored
against the exact (MC) reference with the published arithmetic (scripts/analyze_lta_histogram.py).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
KB = 0.008314462618
NB, NF = 180, 1800
DPHI = 2 * np.pi / NB
CEN = -np.pi + (np.arange(NB) + 0.5) * DPHI
CENTRES = np.linspace(-np.pi, np.pi, 40, endpoint=False)
KAPPA = {300.0: 300.0, 150.0: 254.0}
RNG = np.random.default_rng(20261009)


def load(T, tag):
    files = sorted(glob.glob(os.path.join(ROOT, "results", "numerical_validation", "lta", f"T{int(T)}", tag, "g*_w*.npz")))
    if not files:
        return None
    G = 16
    Cf = np.zeros((G, 40, NF)); C = np.zeros((G, 40, NB)); Mf = np.zeros((G, 40, NB))
    sb = np.zeros(G); sb2 = np.zeros(G); nd = np.zeros(G); sU = np.zeros(G)
    acc = []
    present = np.zeros((G, 40), bool)
    for f in files:
        if ".tmp." in f:
            continue
        d = np.load(f)
        m = json.loads(str(d["meta_json"]))
        g, w = m["group"], m["window"]
        Cf[g, w] = d["C_fine"]; C[g, w] = d["C"]; Mf[g, w] = d["Mf"]
        sb[g] += float(d["sum_bond"]); sb2[g] += float(d["sum_bond2"]); nd[g] += float(d["n_dep"]); sU[g] += float(d["sum_U"])
        acc.append((float(d["n_acc"]) / max(1.0, float(d["n_prop"])), float(d["n_tacc"]) / max(1.0, float(d["n_tprop"]))))
        present[g, w] = True
    groups = np.where(present.all(1))[0]
    return dict(Cf=Cf[groups], C=C[groups], Mf=Mf[groups], sb=sb[groups], sb2=sb2[groups], nd=nd[groups], sU=sU[groups],
                n_groups=len(groups), n_files=int(present.sum()), acc=np.mean(acc, 0).tolist() if acc else None)


class Wham:
    def __init__(self, nb, kappa, kT):
        e = np.linspace(-np.pi, np.pi, nb + 1)
        mid = 0.5 * (e[1:] + e[:-1])
        d = mid[None, :] - CENTRES[:, None]
        d = d - 2 * np.pi * np.floor((d + np.pi) / (2 * np.pi))
        self.E = np.exp(-0.5 * kappa * d ** 2 / kT)
        self.nb = nb

    def solve(self, H, f0=None, tol=1e-10, max_iter=100000):
        N = H.sum(1)
        num = H.sum(0)
        f = np.zeros(len(N)) if f0 is None else f0.copy()
        for it in range(max_iter):
            den = (N * np.exp(f)) @ self.E
            p = np.where(num > 0, num / np.maximum(den, 1e-300), 0.0)
            p /= p.sum()
            fn = -np.log(self.E @ p)
            fn -= fn[0]
            if np.max(np.abs(fn - f)) < tol:
                return p, fn
            f = fn
        return p, f


def centre(F):
    return F - F.mean()


def F_from_p180(p, kT):
    return centre(-kT * np.log(np.maximum(p, 1e-300)))


def hist_pmf(gamma):
    g0 = gamma - gamma.mean()
    Fr = np.cumsum(g0 * DPHI)
    return centre(Fr - 0.5 * g0 * DPHI)


class Prof:
    """One profile kind for one tag, with group jackknife and bootstrap resampling."""

    def __init__(self, label, D, kind, kT, kappa):
        self.label, self.D, self.kind, self.kT = label, D, kind, kT
        self.G = D["n_groups"]
        if kind == "density":
            self.w = Wham(NF, kappa, kT)
        elif kind == "mid180":
            self.w = Wham(NB, kappa, kT)
        self.f0 = None
        self.F = self.of(np.arange(self.G))
        loo = np.array([self.of(np.delete(np.arange(self.G), i)) for i in range(self.G)])
        self.se = np.sqrt((self.G - 1) / self.G * ((loo - loo.mean(0)) ** 2).sum(0))

    def of(self, idx):
        if self.kind == "mf":
            M = self.D["Mf"][idx].sum((0, 1)); C = self.D["C"][idx].sum((0, 1))
            return hist_pmf(np.where(C > 0, M / np.maximum(C, 1), 0.0))
        H = (self.D["Cf"] if self.kind == "density" else self.D["C"])[idx].sum(0)
        p, f = self.w.solve(H, self.f0)
        if self.f0 is None:
            self.f0 = f
        if self.kind == "density":
            p = p.reshape(NB, NF // NB).sum(1)
        return F_from_p180(p, self.kT)


class Fixed:
    def __init__(self, label, F):
        self.label, self.F, self.se, self.G = label, centre(F), np.zeros(NB), 1

    def of(self, idx):
        return self.F


def compare(A, B, nboot=300):
    d = centre(A.F - B.F)
    noise2 = float(np.mean(A.se ** 2 + B.se ** 2))
    rms = float(np.sqrt(np.mean(d ** 2)))
    D = float(np.sqrt(max(0.0, rms ** 2 - noise2)))
    Db = []
    for _ in range(nboot):
        fa = A.of(RNG.integers(0, A.G, A.G)) if A.G > 1 else A.F
        fb = B.of(RNG.integers(0, B.G, B.G)) if B.G > 1 else B.F
        db = centre(fa - fb)
        Db.append(np.sqrt(max(0.0, float(np.mean(db ** 2)) - 2 * noise2)))
    return dict(A=A.label, B=B.label, D=D, D_upper95=float(np.quantile(Db, 0.95)), rms_obs=rms, noise=float(np.sqrt(noise2)),
                maxabs=float(np.max(np.abs(d))), n_boot=nboot)


def barrier(F, kT, a=11.919):
    """run_lta_reference.barrier_stats: mean F over |z| < 0.4 A minus mean F over the cage bins |z| > 4 A,
    z = phi a / 2 pi the distance from the WINDOW plane (phi = 0 is the window, phi = +-pi the cage centre).
    (A first version measured z from the cage centre, inverting the sign; fixed 2026-10-09 before any write-up.)"""
    z = np.abs(CEN) * a / (2 * np.pi)
    return float(F[z < 0.4].mean() - F[z > 4.0].mean())


def level(D, up, tol=0.03, tol_up=0.045, marg=0.06):
    if D <= tol and up <= tol_up:
        return "ADMISSIBLE"
    return "MARGINAL" if D <= marg else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--nboot", type=int, default=300)
    a = ap.parse_args()
    T = a.T
    kT = KB * T
    kappa = KAPPA[T]
    out_dir = os.path.join(ROOT, "results", "numerical_validation", "lta", f"T{int(T)}")
    S = dict(T=T, kT=kT, kappa=kappa, units="kJ/mol", prereg="configs/numerical_validation/lta_validation_prereg.json")
    data = {t: load(T, t) for t in ("MC", "EM2e-4", "EM1e-4", "EM5e-5", "LM2e-4")}
    data = {k: v for k, v in data.items() if v is not None and v["n_groups"] >= 4}
    S["groups"] = {k: v["n_groups"] for k, v in data.items()}
    print("groups:", S["groups"], flush=True)
    assert "MC" in data
    P = {}
    for t, D in data.items():
        for kind in ("density", "mf", "mid180"):
            if kind == "mid180" and t not in ("MC", "EM2e-4"):
                continue
            P[(t, kind)] = Prof(f"{t} {kind}", D, kind, kT, kappa)
            print("profile", t, kind, "noise", float(np.sqrt(np.mean(P[(t, kind)].se ** 2))), flush=True)
    ref = P[("MC", "density")]
    rows = []

    def add(A_, tag):
        c = compare(A_, ref, a.nboot)
        c["tag"] = tag
        c["barrier"] = barrier(A_.F, kT)
        c["barrier_minus_MC"] = c["barrier"] - barrier(ref.F, kT)
        rows.append(c)
        print(f"{A_.label:40s} D {c['D']:.4f} [up {c['D_upper95']:.4f}] rms {c['rms_obs']:.4f} noise {c['noise']:.4f} "
              f"max|d| {c['maxabs']:.3f}  barrier-MC {c['barrier_minus_MC']:+.4f}", flush=True)
        return c

    S["MC_self"] = add(P[("MC", "mf")], "G_LTA_MC_self")
    for (t, kind), pr in P.items():
        if (t, kind) in (("MC", "density"), ("MC", "mf")):
            continue
        add(pr, f"{t}/{kind}")
    refp = np.load(os.path.join(ROOT, "results", "uniform_campaign", "lta", "reference", f"reference_T{int(T)}.npz"), allow_pickle=True)
    from analyze_uniform_lta import circular_interp_ref
    F_pub = circular_interp_ref(np.asarray(refp["F"], float), np.asarray(refp["grid_phi"], float), CEN)
    S["published_reference"] = add(Fixed("published umbrella/WHAM reference (EM dt 2e-4)", F_pub), "G_LTA_reference")
    S["rows"] = rows
    S["barrier_MC"] = barrier(ref.F, kT)
    S["barrier_published"] = barrier(centre(F_pub), kT)
    em = [r for r in rows if r["tag"] == "EM2e-4/density"]
    emm = [r for r in rows if r["tag"] == "EM2e-4/mf"]
    if em and emm:
        l1 = level(em[0]["D"], em[0]["D_upper95"])
        l2 = level(emm[0]["D"], emm[0]["D_upper95"])
        order = ["ADMISSIBLE", "MARGINAL", "FAIL"]
        S["G_LTA_dyn"] = order[max(order.index(l1), order.index(l2))]
    S["G_LTA_reference_verdict"] = level(S["published_reference"]["D"], S["published_reference"]["D_upper95"])
    S["G_LTA_MC_self_verdict"] = "PASS" if S["MC_self"]["D"] <= 0.03 else "FAIL"
    S["bond_variance"] = {t: float((D["sb2"].sum() / D["nd"].sum()) - (D["sb"].sum() / D["nd"].sum()) ** 2) for t, D in data.items()}
    S["bond_variance_exact"] = kT / 400.0
    S["mean_U"] = {t: float(D["sU"].sum() / D["nd"].sum()) for t, D in data.items()}
    S["mc_acceptance"] = data["MC"]["acc"]
    print("verdicts:", S.get("G_LTA_dyn"), S["G_LTA_reference_verdict"], S["G_LTA_MC_self_verdict"], flush=True)
    print("bond variance:", S["bond_variance"], "exact", S["bond_variance_exact"], flush=True)

    # ---- rescoring of the published production runs
    S["rescoring"] = rescore(T, ref.F, F_pub)
    # ---- pooled long-run limits of each published arm vs MC
    S["pooled_limits"] = pooled_limits(T, ref, F_pub)
    S["profiles"] = {f"{t} {k}": v.F.tolist() for (t, k), v in P.items()}
    S["profiles"]["published reference"] = centre(F_pub).tolist()
    S["mc_se"] = ref.se.tolist()
    S["centres"] = CEN.tolist()
    json.dump(S, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=float)
    figure(T, P, ref, F_pub, S, out_dir)


def rescore(T, F_mc, F_pub):
    import analyze_lta_histogram as H
    prod = os.path.join(ROOT, "results", "lta_histogram", f"production_T{int(T)}")
    arms = {a: H.load_npz(os.path.join(prod, f"{a}.npz")) for a in ("abf", "fr_uniform", "fr_sham")
            if os.path.exists(os.path.join(prod, f"{a}.npz"))}
    if len(arms) < 3:
        return dict(missing=True)
    out = {}
    for name, F in (("published reference", F_pub), ("MC exact reference", F_mc)):
        notes = []
        g = H.score_group(arms, dict(F=F, grid_phi=CEN), 20000, 1024, notes, f"T{T}")
        res = dict(median_I_F={a: float(np.median(g[a]["I"])) for a in g}, median_eF_T={a: float(np.median(g[a]["fin"])) for a in g})
        for arm, base in (("fr_uniform", "abf"), ("fr_sham", "abf"), ("fr_uniform", "fr_sham")):
            c, _ = H.contrast(g, arm, base)
            res[f"{arm}_vs_{base}"] = c
        out[name] = res
        print(f"rescore vs {name}: " + "; ".join(f"{k} dI_F {v['d_int']['median']:+.2f}% [{v['d_int']['ci95'][0]:+.2f},{v['d_int']['ci95'][1]:+.2f}] "
                                                  f"de_F(T) {v['d_fin']['median']:+.2f}% [{v['d_fin']['ci95'][0]:+.2f},{v['d_fin']['ci95'][1]:+.2f}]"
                                                  for k, v in res.items() if k.endswith(("abf", "sham"))), flush=True)
    mc = out["MC exact reference"]
    out["LTA_FR_gain_survives"] = bool(mc["fr_uniform_vs_abf"]["d_int"]["ci95"][1] < 0 and mc["fr_uniform_vs_fr_sham"]["d_int"]["ci95"][1] < 0)
    return out


def pooled_limits(T, ref, F_pub):
    prod = os.path.join(ROOT, "results", "lta_histogram", f"production_T{int(T)}")
    out = {}
    lim = {}
    for arm in ("abf", "fr_uniform", "fr_sham"):
        p = os.path.join(prod, f"{arm}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        fs, uc = np.asarray(z["fsum_prod"], float), np.asarray(z["u_counts"], float)
        n = fs.shape[0]

        class L:
            pass
        E = L()
        E.label = f"pooled limit {arm}"
        E.G = n
        E.of = lambda idx, fs=fs, uc=uc: hist_pmf(np.where(uc[idx].sum(0) > 0, fs[idx].sum(0) / np.maximum(uc[idx].sum(0), 1), 0.0))
        E.F = E.of(np.arange(n))
        loo = np.array([E.of(np.delete(np.arange(n), i)) for i in range(n)])
        E.se = np.sqrt((n - 1) / n * ((loo - loo.mean(0)) ** 2).sum(0))
        lim[arm] = E
        c = compare(E, ref, 300)
        cp = compare(E, Fixed("pub", F_pub), 300)
        out[arm] = dict(D_vs_MC=c["D"], up_vs_MC=c["D_upper95"], D_vs_published=cp["D"], noise=c["noise"])
        print(f"pooled limit {arm:10s}: D vs MC {c['D']:.4f} [up {c['D_upper95']:.4f}], vs published ref {cp['D']:.4f}, noise {c['noise']:.4f}", flush=True)
    if "fr_uniform" in lim and "abf" in lim:
        c = compare(lim["fr_uniform"], lim["abf"], 300)
        out["fr_uniform_vs_abf_limit"] = dict(D=c["D"], up=c["D_upper95"], noise=c["noise"])
        print(f"pooled FR limit vs ABF limit: D {c['D']:.4f} [up {c['D_upper95']:.4f}] noise {c['noise']:.4f}", flush=True)
    return out


def figure(T, P, ref, F_pub, S, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    a = 11.919
    zz = CEN * a / (2 * np.pi)
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axs[0]
    ax.plot(zz, ref.F, color="k", lw=2.2, label="MC (exact), fine WHAM")
    ax.plot(zz, centre(F_pub), color="#c0392b", lw=1.2, ls="--", label="published reference (EM dt 2e-4, WHAM)")
    ax.set_xlabel("z = phi a / 2 pi (A)"); ax.set_ylabel("F (kJ/mol)"); ax.set_title(f"(a) ethane/LTA {T:g} K", loc="left")
    ax.legend(fontsize=7)
    ax = axs[1]
    ax.fill_between(zz, -2 * ref.se, 2 * ref.se, color="#dddddd", label="MC +-2 se")
    sty = {("EM2e-4", "density"): ("#c0392b", "-"), ("EM1e-4", "density"): ("#e67e22", "-"), ("EM5e-5", "density"): ("#2a78d6", "-"),
           ("LM2e-4", "density"): ("#1baf7a", "-"), ("EM2e-4", "mf"): ("#c0392b", ":"), ("MC", "mf"): ("k", ":"),
           ("EM2e-4", "mid180"): ("#6c3483", "--"), ("MC", "mid180"): ("#555555", "--")}
    for k, (c, ls) in sty.items():
        if k in P:
            ax.plot(zz, centre(P[k].F - ref.F), color=c, ls=ls, lw=1.2, label=f"{k[0]} {k[1]}")
    ax.plot(zz, centre(centre(F_pub) - ref.F), color="#888888", lw=1.0, label="published reference")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("z (A)"); ax.set_ylabel("F - F_MC (kJ/mol)"); ax.set_title("(b) differences from exact Gibbs", loc="left")
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, "figures", f"fig_lta_validation_T{int(T)}.{ext}"), dpi=160)


if __name__ == "__main__":
    main()
