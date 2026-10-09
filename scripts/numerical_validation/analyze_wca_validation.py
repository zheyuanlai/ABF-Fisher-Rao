#!/usr/bin/env python
"""Analysis of configs/numerical_validation/wca_validation_prereg.json.

    python scripts/numerical_validation/analyze_wca_validation.py

Writes results/numerical_validation/wca/summary.json, tables (markdown) and figures.  Statistic
exactly as frozen: profiles on the 160 bin centres inside the eval window [-0.1, 1.1], each centred
by its window mean; per-bin noise by leave-one-seed-out jackknife; D = debiased RMS; 95 % upper bound
from a seed bootstrap (2000 resamples; in the bootstrap world a pooled profile carries the original
noise plus the resampling noise, so the bootstrap replicate is debiased by 2x the jackknife noise).
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "numerical_validation", "wca")
OLD = os.path.join(ROOT, "results", "wca_replica_ladder", "consistency", "raw")
NB, ZMIN, ZMAX = 160, -0.2, 1.2
DZ = (ZMAX - ZMIN) / NB
CEN = ZMIN + (np.arange(NB) + 0.5) * DZ
WIN = (CEN >= -0.1) & (CEN <= 1.1)
RNG = np.random.default_rng(20261009)


# ------------------------------------------------------------------------------------------ loading
def load_group(group, name):
    """List of per-seed dicts {C, M, (M_impl), meta} with C, M of length 160 (out-of-window samples dropped)."""
    out = []
    for f in sorted(glob.glob(os.path.join(OUT, group, f"{name}_s*.npz"))):
        if ".tmp." in f:
            continue
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta_json"]))
        if group == "em_impl":
            rec = dict(C=d["C"][-1, :NB].astype(float), M=d["M"][-1, :NB].astype(float))
        else:
            rec = dict(C=d["C"][1:-1].astype(float), M=d["M_true"][1:-1].astype(float),
                       M_impl=d["M_impl"][1:-1].astype(float))
            for k in ("sum_U", "n_U", "n_clip_particle_steps", "n_clip_steps", "n_minr_diag", "n_mf_clip",
                      "max_step_displacement", "n_acc", "n_prop", "n_dimer_acc", "n_dimer_prop", "sum_K", "nonfinite"):
                rec[k] = float(d[k])
            rec["sum_p_moments"] = d["sum_p_moments"].astype(float)
            rec["C_all"] = d["C"].astype(float)
            rec["z_series"] = d["z_series"].astype(float)
        rec["meta"] = meta
        out.append(rec)
    return out


def load_old(tag):
    """The 2026-10-04 consistency runs (wca_numba results, production estimator)."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import wca_numba as wn
    out = []
    for f in sorted(glob.glob(os.path.join(OLD, f"{tag}_s*.npz"))):
        r = wn.load_result(f)
        out.append(dict(C=np.asarray(r["C_prod"])[0, -1, :NB].astype(float), M=np.asarray(r["M_prod"])[0, -1, :NB].astype(float),
                        meta=dict(seed=int(r["seed"]), source="2026-10-04 consistency")))
    return out


# ------------------------------------------------------------------------------------------ profiles
def centre(F):
    return F - F[WIN].mean()


def F_density(C):
    return centre(-np.log(np.maximum(C, 1e-300)))


def F_mf(M, C):
    m = M / np.maximum(C, 1e-300)
    F = np.concatenate([[0.0], np.cumsum(0.5 * (m[1:] + m[:-1]) * DZ)])
    return centre(F)


def pooled(recs, kind, idx=None, key="M"):
    idx = range(len(recs)) if idx is None else idx
    C = sum(recs[i]["C"] for i in idx)
    if kind == "density":
        return F_density(C)
    M = sum(recs[i][key] for i in idx)
    return F_mf(M, C)


def jack_se(recs, kind, key="M"):
    n = len(recs)
    Cs = np.array([r["C"] for r in recs])
    Ms = np.array([r[key] for r in recs]) if kind == "mf" else None
    Ct = Cs.sum(0)
    Mt = Ms.sum(0) if kind == "mf" else None
    loo = []
    for i in range(n):
        if kind == "density":
            loo.append(F_density(Ct - Cs[i]))
        else:
            loo.append(F_mf(Mt - Ms[i], Ct - Cs[i]))
    loo = np.array(loo)
    return np.sqrt((n - 1) / n * ((loo - loo.mean(0)) ** 2).sum(0))


class Ens:
    """One ensemble (a set of independent seeds) and one profile kind."""

    def __init__(self, label, recs, kind, key="M"):
        self.label, self.recs, self.kind, self.key = label, recs, kind, key
        self.n = len(recs)
        self.F = pooled(recs, kind, key=key)
        self.se = jack_se(recs, kind, key=key)
        self.Cs = np.array([r["C"] for r in recs])
        self.Ms = np.array([r[key] for r in recs]) if kind == "mf" else None

    def boot(self, idx):
        C = self.Cs[idx].sum(0)
        if self.kind == "density":
            return F_density(C)
        return F_mf(self.Ms[idx].sum(0), C)


def jack_diff_se(A, B):
    """Jackknife se of the per-bin difference A - B when both profiles come from the SAME chains."""
    n = A.n
    loo = np.array([A.boot(np.delete(np.arange(n), i)) - B.boot(np.delete(np.arange(n), i)) for i in range(n)])
    loo = loo - loo[:, WIN].mean(1, keepdims=True)
    return np.sqrt((n - 1) / n * ((loo - loo.mean(0)) ** 2).sum(0))


def compare(A, B, nboot=2000, paired=False):
    """D(A, B).  paired=True: A and B are computed from the same chains (resampled jointly, noise of
    the difference from its own jackknife); otherwise the two ensembles are independent."""
    d = (A.F - B.F)[WIN]
    d = d - d.mean()
    if paired:
        noise2 = float(np.mean(jack_diff_se(A, B)[WIN] ** 2))
    else:
        noise2 = float(np.mean((A.se ** 2 + B.se ** 2)[WIN]))
    rms = float(np.sqrt(np.mean(d ** 2)))
    D = float(np.sqrt(max(0.0, rms ** 2 - noise2)))
    Db = []
    for _ in range(nboot):
        ia = RNG.integers(0, A.n, A.n)
        ib = ia if paired else RNG.integers(0, B.n, B.n)
        db = (A.boot(ia) - B.boot(ib))[WIN]
        db = db - db.mean()
        Db.append(np.sqrt(max(0.0, float(np.mean(db ** 2)) - 2.0 * noise2)))
    Db = np.array(Db)
    return dict(A=A.label, B=B.label, rms_obs=rms, noise=float(np.sqrt(noise2)), D=D, paired=paired,
                D_upper95=float(np.quantile(Db, 0.95)), D_boot_median=float(np.median(Db)),
                maxabs=float(np.max(np.abs(d))), n_A=A.n, n_B=B.n)


def gate_level(D, up, tol=0.005, tol_up=0.0075, marg=0.010):
    if D <= tol and up <= tol_up:
        return "ADMISSIBLE"
    if D <= marg:
        return "MARGINAL"
    return "FAIL"


# ------------------------------------------------------------------------------------------ main
def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    S = dict(prereg="configs/numerical_validation/wca_validation_prereg.json", window=[-0.1, 1.1], bins=NB)
    groups = {}
    mc = load_group("mc", "mc")
    assert len(mc) >= 16, f"MC runs missing ({len(mc)})"
    groups["MC"] = mc
    mala = load_group("mala", "dt0.00025")
    if mala:
        groups["MALA dt 2.5e-4"] = mala
    em = {}
    for dt, tag in ((0.002, "A_dt0.002"), (0.001, "B_dt0.001"), (0.0005, "C_dt0.0005")):
        em[dt] = load_old(tag) + load_group("em_impl", f"dt{dt:g}")
    for dt in (0.00025, 0.000125):
        em[dt] = load_group("em_impl", f"dt{dt:g}")
    for dt, r in em.items():
        if r:
            groups[f"EM_IMPL dt {dt:g}"] = r
    for dt in (0.001, 0.0005, 0.00025):
        r = load_group("lm", f"dt{dt:g}")
        if r:
            groups[f"LM dt {dt:g}"] = r
    for dt in (0.005, 0.0025, 0.00125):
        r = load_group("baoab", f"dt{dt:g}")
        if r:
            groups[f"BAOAB dt {dt:g}"] = r
    S["n_seeds"] = {k: len(v) for k, v in groups.items()}
    print("seeds:", S["n_seeds"], flush=True)

    ref_d = Ens("MC F_density", mc, "density")
    ref_m = Ens("MC F_MF", mc, "mf")
    S["mc_noise_density"] = float(np.sqrt(np.mean(ref_d.se[WIN] ** 2)))
    S["mc_noise_mf"] = float(np.sqrt(np.mean(ref_m.se[WIN] ** 2)))
    rows = []
    # G_MF_formula: MC mean-force route vs MC density route
    g_mf = compare(ref_m, ref_d, paired=True)
    S["G_MF_formula"] = dict(g_mf, verdict="PASS" if g_mf["D"] <= 0.005 else "FAIL")
    print("G_MF_formula", json.dumps(S["G_MF_formula"]), flush=True)
    profiles = {"MC F_density": ref_d.F, "MC F_MF": ref_m.F}
    ses = {"MC F_density": ref_d.se}
    for name, recs in groups.items():
        if name == "MC":
            continue
        Ed = Ens(f"{name} F_density", recs, "density")
        Em = Ens(f"{name} F_MF", recs, "mf")
        cd = compare(Ed, ref_d)
        cm = compare(Em, ref_d)
        ci = compare(Em, Ed, paired=True)
        row = dict(ensemble=name, n=len(recs), D_density_vs_MC=cd["D"], up_density=cd["D_upper95"], rms_density_vs_MC=cd["rms_obs"],
                   D_mf_vs_MC=cm["D"], up_mf=cm["D_upper95"], D_mf_vs_own_density=ci["D"], up_mf_own=ci["D_upper95"],
                   rms_mf_vs_own_density=ci["rms_obs"], noise_density=float(np.sqrt(np.mean(Ed.se[WIN] ** 2))),
                   noise_mf=float(np.sqrt(np.mean(Em.se[WIN] ** 2))))
        blow = sum(1 for r in recs if r.get("nonfinite", 0) > 0)
        row["blowups"] = blow
        lvl_d = gate_level(cd["D"], cd["D_upper95"])
        lvl_m = gate_level(cm["D"], cm["D_upper95"])
        order = ["ADMISSIBLE", "MARGINAL", "FAIL"]
        row["gate"] = order[max(order.index(lvl_d), order.index(lvl_m))] if blow == 0 else "FAIL"
        rows.append(row)
        profiles[f"{name} F_density"] = Ed.F
        profiles[f"{name} F_MF"] = Em.F
        ses[f"{name} F_density"] = Ed.se
        print(f"{name:22s} n={len(recs):3d}  D(dens,MC) {cd['D']:.4f} [up {cd['D_upper95']:.4f}]  D(MF,MC) {cm['D']:.4f} [up {cm['D_upper95']:.4f}]"
              f"  D(MF,own dens) {ci['D']:.4f} (rms {ci['rms_obs']:.4f})  noise {row['noise_density']:.4f}/{row['noise_mf']:.4f}  -> {row['gate']}",
              flush=True)
    S["rows"] = rows
    if "MALA dt 2.5e-4" in groups:
        r = [x for x in rows if x["ensemble"] == "MALA dt 2.5e-4"][0]
        S["G_MC_self_fullsystem"] = dict(D=r["D_density_vs_MC"], up=r["up_density"],
                                         verdict="PASS" if r["up_density"] <= 0.0075 else "FAIL")
    # timestep choice per the frozen rule
    adm = [float(x["ensemble"].split()[-1]) for x in rows if x["ensemble"].startswith("EM_IMPL") and x["gate"] == "ADMISSIBLE"]
    S["timestep_choice"] = dict(admissible_EM_IMPL_dt=sorted(adm), chosen=max(adm) if adm else None,
                                verdict="RESOLVED" if adm else "UNRESOLVED")
    print("timestep choice:", S["timestep_choice"], flush=True)

    # secondary observables
    sec = {}
    for name, recs in groups.items():
        Ct = sum(r["C"] for r in recs)
        e = {}
        e["P_stretched"] = float(Ct[CEN > 0.5].sum() / Ct.sum())
        ps = np.array([r["C"][CEN > 0.5].sum() / r["C"].sum() for r in recs])
        e["P_stretched_se"] = float(ps.std(ddof=1) / np.sqrt(len(ps)))
        lo = (CEN >= -0.1) & (CEN <= 0.25)
        hi = (CEN >= 0.75) & (CEN <= 1.1)
        dfw = np.array([-np.log(r["C"][hi].sum() / r["C"][lo].sum()) for r in recs])
        e["DeltaF_wells"] = float(-np.log(Ct[hi].sum() / Ct[lo].sum()))
        e["DeltaF_wells_se"] = float(dfw.std(ddof=1) / np.sqrt(len(dfw)))
        if "sum_U" in recs[0]:
            Us = np.array([r["sum_U"] / max(r["n_U"], 1) for r in recs])
            e["mean_U"] = float(Us.mean())
            e["mean_U_se"] = float(Us.std(ddof=1) / np.sqrt(len(Us)))
            e["acc"] = float(np.mean([r["n_acc"] / max(r["n_prop"], 1) for r in recs]))
            e["dimer_acc"] = float(np.mean([r["n_dimer_acc"] / max(r["n_dimer_prop"], 1) for r in recs]))
            e["max_step_displacement"] = float(max(r["max_step_displacement"] for r in recs))
            if name.startswith("BAOAB"):
                sp = sum(r["sum_p_moments"] for r in recs)
                ndof = sum(r["C_all"].sum() for r in recs) * 200
                m1, m2, m3, m4 = sp / ndof
                var = m2 - m1 ** 2
                e["p_mean"] = float(m1)
                e["p_var"] = float(var)
                e["p_skew"] = float((m3 - 3 * m1 * var - m1 ** 3) / var ** 1.5)
                e["p_exkurt"] = float(m4 / var ** 2 - 3.0)
        sec[name] = e
    diag = {}
    for dt in (0.002, 0.001, 0.0005, 0.00025, 0.000125):
        recs = load_group("diag", f"dt{dt:g}")
        if not recs:
            continue
        nst = sum(r["C_all"].sum() for r in recs)
        diag[f"{dt:g}"] = dict(n_seeds=len(recs), steps=float(nst),
                               frac_steps_clip_binding=float(sum(r["n_clip_steps"] for r in recs) / nst),
                               clip_particle_steps_per_step=float(sum(r["n_clip_particle_steps"] for r in recs) / nst),
                               frac_steps_pair_inside_min_r=float(sum(r["n_minr_diag"] for r in recs) / nst),
                               frac_deposits_mf_sample_clip=float(sum(r["n_mf_clip"] for r in recs) / nst),
                               max_step_displacement=float(max(r["max_step_displacement"] for r in recs)),
                               mean_U=float(np.mean([r["sum_U"] / r["n_U"] for r in recs])),
                               mean_U_se=float(np.std([r["sum_U"] / r["n_U"] for r in recs], ddof=1) / np.sqrt(len(recs))))
    S["secondary"] = sec
    S["regularisation_EM_IMPL"] = diag
    for k, v in diag.items():
        print("diag dt", k, json.dumps(v), flush=True)
    for k, v in sec.items():
        print("sec", k, json.dumps({a: round(b, 5) for a, b in v.items()}), flush=True)
    S["profiles"] = {k: v.tolist() for k, v in profiles.items()}
    S["centres"] = CEN.tolist()
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    make_figures(S, profiles, ses, rows)


def make_figures(S, profiles, ses, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    ref = profiles["MC F_density"]
    se_ref = ses["MC F_density"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axs[0]
    ax.plot(CEN[WIN], ref[WIN], color="k", lw=2.4, label="MC (exact Gibbs), F_density")
    ax.plot(CEN[WIN], profiles["MC F_MF"][WIN], color="#1baf7a", ls="--", lw=1.4, label="MC, F_MF (mean-force route)")
    em_cols = {0.002: "#c0392b", 0.001: "#e67e22", 0.0005: "#f1c40f", 0.00025: "#2a78d6", 0.000125: "#6c3483"}
    for dt, c in em_cols.items():
        k = f"EM_IMPL dt {dt:g} F_density"
        if k in profiles:
            ax.plot(CEN[WIN], profiles[k][WIN], color=c, lw=1.2, label=f"production EM dt {dt:g}")
    ax.set_xlabel("z"); ax.set_ylabel("F (kT), centred on [-0.1, 1.1]"); ax.set_title("(a) free energy, density route", loc="left")
    ax.legend(fontsize=7)
    ax = axs[1]
    ax.fill_between(CEN[WIN], -2 * se_ref[WIN], 2 * se_ref[WIN], color="#dddddd", label="MC +-2 se")
    for dt, c in em_cols.items():
        k = f"EM_IMPL dt {dt:g} F_density"
        if k in profiles:
            ax.plot(CEN[WIN], (profiles[k] - ref)[WIN], color=c, lw=1.3, label=f"EM dt {dt:g} density")
            km = f"EM_IMPL dt {dt:g} F_MF"
            ax.plot(CEN[WIN], (profiles[km] - ref)[WIN], color=c, lw=1.0, ls=":", label=f"EM dt {dt:g} mean force")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("z"); ax.set_ylabel("F - F_MC (kT)"); ax.set_title("(b) production EM minus exact Gibbs", loc="left")
    ax.legend(fontsize=6, ncol=2)
    ax = axs[2]
    for prefix, c, mk in (("EM_IMPL", "#c0392b", "o"), ("LM", "#2a78d6", "s"), ("BAOAB", "#1baf7a", "^"), ("MALA", "k", "*")):
        rr = [r for r in rows if r["ensemble"].startswith(prefix)]
        if not rr:
            continue
        dts = np.array([float(r["ensemble"].split()[-1]) for r in rr])
        D = np.array([r["D_density_vs_MC"] for r in rr])
        up = np.array([r["up_density"] for r in rr])
        o = np.argsort(dts)
        ax.errorbar(dts[o], np.maximum(D[o], 1e-4), yerr=[np.zeros(len(o)), np.maximum(up[o] - D[o], 0)], color=c, marker=mk,
                    label=f"{prefix}: D(F_density, MC)", capsize=2)
        Dm = np.array([r["D_mf_vs_MC"] for r in rr])
        ax.plot(dts[o], np.maximum(Dm[o], 1e-4), color=c, marker=mk, ls=":", mfc="none", label=f"{prefix}: D(F_MF, MC)")
    ax.axhline(0.005, color="#888", ls="--", lw=0.8)
    ax.text(1.3e-4, 0.0055, "admissibility 0.005", fontsize=7, color="#555")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("time step"); ax.set_ylabel("debiased RMS vs MC (kT)"); ax.set_title("(c) timestep convergence", loc="left")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"fig_wca_timestep_validation.{ext}"), dpi=160)


if __name__ == "__main__":
    main()
