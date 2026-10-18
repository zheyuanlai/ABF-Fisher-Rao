"""Evaluation helpers for the pentane R15 timestep campaign (docs/PENTANE_R15_DT_FR.md).

Evaluation only: the v2 reference (cache/alkanes_cv/ref_pentane_b2_R15_v2_meanforce.npz) is read here and nowhere
in the dynamics.  One mask (the v2 window), one gauge (additive constant aligned on the window).
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(ROOT, "results", "pentane_r15_dt_fr")
REF_V2 = os.path.join(ROOT, "cache", "alkanes_cv", "ref_pentane_b2_R15_v2_meanforce.npz")
REF_SHA256 = "7363c374500f6a2bc7602a64c9a71020b36a8dd52e0e0f23dd6b334eab00d9d9"
N_BOOT = 10000
FAM = ["TT", "TG", "GG same", "GG opp"]


def load_ref():
    import hashlib
    h = hashlib.sha256(open(REF_V2, "rb").read()).hexdigest()
    assert h == REF_SHA256, f"reference checksum mismatch {h}"
    r = np.load(REF_V2)
    edges = r["cond_edges"]; centres = 0.5 * (edges[1:] + edges[:-1])
    wbins = [k for k in range(len(centres)) if float(r["window_lo"]) <= centres[k] <= float(r["window_hi"])]
    return dict(grid=r["grid"], dz=float(r["dz"]), win=r["window_mask"].astype(bool), F=r["F"], Fp=r["Fprime"], Fp_se=r["Fprime_se"],
                cond_dens=r["cond_dens"], cond_basin=r["cond_basin_probs"], cond_weight=r["cond_weight"], wbins=wbins, dphi=float(r["cond_dphi"]),
                window=(float(r["window_lo"]), float(r["window_hi"])), cond_edges=edges)


def l2_aligned(P, Fref, mask):
    diff = P - Fref
    shift = diff[..., mask].mean(-1, keepdims=True)
    return np.sqrt(((diff - shift)[..., mask] ** 2).mean(-1))


def l2_plain(P, ref, mask):
    return np.sqrt(((P - ref)[..., mask] ** 2).mean(-1))


def cond_tv(hist, cond_dens, bins, dphi, min_count=100):
    out = []
    for k in bins:
        h = np.asarray(hist[k], float); c = h.sum()
        if c < min_count:
            out.append(1.0); continue
        p = h / (c * dphi * dphi); r = cond_dens[k]; r = r / (r.sum() * dphi * dphi)
        out.append(0.5 * np.abs(p - r).sum() * dphi * dphi)
    return float(np.mean(out))


def cond_tv9(cb, ref_basin, bins, min_count=100):
    out = []
    for k in bins:
        h = np.asarray(cb[k], float); c = h.sum()
        if c < min_count:
            out.append(1.0); continue
        out.append(0.5 * np.abs(h / c - ref_basin[k] / ref_basin[k].sum()).sum())
    return float(np.mean(out))


def kl_uniform_window(p_hat, grid, dz, win):
    """KL(p_hat restricted+renormalised to the window || uniform on the window)."""
    pw = np.clip(p_hat[..., win], 0, None); mass = pw.sum(-1, keepdims=True) * dz
    pw = pw / np.clip(mass, 1e-300, None); U = 1.0 / (win.sum() * dz)
    return (pw * np.log(np.clip(pw, 1e-300, None) / U)).sum(-1) * dz, mass[..., 0]


def low_support_fraction(eff, win):
    e = np.asarray(eff, float)[..., win]
    med = np.median(e, axis=-1, keepdims=True)
    return np.mean(e < 0.25 * med, axis=-1)


def load_runs(stage, pattern="*.npz"):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, stage, "raw", pattern))):
        if f.endswith(".tmp.npz"):
            continue
        d = np.load(f, allow_pickle=True)
        m = json.loads(str(d["meta_json"]))
        out.append(dict(path=f, d=d, meta=m, name=str(d["name"])))
    return out


def per_seed(run, ref, T_eval=None):
    """Per-seed metric dict list for one run file (all seeds), evaluated on [0, T_eval]."""
    d, m = run["d"], run["meta"]
    times = np.asarray(d["times"], float); dt = float(m["dt"]); N = int(d["n_replicas"])
    T_eval = float(m["T"]) if T_eval is None else T_eval
    sel = times <= T_eval + 1e-9
    t = times[sel]
    pmf = np.asarray(d["pmf"])[sel]; mf = np.asarray(d["mean_force"])[sel]; ph = np.asarray(d["p_hat"])[sel]
    eff = np.asarray(d["eff_counts"])[sel]
    S, R, G = pmf.shape
    win = ref["win"]
    eF = np.stack([l2_aligned(pmf[:, s], ref["F"], win) for s in range(R)])            # (R, S)
    eFp = np.stack([l2_plain(mf[:, s], ref["Fp"], win) for s in range(R)])
    kl, mass = kl_uniform_window(ph, ref["grid"], ref["dz"], win)                       # (S, R)
    lowsup = low_support_fraction(eff, win)                                             # (S, R)
    ess = np.asarray(d["ancestor_ess"])[sel] / N; maxf = np.asarray(d["max_ancestor_frac"])[sel]
    repl = np.asarray(d["repl_cumulative"])[sel] / N
    cb = np.asarray(d["cond_basin"])[sel] if "cond_basin" in d.files else None
    famc = np.asarray(d["fam_count"])[sel] / N; fama = np.asarray(d["fam_distinct_anc"])[sel] / N
    full = T_eval >= float(m["T"]) - 1e-9
    after6 = t >= 6.0 - 1e-9
    rows = []
    for s in range(R):
        r = dict(run=run["name"], arm=m["arm"], rate=float(m["fr_rate"]), dt=dt, T=T_eval, rng=int(m["rng_seed"]), seed=s,
                 dep=m.get("deposit_interval"), t=t, eF=eF[s], eFp=eFp[s],
                 I_F=float(np.trapezoid(eF[s], t)), I_Fp=float(np.trapezoid(eFp[s], t)), eF_end=float(eF[s, -1]), eFp_end=float(eFp[s, -1]),
                 kl=kl[:, s], I_KL=float(np.trapezoid(kl[after6, s], t[after6])), kl_end=float(kl[-1, s]), win_mass_end=float(mass[-1, s]),
                 lowsup=lowsup[:, s], lowsup_end=float(lowsup[-1, s]),
                 ess=ess[:, s], maxf=maxf[:, s], repl=repl[:, s], ess_end=float(ess[-1, s]), maxf_end=float(maxf[-1, s]), repl_end=float(repl[-1, s]),
                 famc=famc[:, s], fama=fama[:, s],
                 fam_first=np.asarray(d["fam_first_discovery"])[s] * dt, b9_first=np.asarray(d["b9_first_discovery"])[s] * dt,
                 fam_trans=int(np.asarray(d["fam_transitions"])[s]), n_trans=int(np.asarray(d["n_transitions"])[s]), round_trips=int(np.asarray(d["n_round_trips"])[s]),
                 clip_frac=float(np.asarray(d["clip_n"])[s] / max(np.asarray(d["clip_evals"])[s], 1)), clip_max_raw=float(np.asarray(d["clip_max_raw"])[s]),
                 max_disp=float(np.asarray(d["max_step_displacement"])[s]), nonfinite=int(np.asarray(d["n_nonfinite_walker_steps"])[s]),
                 had_nan=bool(d["had_nan"]), wall_s=float(d["wall_seconds"]) / R, force_evals=float(int(round(T_eval / dt)) + 1) * N,
                 score_std=float(np.asarray(d["fr_score_std"])[s]), score_absmax=float(np.asarray(d["fr_score_absmax"])[s]))
        if cb is not None:
            r["Dc9"] = np.array([cond_tv9(cb[k, s], ref["cond_basin"], ref["wbins"]) for k in range(len(t))])
            r["Dc9_end"] = float(r["Dc9"][-1]); r["cond_basin_end"] = cb[-1, s]
        if full:
            r["Dcond_end"] = cond_tv(np.asarray(d["cond_hist"])[s], ref["cond_dens"], ref["wbins"], ref["dphi"])
            r["birth_hist"] = np.asarray(d["birth_hist"])[s]; r["death_hist"] = np.asarray(d["death_hist"])[s]
            r["fam_birth"] = np.asarray(d["fam_birth"])[s]; r["fam_death"] = np.asarray(d["fam_death"])[s]
            r["pmf_end"] = pmf[-1, s]; r["mf_end"] = mf[-1, s]; r["p_hat_end"] = ph[-1, s]
            r["fsum_prod"] = np.asarray(d["fsum_prod"])[s]; r["csum_prod"] = np.asarray(d["csum_prod"])[s]
            # raw deposit bias: binned conditional mean of the production deposits vs F'_ref (bins with >= 200 deposits)
            Cb = r["csum_prod"]; ok = (Cb >= 200) & win
            if ok.any():
                bias = r["fsum_prod"][ok] / Cb[ok] - ref["Fp"][ok]
                r["dep_bias_rms"] = float(np.sqrt(np.average(bias ** 2, weights=Cb[ok])))
                r["dep_bias_bins"] = int(ok.sum())
                r["dep_bias_profile"] = np.where(ok, np.divide(r["fsum_prod"], np.clip(Cb, 1, None)) - ref["Fp"], np.nan)
            # error split: compact part of the window (R < 2.4) vs the rest, final aligned profile
            P = pmf[-1, s]; dev = P - ref["F"]; dev = dev - dev[win].mean()
            cmp_ = win & (ref["grid"] < 2.4); rest = win & (ref["grid"] >= 2.4)
            r["eF_compact"] = float(np.sqrt(np.mean(dev[cmp_] ** 2))); r["eF_rest"] = float(np.sqrt(np.mean(dev[rest] ** 2)))
        rows.append(r)
    return rows


FAM_OF9 = [0, 1, 1, 1, 2, 3, 1, 3, 2]      # basin9 = 3*b(phi1) + b(phi2), b: 0 T, 1 G+, 2 G- -> TT / TG / GG same / GG opp


def fam4_from9(p9):
    p9 = np.asarray(p9, float); out = np.zeros(p9.shape[:-1] + (4,))
    for i, f in enumerate(FAM_OF9):
        out[..., f] += p9[..., i]
    return out


def deposit_family_composition(run):
    """Final deposit-weighted torsional-family composition per conditional R bin (summed over seeds) and deposit share."""
    cb = np.asarray(run["d"]["cond_basin"])[-1].sum(0)                       # (12, 9)
    tot = cb.sum(-1, keepdims=True)
    return fam4_from9(cb / np.clip(tot, 1, None)), tot[:, 0] / max(cb.sum(), 1)


def boot_median_ci(x, level=0.95, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(N_BOOT, x.size), replace=True), axis=1)
    a = (1 - level) / 2
    return float(np.median(x)), float(np.quantile(meds, a)), float(np.quantile(meds, 1 - a))


def paired(xs, ys, key):
    """Paired relative change x/y - 1 over matched (rng, seed); returns dict with median, CIs, wins."""
    yk = {(r["rng"], r["seed"]): r[key] for r in ys}
    rel = np.array([r[key] / yk[(r["rng"], r["seed"])] - 1.0 for r in xs if (r["rng"], r["seed"]) in yk])
    med, lo, hi = boot_median_ci(rel); _, lo90, hi90 = boot_median_ci(rel, 0.90); _, lo975, hi975 = boot_median_ci(rel, 0.975)
    return dict(median=med, ci95=[lo, hi], ci90=[lo90, hi90], ci975=[lo975, hi975], wins=int(np.sum(rel < 0)), n=int(rel.size), values=rel.tolist())


def unpaired_ratio(xs, ys, key, seed=0):
    """median(x)/median(y) - 1 with a bootstrap CI resampling each group independently."""
    x = np.array([r[key] for r in xs], float); y = np.array([r[key] for r in ys], float)
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(x, (N_BOOT, x.size)), 1) / np.median(rng.choice(y, (N_BOOT, y.size)), 1) - 1
    return dict(ratio_minus_1=float(np.median(x) / np.median(y) - 1), ci95=[float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))],
                n_x=int(x.size), n_y=int(y.size), median_x=float(np.median(x)), median_y=float(np.median(y)))


def med(rows, key):
    v = np.array([r[key] for r in rows], float)
    return float(np.median(v)) if v.size else float("nan")
