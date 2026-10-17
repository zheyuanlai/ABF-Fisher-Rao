"""Loaders: paired ABF vs ABF+FR runs -> common per-seed error / occupancy arrays (two arms only; any other arm in a
historical file is never read).

Historical raw npz files are git-ignored and live in the main checkout; set ABFFR_DATA_ROOT to read them from a
worktree (default: this repository's root).
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_ROOT = os.environ.get("ABFFR_DATA_ROOT", ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import metrics as mt  # noqa: E402

KB = 0.0083144626
TWO_PI = 2.0 * math.pi


def dpath(rel):
    for base in (ROOT, DATA_ROOT):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return os.path.join(DATA_ROOT, rel)


def glob_both(rel_dir, pattern):
    """Files matching pattern in rel_dir under ROOT and DATA_ROOT (ROOT wins on a name clash)."""
    found = {}
    for base in (DATA_ROOT, ROOT):
        for f in glob.glob(os.path.join(base, rel_dir, pattern)):
            if not f.endswith((".ckpt.npz", ".tmp.npz")):
                found[os.path.basename(f)] = f
    return [found[k] for k in sorted(found)]


# ---------------------------------------------------------------------------------------------------------------
# ERI-engine cells (ITE 300 K / 150 K, ERI 600 K): files <dir>/<abf|fr>_N<K>_s<seed>.npz
# ---------------------------------------------------------------------------------------------------------------
def load_eri_cell(run_dir, K, refbase, variant, read_bins=None, arms=("abf", "fr"), fr_glob=None, t_max=None,
                  keep_runs=False, abf_dir=None):
    """read_bins: None = as run; an integer dividing the run grid = coarsen the cumulative accumulators exactly
    (sum of adjacent bins) before the read-out (a read-out only: the dynamics used the run grid)."""
    mt.set_reference_base(dpath(refbase))
    out = {"arms": {}, "seeds": None}
    globs = {"abf": f"abf_N{K}_s*.npz", "fr": fr_glob or f"fr_N{K}_s*.npz"}
    per = {}
    for arm in arms:
        files = glob_both(abf_dir if (arm == "abf" and abf_dir) else run_dir, globs[arm])
        for f in files:
            s = int(re.search(r"_s(\d+)\.npz$", f).group(1))
            per.setdefault(s, {})[arm] = f
    seeds = sorted(s for s in per if all(a in per[s] for a in arms))
    if len(seeds) != len(per):
        raise RuntimeError(f"{run_dir}: unpaired seeds {sorted(set(per) - set(seeds))} (no discarding allowed)")
    out["seeds"] = seeds
    refs = {}
    for arm in arms:
        E = []
        G = []
        W = []
        U = []
        runs = []
        for s in seeds:
            r = mt.load_run(per[s][arm])
            c = r["cfg"]
            ng = r["sv_C"].shape[1]
            if read_bins and read_bins != ng:
                k = ng // read_bins
                assert ng % read_bins == 0
                r["sv_C"] = r["sv_C"].reshape(len(r["sv_C"]), read_bins, k).sum(2)
                r["sv_M"] = r["sv_M"].reshape(len(r["sv_M"]), read_bins, k).sum(2)
                ng = read_bins
            if ng not in refs:
                refs[ng] = mt.reference_for_grid(ng, variant)
            Fr, Gr = refs[ng]
            if t_max is not None:
                sel = r["times"] <= t_max + 1e-9
                for key in ("save_steps", "times", "sv_C", "sv_M", "sv_h", "sv_i", "sv_u", "sv_cond"):
                    r[key] = r[key][sel]
            e = mt.errors(r, Fr, Gr)
            E.append(e["eF"])
            G.append(e["eG"])
            W.append(r["sv_i"][:, 3] / float(c["N"]))                   # V_NWIN: |phi| < window_phi at the save
            U.append(np.column_stack([r["sv_i"][:, 2], r["sv_u"][:, 2], r["sv_u"][:, 3], r["sv_i"][:, 1]]))
            if keep_runs:
                runs.append(r)
            t = r["times"]
        out["arms"][arm] = dict(eF=np.array(E).T, eG=np.array(G).T, win=np.array(W).T, gen=np.array(U),
                                cfg=c, runs=runs if keep_runs else None,
                                nfe=None)
    c = out["arms"]["abf"]["cfg"]
    cf = out["arms"][arms[-1]]["cfg"]
    out.update(t=t, T=float(t[-1]), K=int(c["N"]), dt=float(c["dt"]), bins=ng, run_bins=int(c["n_grid"]),
               t_fr=float(cf["fr_start_steps"]) * float(c["dt"]), t_ramp=float(c["warmup_steps"]) * float(c["dt"]),
               t_stop=(float(cf["fr_stop_steps"]) * float(c["dt"]) if int(cf.get("fr_stop_steps", -1)) >= 0 else None),
               rate=float(cf["fr_rate"]), T_K=float(c["T_K"]), kT=KB * float(c["T_K"]),
               win_share=2.0 * float(c["window_phi"]) / TWO_PI, F_ref=refs[ng][0], G_ref=refs[ng][1])
    return out


# ---------------------------------------------------------------------------------------------------------------
# LTA histogram production (src/lta/core_lta.py output: one npz per arm with all 16 seeds)
# ---------------------------------------------------------------------------------------------------------------
def lta_reference(T_K):
    s = json.load(open(dpath(f"results/numerical_validation/lta/T{int(T_K)}/summary.json")))
    F = np.asarray(s["profiles"]["MC density"], float)
    cen = np.asarray(s["centres"], float)
    C = np.zeros(180)
    M = np.zeros(180)
    files = sorted(glob.glob(dpath(f"results/numerical_validation/lta/T{int(T_K)}/MC") + "/*.npz"))
    for f in files:
        z = np.load(f, allow_pickle=False)
        C += z["C"]
        M += z["Mf"]
    G = M / C
    return F - F.mean(), G, cen, len(files)


def load_lta(T_K):
    camp = json.load(open(dpath("configs/lta_histogram/campaign.json")))
    smp = camp["sampler"]
    Fr, Gr, cen, nfiles = lta_reference(T_K)
    out = {"arms": {}}
    seeds = None
    for arm, name in (("abf", "abf"), ("fr", "fr_uniform")):
        z = np.load(dpath(f"results/lta_histogram/production_T{int(T_K)}/{name}.npz"), allow_pickle=True)
        assert np.abs(np.asarray(z["grid"]) - cen).max() < 1e-12
        sd = [int(x) for x in z["seeds"]]
        if seeds is None:
            seeds = sd
        assert sd == seeds, "LTA arms must hold the same seeds in the same order"
        pmf = np.asarray(z["pmf"], float)
        mf = np.asarray(z["mean_force"], float)
        d = pmf - Fr[None, None, :]
        d = d - d.mean(-1, keepdims=True)
        eF = np.sqrt((d * d).mean(-1))
        eG = np.sqrt(((mf - Gr[None, None, :]) ** 2).mean(-1))
        meta = json.loads(str(z["meta"]))
        out["arms"][arm] = dict(eF=eF, eG=eG, win=np.asarray(z["frac_window"], float), meta=meta,
                                gen=None, ess=np.asarray(z["ancestor_ess"], float) / smp["n_replicas"],
                                repl=np.asarray(z["repl_cumulative"], float))
        t = np.asarray(z["times"], float)
        a_ = float(z["a_pseudo"])
    dt = float(smp["dt"])
    out.update(seeds=seeds, t=t, T=float(t[-1]), K=int(smp["n_replicas"]), dt=dt, bins=int(smp["n_grid"]),
               run_bins=int(smp["n_grid"]), t_fr=smp["fr_start_steps"] * dt, t_ramp=smp["abf_warmup_steps"] * dt,
               t_stop=None, rate=float(camp["per_T"][str(int(T_K))]["fr_rate"]), T_K=float(T_K), kT=KB * T_K,
               win_share=2 * 1.5 / a_, F_ref=Fr, G_ref=Gr, ref_files=nfiles)
    return out


SYSTEMS = {
    "LTA150": dict(label="LTA 150 K", loader=lambda: load_lta(150),
                   model="rigid all-silica LTA, 2-bead UA ethane (dilute), EM dt 2e-4, exact MC reference",
                   status="historical (2026-10-03/04), 16 paired seeds; FR rate 0.2"),
    "LTA300": dict(label="LTA 300 K", loader=lambda: load_lta(300),
                   model="rigid all-silica LTA, 2-bead UA ethane (dilute), EM dt 2e-4, exact MC reference",
                   status="historical (2026-10-03/04), 16 paired seeds; FR rate 0.2"),
    "ITE300": dict(label="ITE 300 K", loader=lambda: load_eri_cell("results/overnight/ite/prod_confirm", 1024,
                                                                   "results/overnight/ite/reference", "ite_itq3"),
                   model="rigid Si-ITE (ITQ-3), Dubbeldam-2004 UA ethane (dilute), EM dt 1e-4, exact MC reference",
                   status="exploratory block X1 (2026-10-16), 16 paired seeds; FR rate 2.0"),
    "ITE150": dict(label="ITE 150 K", loader=lambda: load_eri_cell("results/overnight/ite150/prod_confirm", 256,
                                                                   "results/overnight/ite150/reference", "ite_itq3"),
                   model="rigid Si-ITE (ITQ-3), Dubbeldam-2004 UA ethane (dilute), EM dt 1e-4, smooth MC reference (360 bins)",
                   status="exploratory block X2 (2026-10-16), 16 paired seeds; FR rate 0.5"),
    "ITE150c": dict(label="ITE 150 K (180-bin read-out)",
                    loader=lambda: load_eri_cell("results/overnight/ite150/prod_confirm", 256,
                                                 "results/overnight/ite150/reference", "ite_itq3", read_bins=180),
                    model="as ITE150; the 360-bin accumulators summed pairwise and scored vs the 180-bin MC mean force",
                    status="read-out sensitivity of X2 (same trajectories)"),
}
