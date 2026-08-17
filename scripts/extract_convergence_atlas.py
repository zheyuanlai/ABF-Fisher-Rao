"""Collect every study's L2(F)-versus-time curve into ONE artifact, on one convention.

    python scripts/extract_convergence_atlas.py

The campaign already scores  e_F(t) = || F_hat_t - F_ref - c_t ||  online, in each system's
own core, and stores it per seed.  What it has never had is those curves side by side.  This
script gathers them, without recomputing dynamics, into

    results/convergence_atlas/atlas.npz      per-panel  times (T,)  +  per-arm (n_seeds, T)
    results/convergence_atlas/atlas.json     panel metadata: units, t_FR, provenance, caveats

CONVENTION (verified identical up to endpoint weighting across cores, see docstring notes):
every core removes the arbitrary additive constant on the evaluation window and reports an
interior-window RMS error, so panels are directly comparable in shape.  The VERTICAL SCALE is
NOT comparable between panels -- different systems, different CV units, different windows --
which is why nothing here ever pools across panels.

TWO PROVENANCE FACTS THAT DECIDE WHICH FILES ARE READ:

  * WCA.  `results/wca_production/` was scored against the SUPERSEDED reference and stores no
    profile time series, so its curves can be neither reused nor rescored.  The flagship panel
    therefore reads `results/wca_caseix_hp/` and `results/wca_five_arm/`, whose runs store
    `pmf_t` and carry the corrected high-precision reference in the file.  Verified: the stored
    `l2_f_t` reproduces EXACTLY from `pmf_t` + `reference_free_energy` on the [-0.1, 1.1]
    window (max abs diff 0.0 over all 80 + 80 runs).

  * 2-D toy.  The FR arms were tuned over 36 configurations; ABF has exactly ONE.  A
    best-config-FR-vs-only-config-ABF panel would be a selection artifact, so this script emits
    ALL 36 FR configs per arm and marks the selected one, leaving the selection visible.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "convergence_atlas")

# Canonical arm names, so every panel speaks the same language.
ABF, MFR, MFR_ORACLE = "abf", "mfr", "mfr_oracle"
SHAM, SHAM_ORACLE = "sham", "sham_oracle"
COUNT_BAL, BOOK_LAP = "count_balancing", "book_laplacian"
FR_UNIFORM, OPES, MFR_ACTIVE = "fr_uniform", "opes", "mfr_active"


def _seed_of(path):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _stack(per_seed):
    """dict seed -> (T,) curve  ->  (n_seeds, T) ordered by seed, plus the seed list."""
    seeds = sorted(per_seed)
    return np.asarray([per_seed[s] for s in seeds], dtype=float), seeds


# --------------------------------------------------------------------------- WCA
def _wca_dir(raw_dir, arm_map, panel, label, note):
    times, arms, seedmap = None, {}, {}
    for tag, canon in arm_map.items():
        per_seed = {}
        for f in sorted(glob.glob(os.path.join(raw_dir, f"*__{tag}__*.npz"))):
            z = np.load(f, allow_pickle=True)
            s = _seed_of(f)
            if s is None:
                continue
            per_seed[s] = np.asarray(z["l2_f_t"], dtype=float)
            if times is None:
                times = np.asarray(z["times"], dtype=float)
                spec = json.loads(str(z["spec_json"]))
                dt_total = float(times[-1])
                t_fr = dt_total * spec["fr_start_steps"] / spec["n_steps"]
                seedmap["_t_fr"] = t_fr
                seedmap["_ref_label"] = str(z["reference_label"]) if "reference_label" in z.files else ""
        if per_seed:
            arms[canon], sl = _stack(per_seed)
            seedmap[canon] = sl
    return dict(panel=panel, label=label, times=times, arms=arms, seeds=seedmap,
                t_fr=seedmap.get("_t_fr"), x_label="reduced time", y_label=r"$L_2(F)$  [$k_BT$]",
                ref_label=seedmap.get("_ref_label", ""), note=note)


def wca_starved():
    return _wca_dir(
        os.path.join(ROOT, "results/wca_caseix_hp/sham/raw"),
        {"abf": ABF, "fr_estimated": MFR, "fr_oracle": MFR_ORACLE,
         "sham_practical": SHAM, "sham_oracle": SHAM_ORACLE},
        "wca_starved", "WCA dimer, starved cell  (beta=1, h=2)",
        "Confirmatory Case IX at the corrected high-precision reference; 16 paired seeds.")


def wca_five_arm():
    return _wca_dir(
        os.path.join(ROOT, "results/wca_five_arm/confirm/raw"),
        {"five_abf": ABF, "five_fr_estimated": MFR, "five_count_balancing": COUNT_BAL,
         "five_book_laplacian": BOOK_LAP, "five_sham_practical": SHAM},
        "wca_five_arm", "WCA dimer, five-arm mechanism test",
        "Same cell and reference as the flagship. count_balancing is the "
        "non-Fisher-Rao reallocation rule: if it tracks mFR, the gain is not FR-specific.")


# --------------------------------------------------------------------------- gateway
def gateway(init="left"):
    f = os.path.join(ROOT, "results/gateway_anchor/confirmatory_v2/raw.npz")
    d = np.load(f, allow_pickle=True)
    sel_init = d["init"].astype(str) == init
    method = d["method"].astype(str)
    seed = d["seed"].astype(int)
    canon = {"abf": ABF, "fr_estimated": MFR, "fr_oracle": MFR_ORACLE,
             "sham_practical": SHAM, "sham_oracle": SHAM_ORACLE}
    times, arms, seedmap = None, {}, {}
    for tag, c in canon.items():
        m = sel_init & (method == tag)
        if not m.any():
            continue
        order = np.argsort(seed[m])
        arms[c] = np.asarray(d["l2_f_t"][m], dtype=float)[order]
        seedmap[c] = seed[m][order].tolist()
        if times is None:
            times = np.asarray(d["t"][m][0], dtype=float)
    prereg = json.load(open(os.path.join(ROOT, "results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json")))
    t_fr = float(times[-1]) * float(prereg["sampler"]["ramp_fraction"])  # FR ramps in over this
    tag = "primary anchor" if init == "left" else "mechanism control"
    return dict(panel=f"gateway_{init}", label=f"Entropic gateway, init={init}  [{tag}]",
                times=times, arms=arms, seeds=seedmap, t_fr=t_fr,
                x_label="reduced time", y_label=r"$L_2(F)$  [$k_BT$]", ref_label="analytic",
                note="Constructed establishment-limited regime; 32 preregistered fresh seeds. "
                     "FR ramps in over the first 10 % of the run rather than switching on.")


# --------------------------------------------------------------------------- entropic bottleneck
def entropic_bottleneck(beta):
    f = os.path.join(ROOT, "results/entropic_bottleneck/summaries/arrays.npz")
    d = np.load(f)
    stage = "stage3_beta"
    canon = {"abf": ABF, "fr_estimated": MFR}
    times, arms, seedmap = None, {}, {}
    for tag, c in canon.items():
        key = f"{stage}|{tag}|beta{beta}|oin25|gamma15"
        if f"{key}::l2_f_t" not in d.files:
            continue
        arms[c] = np.asarray(d[f"{key}::l2_f_t"], dtype=float)
        seedmap[c] = np.asarray(d[f"{key}::seeds"]).tolist()
        if times is None:
            times = np.asarray(d[f"{key}::t"], dtype=float)
    if not arms:
        return None
    return dict(panel=f"eb_beta{beta}", label=f"Entropic bottleneck, beta={beta}",
                times=times, arms=arms, seeds=seedmap,
                t_fr=float(times[-1]) * 0.10,   # ramp_fraction = 0.10, fr_burnin = 0
                x_label="reduced time", y_label=r"$L_2(F)$  [$k_BT$]", ref_label="analytic",
                note="One model family, one estimator, one plotting convention; only beta changes.")


# --------------------------------------------------------------------------- alkanes
def _alkanes(pattern, arm_map, panel, label, note, y_label=r"$L_2(F)$  [$k_BT$]"):
    times, arms, seedmap, t_fr = None, {}, {}, None
    for tag, c in arm_map.items():
        files = sorted(glob.glob(pattern.format(arm=tag)))
        if not files:
            continue
        z = np.load(files[0], allow_pickle=True)
        arms[c] = np.asarray(z["l2_F_t"], dtype=float)
        seedmap[c] = np.asarray(z["seeds"]).tolist()
        if times is None:
            times = np.asarray(z["times"], dtype=float)
            spec = json.loads(str(z["spec_json"]))
            t_fr = float(times[-1]) * spec["fr_start_steps"] / spec["n_steps"]
    if not arms:
        return None
    return dict(panel=panel, label=label, times=times, arms=arms, seeds=seedmap, t_fr=t_fr,
                x_label="time [ps]", y_label=y_label, ref_label="umbrella/TI reference", note=note)


def butane_phi1():
    return _alkanes(
        os.path.join(ROOT, "results/alkanes/production/raw/b1__butane__{arm}__trans__b1__*.npz"),
        {"abf": ABF, "fr_estimated": MFR, "fr_oracle": MFR_ORACLE,
         "fr_uniform": FR_UNIFORM, "opes": OPES},
        "butane_phi1", "Butane, xi = phi1",
        "ABF is already sufficient here; the FR mechanism nearly switches itself off.")


def pentane_phi1():
    return _alkanes(
        os.path.join(ROOT, "results/alkanes/production/raw/p1__pentane__{arm}__trans__b1__*.npz"),
        {"abf": ABF, "fr_estimated": MFR, "fr_oracle": MFR_ORACLE, "fr_active": MFR_ACTIVE,
         "fr_uniform": FR_UNIFORM, "opes": OPES},
        "pentane_phi1", "Pentane, xi = phi1",
        "Gentle mFR is equivalent; the 'active' arm raises the FR rate and harms a hidden conditional.")


def pentane_r15():
    return _alkanes(
        os.path.join(ROOT, "results/alkanes_cv_extension/r15_methods/raw/production__dist__pentane__{arm}__trans__b2__*.npz"),
        {"abf": ABF, "fr_estimated": MFR, "fr_oracle": MFR_ORACLE, "fr_active": MFR_ACTIVE,
         "fr_uniform": FR_UNIFORM, "opes": OPES},
        "pentane_r15", "Pentane, xi = R15 distance",
        "The only genuinely starved cell in the campaign -- and mFR still fails, oracle included: "
        "reallocation converts a support deficit into a diversity deficit.")


def pentane_2d():
    return _alkanes(
        os.path.join(ROOT, "results/alkanes_cv_extension/2d_methods/raw/production__joint2d__pentane__{arm}__trans__b2__*.npz"),
        {"abf": ABF, "fr_estimated": MFR, "fr_active": MFR_ACTIVE, "opes": OPES},
        "pentane_2d", "Pentane, xi = (phi1, phi2)",
        "Two-dimensional CV; a genuine null.")


# --------------------------------------------------------------------------- 2-D toy
def toy2d():
    """ABF has ONE config; each FR arm has 36. Emit all of them so the selection is visible."""
    f = os.path.join(ROOT, "results/two_dim_xi_x/production_gpu/production_gpu_runs_long.csv")
    # best_configs.csv identifies a config by its HYPERPARAMETERS, not by config_id.
    knobs = ("gamma", "eta", "burnin_fraction", "fr_every")

    def _key(r):
        return tuple(f"{float(r[k]):g}" for k in knobs)

    best = {r["method"]: _key(r) for r
            in csv.DictReader(open(os.path.join(ROOT, "results/two_dim_xi_x/production_gpu/best_configs.csv")))
            if r["selected"] == "True"}
    canon = {"abf_only": ABF, "abf_fr_estimated": MFR, "abf_fr_oracle": MFR_ORACLE,
             "abf_fr_uniform": FR_UNIFORM}
    inv = {v: k for k, v in canon.items()}
    # (canonical arm, config key, seed) -> {t: l2}
    acc, cfg_id = {}, {}
    for r in csv.DictReader(open(f)):
        c = canon.get(r["method"])
        if c is None:
            continue
        k = _key(r)
        cfg_id[(c, k)] = r["config_id"]
        acc.setdefault((c, k), {}).setdefault(int(r["seed"]), {})[float(r["t"])] = float(r["l2_F"])
    all_t = sorted({t for v in acc.values() for s in v.values() for t in s})
    times = np.asarray(all_t, dtype=float)
    arms, seedmap, configs, alt = {}, {}, {}, {}
    for (c, k), per_seed in sorted(acc.items()):
        if k == best.get(inv[c]):
            key = c                                   # the config the study SELECTED
        else:
            alt[c] = alt.get(c, 0) + 1
            key = f"{c}__alt{alt[c]:02d}"             # one of the 35 it did not
        curves = {s: np.asarray([per_seed[s].get(t, np.nan) for t in all_t]) for s in per_seed}
        arms[key], seedmap[key] = _stack(curves)
        configs[key] = dict(zip(knobs, k), config_id=cfg_id[(c, k)])
    return dict(panel="toy2d", label="2-D metastability toy  (xi = x)", times=times, arms=arms,
                seeds=seedmap, t_fr=None, x_label="time", y_label=r"$L_2(F)$",
                ref_label="numerical reference", configs=configs,
                note="SELECTION CAVEAT: each FR arm was tuned over 36 configurations, ABF over 1. "
                     "The unsuffixed FR arm is the SELECTED config; '__cfgN' arms are the other 35. "
                     "A best-of-36 vs best-of-1 contrast is not an unbiased comparison.")


# --------------------------------------------------------------------------- alanine
def alanine():
    """Rebuild eF(t) from the stored FES time series, primary = kernel-matched, equilibrium weight."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    try:
        from alanine.metrics_ala import aligned_l2, build_masks, smooth_reference
    except Exception as e:                                        # torch/mdtraj may be absent
        return dict(panel="alanine", label="Alanine dipeptide (phi, psi)", times=None, arms={},
                    seeds={}, t_fr=None, x_label="time [ps]", y_label=r"$e_F$  [$k_BT$]",
                    ref_label="", note=f"SKIPPED: {type(e).__name__}: {e}")
    ref_f = os.path.join(ROOT, "results/alanine/reference/reference.npz")
    F_ref = np.load(ref_f, allow_pickle=True)["F"]
    meta = json.load(open(os.path.join(ROOT, "results/alanine/reference/meta.json")))
    kT = float(meta.get("kT", meta.get("kT_kJ_per_mol", 2.494)))
    pack = build_masks(F_ref, kT)
    n_grid = F_ref.shape[0]
    F_sm = smooth_reference(F_ref, 0.08, n_grid)
    w = pack["weights"]["equilibrium"]

    sources = {
        ABF:  "results/alanine_oracle/pilot/N4096/raw/N4096__abf__c7eq__*.npz",
        MFR_ORACLE: "results/alanine_oracle/pilot/N4096/raw/N4096__fr_oracle__c7eq__*.npz",
        "mfr_oracle_r015": "results/alanine_oracle/rate_ladder/rate015/raw/rate015__fr_oracle__*.npz",
        "mfr_oracle_r045": "results/alanine_oracle/rate_ladder/rate045/raw/rate045__fr_oracle__*.npz",
    }
    times, arms, seedmap = None, {}, {}
    for c, pat in sources.items():
        files = sorted(glob.glob(os.path.join(ROOT, pat)))
        if not files:
            continue
        z = np.load(files[0], allow_pickle=True)
        pmf = z["pmf"]                                            # (T, R, n, n)
        tt = np.asarray(z["times"], dtype=float)
        curves = np.asarray([[aligned_l2(pmf[ti, r], F_sm, w) for ti in range(pmf.shape[0])]
                             for r in range(pmf.shape[1])], dtype=float)
        arms[c] = curves
        seedmap[c] = np.asarray(z["seeds"]).tolist()
        if times is None:
            times = tt
    if not arms:
        return None
    return dict(panel="alanine", label="Alanine dipeptide, xi = (phi, psi)  [vacuum]",
                times=times, arms=arms, seeds=seedmap, t_fr=20.0,
                x_label="time [ps]", y_label=r"$e_F$ (kernel-matched, eq. weight)  [$k_BT$]",
                ref_label="long-run reference FES",
                note="ORACLE mFR at three FR rates. The atomistic neutrality control: the "
                     "mechanism fires, and the free-energy error does not move.")


PANELS = [
    wca_starved, wca_five_arm,
    lambda: gateway("left"), lambda: gateway("one_right"),
    lambda: entropic_bottleneck(2), lambda: entropic_bottleneck(4),
    lambda: entropic_bottleneck(8), lambda: entropic_bottleneck(12),
    toy2d, butane_phi1, pentane_phi1, pentane_r15, pentane_2d, alanine,
]

# Studies that exist in the campaign but CANNOT appear here, and why. Recorded in the artifact
# so a reader of the atlas alone cannot mistake absence for an oversight.
NO_ARM = {
    "methane_water": "ABF-sufficient at the preregistered screen; no mFR arm was ever licensed.",
    "nacl_water": "ABF-only screen found no persistent population deficit; protocol did not license mFR.",
    "valine": "ABF reaches and establishes every region before mFR could act; stopped at the ABF-only gate.",
    "deca_alanine": "ABF baseline invalid (retracted screen); a contrast against it would be meaningless.",
    "c60_water": "Suspended before any production data; zero-data state preserved.",
    "wca_production_v1": "Scored against the SUPERSEDED reference and stores no profile time series, "
                         "so it can be neither reused nor rescored. Superseded by wca_caseix_hp.",
}


def main():
    os.makedirs(OUT, exist_ok=True)
    blob, meta = {}, []
    for fn in PANELS:
        try:
            p = fn()
        except Exception as e:
            print(f"  !! {fn}: {type(e).__name__}: {e}")
            continue
        if p is None or p.get("times") is None or not p["arms"]:
            nm = p["panel"] if p else str(fn)
            print(f"  -- {nm}: no data ({p.get('note', '') if p else ''})")
            continue
        blob[f"{p['panel']}::times"] = p["times"]
        for arm, curves in p["arms"].items():
            blob[f"{p['panel']}::{arm}"] = curves
        meta.append(dict(panel=p["panel"], label=p["label"], t_fr=p["t_fr"],
                         x_label=p["x_label"], y_label=p["y_label"],
                         ref_label=p.get("ref_label", ""), note=p["note"],
                         n_times=int(len(p["times"])), t_max=float(p["times"][-1]),
                         arms={a: dict(n_seeds=int(c.shape[0]), seeds=p["seeds"].get(a, []))
                               for a, c in p["arms"].items()},
                         configs=p.get("configs")))
        shown = [a for a in p["arms"] if "__alt" not in a]
        extra = len(p["arms"]) - len(shown)
        print(f"  ok {p['panel']:16s} T={len(p['times']):3d}  arms={shown}"
              + (f"  (+{extra} untuned-alternative configs)" if extra else ""))
    np.savez_compressed(os.path.join(OUT, "atlas.npz"), **blob)
    json.dump(dict(panels=meta, excluded=NO_ARM), open(os.path.join(OUT, "atlas.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(OUT, 'atlas.npz')}  ({len(meta)} panels)")


if __name__ == "__main__":
    main()
