"""Score the WCA IO-ABF runs post hoc against the current reference, and report.

The sampler stores ``pmf`` at every save, so the free-energy error curve is
recovered without re-running dynamics -- which is the whole point of that
storage decision, and why a future reference change costs nothing here.

**The reference is `cache/phase_hp_v3`, never the default cache.** The default
`cache/wca_ti_reference.npz` is the build the Stage-A audit found wrong by 24.8
sigma at z = 0.255; loading it here would silently score every arm against a
known-defective curve.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import wca_abffr_core as core                                    # noqa: E402
from scipy.stats import spearmanr                                # noqa: E402

OUT = os.path.join(ROOT, "results", "io_abf_overnight", "wca")
REF = os.path.join(ROOT, "cache", "phase_hp_v3",
                   "wca_ti_b1_h2_w2_n10_a1.5_g160.npz")


def reference():
    with np.load(REF, allow_pickle=True) as d:
        return {"grid": d["grid"], "free_energy": d["free_energy"],
                "mean_force": d["mean_force"], "label": str(d["label"])}


def score(phase="screening"):
    sim = core.SimConfig()
    ref = reference()
    grid = ref["grid"]
    emask = core.eval_window_mask_np(grid, sim)
    full = np.ones_like(emask, dtype=bool)
    files = sorted(glob.glob(os.path.join(OUT, phase, "*.npz")))
    if not files:
        raise SystemExit(f"no records in {os.path.join(OUT, phase)}")
    recs = []
    for p in files:
        with np.load(p, allow_pickle=True) as d:
            r = {k: d[k] for k in d.files}
        pmf = np.asarray(r["pmf"], dtype=float)            # (n_saves, G)
        e = np.array([core.profile_l2_error_np(
            core.align_additive_constant_np(pmf[i], ref["free_energy"], grid, emask),
            ref["free_energy"], grid, emask) for i in range(pmf.shape[0])])
        ef = np.array([core.profile_l2_error_np(
            core.align_additive_constant_np(pmf[i], ref["free_energy"], grid, full),
            ref["free_energy"], grid, full) for i in range(pmf.shape[0])])
        r["l2_f_t"], r["l2_f_full_t"] = e, ef
        r["t"] = np.asarray(r["times"], dtype=float)
        recs.append(r)
    return recs, emask, grid


def report(phase="screening"):
    recs, emask, grid = score(phase)
    arm = str(recs[0].get("io_arm", "A0"))
    t = recs[0]["t"]
    E = np.stack([r["l2_f_t"] for r in recs])
    EF = np.stack([r["l2_f_full_t"] for r in recs])
    T = float(t[-1])

    a_cell = recs[0]["io_a_cell"]
    scored = a_cell > 0

    def spread(name):
        v = np.stack([r[f"io_{name}"] for r in recs])[:, scored]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size < 4:
            return dict(q10=float("nan"), q90=float("nan"), ratio=float("nan"))
        q10, q90 = float(np.quantile(v, .1)), float(np.quantile(v, .9))
        return dict(q10=q10, q90=q90, ratio=q90 / max(q10, 1e-300))

    tau = np.stack([r["io_tau"] for r in recs])[:, scored]
    valid = float(np.mean(np.isfinite(tau) & (tau > 0)))

    g_t = np.stack([r["io_gamma_t"] for r in recs])
    sp = float("nan")
    if g_t.ndim == 3 and g_t.shape[1] >= 4:
        n = g_t.shape[1]
        a_ = np.log(np.maximum(g_t[:, :n // 3].mean(1)[:, scored], 1e-300))
        b_ = np.log(np.maximum(g_t[:, -n // 3:].mean(1)[:, scored], 1e-300))
        sp = float(np.median([spearmanr(a_[i], b_[i]).statistic
                              for i in range(a_.shape[0])]))

    s2, tt, gg = spread("sigma2"), spread("tau"), spread("gamma")
    out = dict(
        phase=phase, arm=arm, n_seeds=len(recs), T_total=T,
        reference=os.path.relpath(REF, ROOT), reference_label=reference()["label"],
        n_cells=int(a_cell.size), n_scored_cells=int(scored.sum()),
        obs_every=int(recs[0].get("io_obs_every", -1)),
        eps1=float(np.median(E[:, int(np.argmin(abs(t - 0.4 * T)))])),
        eps2=float(np.median(E[:, int(np.argmin(abs(t - 0.6 * T)))])),
        final_l2_f_median=float(np.median(E[:, -1])),
        final_l2_f_full_median=float(np.median(EF[:, -1])),
        valid_tau_fraction=valid, gamma_unresolved=bool(valid < 0.80),
        R_sigma2=s2, R_tau=tt, R_gamma=gg,
        spearman_gamma_early_late=sp,
        dominant_source=("sigma2" if s2["ratio"] > 3 * tt["ratio"]
                         else "tau" if tt["ratio"] > 3 * s2["ratio"] else "both"))
    os.makedirs(os.path.join(OUT, "analysis"), exist_ok=True)
    with open(os.path.join(OUT, "analysis", f"{phase}_report.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    np.savez_compressed(os.path.join(OUT, "analysis", f"{phase}_curves.npz"),
                        t=t, l2_f=E, l2_f_full=EF,
                        sigma2=np.stack([r["io_sigma2"] for r in recs]),
                        tau=np.stack([r["io_tau"] for r in recs]),
                        gamma=np.stack([r["io_gamma"] for r in recs]),
                        a_cell=a_cell, cell_edges=recs[0]["io_cell_edges"])
    print(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="screening")
    report(ap.parse_args().phase)
