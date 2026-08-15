"""Reference consensus, acceptance, reproduction gate, states, Gate A -- SPEC §5/§13.

Consumes results/c60/reference/build{1,2,3}/windows.npz (+ spotcheck_openmm.json when
present) and writes results/c60/reference/{consensus.npz, RESULT.md, result.json}.

Refuses to run acceptance on fewer than 3 builds -- a partial reference is reported as
partial, never scored (the campaign's partial-block rule).

Everything decided here comes from the reference alone: no ABF datum exists yet.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from c60 import system as csys  # noqa: E402

REF = os.path.join(os.path.dirname(__file__), "..", "results", "c60", "reference")

KT = csys.kT_kJ()
BETA = 1.0 / KT

# ---- frozen thresholds (SPEC §5, §13; Amendment 3) ------------------------------------------
R_REF_MAX = 0.5
CONTACT_WINDOW_NM = (0.94, 1.00)
DELTA_G_TARGET_KJ = -16.0
DELTA_G_TOL_KJ = 3.0
BARRIER_MIN_KJ = 1.0
BARRIER_SEARCH_HI_NM = 1.40
MERGE_KT = 2.0
GATE_A_TV_MIN = 0.30
MIN_SAMPLES_PER_LABEL = 12
THERMAL_KT = 15.0


def load_build(k):
    path = os.path.join(REF, f"build{k}", "windows.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)


JAM_FORCE = 1.0e4       #: per-site force above this at production start = wedged water


def window_stats(bz):
    """Per-window pooled mean force, block SEM, and family means/spread.

    Replicas with a jammed water (per-site force > JAM_FORCE at the equilibration boundary,
    recorded by the driver) are EXCLUDED and counted -- a wedged water biases f by its full
    ~1e4 kJ/mol/nm cage-separating force and no amount of averaging dilutes it."""
    d_grid = bz["d_grid"]
    fam = bz["family"]
    fb = np.array(bz["f_block_means"], dtype=np.float64)      # (816, n_blocks)
    if "max_force_post_equil" in bz.files:
        jammed = np.asarray(bz["max_force_post_equil"]) > JAM_FORCE
        fb[jammed] = np.nan
    n_w = len(d_grid)
    mean = np.zeros(n_w); sem = np.zeros(n_w)
    fam_means = np.zeros((n_w, 4)); fam_spread = np.zeros(n_w)
    n_jammed = 0
    for w in range(n_w):
        rows = slice(w * 12, (w + 1) * 12)
        blocks = fb[rows]                          # (12, n_blocks)
        rep_means = np.nanmean(blocks, axis=1)
        ok = np.isfinite(rep_means)
        n_jammed += int((~ok).sum())
        mean[w] = np.nanmean(rep_means)
        sem[w] = (np.nanstd(rep_means[ok], ddof=1) / np.sqrt(max(1, ok.sum()))
                  if ok.sum() > 1 else np.nan)
        for f in range(4):
            fam_rows = fb[w * 12 + f * 3: w * 12 + (f + 1) * 3]
            fam_means[w, f] = np.nanmean(fam_rows)
        fam_spread[w] = np.nanmax(fam_means[w]) - np.nanmin(fam_means[w])
    if n_jammed:
        print(f"[window_stats] excluded {n_jammed} jammed replicas (> {JAM_FORCE:.0e})")
    return d_grid, mean, sem, fam_means, fam_spread


def integrate_F(d_grid, fprime):
    """F(d) = -int_d^{top} F'(s) ds, anchored F(top) = 0 (the paper's convention)."""
    F = np.zeros_like(fprime)
    for i in range(len(d_grid) - 2, -1, -1):
        F[i] = F[i + 1] - 0.5 * (fprime[i] + fprime[i + 1]) * (d_grid[i + 1] - d_grid[i])
    return F


def find_states(d_grid, F):
    """Amendment 3: minima, merge across < 2 kT barriers (from the higher minimum),
    boundaries at intervening maxima; tercile fallback."""
    n = len(F)
    minima = [i for i in range(1, n - 1) if F[i] <= F[i - 1] and F[i] <= F[i + 1]]
    if not minima:
        minima = [int(np.argmin(F))]
    # iterative merge
    while len(minima) > 1:
        merged = False
        for a, b in zip(minima[:-1], minima[1:]):
            barrier = F[a:b + 1].max()
            if barrier - max(F[a], F[b]) < MERGE_KT * KT:
                keep = a if F[a] <= F[b] else b
                minima = [m for m in minima if m not in (a, b)] + [keep]
                minima.sort()
                merged = True
                break
        if not merged:
            break
    if len(minima) < 2:
        edges = np.linspace(d_grid[0], d_grid[-1], 4)
        return [(edges[k], edges[k + 1]) for k in range(3)], minima, "tercile-fallback"
    bounds = [d_grid[0]]
    for a, b in zip(minima[:-1], minima[1:]):
        bounds.append(d_grid[a + int(np.argmax(F[a:b + 1]))])
    bounds.append(d_grid[-1])
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)], minima, "minima"


def main():
    builds = {k: load_build(k) for k in (1, 2, 3)}
    present = [k for k, v in builds.items() if v is not None]
    out = dict(builds_present=present)
    os.makedirs(REF, exist_ok=True)

    stats = {k: window_stats(builds[k]) for k in present}
    Fs = {}
    for k in present:
        d_grid, mean, sem, fam_means, fam_spread = stats[k]
        Fs[k] = integrate_F(d_grid, mean)

    if len(present) < 3:
        out["status"] = f"PARTIAL ({len(present)}/3 builds) -- acceptance NOT evaluated"
        with open(os.path.join(REF, "result.json"), "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        print(out["status"])
        return

    d_grid = stats[present[0]][0]
    F_all = np.stack([Fs[k] for k in present])
    F_cons = F_all.mean(axis=0)
    fprime_cons = np.stack([stats[k][1] for k in present]).mean(axis=0)
    fam_means_all = np.stack([stats[k][3] for k in present])   # (3, n_w, 4)
    fam_spread_cons = fam_means_all.mean(axis=0).max(axis=1) - \
        fam_means_all.mean(axis=0).min(axis=1)
    sem_cons = np.stack([stats[k][2] for k in present]).mean(axis=0) / np.sqrt(3)

    # ---- acceptance -------------------------------------------------------------------------
    span = float(F_cons.max() - F_cons.min())
    pair_l2 = max(float(np.sqrt(np.mean((F_all[i] - F_all[j]) ** 2)))
                  for i in range(3) for j in range(i + 1, 3))
    R_ref = pair_l2 / (0.10 * span)
    out["acceptance"] = dict(span_kJ=span, max_pairwise_L2=pair_l2, R_ref=R_ref,
                             passes=bool(R_ref <= R_REF_MAX))

    # ---- reproduction gate (SPEC §5 i,ii,iv; iii is a unit test) ----------------------------
    i_min = int(np.argmin(F_cons))
    d_contact = float(d_grid[i_min])
    dG = float(F_cons[i_min] - F_cons[-1])
    in_range = (d_grid > d_contact) & (d_grid <= BARRIER_SEARCH_HI_NM)
    barrier_height = float(F_cons[in_range].max() - F_cons[-1]) if in_range.any() else 0.0
    repro = dict(
        contact_min_nm=d_contact,
        contact_in_window=bool(CONTACT_WINDOW_NM[0] <= d_contact <= CONTACT_WINDOW_NM[1]),
        delta_G_kJ=dG,
        delta_G_ok=bool(abs(dG - DELTA_G_TARGET_KJ) <= DELTA_G_TOL_KJ),
        barrier_height_kJ=barrier_height,
        barrier_ok=bool(barrier_height >= BARRIER_MIN_KJ),
    )
    repro["passes"] = bool(repro["contact_in_window"] and repro["delta_G_ok"]
                           and repro["barrier_ok"])
    out["reproduction_gate"] = repro

    # ---- states (Amendment 3) ---------------------------------------------------------------
    states, minima_idx, rule = find_states(d_grid, F_cons)
    out["states"] = dict(rule=rule, boundaries=[list(s) for s in states],
                         minima_nm=[float(d_grid[i]) for i in minima_idx])

    # ---- Amendment 16.7/16.8: the lambda table, BEFORE any occupancy exists -----------------
    # lambda_k(N) = N * Q*_k, bracketed by the unbiased reference weights (B = 0, the screen's
    # t = 0 target) and the fully flattened bias (width fractions, the late-time target).
    # Recorded here, at reference acceptance, so the partition and its power are frozen
    # mechanically before any screen cell can launch.  A state below 16 at some N cannot carry
    # a headline Gate C deficit there (16.7); N in {8, 16} cannot classify at all (16.8).
    w_boltz = np.exp(-BETA * (F_cons - F_cons.min()))
    w_boltz /= w_boltz.sum()
    lam = {}
    for k, (a, b) in enumerate(states):
        m = (d_grid >= a) & (d_grid <= b)
        q_unbiased = float(w_boltz[m].sum())
        q_flat = float((b - a) / (d_grid[-1] - d_grid[0]))
        lam[f"state{k}"] = {
            "bounds_nm": [float(a), float(b)],
            "Q_unbiased": q_unbiased, "Q_flat": q_flat,
            **{f"lambda_N{N}": dict(unbiased=N * q_unbiased, flat=N * q_flat,
                                    clears_16=bool(min(N * q_unbiased, N * q_flat) >= 16.0))
               for N in (8, 16, 32, 64)}}
    out["lambda_table"] = lam
    out["executable_cells"] = dict(
        note="Amendment 16.8: N in {8,16} struck (Q*<=1 makes the 16.7 floor unsatisfiable); "
             "N=32 runs only if N=64 is establishment-limited",
        N64_classifiable_states=[k for k, v in lam.items()
                                 if v["lambda_N64"]["clears_16"]],
        N32_classifiable_states=[k for k, v in lam.items()
                                 if v["lambda_N32"]["clears_16"]])

    # ---- thermal window ---------------------------------------------------------------------
    mask = F_cons - F_cons.min() <= THERMAL_KT * KT
    # largest contiguous interval containing the argmin
    lo = i_min
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    hi = i_min
    while hi < len(mask) - 1 and mask[hi + 1]:
        hi += 1
    out["omega_thermal_nm"] = [float(d_grid[lo]), float(d_grid[hi])]

    # ---- Gate A (corrected orientation) + R_orth --------------------------------------------
    # joint samples: (window d, n_gap) reweighted by p(d) ~ exp(-beta F) within the thermal
    # window; labels = n_gap terciles of the weighted sample
    xi_s, ng_s, w_s = [], [], []
    for k in present:
        bz = builds[k]
        ng = bz["ngap"]                        # (n_snap, 816)
        dv = bz["d_values"]
        for w in range(len(d_grid)):
            if not (lo <= w <= hi):
                continue
            rows = slice(w * 12, (w + 1) * 12)
            vals = ng[:, rows].reshape(-1)
            xi_s.append(np.full(vals.shape, d_grid[w]))
            ng_s.append(vals)
            pw = np.exp(-BETA * (F_cons[w] - F_cons.min()))
            w_s.append(np.full(vals.shape, pw / max(1, len(vals))))
    xi_s = np.concatenate(xi_s); ng_s = np.concatenate(ng_s); w_s = np.concatenate(w_s)

    # R_orth: across-d sd of E[n_gap|d] vs mean within-d sd (unweighted across windows)
    e_by_d, sd_by_d = [], []
    for w in range(lo, hi + 1):
        m = xi_s == d_grid[w]
        e_by_d.append(ng_s[m].mean())
        sd_by_d.append(ng_s[m].std(ddof=1))
    R_orth = float(np.std(e_by_d, ddof=1) / np.mean(sd_by_d))
    out["R_orth"] = R_orth

    q1, q2 = np.quantile(ng_s, [1 / 3, 2 / 3])   # unweighted terciles of the sampled n_gap
    labels = np.digitize(ng_s, [q1, q2])
    edges = np.linspace(d_grid[lo], d_grid[hi], 41)
    tv_max, tv_detail = 0.0, {}
    hists = {}
    for lab in range(3):
        m = labels == lab
        if m.sum() < MIN_SAMPLES_PER_LABEL:
            hists[lab] = None
            continue
        h, _ = np.histogram(xi_s[m], bins=edges, weights=w_s[m])
        tot = h.sum()
        hists[lab] = h / tot if tot > 0 else None
    for a in range(3):
        for b in range(a + 1, 3):
            if hists.get(a) is None or hists.get(b) is None:
                tv_detail[f"{a}-{b}"] = "NOT COMPUTABLE"
                continue
            tv = 0.5 * float(np.abs(hists[a] - hists[b]).sum())
            tv_detail[f"{a}-{b}"] = tv
            tv_max = max(tv_max, tv)
    out["gate_A"] = dict(statistic="max TV(p(xi|Y=a), p(xi|Y=b)), Y = n_gap terciles",
                         tv=tv_detail, tv_max=tv_max,
                         passes=bool(tv_max >= GATE_A_TV_MIN), R_orth_caveat=R_orth)

    # ---- Gate 0 preview from family spread (binding Gate 0 = the §6 pools) ------------------
    denom = np.maximum(np.abs(fprime_cons), 1e-9)
    g_prof = fam_spread_cons / denom
    core = (d_grid >= d_contact) & (d_grid <= BARRIER_SEARCH_HI_NM)
    out["gate0_preview"] = dict(
        mean_spread_over_Fprime=float(np.mean(g_prof[np.abs(fprime_cons) > 1.0])),
        barrier_region_max=float(g_prof[core].max()) if core.any() else None,
        note="preview from 4x3 reference families; binding Gate 0 is the 32-replica pools")

    # ---- OpenMM spot-check ------------------------------------------------------------------
    spot_path = os.path.join(REF, "spotcheck_openmm.json")
    if os.path.exists(spot_path):
        with open(spot_path) as fh:
            spots = json.load(fh)["results"]
        checks = []
        for key, s in spots.items():
            w = int(np.argmin(np.abs(d_grid - s["d_nm"])))
            diff = abs(s["mean"] - fprime_cons[w])
            comb = float(np.sqrt(s["block_sem"] ** 2 + sem_cons[w] ** 2))
            checks.append(dict(d_nm=s["d_nm"], openmm=s["mean"], torch=float(fprime_cons[w]),
                               diff=diff, comb_sem=comb, ok=bool(diff <= 3.0 * comb)))
        out["spotcheck"] = dict(points=checks, passes=bool(all(c["ok"] for c in checks)))
    else:
        out["spotcheck"] = "MISSING -- acceptance incomplete without it"

    np.savez(os.path.join(REF, "consensus.npz"), d_grid=d_grid, F=F_cons,
             Fprime=fprime_cons, sem=sem_cons, F_builds=F_all,
             fam_spread=fam_spread_cons)
    with open(os.path.join(REF, "result.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    lines = [f"# C60 reference analysis (SPEC §5)\n",
             f"* acceptance: R_ref = {R_ref:.4f} (<= {R_REF_MAX}) -> "
             f"{'PASS' if out['acceptance']['passes'] else 'FAIL'}",
             f"* reproduction: contact {d_contact:.3f} nm in {CONTACT_WINDOW_NM}: "
             f"{repro['contact_in_window']}; dG {dG:.2f} kJ/mol vs {DELTA_G_TARGET_KJ}"
             f"+-{DELTA_G_TOL_KJ}: {repro['delta_G_ok']}; barrier {barrier_height:.2f} kJ/mol "
             f">= {BARRIER_MIN_KJ}: {repro['barrier_ok']} -> "
             f"{'PASS' if repro['passes'] else 'FAIL'}",
             f"* states ({rule}): {out['states']['boundaries']}",
             f"* lambda table (16.7/16.8, frozen at acceptance): " + "; ".join(
                 f"{k}: N64 {v['lambda_N64']['unbiased']:.1f}/{v['lambda_N64']['flat']:.1f} "
                 f"(clears16={v['lambda_N64']['clears_16']})" for k, v in lam.items()),
             f"* executable cells: {out['executable_cells']}",
             f"* Omega_thermal: {out['omega_thermal_nm']} nm",
             f"* Gate A: tv_max = {tv_max:.3f} (>= {GATE_A_TV_MIN}) -> "
             f"{'PASS' if out['gate_A']['passes'] else 'FAIL'}   R_orth = {R_orth:.2f}",
             f"* Gate 0 preview: {out['gate0_preview']}",
             f"* spot-check: {out['spotcheck'] if isinstance(out['spotcheck'], str) else ('PASS' if out['spotcheck']['passes'] else 'FAIL')}"]
    with open(os.path.join(REF, "RESULT.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
