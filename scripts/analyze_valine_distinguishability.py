#!/usr/bin/env python
"""The global distinguishability gate: can the full 3-D state be inferred from xi = (phi, chi1)?

This is the replacement for decision-doc gate V2, which became tautological once chi1 was
promoted into the CV.  It is also the *global* version of the sec.32 screen, which tested the
omitted coordinate at six anchors only.

Why it decides something.  Marginal mFR sees exactly ``p(xi)`` and nothing else.  If two
metastable states of ``(phi, psi, chi1)`` project onto overlapping regions of ``(phi, chi1)``,
then a population deficit in one of them is not a resolvable feature of ``p(xi)``: the score
cannot tell which state is starved, so it cannot preferentially clone into it.  Passing this
gate is what makes a V3 deficit *actionable* rather than merely present.

Three measurements, and the first is the one that does not depend on a prior
---------------------------------------------------------------------------
1. **Pairwise overlap** ``OVL(i,j) = sum_z min(p(z|B_i), p(z|B_j))``.  Prior-free: it compares
   the state-conditioned footprints directly, so the non-Boltzmann weighting of the exploration
   cloud cannot bias it.  This is the headline number.
2. **Cross-validated classification accuracy** of the state label from ``(phi, chi1)``.
   Folds are split **by walker, never by frame** -- consecutive frames of one walker are
   strongly correlated, and splitting by frame reports an accuracy inflated toward 100 % that
   says nothing about generalisation.  Reported under a uniform prior (balanced accuracy, which
   is prior-free) and under the empirical prior, because the two answer different questions.
3. **Conditional entropy** ``H(B | phi, chi1)`` in bits, under the uniform prior.

Plus the global ``p(psi | phi, chi1)`` check, which is run on the MBAR-weighted pilot ensemble
when it is available -- that is the only *Boltzmann*-weighted ``(phi, psi, chi1)`` sample the
study has, and mode populations from the relaxation cloud would not be trustworthy.

Usage
-----
    python scripts/analyze_valine_distinguishability.py \
        --state-map results/valine/state_map \
        --pilot results/valine/pilot_reference
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from valine.states import to_cell                                            # noqa: E402
from valine.umbrella import count_states                                     # noqa: E402

KB = 0.008314462618
TWO_PI = 2.0 * math.pi

#: pass rule from the screening plan
ACC_THRESHOLD = 0.80
OVERLAP_THRESHOLD = 0.30


def conditional_histograms(cells, labels, n_states, n, weights=None, alpha=0.5):
    """``p(z | B_k)`` on an ``n x n`` grid, Laplace-smoothed.

    ``alpha`` matters: an unsmoothed histogram assigns probability exactly zero to any cell a
    state never visited, so a single test sample there produces ``log 0`` and the classifier
    becomes undefined rather than merely wrong.
    """
    lin = cells[:, 0] * n + cells[:, 1]
    H = np.zeros((n_states, n * n))
    for k in range(n_states):
        m = labels == k
        H[k] = np.bincount(lin[m], weights=None if weights is None else weights[m],
                           minlength=n * n)
    H += alpha
    return H / H.sum(1, keepdims=True)


def overlap_matrix(P):
    """``OVL(i,j) = sum_z min(p_i, p_j)`` -- 1 means identical, 0 means disjoint support."""
    K = P.shape[0]
    O = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            O[i, j] = O[j, i] = float(np.minimum(P[i], P[j]).sum())
    return O


def cv_classify(cells, labels, walker, n_states, n, n_folds=5, alpha=0.5):
    """Cross-validated Bayes classification of the state from the selected CV.

    Folds are over WALKERS.  Returns overall accuracy, balanced (uniform-prior) accuracy, the
    confusion matrix, and ``H(B | z)`` in bits under the uniform prior.
    """
    uw = np.unique(walker)
    fold_of_walker = {w: k % n_folds for k, w in enumerate(uw)}
    fold = np.array([fold_of_walker[w] for w in walker])
    conf = np.zeros((n_states, n_states), dtype=np.int64)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if not te.any() or not tr.any():
            continue
        P = conditional_histograms(cells[tr], labels[tr], n_states, n, alpha=alpha)
        prior = np.bincount(labels[tr], minlength=n_states).astype(float)
        prior = np.maximum(prior, 1.0)
        prior /= prior.sum()
        lin = cells[te, 0] * n + cells[te, 1]
        post = np.log(P[:, lin]) + np.log(prior)[:, None]
        np.add.at(conf, (labels[te], post.argmax(0)), 1)
    acc = float(np.trace(conf) / max(conf.sum(), 1))
    per = np.divide(np.diag(conf), conf.sum(1), out=np.zeros(n_states),
                    where=conf.sum(1) > 0)
    bal = float(per[conf.sum(1) > 0].mean())

    # H(B | z) under a uniform prior, from the full-sample conditionals
    P = conditional_histograms(cells, labels, n_states, n, alpha=alpha)
    joint = P / n_states                                     # uniform prior over states
    pz = joint.sum(0)
    ok = pz > 0
    post = joint[:, ok] / pz[ok]
    Hcond = float(-(pz[ok] * (post * np.log2(np.clip(post, 1e-300, None))).sum(0)).sum())
    return dict(accuracy=acc, balanced_accuracy=bal, per_state_recall=per.tolist(),
                confusion=conf.tolist(), H_given_cv_bits=Hcond,
                H_prior_bits=float(math.log2(n_states)))


def hidden_psi_modes(phi, psi, chi1, logw, n_cv=18, n_psi=36, kT=1.0, sep_kT=3.0,
                     min_pop=0.05, min_eff=40.0):
    """Count metastable psi modes inside every populated ``(phi, chi1)`` cell.

    The candidate CV fails if a *substantial* part of the populated selected-CV plane holds two
    slowly interconverting psi states -- there ABF would average its mean force over basins that
    do not interconvert, and marginal mFR could not even see the problem.

    ``logw`` are MBAR log weights, so mode populations are Boltzmann.  Cells whose effective
    sample size is below ``min_eff`` are reported as undetermined rather than as single-moded:
    an under-sampled cell trivially shows one mode, which would bias the verdict toward PASS.
    """
    beta = 1.0 / kT
    ci = to_cell(phi, n_cv)
    cj = to_cell(chi1, n_cv)
    w = np.exp(logw - logw.max())
    lin = ci * n_cv + cj
    edges = np.linspace(-math.pi, math.pi, n_psi + 1)
    pb = np.clip(np.digitize((psi + math.pi) % TWO_PI - math.pi, edges) - 1, 0, n_psi - 1)

    cell_w = np.bincount(lin, weights=w, minlength=n_cv * n_cv)
    total_w = cell_w.sum()
    rows = []
    for c in np.flatnonzero(cell_w > 0):
        m = lin == c
        ww = w[m]
        n_eff = float(ww.sum() ** 2 / np.maximum((ww ** 2).sum(), 1e-300))
        h = np.bincount(pb[m], weights=ww, minlength=n_psi)
        with np.errstate(divide="ignore"):
            F = -np.log(np.where(h > 0, h, np.nan)) / beta
        F -= np.nanmin(F)
        st = count_states(F, beta, kT, sep_kT=sep_kT, min_pop=min_pop)
        rows.append(dict(cell=int(c), phi_bin=int(c // n_cv), chi_bin=int(c % n_cv),
                         weight=float(cell_w[c] / total_w), n_eff=n_eff,
                         n_psi_states=int(st["n_states"]) if n_eff >= min_eff else -1,
                         populations=st["populations"]))
    multi = sum(r["weight"] for r in rows if r["n_psi_states"] >= 2)
    undet = sum(r["weight"] for r in rows if r["n_psi_states"] < 0)
    return dict(n_cells=len(rows), n_cv=n_cv, n_psi_bins=n_psi,
                weight_multi_modal=float(multi), weight_undetermined=float(undet),
                worst=sorted([r for r in rows if r["n_psi_states"] >= 2],
                             key=lambda r: -r["weight"])[:10])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-map", default="results/valine/state_map")
    ap.add_argument("--pilot", default="results/valine/pilot_reference")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cv-cells", type=int, default=36, help="grid for p(phi,chi1|B_k)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=300.0)
    a = ap.parse_args()
    out_dir = a.out or a.state_map
    kT = KB * a.temperature

    st = np.load(os.path.join(a.state_map, "states.npz"), allow_pickle=True)
    ex = np.load(os.path.join(a.state_map, "explore.npz"), allow_pickle=True)
    sm_meta = json.load(open(os.path.join(a.state_map, "meta.json")))
    theta = ex["theta"]                                    # (W, T, 3) phi, psi, chi1
    lab = st["frame_labels"]                               # (W, T)
    n_states = int(st["centres"].shape[0])
    W, T = lab.shape
    print(f"state map: {n_states} states, {W} walkers x {T} frames")

    keep = lab >= 0
    walker = np.repeat(np.arange(W), T)[keep.reshape(-1)]
    labels = lab.reshape(-1)[keep.reshape(-1)].astype(np.int64)
    ang = theta.reshape(-1, 3)[keep.reshape(-1)].astype(np.float64)
    cells = np.stack([to_cell(ang[:, 0], a.cv_cells), to_cell(ang[:, 2], a.cv_cells)], 1)
    print(f"assigned frames: {labels.size:,} ({100 * labels.size / lab.size:.1f} %)")

    # ---------------------------------------------------------------- 1. overlap (prior-free)
    P = conditional_histograms(cells, labels, n_states, a.cv_cells)
    O = overlap_matrix(P)
    iu = np.triu_indices(n_states, 1)
    worst = int(np.argmax(O[iu]))
    print(f"\npairwise (phi,chi1) footprint overlap: max {O[iu].max():.3f} "
          f"(states B{iu[0][worst]}-B{iu[1][worst]}), median {np.median(O[iu]):.3f}")
    bad = [(int(i), int(j), float(O[i, j])) for i, j in zip(*iu) if O[i, j] > OVERLAP_THRESHOLD]
    for i, j, v in sorted(bad, key=lambda t: -t[2])[:8]:
        print(f"    B{i} vs B{j}: OVL {v:.3f}")

    # ---------------------------------------------------------------- 2-3. classifier
    cls = cv_classify(cells, labels, walker, n_states, a.cv_cells, n_folds=a.folds)
    print(f"\ncross-validated (by walker, {a.folds} folds) state recovery from (phi, chi1):")
    print(f"    accuracy           {cls['accuracy']:.4f}")
    print(f"    balanced accuracy  {cls['balanced_accuracy']:.4f}")
    print(f"    H(B | phi,chi1)    {cls['H_given_cv_bits']:.3f} bits  "
          f"(prior H(B) = {cls['H_prior_bits']:.3f} bits, uniform)")
    lo = np.argsort(cls["per_state_recall"])[:5]
    print("    weakest states: " + ", ".join(
        f"B{int(k)} {cls['per_state_recall'][int(k)]:.2f}" for k in lo))

    # ---------------------------------------------------------------- 4. hidden psi, globally
    psi_res, psi_source = None, "none"
    pilot_npz = os.path.join(a.pilot, "pilot_reference.npz")
    if os.path.exists(pilot_npz):
        pf = np.load(pilot_npz, allow_pickle=True)
        if "mbar_logw" in pf.files:
            psi_source = "pilot MBAR (Boltzmann-weighted)"
            psi_res = hidden_psi_modes(pf["mbar_phi"].astype(np.float64),
                                       pf["mbar_psi"].astype(np.float64),
                                       pf["mbar_chi1"].astype(np.float64),
                                       pf["mbar_logw"].astype(np.float64), kT=kT)
    if psi_res is None:
        # Fall back to the relaxation cloud.  Mode EXISTENCE is still meaningful there; mode
        # POPULATIONS are not, so this is explicitly labelled and is not used for a verdict.
        psi_source = "state-map relaxation cloud (NOT Boltzmann-weighted -- existence only)"
        psi_res = hidden_psi_modes(ang[:, 0], ang[:, 1], ang[:, 2],
                                   np.zeros(ang.shape[0]), kT=kT)
    print(f"\nglobal p(psi | phi, chi1) [{psi_source}]:")
    print(f"    selected-CV weight with >=2 metastable psi states: "
          f"{psi_res['weight_multi_modal']:.4f}")
    print(f"    weight undetermined (too few effective samples):   "
          f"{psi_res['weight_undetermined']:.4f}")

    # ---------------------------------------------------------------- verdict
    pass_acc = cls["balanced_accuracy"] > ACC_THRESHOLD
    pass_ovl = O[iu].max() <= OVERLAP_THRESHOLD if iu[0].size else True
    pass_psi = psi_res["weight_multi_modal"] < 0.10
    verdict = "PASS" if (pass_acc and pass_ovl and pass_psi) else "FAIL"
    print(f"\nGATE distinguishability ({verdict})")
    print(f"    balanced accuracy > {ACC_THRESHOLD}:            "
          f"{pass_acc}  ({cls['balanced_accuracy']:.4f})")
    print(f"    no pair overlapping > {OVERLAP_THRESHOLD}:          "
          f"{pass_ovl}  (max {O[iu].max():.3f})" if iu[0].size else "    (single state)")
    print(f"    <10 % of selected-CV weight bimodal in psi: "
          f"{pass_psi}  ({psi_res['weight_multi_modal']:.4f})")
    if verdict == "PASS":
        print("  -> a V3 population deficit would be VISIBLE to a marginal FR score.")
    else:
        print("  -> mFR could not selectively repair a deficit even though chi1 is in the CV.")

    res = dict(state_map=a.state_map, n_states=n_states, cv_cells=a.cv_cells,
               n_assigned_frames=int(labels.size),
               overlap_max=float(O[iu].max()) if iu[0].size else 0.0,
               overlap_median=float(np.median(O[iu])) if iu[0].size else 0.0,
               overlap_pairs_above_threshold=bad, overlap_matrix=O.tolist(),
               classification=cls, hidden_psi=psi_res, hidden_psi_source=psi_source,
               thresholds=dict(balanced_accuracy=ACC_THRESHOLD, overlap=OVERLAP_THRESHOLD,
                               psi_multimodal_weight=0.10),
               gate=dict(accuracy=bool(pass_acc), overlap=bool(pass_ovl),
                         hidden_psi=bool(pass_psi), verdict=verdict),
               state_centres_deg=np.degrees(st["centres"]).round(1).tolist(),
               state_map_config=sm_meta.get("clustering"))
    path = os.path.join(out_dir, "distinguishability.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
