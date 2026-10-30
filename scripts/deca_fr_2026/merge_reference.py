"""Merge the independent umbrella builds into the 2026-10 deca-alanine reference.

    python scripts/deca_fr_2026/merge_reference.py --dir results/deca_fr_2026/reference

Produces ``reference.npz`` (grid, F_ref, F_builds, F'_ref from conditional mean forces,
F'_builds, per-checkpoint F for drift, the weighted joint of (R, Y) and of (R, n_hbonds),
(R, alpha_frac) for the conditional-fidelity metric) and ``summary.json`` with the acceptance
statistics, the comparison against the recorded 2026-08 numbers, and Gate A.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                                          # noqa: E402
from deca.labels import conditional_tv                                     # noqa: E402
from deca.umbrella import UmbrellaConfig, mbar_weights, pmf_from_weights, window_centers  # noqa: E402
from deca.umbrella import POOL_PRESETS                                        # noqa: E402

KB = 0.008314462618
EFFECT_SIZE_PCT = 10.0
OLD = dict(span_kJ=179.59878070490453, pairwise_l2_max=0.6044857339732211, drift_l2_kJ=3.1810647541490846,
           argmin_nm=1.6372093023255814, gate_a_tv=0.753736636252673)
HB_VALUES = np.arange(0, 8)                       # i->i+4 hydrogen bonds: 0..7 possible
AF_EDGES = np.linspace(0.0, 1.0, 11)              # alpha fraction deciles


def l2(a, b, mask, dz):
    d = (a - a[mask].mean()) - (b - b[mask].mean())
    return float(np.sqrt((d[mask] ** 2).sum() * dz))


def weighted_joint(xi, w, lab, edges_lab, grid, lo, hi):
    """Weighted 2-D histogram of (R cell, label bin) on the evaluation grid; drops out-of-domain."""
    n = grid.size
    dz = (hi - lo) / n
    inside = (xi >= lo) & (xi < hi)
    cell = np.clip(((xi[inside] - lo) / dz).astype(int), 0, n - 1)
    lb = np.clip(np.digitize(lab[inside], edges_lab) - 1, 0, len(edges_lab) - 2)
    J = np.zeros((len(edges_lab) - 1, n))
    np.add.at(J, (lb, cell), w[inside])
    return J


def preset_audit(grid, audit_s, audit_c, intervals=((2.5, 2.7), (2.7, 2.9), (2.9, 3.1), (3.1, 3.3), (3.3, 3.5))):
    """Conditional mean force per starting preset on coarse R intervals (kJ/mol/nm)."""
    out = {}
    for lo, hi in intervals:
        m = (grid >= lo) & (grid < hi)
        out[f"{lo}-{hi}"] = [float(audit_s[p][m].sum() / max(audit_c[p][m].sum(), 1.0)) for p in range(audit_s.shape[0])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/deca_fr_2026/reference")
    ap.add_argument("--raw-dir", default=None, help="where build*.npz live (default <dir>/raw)")
    ap.add_argument("--out", default=None, help="where reference.npz / summary.json go (default <dir>)")
    ap.add_argument("--exclude-presets", type=int, nargs="*", default=[],
                    help="drop every replica that started from these diverse_pool presets (0-5)")
    args = ap.parse_args()
    raw_dir = args.raw_dir or os.path.join(args.dir, "raw")
    out_dir = args.out or args.dir
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(raw_dir, "build[0-9].npz")))
    if not files:
        raise SystemExit("no build files")
    n_presets = len(POOL_PRESETS)
    audit_s = np.zeros((n_presets, 0)); audit_c = np.zeros((n_presets, 0)); n_excl_preset = 0
    cfg0 = json.loads(str(np.load(files[0], allow_pickle=True)["config_json"]))
    cfg = UmbrellaConfig(**cfg0)
    cen = window_centers(cfg)
    beta = cfg.beta
    kT = KB * cfg.temperature

    Fs, Fps, cnts, joints_y, joints_hb, joints_af = [], [], [], [], [], []
    xi_cat, w_cat, y_cat = [], [], []
    ckpt_F = {}                                   # ns -> list of per-build F
    info_builds = []
    for fpath in files:
        z = np.load(fpath, allow_pickle=True)
        b = int(z["build_index"])
        xi, y, keep = z["xi"].astype(np.float64), z["y"].astype(int), np.asarray(z["keep"], bool)
        preset = np.arange(keep.size) % n_presets
        # per-preset conditional mean force (audit of the pool, before any exclusion)
        if audit_s.shape[1] == 0:
            audit_s = np.zeros((n_presets, z["floc_sum"].shape[1])); audit_c = np.zeros_like(audit_s)
        for pi in range(n_presets):
            mp = (preset == pi) & keep
            audit_s[pi] += z["floc_sum"][mp].sum(0); audit_c[pi] += z["floc_cnt"][mp].sum(0)
        if args.exclude_presets:
            drop = np.isin(preset, args.exclude_presets) & keep
            n_excl_preset += int(drop.sum())
            keep = keep & ~drop
        aux_in = {"y": y.astype(np.float64), "hb": z["n_hbonds"].astype(np.float64),
                  "af": z["alpha_frac"].astype(np.float64)}
        xa, w, info, aux = mbar_weights(xi, cen, cfg.n_rep, cfg.k_umbrella, beta, keep=keep, aux=aux_in)
        grid, dz, p, F, counts, n_drop = pmf_from_weights(xa, w, cfg)
        Fs.append(F)
        # conditional mean force from every production step of the kept replicas
        fs, fc = z["floc_sum"][keep].sum(0), z["floc_cnt"][keep].sum(0)
        Fps.append(np.where(fc > 0, fs / np.maximum(fc, 1), np.nan))
        cnts.append(fc)
        joints_y.append(weighted_joint(xa, w, aux["y"], np.arange(-0.5, 9.5, 1.0), grid, cfg.R_lo, cfg.R_hi))
        joints_hb.append(weighted_joint(xa, w, aux["hb"], np.arange(-0.5, 8.5, 1.0), grid, cfg.R_lo, cfg.R_hi))
        joints_af.append(weighted_joint(xa, w, aux["af"], AF_EDGES, grid, cfg.R_lo, cfg.R_hi))
        xi_cat.append(xa); w_cat.append(w / w.sum()); y_cat.append(aux["y"].astype(int))
        info_builds.append(dict(build=b, n_kept=int(keep.sum()), n_excluded=int((~keep).sum()),
                                n_samples=int(xa.size), n_dropped=int(n_drop), stride=info["stride"],
                                runtime_h=float(z["runtime_seconds"]) / 3600.0,
                                pull_error_max_nm=float(z["pull_error_nm"].max())))
        # checkpoint PMFs from time prefixes of the samples (samples are appended in time order)
        ck = np.load(os.path.join(raw_dir, f"build{b}_checkpoints.npz"))
        for ns, ns_s in zip(ck["ns"], ck["n_sample"]):
            if int(ns_s) >= xi.shape[0]:
                F_c = F
            else:
                xa_c, w_c, _, _ = mbar_weights(xi[:int(ns_s)], cen, cfg.n_rep, cfg.k_umbrella, beta, keep=keep)
                _, _, _, F_c, _, _ = pmf_from_weights(xa_c, w_c, cfg)
            ckpt_F.setdefault(float(ns), []).append(F_c)
        print(f"  build {b}: {info_builds[-1]}", flush=True)

    mask = (grid >= cfg.R_lo) & (grid <= cfg.R_hi)
    A = np.stack([F - F[mask].mean() for F in Fs])
    n = len(Fs)
    M = np.full((n, n), np.nan)
    for a in range(n):
        for b2 in range(n):
            if a != b2:
                M[a, b2] = l2(A[a], A[b2], mask, dz)
    F_ref = A.mean(0); F_ref -= F_ref[mask].mean()
    span = float(F_ref[mask].max() - F_ref[mask].min())
    tolerable = EFFECT_SIZE_PCT / 100.0 * span
    ratio = float(np.nanmax(M) / tolerable) if n > 1 else float("nan")

    # F' reference: pooled over builds; per-build versions for the spread
    FS = np.stack([np.nan_to_num(fp) * c for fp, c in zip(Fps, cnts)]).sum(0)
    CN = np.stack(cnts).sum(0)
    Fp_ref = np.where(CN > 0, FS / np.maximum(CN, 1), np.nan)
    Fp_builds = np.stack(Fps)
    Mp = np.full((n, n), np.nan)
    for a in range(n):
        for b2 in range(n):
            if a != b2:
                ok = np.isfinite(Fp_builds[a]) & np.isfinite(Fp_builds[b2]) & mask
                Mp[a, b2] = float(np.sqrt(((Fp_builds[a] - Fp_builds[b2])[ok] ** 2).sum() * dz))
    # integrate F' (trapezoid on the cell centres) and compare with the MBAR F
    fp = np.nan_to_num(Fp_ref)
    F_from_Fp = np.concatenate([[0.0], np.cumsum(0.5 * (fp[1:] + fp[:-1]) * dz)])
    F_from_Fp -= F_from_Fp[mask].mean()
    consistency_l2 = l2(F_from_Fp, F_ref, mask, dz)
    dF_num = np.gradient(F_ref, grid)
    ok = np.isfinite(Fp_ref) & mask
    fp_vs_grad = float(np.sqrt(((Fp_ref - dF_num)[ok] ** 2).mean()) / np.sqrt((dF_num[ok] ** 2).mean()))

    # drift between consecutive checkpoints of the consensus
    ns_list = sorted(ckpt_F)
    Fck = []
    for ns in ns_list:
        Fm = np.stack([F - F[mask].mean() for F in ckpt_F[ns]]).mean(0)
        Fck.append(Fm - Fm[mask].mean())
    drifts = [l2(Fck[i], Fck[i - 1], mask, dz) for i in range(1, len(Fck))]

    # Gate A and the joints
    xi_all, w_all, y_all = np.concatenate(xi_cat), np.concatenate(w_cat) / n, np.concatenate(y_cat)
    edges = np.linspace(cfg.R_lo, cfg.R_hi, 41)
    tv, occ, p_cond = conditional_tv(xi_all, y_all, w_all, edges, min_count=1e-3 * w_all.sum())
    with np.errstate(invalid="ignore"):
        tv_max = float(np.nanmax(tv)) if np.isfinite(tv).any() else float("nan")
    Jy, Jhb, Jaf = (np.stack(j).sum(0) for j in (joints_y, joints_hb, joints_af))

    summary = dict(
        n_builds=n, ns_per_replica=float(cfg.n_prod_steps * cfg.dt / 1000.0),
        aggregate_ns=float(cfg.n_prod_steps * cfg.dt / 1000.0 * cfg.n_windows * cfg.n_rep * n),
        builds=info_builds, pairwise_l2_max=float(np.nanmax(M)) if n > 1 else None,
        pairwise_l2_matrix=M.tolist(), F_span_kJ=span, F_span_kT=span / kT,
        resolvable_effect_kJ=tolerable, ratio=ratio,
        reference_accepted=bool(n > 1 and ratio < 1.0),
        F_argmin_nm=float(grid[mask][F_ref[mask].argmin()]),
        Fprime_pairwise_l2_max=float(np.nanmax(Mp)) if n > 1 else None,
        Fprime_min_count_in_domain=float(CN[mask].min()),
        F_from_Fprime_vs_F_mbar_l2=consistency_l2,
        Fprime_vs_numerical_gradient_rel_rms=fp_vs_grad,
        checkpoint_ns=ns_list, consensus_drift_l2=drifts,
        drift_over_effect=(drifts[-1] / tolerable if drifts else None),
        gate_a_max_pairwise_tv=tv_max, gate_a_labels_occupied=int((occ > 0).sum()),
        gate_a_pass=bool(np.isfinite(tv_max) and tv_max >= 0.30),
        old_2026_08=OLD, config_hash=cfg.config_hash(), git=C.git_commit(),
        excluded_presets=list(args.exclude_presets), n_replicas_excluded_by_preset=n_excl_preset,
        preset_audit=preset_audit(grid, audit_s, audit_c))
    np.savez_compressed(os.path.join(out_dir, "reference.npz"),
                        grid=grid, dz=dz, mask=mask, F_ref=F_ref, F_builds=A,
                        Fprime_ref=Fp_ref, Fprime_builds=Fp_builds, Fprime_counts=CN,
                        F_from_Fprime=F_from_Fp, F_checkpoints=np.stack(Fck), checkpoint_ns=np.array(ns_list),
                        joint_y=Jy, joint_hb=Jhb, joint_af=Jaf, af_edges=AF_EDGES,
                        gate_a_tv=tv, gate_a_occupancy=occ, gate_a_edges=edges,
                        xi_all=xi_all.astype(np.float32), weights=w_all, y_all=y_all.astype(np.int8))
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n" + "=" * 78)
    for k in ("n_builds", "ns_per_replica", "aggregate_ns", "pairwise_l2_max", "F_span_kJ", "F_span_kT",
              "ratio", "reference_accepted", "F_argmin_nm", "Fprime_pairwise_l2_max",
              "F_from_Fprime_vs_F_mbar_l2", "Fprime_vs_numerical_gradient_rel_rms", "consensus_drift_l2",
              "drift_over_effect", "gate_a_max_pairwise_tv", "gate_a_pass"):
        print(f"  {k:38s} {summary[k]}")
    print(f"  2026-08 recorded: span {OLD['span_kJ']:.1f} kJ, L2 {OLD['pairwise_l2_max']:.3f}, "
          f"drift {OLD['drift_l2_kJ']:.2f}, argmin {OLD['argmin_nm']:.3f}, GateA {OLD['gate_a_tv']:.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
