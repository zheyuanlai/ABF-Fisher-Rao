#!/usr/bin/env python
"""v3 offline discretization benchmark (Q-D / prediction P6).

Frozen protocol: docs/V3_PREREGISTRATION.md, Appendix A.1-A.2 and Amendment 3.

Runs on the hashed K=1024 clouds only -- no MD.  For every cloud, subsample
size, and registered dose it applies BD-standard, BD-paired and FT at the *same
nominal FR time* (Amendment 3: theta = 1 - exp(-dtau)) and measures how much of
the continuum contraction each realization delivers per unit genealogy cost.

Two evaluation rules matter and are frozen:

* the KL used to score the operators is computed with an **independent
  leave-one-out-bandwidth KDE**, never the eta = 0.10 KDE the operator itself
  used to build its weights -- otherwise the discretization that best overfits
  the operator's own density estimate wins artificially;
* a bandwidth-free companion (1-Wasserstein to q) is reported alongside.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from abffr import fr_v3  # noqa: E402

K_VALUES = (64, 128, 256, 512, 1024)
P_MAX_VALUES = (0.02, 0.05, 0.10)
OPERATORS = ("bd_standard", "bd_paired", "ft")
ETA_OPERATOR = 0.10                 # the registered operator KDE bandwidth
LOO_BANDWIDTHS = np.geomspace(0.03, 0.60, 12)
MIN_KL_DROP = 0.01                  # C_gene exclusion rule (A / v3.1)


def _kde_on_grid(z, grid, bw):
    d = (grid[None, :] - z[:, None]) / bw
    return np.exp(-0.5 * d * d).sum(0) / (len(z) * bw * np.sqrt(2 * np.pi))


def _loo_bandwidth(z):
    """Leave-one-out ML bandwidth -- independent of the operator's own KDE."""
    d = (z[None, :] - z[:, None])
    best, best_ll = LOO_BANDWIDTHS[0], -np.inf
    for bw in LOO_BANDWIDTHS:
        k = np.exp(-0.5 * (d / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
        np.fill_diagonal(k, 0.0)
        ll = np.log(k.sum(1) / (len(z) - 1) + 1e-300).sum()
        if ll > best_ll:
            best_ll, best = ll, bw
    return float(best)


def _normalize(p, grid):
    return p / max(np.trapezoid(p, grid), 1e-300)


def _kl(p, q, grid):
    p = _normalize(np.maximum(p, 1e-300), grid)
    q = _normalize(np.maximum(q, 1e-300), grid)
    return float(np.trapezoid(p * (np.log(p) - np.log(q)), grid))


def _w1(z, q, grid):
    """Bandwidth-free companion: 1-Wasserstein from the empirical cloud to q."""
    qn = _normalize(np.maximum(q, 1e-300), grid)
    cdf_q = np.concatenate([[0.0], np.cumsum(0.5 * (qn[1:] + qn[:-1]) * np.diff(grid))])
    cdf_q /= cdf_q[-1]
    cdf_e = np.searchsorted(np.sort(z), grid, side="right") / len(z)
    return float(np.trapezoid(np.abs(cdf_e - cdf_q), grid))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--infra", default="results/v3/infrastructure")
    p.add_argument("--out", default="results/v3/offline_benchmark")
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--max-clouds", type=int, default=None)
    args = p.parse_args(argv)

    infra = pathlib.Path(args.infra)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man = json.loads((infra / "cloud_manifest.json").read_text())
    clouds = man["clouds"][:args.max_clouds] if args.max_clouds else man["clouds"]
    print(f"[offline] {len(clouds)} clouds x {len(K_VALUES)} K x "
          f"{len(P_MAX_VALUES)} doses x {len(OPERATORS)} operators x "
          f"{args.seeds} seeds")

    rows = []
    for ci, c in enumerate(clouds):
        d = np.load(infra / "clouds" / c["file"])
        z_full, grid, q = d["x"], d["x_grid"], d["q"]
        # Deterministic subsampling: a fixed permutation keyed by the cloud file,
        # so which particles a K-subsample contains is not a free choice.
        perm = np.random.default_rng(
            abs(hash(c["file"])) % (2 ** 32)).permutation(len(z_full))
        for K in K_VALUES:
            z = z_full[perm[:K]]
            bw_eval = _loo_bandwidth(z)
            p_before = _kde_on_grid(z, grid, bw_eval)
            kl_before = _kl(p_before, q, grid)
            w1_before = _w1(z, q, grid)

            # the operator's own KDE (eta = 0.10), used only for the score
            p_op = _normalize(_kde_on_grid(z, grid, ETA_OPERATOR), grid)
            qn = _normalize(np.maximum(q, 1e-300), grid)
            log_p_part = np.log(np.maximum(np.interp(z, grid, p_op), 1e-300))
            log_q_part = np.log(np.maximum(np.interp(z, grid, qn), 1e-300))
            score = fr_v3.FRScore(
                log_p=torch.tensor(log_p_part, dtype=torch.float64),
                log_q=torch.tensor(log_q_part, dtype=torch.float64))

            for p_max in P_MAX_VALUES:
                dtau = fr_v3.bd_timestep(score, p_max)
                theta = fr_v3.theta_from_dtau(dtau)
                for op in OPERATORS:
                    kls, w1s, esss, wmaxs, repls = [], [], [], [], []
                    for s in range(args.seeds):
                        g = torch.Generator(); g.manual_seed(1000 * s + 7)
                        if op == "bd_standard":
                            src, _ = fr_v3.bd_standard(score, dtau, g)
                        elif op == "bd_paired":
                            src, _ = fr_v3.bd_paired(score, dtau, g)
                        else:
                            src = fr_v3.ft_step_fixed(score, theta, g)
                        idx = src.numpy()
                        z_after = z[idx]
                        kls.append(_kl(_kde_on_grid(z_after, grid, bw_eval), q, grid))
                        w1s.append(_w1(z_after, grid=grid, q=q))
                        counts = np.bincount(idx, minlength=K).astype(float)
                        w = counts / K
                        esss.append(1.0 / max((w * w).sum(), 1e-300) / K)
                        wmaxs.append(w.max())
                        repls.append(int(K - (counts > 0).sum()))
                    kl_after = float(np.mean(kls))
                    drop = kl_before - kl_after
                    rows.append(dict(
                        cloud=c["file"], family=c["family"], seed=c["seed"],
                        t_over_T=c["t_over_T"], K=K, p_max=p_max, dtau=dtau,
                        theta=theta, operator=op, bw_eval=bw_eval,
                        kl_before=kl_before, kl_after=kl_after, kl_drop=drop,
                        kl_after_sd=float(np.std(kls)),
                        w1_before=w1_before, w1_after=float(np.mean(w1s)),
                        ess_anc_frac=float(np.mean(esss)),
                        wmax=float(np.mean(wmaxs)),
                        replacements=float(np.mean(repls)),
                        c_gene=((1.0 - float(np.mean(esss))) / drop
                                if drop >= MIN_KL_DROP else np.nan)))
        print(f"[offline] cloud {ci + 1}/{len(clouds)}  {c['file']}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "offline_benchmark.csv", index=False)
    excluded = int(df["c_gene"].isna().sum())
    print(f"\n[offline] wrote {len(df)} rows -> {out/'offline_benchmark.csv'}")
    print(f"[offline] C_gene excluded (KL drop < {MIN_KL_DROP}): "
          f"{excluded}/{len(df)} cells -- reported, not silently dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
