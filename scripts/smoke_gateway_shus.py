"""Engineering smoke run (NOT Stage-1 science): plain SHUS + one FR sanity arm on the
default gateway cell, batched seeds x arms on one GPU, records saved via the schema.

Checks: GPU execution path, throughput, e_F(t) decreasing, marginal flattening,
FR machinery firing inside its window with healthy ancestry.

Usage:  python scripts/smoke_gateway_shus.py [--steps 100000] [--seeds 4] [--cpu]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from abpfr.io import save_run
from abpfr.systems import gateway as gw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--walkers", type=int, default=1024)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="results/smoke")
    args = ap.parse_args()

    device = torch.device("cpu") if args.cpu else gw.DEVICE
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    cfg = gw.GatewayConfig(K=args.walkers, n_steps=args.steps)
    seeds = list(range(args.seeds))
    fr = gw.Method("shus_fr", use_fr=True, theta=0.10, t_on_frac=0.10,
                   t_off_frac=0.50, fr_every_blocks=5)
    methods = [gw.SHUS, fr]
    print(f"cell: beta={cfg.beta} H={cfg.H} r={cfg.r} s={cfg.s} "
          f"(barrier {cfg.barrier_kT():.2f} kT), K={cfg.K}, T={cfg.T_total:.1f}, "
          f"B={len(seeds)} seeds x M={len(methods)} arms")

    t0 = time.time()
    recs = gw.simulate_batch([cfg] * len(seeds), seeds, methods, batch_seed=12345,
                             device=device, progress=args.steps // 10)
    wall = time.time() - t0
    rate = args.steps * len(seeds) * len(methods) * cfg.K / wall
    print(f"wall {wall:.1f}s  ({args.steps} steps, {rate/1e6:.1f}M walker-steps/s)")

    os.makedirs(args.out, exist_ok=True)
    for rec in recs:
        name = f"{rec['method']['name']}_seed{rec['seed']}"
        arrays = {k: rec[k] for k in
                  ("time", "pmf_t", "marginal_t", "x_grid", "F_ref", "l2_f_t",
                   "l2_fp_t", "kl_u_t", "tv_u_t", "ess_anc_t", "wmax_t", "P_regions",
                   "Q_regions", "event_time", "event_theta", "event_ess_fr",
                   "event_turnover")}
        meta = {"reference_id": rec["reference_id"], "eval_window": rec["eval_window"],
                "config": rec["config"], "method": rec["method"], "seed": rec["seed"],
                "batch_seed": rec["batch_seed"], "purpose": "smoke"}
        save_run(os.path.join(args.out, name), arrays, meta)

    # console summary per arm
    for mname in [m.name for m in methods]:
        rows = [r for r in recs if r["method"]["name"] == mname]
        e0 = np.median([r["l2_f_t"][0] for r in rows])
        eT = np.median([r["l2_f_t"][-1] for r in rows])
        iF = np.median([r["int_l2_f"] for r in rows])
        klT = np.median([r["kl_u_t"][-1] for r in rows])
        essmin = np.median([r["ess_anc_t"].min() / r["config"]["K"] for r in rows])
        print(f"  {mname:10s} e0={e0:.3f} eT={eT:.3f} I_F={iF:.2f} "
              f"KL_T={klT:.4f} min ESS_anc/K={essmin:.2f}")

    # quick figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        colors = {"shus": "C0", "shus_fr": "C1"}
        for rec in recs:
            m = rec["method"]["name"]
            t = rec["time"]
            kw = dict(color=colors.get(m, "k"), alpha=0.6, lw=1.2)
            axes[0, 0].plot(t, rec["l2_f_t"], **kw)
            axes[0, 1].semilogy(t, np.maximum(rec["kl_u_t"], 1e-6), **kw)
            axes[1, 0].plot(t, rec["P_regions"][:, 2], **kw)
            axes[1, 1].plot(t, rec["ess_anc_t"] / rec["config"]["K"], **kw)
        for m, c in colors.items():
            axes[0, 0].plot([], [], color=c, label=m)
        won = fr.t_on_frac * cfg.T_total
        woff = fr.t_off_frac * cfg.T_total
        for ax in axes.flat:
            ax.axvspan(won, woff, color="C1", alpha=0.08)
        axes[0, 0].set_ylabel(r"$e_F(t)$  [$L^2$, gauge-opt]")
        axes[0, 0].legend()
        axes[0, 1].set_ylabel(r"KL$(\hat p_t\,\|\,u)$")
        axes[1, 0].set_ylabel(r"right-basin occupancy $P_+$")
        axes[1, 1].set_ylabel(r"ESS$_{\rm anc}/K$ (windowed)")
        for ax in axes[1]:
            ax.set_xlabel("t")
        fig.suptitle("SMOKE: gateway mollified SHUS vs SHUS+temporary FR "
                     f"(cell r={cfg.r}, s={cfg.s}; FR window shaded)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "smoke_overview.png"), dpi=130)
        print(f"figure -> {os.path.join(args.out, 'smoke_overview.png')}")
    except Exception as e:  # plotting must never fail the smoke
        print(f"(no figure: {e})")


if __name__ == "__main__":
    main()
