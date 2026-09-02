#!/usr/bin/env python
"""Figure A of the transport campaign: what one horizontal-transport event literally does.

Runs plain ABF on the production cell for t = 5 (one seed, 'left' init) to obtain a
representative mid-run population, then applies ``horizontal_ot_map`` ONCE at alpha* (from
Stage A) and at alpha = 1, and draws (x_i, y_i) -> (x_i', y_i) with horizontal segments: every
segment is horizontal because y is untouched.  A second panel shows the exact conditional
distortion D_move of that single event per walker against x'.  Illustration only -- no
error metric, no arm comparison.

    CUDA_VISIBLE_DEVICES=3 python scripts/plot_gateway_transport_event.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gateway_core as gw  # noqa: E402
from run_gateway_bandwidth_audit import build_config  # noqa: E402

BASE_PREREG = os.path.join(ROOT, "results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json")
CAL_SEL = os.path.join(ROOT, "results/transport_campaign/gateway_horizontal/calibration/alpha_selection.json")
OUT = os.path.join(ROOT, "results/transport_campaign/gateway_horizontal/production/figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=480)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = json.load(open(BASE_PREREG))
    sampler, cell = dict(base["sampler"]), base["cell"]
    n_steps = int(round(a.t / sampler["dt"]))
    sampler.update(n_steps=n_steps, save_every=n_steps)
    alpha_star = float(json.load(open(CAL_SEL))["alpha_star"])
    cfg = build_config(sampler, cell, "left", sampler["h"])
    recs = gw.simulate_batch(gw.BatchSpec(configs=[cfg], seeds=[a.seed], methods=[gw.ABF], batch_seed=1),
                             store_final_state=True)
    X = torch.as_tensor(recs[0]["X_final"]).unsqueeze(0)
    Y = np.asarray(recs[0]["Y_final"])
    u = gw.uniform_quantiles(X.shape[1], X.device, X.dtype)
    dev = torch.device("cpu")
    X, u = X.to(dev), u.to(dev)
    om = lambda x: gw.omega_of(x, torch.tensor(cfg.omega_out), torch.tensor(cfg.omega_in), torch.tensor(cfg.s))  # noqa: E731
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True)
    for j, (al, title) in enumerate(((alpha_star, f"matched, alpha = {alpha_star:g}"), (1.0, "full, alpha = 1"))):
        Xn = gw.horizontal_ot_map(X, al, u)
        x0, x1 = X.numpy()[0], Xn.numpy()[0]
        ax = axes[0, j]
        sub = np.random.default_rng(0).choice(len(x0), size=min(400, len(x0)), replace=False)
        for i in sub:
            ax.plot([x0[i], x1[i]], [Y[i], Y[i]], color="#1b9e77", lw=0.5, alpha=0.6)
        ax.scatter(x0[sub], Y[sub], s=5, color="#4d4d4d", label="before", zorder=3)
        ax.scatter(x1[sub], Y[sub], s=5, color="#d95f02", label="after", zorder=4)
        ax.set_title(f"one horizontal-OT event at t = {a.t:g} ({title})", fontsize=9)
        ax.set_ylabel("y"); ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.axvspan(-cfg.s, cfg.s, color="#7570b3", alpha=0.12, lw=0)
        dm = gw.d_move(om(X), om(Xn)).numpy()[0]
        ax2 = axes[1, j]
        ax2.scatter(x1, dm + 1e-6, s=4, color="#7570b3", alpha=0.7)
        ax2.set_yscale("log"); ax2.set_xlabel("x' (after the event)")
        ax2.set_ylabel(r"$D_{\rm move}$ (nats)")
        ax2.axvspan(-cfg.s, cfg.s, color="#7570b3", alpha=0.12, lw=0)
        ax2.set_title(f"mean |dx| {np.abs(x1 - x0).mean():.3f}, mean D_move {dm.mean():.3g}, max {dm.max():.3g}", fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=0.25, which="both")
    os.makedirs(a.out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"fig_A_transport_event.{ext}"), dpi=200, bbox_inches="tight")
    print(f"wrote {os.path.relpath(a.out, ROOT)}/fig_A_transport_event.{{png,pdf}} (seed {a.seed}, t {a.t:g}, alpha* {alpha_star:g})")


if __name__ == "__main__":
    main()
