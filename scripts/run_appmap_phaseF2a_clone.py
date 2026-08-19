"""Phase F2a: clone decorrelation in the hidden coordinate, on the F1-eligible cells.

Frozen in docs/PREREGISTRATION_APPLICATION_MAP.md (Phase F) before this run and
before any F2 row: the Q4a instrument, split into the channel-identity measure a
population correction is actually carried by, and the within-fiber measure that says
whether siblings are statistically redundant.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abpfr.systems import bichannel as bc

CELLS = ((2.0, 0.0), (2.5, 0.0), (2.0, 0.5))
SEEDS = [0, 1, 2, 3]
T0 = 200.0
LAG_MAX = 400.0
COMMON = dict(K=1024, dt=1e-3, n_steps=800_000, block=20, eps_bw=0.06, eta_bw=0.25,
              n_saves=400, profile_every=8, joint_every=40, ess_window_steps=4000,
              n_strata=32)
STRIDE_T = 49 * COMMON["block"] * COMMON["dt"]      # the frozen F2 event stride
WINDOW_T = 195.0                                    # the frozen F2 window length
OUT = "results/appmap_phaseF2a_clone"


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(OUT, exist_ok=True)
    print(f"F2a: {len(CELLS)} cells x {len(SEEDS)} seeds, t0={T0}, lag_max={LAG_MAX}; "
          f"F2 event stride = {STRIDE_T:.2f}, window = {WINDOW_T:.0f}")
    summary = {}
    for (hp, dl) in CELLS:
        cfg = bc.BiChannelConfig(beta=4.0, Hperp=hp, Delta=dl, **COMMON)
        t0 = time.time()
        r = bc.clone_decorrelation(cfg, SEEDS, t0=T0, lag_max=LAG_MAX,
                                   batch_seed=20260955, device=device)
        tag = f"hp{hp:g}_d{dl:g}"
        np.savez_compressed(os.path.join(OUT, tag + ".npz"), **r)
        med = lambda k: float(np.nanmedian(r[k]))
        summary[tag] = {k: med(k) for k in ("tau_chan", "tau_psi", "tau_phi")}
        summary[tag]["same_ind_0"] = float(np.median(r["same_ind"][:, 0]))
        summary[tag]["d_ind_psi_0"] = float(np.median(r["d_ind_psi"][:, 0]))
        print(f"\n--- {tag}  (wall {time.time()-t0:.0f}s) ---")
        print(f"  tau_clone^chan = {med('tau_chan'):8.2f}   "
              f"tau_clone^psi = {med('tau_psi'):7.2f}   "
              f"tau_clone^phi = {med('tau_phi'):7.2f}")
        print(f"  independent baseline at lag 0: same-channel {summary[tag]['same_ind_0']:.3f}, "
              f"RMS psi separation {summary[tag]['d_ind_psi_0']:.3f} rad")
        lag = r["lag"]
        idx = [np.argmin(np.abs(lag - x)) for x in (0.0, 1.0, 5.0, 20.0, 100.0, 400.0)]
        print("     lag:", " ".join(f"{lag[i]:8.1f}" for i in idx))
        for k in ("m_chan", "m_psi", "m_phi"):
            v = np.median(r[k], axis=0)
            print(f"  {k:>7}:", " ".join(f"{v[i]:8.3f}" for i in idx))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"stride_T": STRIDE_T, "window_T": WINDOW_T, "cells": summary},
                  f, indent=2)
    print("\nReading rule (frozen): conditional cloning can carry a correction only if "
          "tau_clone^chan >> the event stride; tau_clone^psi ~ stride means siblings "
          "are not redundant for within-fiber statistics.")


if __name__ == "__main__":
    main()
