"""Which fiber mode has to be promoted -- predicted from the run's own statistics.

The manifold phase's spectral estimate of the free-energy damage caused by a
lift error is

    C_eff  ~  sum_j  c_j^2 / lambda_j^2,

i.e. a mode's contribution scales with the SQUARE of its relaxation time and
with how strongly its conditional actually moves as z moves.  That gives a
concrete, measurable ranking without any oracle:

    S_k  =  E_z [ D_KL( p(y_k | z) || p(y_k | z + dz) ) ] / dz^2      (sensitivity)
    tau_k                                                             (relaxation time)
    damage_k  ~  S_k * tau_k^2.

`S_k` comes from the joint histogram a thermodynamic-integration run already
accumulates; `tau_k` from watching a constrained window relax.  The hexane
experiment measures the ACTUAL benefit of promoting each mode, so the ranking
can be checked rather than assumed.
"""
from __future__ import annotations

import argparse, json, math, os, sys

import numpy as np


def sensitivity(Hjoint, k, dz_bins=1):
    """S_k from a (nz, n1, ..., nF) joint count table."""
    nfib = Hjoint.ndim - 1
    H = Hjoint
    for ax in reversed(range(nfib)):
        if ax != k:
            H = H.sum(axis=ax + 1)
    p = H / np.maximum(H.sum(-1, keepdims=True), 1e-12)
    p = 0.98 * p + 0.02 / p.shape[-1]
    q = np.roll(p, -dz_bins, axis=0)
    kl = (p * (np.log(p) - np.log(q))).sum(-1)
    w = H.sum(-1); w = w / w.sum()
    dz = dz_bins * 2 * math.pi / p.shape[0]
    return float((kl * w).sum() / dz ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="HEX")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--tau", default=None)
    a = ap.parse_args()
    ref = a.ref or f"results/mol/ref/{a.system}_ref.npz"
    d = np.load(ref)
    Hj = d["Hjoint"].sum(0) if ("Hjoint" in d and d["Hjoint"].size) else d["H2"].sum(0)
    nfib = Hj.ndim - 1
    S = [sensitivity(Hj, k) for k in range(nfib)]
    tp = a.tau or f"results/mol/{a.system}_fiber_time.npz"
    tau = np.load(tp)["tau"] if os.path.exists(tp) else np.ones(nfib)
    tau = np.atleast_1d(tau)
    dmg = [S[k] * float(tau[min(k, len(tau) - 1)]) ** 2 for k in range(nfib)]
    out = {"system": a.system, "S": S, "tau": tau.tolist(), "damage": dmg,
           "ranking": [int(i) + 1 for i in np.argsort(dmg)[::-1]]}
    os.makedirs("results/mol", exist_ok=True)
    json.dump(out, open(f"results/mol/{a.system}_mode_diagnostic.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
