"""Pick each arm's winner from the screen, freeze it, and emit the confirmation spec.

Selection is by median I_F over the 8 screening seeds -- the time-averaged error,
not the endpoint, because the manifold phase showed an endpoint at one budget can
rank two arms the opposite way from their whole trajectory.  The frozen choice is
written out before any confirmation seed is touched.
"""
from __future__ import annotations

import argparse, glob, json, os, sys

import numpy as np


def best(path, key="I_F"):
    d = np.load(path)
    n_cfg = int(d["n_cfg"]); ns = int(d["n_seed"])
    v = np.median(d[key].reshape(n_cfg, ns), axis=1)
    i = int(np.argmin(v))
    g = d["cfg_grid"][i]
    return dict(i=i, kappa=float(g[0]), theta=float(g[1]), decay=float(g[2]),
                I_F=float(v[i]), all=v.tolist(), grid=d["cfg_grid"].tolist(),
                e_F=float(np.median(d["e_F_final"].reshape(n_cfg, ns)[i])),
                dcond=float(np.median(d["dcond"][-1].reshape(n_cfg, ns)[i])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="PEN")
    ap.add_argument("--dir", default="results/mol/campaign")
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=90_000)
    ap.add_argument("--out", default="results/mol/frozen.json")
    ap.add_argument("--spec", default="results/mol/confirm_spec.json")
    a = ap.parse_args()
    froz = {}
    for p in sorted(glob.glob(os.path.join(a.dir, f"{a.system}_*_screen*.npz"))):
        name = os.path.basename(p)[:-4]
        arm = name[len(a.system) + 1:].split("_screen")[0]
        suffix = name.split("_screen")[1]
        try:
            b = best(p)
        except Exception:
            continue
        b["file"] = name
        b["suffix"] = suffix
        cur = froz.get(arm)
        if cur is None or b["I_F"] < cur["I_F"]:
            froz[arm] = b
    Z0 = {"ALA": -0.5236}
    base = dict(system=a.system, seeds=a.seeds, seed0=a.seed0, N=256,
                steps=a.steps, bw_mf=0.05, save_every=10_000, tag="confirm",
                z0=Z0.get(a.system, 0.0))
    spec = []
    for arm, b in froz.items():
        kw = {**base, "arm": arm, "kappa": str(b["kappa"]), "theta": str(b["theta"])}
        s = b["suffix"]
        if s.startswith("_w"):
            kw["n_windows"] = int(s[2:])
        if s.startswith("_n"):
            kw["abf_nmin"] = float(s[2:])
        if s.startswith("_bz"):
            kw["lift_bw_z"] = float(s[3:])
            kw["decay"] = str(b["decay"])
            kw["lift_start"] = 50_000.0 if arm == "wfr_lref" else 0.0
        spec.append(kw)
    order = ["wfr_rot", "wfr_shake", "wfr_ymap", "wfr_yref", "wfr_lmap", "wfr_lref",
             "wfr_ymh", "wfr_lmh", "wfr_qref", "ti_cold", "ti_warm", "abf"]
    spec.sort(key=lambda k: order.index(k["arm"]) if k["arm"] in order else 99)
    json.dump(spec, open(a.spec, "w"), indent=1)

    if a.system != "PEN":
        json.dump(froz, open(a.out, "w"), indent=1)
        _print(froz, order, spec, a)
        return
    # transport-rate stress test: every lift at its own frozen theta, kappa swept
    ks = "0.0375,0.075,0.15,0.3,0.6,1.2,2.4"
    kspec = [dict(system=a.system, arm=arm, seeds=16, seed0=70_000, N=256,
                  steps=100_000, bw_mf=0.05, save_every=10_000, tag="kappa",
                  kappa=ks, theta=str(froz[arm]["theta"]),
                  **({"lift_bw_z": float(froz[arm]["suffix"][3:]),
                      "decay": str(froz[arm]["decay"])}
                     if froz[arm]["suffix"].startswith("_bz") else {}))
             for arm in ["wfr_rot", "wfr_shake", "wfr_ymap", "wfr_yref", "wfr_ymh",
                         "wfr_lmh"] if arm in froz]
    json.dump(kspec, open("results/mol/kappa_spec.json", "w"), indent=1)

    # mechanism ablations at the confirmation budget, on the frozen naive and
    # oracle-lift settings
    src = {"w_only": "wfr_rot", "fr_only": "wfr_rot", "w_count": "wfr_rot",
           "w_only_y": "wfr_yref", "wfr_flow": "wfr_rot", "wfr_flow_y": "wfr_yref"}
    aspec = [dict(system=a.system, arm=arm, seeds=16, seed0=90_000, N=256,
                  steps=a.steps, bw_mf=0.05, save_every=10_000, tag="confirm",
                  kappa=str(froz[s2]["kappa"]), theta=str(froz[s2]["theta"]))
             for arm, s2 in src.items() if s2 in froz]
    json.dump(aspec, open("results/mol/ablation_spec.json", "w"), indent=1)

    # hexane: which fiber mode has to be promoted?  Same kappa/theta for every
    # arm, taken from pentane's frozen naive arm, so no arm is tuned against
    # another on a system with no screen of its own.
    kh = str(froz.get("wfr_rot", {"kappa": 0.3})["kappa"])
    th = str(froz.get("wfr_rot", {"theta": 0.3})["theta"])
    hb = dict(system="HEX", seeds=16, seed0=50_000, N=256, steps=200_000,
              bw_mf=0.05, save_every=10_000, kappa=kh, theta=th)
    hspec = [{**hb, "arm": "wfr_rot", "tag": "hex"},
             {**hb, "arm": "wfr_ymh", "promote": "1", "tag": "hex_p1"},
             {**hb, "arm": "wfr_ymh", "promote": "2", "tag": "hex_p2"},
             {**hb, "arm": "wfr_ymh", "promote": "1,2", "tag": "hex_p12"},
             {**hb, "arm": "ti_cold", "n_windows": 64, "tag": "hex"},
             {**hb, "arm": "abf", "tag": "hex"}]
    json.dump(hspec, open("results/mol/hexane_spec.json", "w"), indent=1)
    json.dump(froz, open(a.out, "w"), indent=1)
    _print(froz, order, spec, a)


def _print(froz, order, spec, a):
    print("| arm | frozen | screen I_F | screen e_F | screen D_cond |")
    print("|---|---|---|---|---|")
    for arm in order:
        if arm not in froz:
            continue
        b = froz[arm]
        extra = b["suffix"].lstrip("_") or "-"
        print(f"| {arm} | kappa={b['kappa']:g} theta={b['theta']:g} "
              f"decay={b['decay']:g} {extra} | {b['I_F']:.4f} | {b['e_F']:.4f} "
              f"| {b['dcond']:.4f} |")
    print(f"\n{len(spec)} confirmation runs -> {a.spec}")


if __name__ == "__main__":
    main()
