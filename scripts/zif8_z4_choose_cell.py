#!/usr/bin/env python
"""Apply the FROZEN Z4 cell rule (docs/ZIF8_OT_Z4Z5.md): establishment_limited > intermediate; ties ->
replica count nearest 128; discovery/conditional-limited excluded; none -> no Z5.  Also reports the
384 x 300 ps control (corrected 16-seed ABF production) for T_cover / T_marg (gate: legacy mixture)."""
import glob, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
Z4 = os.path.join(ROOT, "results/ot_repair_campaign/zif8/Z4")
cells = {}
for f in sorted(glob.glob(os.path.join(Z4, "B*.json"))):
    d = json.load(open(f)); cells[d["cell"]] = d
try:                                                                       # control from the corrected production ABF
    from run_zif8_screen import classify
    from zif8.core_zif8 import ZIF8SimConfig
    pre = json.load(open(os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json")))
    z = np.load(os.path.join(ROOT, "results/information_campaign/corrected/abf.npz"), allow_pickle=True)
    s = {k: v for k, v in pre["sampler"].items() if not k.startswith("_")}; s["abf_bandwidth_A"] = 0.10
    out = {k: z[k] for k in z.files if k != "meta"}
    ctrl = classify(out, ZIF8SimConfig(**s), pre, None)
    ctrl["verdict"] += "_(gate=legacy_mixture,not_used)"; ctrl["n_replicas"] = 384; ctrl["cell"] = "control_384x300ps"
    cells["control"] = {k: v for k, v in ctrl.items() if k not in ("js_series", "tv_series", "times")}
except Exception as exc:
    cells["control"] = dict(error=repr(exc))
rank = {"establishment_limited": 0, "intermediate": 1}
cands = [(rank[c["verdict"]], abs(c["n_replicas"] - 128), name) for name, c in cells.items() if name != "control" and c.get("verdict") in rank]
choice = sorted(cands)[0][2] if cands else None
res = dict(choice=choice, go_z5=choice is not None, n_replicas=(cells[choice]["n_replicas"] if choice else None),
           table={n: dict(n_replicas=c.get("n_replicas"), verdict=c.get("verdict"), T_cover=c.get("T_cover"), T_marg=c.get("T_marg"), T_gate=c.get("T_gate"), T=c.get("T"),
                          transits=c.get("transit_events"), unvisited=c.get("unvisited_bins")) for n, c in cells.items()},
           rule="establishment_limited > intermediate; ties -> |N - 128| smallest; discovery/conditional excluded; none -> no Z5")
json.dump(res, open(os.path.join(Z4, "cell_choice.json"), "w"), indent=1, default=float)
print(f"{'cell':8s} {'N':>4s} {'T_cover/T':>10s} {'T_marg/T':>9s} {'T_gate/T':>9s} {'transits':>8s} verdict")
for n, c in res["table"].items():
    T = c["T"] or float("nan")
    f = lambda x: (f"{x / T:9.2f}" if x is not None and np.isfinite(x) else "      inf")
    print(f"{n:8s} {str(c['n_replicas']):>4s} {f(c['T_cover'])} {f(c['T_marg'])} {f(c['T_gate'])} {str(c['transits']):>8s} {c['verdict']}")
print(f"CHOICE: {choice}  (go_z5 = {res['go_z5']})")
