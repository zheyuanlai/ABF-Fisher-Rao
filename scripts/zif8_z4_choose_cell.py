#!/usr/bin/env python
"""Apply the FROZEN Z4 cell rule (docs/ZIF8_OT_Z4Z5.md): establishment_limited > intermediate; ties ->
replica count nearest 128; discovery/conditional-limited excluded; none -> no Z5.  Also reports the
384 x 300 ps control (corrected 16-seed ABF production) for T_cover / T_marg (gate: legacy mixture)."""
import glob, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
Z4 = os.path.join(ROOT, "results/ot_repair_campaign/zif8/Z4")
from run_zif8_screen import relative_time
from zif8.core_zif8 import js_divergence
PRE0 = json.load(open(os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json")))
REFC = np.load(os.path.join(ROOT, "cache/zif8/gate_reference_v2_T300.npz"))["gate_hist_window_xi"].astype(float).reshape(8, 24, 4).sum(-1)


def regate(npz_path, cls):
    """Amendment A1 (docs/ZIF8_OT_Z4Z5.md): T_gate from the CUMULATIVE in-band gate histogram rebinned to
    24 x 0.1 A (finite-sample JS floor ~0.005 at n >= 5000 instead of 0.1-0.2 on 96-bin per-save blocks
    with a few hundred samples, which read as 'degrading' at every small budget)."""
    z = np.load(npz_path, allow_pickle=True); blocks = np.asarray(z["gate_hist_block"], float); t = np.asarray(z["times"], float)
    cum = np.cumsum(blocks.sum(1), axis=0)                                # (T, nx, 96) pooled over seeds
    ser = np.full(cum.shape[0], np.nan)
    for i in range(cum.shape[0]):
        hc = cum[i].reshape(8, 24, 4).sum(-1); ok = hc.sum(-1) >= 200
        if ok.any():
            ser[i] = float(np.mean(js_divergence(hc[ok], REFC[ok])))
    r = PRE0["screen"]; T_gate, J0, Ji, thr, st = relative_time(t, ser, r["relative_fraction"], r["hold_frac"], t0=cls["t_warmup"])
    T = cls["T"]; q = 0.25 * T; nvis_ok = cls["unvisited_bins"] < 1
    # Amendment A2 (docs/ZIF8_OT_Z4Z5.md): the Z4 runs' in-band gate histograms are still a periodic-image
    # mixture (the sampler's unwrapped phi was accumulated from the WRAPPED start position; fixed for Z5),
    # so T_gate cannot be classified from them.  Z4 verdicts use T_cover and T_marg only; the gate clock is
    # measured in Z5 with the corrected band.  The (invalid) J_gate series is kept for the record.
    if cls["T_cover"] > 0.5 * T or not nvis_ok:
        v = "discovery_limited"
    elif cls["T_cover"] < q and cls["T_marg"] < q:
        v = "abf_sufficient"
    elif cls["T_cover"] < q and q <= cls["T_marg"] <= 0.8 * T:
        v = "establishment_limited"
    else:
        v = "intermediate"
    return dict(cls, T_gate=T_gate, gate_status=st, gate=dict(J0=J0, J_inf=Ji, threshold=thr, final=float(ser[-1]) if np.isfinite(ser[-1]) else None),
                verdict_block96=cls["verdict"], verdict=v, gate_series_cum24=ser.tolist())


cells = {}
for f in sorted(glob.glob(os.path.join(Z4, "B*.json"))):
    d = json.load(open(f)); cells[d["cell"]] = regate(f.replace(".json", ".npz"), d)
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
           table={n: dict(n_replicas=c.get("n_replicas"), verdict=c.get("verdict"), verdict_block96=c.get("verdict_block96"), T_cover=c.get("T_cover"), T_marg=c.get("T_marg"), T_gate=c.get("T_gate"), T=c.get("T"),
                          marg_status=c.get("marg_status"), gate_status=c.get("gate_status"), gate_final=(c.get("gate") or {}).get("final"), transits=c.get("transit_events"), unvisited=c.get("unvisited_bins")) for n, c in cells.items()},
           rule="A2: verdict from T_cover/T_marg only (Z4 gate histograms invalid); establishment_limited > intermediate; ties -> |N - 128| smallest; discovery excluded; none -> no Z5")
json.dump(res, open(os.path.join(Z4, "cell_choice.json"), "w"), indent=1, default=float)
print(f"{'cell':8s} {'N':>4s} {'T_cover/T':>10s} {'T_marg/T':>9s} {'T_gate/T':>9s} {'J_gate_end':>10s} {'transits':>8s} verdict (block-96 verdict)")
for n, c in res["table"].items():
    T = c["T"] or float("nan")
    f = lambda x: (f"{x / T:9.2f}" if x is not None and np.isfinite(x) else "      inf")
    gf = c.get("gate_final"); print(f"{n:8s} {str(c['n_replicas']):>4s} {f(c['T_cover'])} {f(c['T_marg'])} {f(c['T_gate'])} {(f'{gf:10.4f}' if gf is not None else '       n/a')} {str(c['transits']):>8s} {c['verdict']} ({c.get('verdict_block96')})")
print(f"CHOICE: {choice}  (go_z5 = {res['go_z5']})")
