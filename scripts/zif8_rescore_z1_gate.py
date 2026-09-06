#!/usr/bin/env python
"""Re-score the Z1 conditional gate against the corrected gate reference (gate_reference_v2_T300.npz)."""
import json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
z = np.load(os.path.join(ROOT, "results/ot_repair_campaign/zif8/Z123/z123.npz")); J = json.load(open(os.path.join(ROOT, "results/ot_repair_campaign/zif8/Z123/z123.json")))
v2 = np.load(os.path.join(ROOT, "cache/zif8/gate_reference_v2_T300.npz"), allow_pickle=True); old = np.load(os.path.join(ROOT, "results/uniform_campaign/zif8/reference/reference_T300.npz"), allow_pickle=True)
edges = z["gate_edges"]; cg = 0.5 * (edges[1:] + edges[:-1]); gh = z["ghist_z1"]                        # (2 halves, S, 96)
sites = J["meta"]["sites"]; out = {}
def stats(h):
    p = h / max(h.sum(), 1); m = float(np.sum(p * cg)); return p, m, float(np.sqrt(max(np.sum(p * cg ** 2) - m ** 2, 0)))
print(f"{'site':13s} {'<A> Z1':>7s} {'<A> v2 ref':>10s} {'<A> old ref':>11s} {'TV v2':>6s} {'TV old':>6s} {'TV halves':>9s}")
for si, st in enumerate(sites):
    if st["ref_bins"] is None:
        continue
    p_rec, m_rec, sd_rec = stats(gh[0, si] + gh[1, si]); p_a, _, _ = stats(gh[0, si]); p_b, _, _ = stats(gh[1, si])
    p_v2, m_v2, sd_v2 = stats(np.asarray(v2["gate_hist_window_xi"])[st["ref_bins"]].sum(0)); p_old, m_old, _ = stats(np.asarray(old["gate_hist_window_xi"])[st["ref_bins"]].sum(0))
    tv = lambda a, b: 0.5 * float(np.abs(a - b).sum())
    out[st["name"]] = dict(A_z1=m_rec, A_v2=m_v2, A_old=m_old, sd_z1=sd_rec, sd_v2=sd_v2, tv_v2=tv(p_rec, p_v2), tv_old=tv(p_rec, p_old), tv_halves=tv(p_a, p_b), dA_v2=m_rec - m_v2)
    print(f"{st['name']:13s} {m_rec:7.4f} {m_v2:10.4f} {m_old:11.4f} {tv(p_rec, p_v2):6.3f} {tv(p_rec, p_old):6.3f} {tv(p_a, p_b):9.3f}")
gate = dict(max_abs_dA_v2=max(abs(r["dA_v2"]) for r in out.values()), max_tv_v2_minus_halves=max(r["tv_v2"] - r["tv_halves"] for r in out.values()))
gate["pass_conditional_v2"] = gate["max_abs_dA_v2"] <= 0.02 and gate["max_tv_v2_minus_halves"] <= 0.05
print("Z1 conditional gate vs corrected reference:", gate)
json.dump(dict(sites=out, gate=gate, v2_split_half_js=float(v2["split_half_js"]), v2_frames_at_image=float(v2["frames_at_image_window"])), open(os.path.join(ROOT, "results/ot_repair_campaign/zif8/Z123/z1_gate_rescored_v2.json"), "w"), indent=1)
