#!/usr/bin/env python
"""Analyzer for the ZIF-8 Z5 block (docs/ZIF8_OT_Z4Z5.md).
  --stage calibration  blind marginal-action rule -> <root>/calibration/alpha_star.json
  --stage pilot|confirmatory  six arms on the compute axis: I_F^(C), e_F(C*) at h_read 0.05 from raw
        accumulators vs the umbrella F (full + split halves), D_gate^(C*) vs gate_reference_v2, paired
        contrasts, time-to-accuracy, genealogy, go rule -> REPORT.md, summary.json, go_nogo.json
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys
import numpy as np, torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, os.path.join(ROOT, "src"))
from alkanes import periodic as per
from zif8.core_zif8 import mean_force_regularized, js_divergence
CAMP = os.path.join(ROOT, "results/ot_repair_campaign/zif8/Z5")
REF = os.path.join(ROOT, "results/uniform_campaign/zif8/reference/reference_T300.npz"); GREF = os.path.join(ROOT, "cache/zif8/gate_reference_v2_T300.npz")
LABEL = {"abf": "A", "fr": "F", "ot": "T", "abf_r": "R", "fr_r": "F+R", "ot_r": "T+R"}
CONTRASTS = [("T", "A"), ("T", "F"), ("F", "A"), ("R", "A"), ("F+R", "F"), ("T+R", "T"), ("T+R", "R"), ("T+R", "F+R"), ("T+R", "A")]
N_BOOT, H_READ, MIN_COUNT = 10000, 0.05, 20.0


def load(raw):
    runs = {}
    for f in sorted(glob.glob(os.path.join(raw, "*.npz"))):
        d = np.load(f, allow_pickle=True); meta = json.loads(str(d["meta"])); runs[(meta["arm"], meta["ot"]["alpha"])] = (d, meta)
    return runs


def pmf_series(d, k_phi):
    fs = np.asarray(d["raw_fsum_t"], float); cs = np.asarray(d["raw_csum_t"], float); T, R, G = fs.shape
    grid, dphi = per.periodic_grid(G, dtype=torch.float64); K = per.wrapped_gaussian_kernel_matrix(grid, H_READ * k_phi)
    mf = mean_force_regularized(torch.as_tensor(fs.reshape(-1, G)), torch.as_tensor(cs.reshape(-1, G)), K, MIN_COUNT)
    return per.free_energy_from_mean_force(mf, grid, dphi).numpy().reshape(T, R, G)


def eF(pmf, F_ref):
    dd = pmf - F_ref[None, None, :]; dd = dd - dd.mean(-1, keepdims=True); return np.sqrt((dd * dd).mean(-1))     # (T, R)


def boot(x, level=0.95, seed=20260829):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed); meds = np.median(rng.choice(x, size=(N_BOOT, x.size)), axis=1); a = (1 - level) / 2
    return float(np.median(x)), float(np.quantile(meds, a)), float(np.quantile(meds, 1 - a))


def stage_calibration(root):
    runs = load(os.path.join(root, "calibration", "raw")); rows = {}
    for (arm, alpha), (d, meta) in runs.items():
        steps = np.asarray(d["steps"]); kl = np.asarray(d["kl_uniform"], float); sel = steps >= meta["sim"]["fr_start_steps"]
        J = np.trapezoid(kl[sel], steps[sel], axis=0)
        key = LABEL[arm] + (f"(a{alpha:g})" if alpha > 0 else "")
        rows[key] = dict(arm=arm, alpha=alpha, J_median=float(np.median(J)), capped=float(np.mean(d["ot_capped_frac"])) if "ot_capped_frac" in d.files else 0.0,
                         moved=float(np.mean(d["ot_moved_frac"])) if "ot_moved_frac" in d.files else 0.0, absdphi=float(np.mean(d["ot_absdphi_mean"])) if "ot_absdphi_mean" in d.files else 0.0,
                         nan=bool(not np.isfinite(np.asarray(d["pmf"])[-1]).all()), kl_final=float(np.median(kl[-1])))
    J_F = rows["F"]["J_median"]
    cands = sorted(((abs(math.log(v["J_median"] / J_F)), v["alpha"], k) for k, v in rows.items() if v["arm"] == "ot" and not v["nan"] and v["capped"] <= 0.5 and v["J_median"] > 0))
    best = cands[0]; ratio = rows[best[2]]["J_median"] / J_F
    sel = dict(alpha_star=rows[best[2]]["alpha"], ratio=ratio, in_band=(0.9 <= ratio <= 1.1), J_F=J_F, J_A=rows["A"]["J_median"], table=rows)
    json.dump(sel, open(os.path.join(root, "calibration", "alpha_star.json"), "w"), indent=1)
    for k, v in rows.items():
        print(f"  {k:10s} J {v['J_median']:9.1f} ratio {v['J_median'] / J_F:6.3f} KL_final {v['kl_final']:.4f} |dphi| {v['absdphi']:.4f} capped {v['capped']:.3f} moved {v['moved']:.3f} NaN {v['nan']}")
    print(f"alpha* = {sel['alpha_star']} (ratio {ratio:.3f}, {'in band' if sel['in_band'] else 'fallback: closest'})")


def stage_arms(root, stage):
    raw = os.path.join(root, stage, "raw"); out_dir = os.path.join(root, stage); runs = load(raw)
    ref = np.load(REF, allow_pickle=True); k_phi = float(ref["k_phi"]); F_full = np.asarray(ref["F"], float); F_split = np.asarray(ref["F_split"], float)
    g2 = np.load(GREF, allow_pickle=True); gref = np.asarray(g2["gate_hist_window_xi"], float)
    per_arm = {}
    for (arm, alpha), (d, meta) in runs.items():
        lab = LABEL[arm]; assert lab not in per_arm, f"two runs for {lab}"
        steps = np.asarray(d["steps"]); N = meta["n_replicas"]; inner = np.asarray(d["series_inner_steps"], float) if "series_inner_steps" in d.files else np.zeros(len(steps))
        C = N * steps + inner; pm = pmf_series(d, k_phi)
        e = eF(pm, F_full).T; eA = eF(pm, F_split[0]).T; eB = eF(pm, F_split[1]).T                                           # (R, T)
        blocks = np.asarray(d["gate_hist_block"], float)                                                                    # (T, R, nx, na)
        cum = np.cumsum(blocks, axis=0)
        Dg = np.zeros((blocks.shape[1], blocks.shape[0]))
        for r in range(blocks.shape[1]):
            for t in range(blocks.shape[0]):
                h = cum[t, r]; ok = (h.sum(-1) >= 200) & (gref.sum(-1) >= 200)
                Dg[r, t] = float(np.mean(js_divergence(h[ok], gref[ok]))) if ok.any() else np.nan
        per_arm[lab] = dict(C=C, e=e, eA=eA, eB=eB, Dg=Dg, N=N, n_steps=meta["n_steps"], inner_total=int(d["inner_steps_total"]) if "inner_steps_total" in d.files else 0,
                            ess=np.asarray(d["ancestor_ess"], float), wmax=np.asarray(d["max_ancestor_frac"], float), transits=int(np.asarray(d["n_crossings"]).sum()),
                            repl=int(np.asarray(d["total_replacement_events"]).sum()), wall_min=meta["wall_seconds"] / 60, nan=bool(not np.isfinite(pm[-1]).all()),
                            moved=float(np.mean(d["ot_moved_frac"])) if "ot_moved_frac" in d.files else 0.0, capped=float(np.mean(d["ot_capped_frac"])) if "ot_capped_frac" in d.files else 0.0)
    A = per_arm["A"]; C_star = A["N"] * A["n_steps"]; Cg = np.linspace(0, C_star, 3001)
    def on_C(p, y):
        I = np.zeros(y.shape[0]); fin = np.zeros(y.shape[0])
        for r in range(y.shape[0]):
            yi = np.interp(Cg, p["C"], np.nan_to_num(y[r], nan=np.nanmax(y[r]) if np.isfinite(y[r]).any() else 1.0)); I[r] = np.trapezoid(yi, Cg) / C_star; fin[r] = float(np.interp(C_star, p["C"], y[r]))
        return I, fin
    for lab, p in per_arm.items():
        p["I_F"], p["eF_star"] = on_C(p, p["e"]); p["I_FA"], p["eA_star"] = on_C(p, p["eA"]); p["I_FB"], p["eB_star"] = on_C(p, p["eB"]); p["I_D"], p["D_star"] = on_C(p, p["Dg"])
    def contrast(X, Y, key):
        x, y = per_arm[X][key], per_arm[Y][key]; n = min(len(x), len(y)); rel = (x[:n] - y[:n]) / y[:n]; med, lo, hi = boot(rel)
        return dict(median=med, ci95=[lo, hi], wins=int(np.sum(x[:n] < y[:n])), n=n)
    con = {}
    for X, Y in CONTRASTS:
        if X in per_arm and Y in per_arm:
            c = {k: contrast(X, Y, k) for k in ("I_F", "eF_star", "I_FA", "I_FB", "D_star", "I_D")}
            c["positive"] = bool(c["I_F"]["median"] <= -0.10 and c["I_F"]["ci95"][1] < 0 and (not np.isfinite(c["D_star"]["ci95"][1]) or c["D_star"]["ci95"][1] <= 0.10))
            c["robust_sign"] = bool(np.sign(c["I_F"]["median"]) == np.sign(c["I_FA"]["median"]) == np.sign(c["I_FB"]["median"]))
            con[f"{X} vs {Y}"] = c
    eps_A = float(np.median(A["eF_star"])); tta = {}
    for lab, p in per_arm.items():
        rr = []
        for r in range(p["e"].shape[0]):
            e = p["e"][r]; hit = next((p["C"][t] for t in range(len(e)) if e[t] <= eps_A and (t == len(e) - 1 or e[t + 1] <= eps_A)), None); rr.append(hit / C_star if hit else np.nan)
        tta[lab] = dict(median=(float(np.nanmedian(rr)) if np.isfinite(rr).any() else float("nan")), reached=int(np.isfinite(rr).sum()), n=len(rr))
    geneal = {lab: dict(ess_min=float(np.nanmin(p["ess"]) / p["N"]) if np.isfinite(p["ess"]).any() else None, wmax=float(np.nanmax(p["wmax"])) if np.isfinite(p["wmax"]).any() else None) for lab, p in per_arm.items()}
    geneal_ok = all((g["ess_min"] is None or g["ess_min"] >= 0.30) and (g["wmax"] is None or g["wmax"] <= 0.05) for g in geneal.values())
    go = bool("T+R vs A" in con and con["T+R vs A"]["positive"] and "T+R vs T" in con and con["T+R vs T"]["I_F"]["ci95"][1] < 0 and geneal_ok)
    summ = dict(stage=stage, C_star=C_star, eps_A=eps_A, go_confirmatory=go, genealogy=geneal, genealogy_ok=geneal_ok,
                arms={lab: dict(I_F=float(np.median(p["I_F"])), eF_star=float(np.median(p["eF_star"])), eF_end=float(np.median(p["e"][:, -1])), D_star=float(np.nanmedian(p["D_star"])),
                                C_end_over_Cstar=float(p["C"][-1] / C_star), inner_total=p["inner_total"], transits=p["transits"], repl=p["repl"], wall_min=p["wall_min"], nan=p["nan"],
                                moved_frac=p["moved"], capped_frac=p["capped"], tta=tta[lab], I_F_per_seed=p["I_F"].tolist(), eF_star_per_seed=p["eF_star"].tolist(), D_star_per_seed=p["D_star"].tolist()) for lab, p in per_arm.items()},
                contrasts=con)
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=float); json.dump(dict(go=go), open(os.path.join(out_dir, "go_nogo.json"), "w"))
    L = [f"# ZIF-8 Z5 {stage}: six arms on the compute axis (C* = {C_star / 1e6:.1f} M walker-steps; h_read 0.05 A; eps_A = {eps_A:.4f} kJ/mol)", "",
         "| arm | I_F^(C) | e_F(C*) | e_F(end) | D_gate(C*) | C(eps_A)/C* | inner/seed | transits | ESS_min/N | moved | capped | wall |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab in ("A", "F", "T", "R", "F+R", "T+R"):
        if lab in per_arm:
            a = summ["arms"][lab]; g = geneal[lab]
            L.append(f"| {lab} | {a['I_F']:.4f} | {a['eF_star']:.4f} | {a['eF_end']:.4f} | {a['D_star']:.4f} | {a['tta']['median']:.2f} ({a['tta']['reached']}/{a['tta']['n']}) | {a['inner_total'] / 1e6:.1f} M | {a['transits']} | "
                     f"{'-' if g['ess_min'] is None else f'{g[chr(101)+chr(115)+chr(115)+chr(95)+chr(109)+chr(105)+chr(110)]:.3f}'} | {a['moved_frac']:.2f} | {a['capped_frac']:.2f} | {a['wall_min']:.0f} |")
    L += ["", "| contrast | dI_F^(C) [CI95] wins | d e_F(C*) | d D_gate(C*) | positive? | sign robust (split halves) |", "|---|---|---|---|---|---|"]
    fmt = lambda c: f"{100 * c['median']:+.1f}% [{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}] {c['wins']}/{c['n']}"    # noqa: E731
    for k, c in con.items():
        L.append(f"| {k} | {fmt(c['I_F'])} | {fmt(c['eF_star'])} | {fmt(c['D_star'])} | {'YES' if c['positive'] else 'no'} | {c['robust_sign']} |")
    L += ["", f"Genealogy floors met: {geneal_ok}.  Go to confirmatory (T+R positive vs A AND T+R vs T CI95 upper < 0): **{go}**"]
    open(os.path.join(out_dir, "REPORT.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=["calibration", "pilot", "confirmatory"]); ap.add_argument("--root", default=CAMP)
    a = ap.parse_args()
    stage_calibration(a.root) if a.stage == "calibration" else stage_arms(a.root, a.stage)
