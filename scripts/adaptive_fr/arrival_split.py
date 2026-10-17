"""Experiment D5 (EXPLORATORY; docs/adaptive_fr_campaign/MECHANISM_ARRIVAL_PREREG.md): a direct test of the arrival-law
mechanism for continuous FR's late mean-force bias, from the read-only provenance record (engine ``diag_arr``) on bitwise
re-runs of BOTH arms of the two D2 blocks (ABF vs the same ABF + FR; no other arm).

Per block: identity gate; P1 per-provenance deviations d_p = <f>_p - Gamma_ref on the flank in ABF (cage vs window
provenance) and the eta conditional by provenance; P2 the same deviations and the provenance weights in FR, and the
arrival-law prediction pred_c = sum_p w_p^FR d_p^ABF of FR's late coarse-bin error; P3 decay of d_cage with time since
leaving the cage; P4 the carrier coordinate (ITE): which fibre / transverse coordinate differs by provenance in ABF and is
displaced in FR; P5 the sub-bin alternative s_p = Gamma'_ref <phi - centre>_p; P6 entry-side record.  Seed-jackknife s.e.

    ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao python scripts/adaptive_fr/arrival_split.py
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "convergence_fr"))
import systems as S  # noqa: E402
from eri import metrics as mt  # noqa: E402

NG, NC = 180, 18
Q = {"count": 0, "f": 1, "eta": 2, "ux2": 3, "bond": 4, "uz2": 5, "cy": 6, "sy": 7, "cz": 8, "sz": 9, "dphi": 10}
COORDS = ("eta", "ux2", "bond", "uz2", "cy", "sy", "cz", "sz")
IDENT_KEYS = ("sv_C", "sv_M", "sv_h", "sv_i", "sv_u", "sv_cond", "q_final")
GRID = (np.arange(NG) + 0.5) / NG * 2 * np.pi - np.pi
CGRID = (np.arange(NC) + 0.5) / NC * 2 * np.pi - np.pi
FLANK_C = np.nonzero((np.abs(CGRID) >= 0.5) & (np.abs(CGRID) < 2.0))[0]          # coarse bins 3-7, 10-14
FLANK_F = (np.abs(GRID) >= 0.5) & (np.abs(GRID) < 2.0)
BLOCKS = {
    "ERI256": dict(dir="results/adaptive_fr_campaign/D5_eri600_K256_arr", hist="results/convergence_fr_campaign/ERI_K256",
                   sel="results/adaptive_fr_campaign/D3_eri600_K256_sel", K=256, refbase="results/eri_campaign/reference",
                   variant="gard_tait", seeds=range(1_111_000, 1_111_016), label="ERI 600 K, K256, rate 2, continuous FR"),
    "ITE300_K256": dict(dir="results/adaptive_fr_campaign/D5_ite300_K256_arr", hist="results/convergence_fr_campaign/M300_256_rate1",
                        sel="results/adaptive_fr_campaign/D3_ite300_K256_sel", K=256, refbase="results/overnight/ite/reference",
                        variant="ite_itq3", seeds=range(1_121_000, 1_121_016), label="ITE 300 K, K256, rate 1, continuous FR"),
}
SD_FLOOR_FRAC = 0.01
PROV = ("cage", "window", "none")


def load_block(name):
    B = BLOCKS[name]
    out = {}
    for arm in ("fr", "abf"):
        new = [mt.load_run(f) for f in S.glob_both(B["dir"], f"{arm}_N{B['K']}_s*.npz")]
        old = {int(re.search(r"_s(\d+)\.npz$", f).group(1)): f for f in S.glob_both(B["hist"], f"{arm}_N{B['K']}_s*.npz")}
        assert [int(r["cfg"]["seed"]) for r in new] == list(B["seeds"]), (name, arm)
        ident = {}
        for r in new:
            h = np.load(old[int(r["cfg"]["seed"])], allow_pickle=False)
            ident[int(r["cfg"]["seed"])] = all(bool(np.array_equal(r[k], h[k])) for k in IDENT_KEYS)
        out[arm] = new
        out[f"identity_{arm}"] = ident
    # ABF within-bin sd of eta, ux2, bond, uz2 per coarse bin from the D4 record's late-half histograms (same trajectories)
    sel = [np.load(f, allow_pickle=False) for f in S.glob_both(B["sel"], f"abf_N{B['K']}_s*.npz")]
    H = sum(z["sel_fh"][1] for z in sel)
    ranges = sel[0]["sel_fh_ranges"]
    sd = {}
    for ci, coord in enumerate(("eta", "ux2", "bond", "uz2")):
        lo, hi = ranges[ci]
        mid = lo + (np.arange(H.shape[2]) + 0.5) / H.shape[2] * (hi - lo)
        h = H[ci].reshape(NC, -1, H.shape[2]).sum(1)
        tot = np.maximum(h.sum(1), 1e-300)
        m = (h * mid).sum(1) / tot
        v = np.sqrt(np.maximum((h * mid ** 2).sum(1) / tot - m ** 2, 0.0))
        sd[coord] = np.maximum(v, SD_FLOOR_FRAC * (hi - lo))
    out["abf_sd_coarse"] = sd
    return out


def coarse_dev(P, Gr, p):
    """Coarse-bin deviation of provenance p (exact: the class's own fine-bin counts weight both <f> and Gamma_ref)."""
    cnt = P[p, :, :, Q["count"]].sum(0)                       # (NG) over bands
    fs = P[p, :, :, Q["f"]].sum(0)
    Cc = cnt.reshape(NC, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = fs.reshape(NC, -1).sum(1) / Cc.sum(1)
        gref = (Cc * Gr.reshape(NC, -1)).sum(1) / Cc.sum(1)
    return g - gref, Cc.sum(1)


def coarse_mean(P, p, qname):
    cnt = P[p, :, :, Q["count"]].sum(0).reshape(NC, -1)
    s = P[p, :, :, Q[qname]].sum(0).reshape(NC, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s.sum(1) / cnt.sum(1)


def wrms(D, w):
    w = np.asarray(w, float)
    m = (w > 0) & np.isfinite(D)
    return float(np.sqrt((w[m] * D[m] ** 2).sum() / w[m].sum())) if m.any() else float("nan")


def stats(blk, idx, Fr, Gr):
    fr = [blk["fr"][i] for i in idx]
    ab = [blk["abf"][i] for i in idx]
    nsteps = int(fr[0]["save_steps"][-1])
    ih = int(np.searchsorted(fr[0]["save_steps"], nsteps // 2))
    PA = sum(r["arr_P"][1] for r in ab)                       # late half: (3, nb, NG, NQ)
    PF = sum(r["arr_P"][1] for r in fr)
    EA = sum(r["arr_E"][1] for r in ab)
    EF = sum(r["arr_E"][1] for r in fr)
    for r in fr + ab:                                         # the record partitions the late deposits exactly
        Cl = r["sv_C"][-1] - r["sv_C"][ih]
        assert np.array_equal(r["arr_P"][1][..., 0].sum((0, 1)), Cl) and np.array_equal(r["arr_E"][1][..., 0].sum((0, 1)), Cl)
    out = {}
    sdc = blk["abf_sd_coarse"]
    # ---------------- P1 / P2: per-provenance deviations and weights, coarse level ----------------
    dev, wgt, cnt = {}, {}, {}
    for arm, P in (("abf", PA), ("fr", PF)):
        tot = P[:, :, :, 0].sum((0, 1)).reshape(NC, -1).sum(1)
        for p, pn in enumerate(PROV):
            d, c = coarse_dev(P, Gr, p)
            dev[(arm, pn)] = d
            cnt[(arm, pn)] = c
            wgt[(arm, pn)] = c / np.maximum(tot, 1)
        # the arm's own late coarse error (all deposits), exact
        d_all, c_all = coarse_dev(P.sum(0, keepdims=True), Gr, 0)
        dev[(arm, "all")] = d_all
        cnt[(arm, "all")] = c_all
    pred = sum(wgt[("fr", pn)] * np.nan_to_num(dev[("abf", pn)]) for pn in PROV)      # sum_p w_p^FR d_p^ABF
    pred_minus_abf = pred - dev[("abf", "all")]
    # exact identity: e^FR - e^ABF = sum_p (w_p^FR - w_p^ABF) d_p^ABF  [weight term]  +  sum_p w_p^FR (d_p^FR - d_p^ABF)  [state term]
    term_w = sum((wgt[("fr", pn)] - wgt[("abf", pn)]) * np.nan_to_num(dev[("abf", pn)]) for pn in PROV)
    term_d = sum(wgt[("fr", pn)] * np.nan_to_num(dev[("fr", pn)] - dev[("abf", pn)]) for pn in PROV)
    fl = FLANK_C
    wfl = cnt[("fr", "all")][fl]
    out["P12"] = dict(
        dev={f"{a}_{p}": dev[(a, p)] for a in ("abf", "fr") for p in PROV + ("all",)},
        weight={f"{a}_{p}": wgt[(a, p)] for a in ("abf", "fr") for p in PROV},
        deposits={f"{a}_{p}": cnt[(a, p)] for a in ("abf", "fr") for p in PROV + ("all",)},
        pred_fr_error=pred, pred_minus_abf_residual=pred_minus_abf, excess=dev[("fr", "all")] - dev[("abf", "all")],
        weight_term=term_w, state_term=term_d,
        flank=dict(
            abf_balance=float(np.average(np.nan_to_num(dev[("abf", "all")])[fl], weights=cnt[("abf", "all")][fl])),
            abf_dev_cage_rms=wrms(dev[("abf", "cage")][fl], cnt[("abf", "cage")][fl]),
            abf_dev_window_rms=wrms(dev[("abf", "window")][fl], cnt[("abf", "window")][fl]),
            fr_dev_cage_rms=wrms(dev[("fr", "cage")][fl], cnt[("fr", "cage")][fl]),
            fr_dev_window_rms=wrms(dev[("fr", "window")][fl], cnt[("fr", "window")][fl]),
            dev_diff_cage_rms=wrms(dev[("fr", "cage")][fl] - dev[("abf", "cage")][fl], wfl),
            dev_diff_window_rms=wrms(dev[("fr", "window")][fl] - dev[("abf", "window")][fl], wfl),
            w_cage_abf=float(np.average(wgt[("abf", "cage")][fl], weights=cnt[("abf", "all")][fl])),
            w_cage_fr=float(np.average(wgt[("fr", "cage")][fl], weights=wfl)),
            w_window_abf=float(np.average(wgt[("abf", "window")][fl], weights=cnt[("abf", "all")][fl])),
            w_window_fr=float(np.average(wgt[("fr", "window")][fl], weights=wfl)),
            obs_fr_rms=wrms(dev[("fr", "all")][fl], wfl), obs_abf_rms=wrms(dev[("abf", "all")][fl], cnt[("abf", "all")][fl]),
            pred_rms=wrms(pred[fl], wfl), resid_rms=wrms(dev[("fr", "all")][fl] - pred[fl], wfl),
            obs_excess_rms=wrms(dev[("fr", "all")][fl] - dev[("abf", "all")][fl], wfl),
            resid_excess_rms=wrms((dev[("fr", "all")][fl] - dev[("abf", "all")][fl]) - pred_minus_abf[fl], wfl),
            sign_agreement=float(np.mean(np.sign(pred[fl]) == np.sign(dev[("fr", "all")][fl]))),
            weight_term_rms=wrms(term_w[fl], wfl), state_term_rms=wrms(term_d[fl], wfl),
            identity_check=float(np.abs((term_w + term_d - (dev[("fr", "all")] - dev[("abf", "all")]))[fl]).max()),
        ))
    # per-side signed flank means (left flank phi < 0: bins 3-7; right: 10-14)
    left, right = fl[:5], fl[5:]
    side = {}
    for a in ("abf", "fr"):
        for p in ("cage", "window", "all"):
            d = np.nan_to_num(dev[(a, p)])
            c = cnt[(a, p)]
            side[f"{a}_{p}"] = [float(np.average(d[left], weights=np.maximum(c[left], 1e-12))), float(np.average(d[right], weights=np.maximum(c[right], 1e-12)))]
    side["pred"] = [float(np.average(pred[left], weights=wfl[:5])), float(np.average(pred[right], weights=wfl[5:]))]
    side["weight_term"] = [float(np.average(term_w[left], weights=wfl[:5])), float(np.average(term_w[right], weights=wfl[5:]))]
    side["state_term"] = [float(np.average(term_d[left], weights=wfl[:5])), float(np.average(term_d[right], weights=wfl[5:]))]
    out["P12"]["flank_sides_left_right"] = side
    # ---------------- P1 / P4: conditionals by provenance (ABF) and FR - ABF displacement ----------------
    P4 = {}
    for coord in COORDS:
        mA = {p: coarse_mean(PA, i, coord) for i, p in enumerate(PROV)}
        mF = {p: coarse_mean(PF, i, coord) for i, p in enumerate(PROV)}
        allA = coarse_mean(PA.sum(0, keepdims=True), 0, coord)
        allF = coarse_mean(PF.sum(0, keepdims=True), 0, coord)
        s = sdc.get(coord, np.ones(NC))                        # transverse phases: natural units (no sd available)
        diff = (mA["cage"] - mA["window"]) / s
        disp = (allF - allA) / s
        # sign test: is the FR displacement toward the cage law? (+1 where sign(disp) == sign(mA_cage - allA))
        toward = np.sign(disp) * np.sign(mA["cage"] - allA)
        P4[coord] = dict(abf_cage_minus_window=diff, fr_minus_abf=disp,
                         flank_diff_rms=wrms(diff[fl], cnt[("abf", "all")][fl]), flank_disp_rms=wrms(disp[fl], wfl),
                         flank_toward_cage_share=float(np.average(toward[fl] > 0, weights=wfl)),
                         standardised=coord in sdc)
    out["P4"] = P4
    # ---------------- P3: decay with time since leaving the cage (flank, cage provenance), per band ----------------
    nb = PA.shape[1]
    P3 = {}
    for arm, P in (("abf", PA), ("fr", PF)):
        rows = []
        for b in range(nb):
            Pb = P[:, b:b + 1]
            d, c = coarse_dev(Pb, Gr, 0)
            e = coarse_mean(Pb, 0, "eta") - coarse_mean(PA.sum(0, keepdims=True), 0, "eta")
            rows.append(dict(dev_rms=wrms(d[fl], c[fl]), dev_left=float(np.average(np.nan_to_num(d[left]), weights=np.maximum(c[left], 1e-12))),
                             dev_right=float(np.average(np.nan_to_num(d[right]), weights=np.maximum(c[right], 1e-12))),
                             eta_rms_std=wrms(e[fl] / sdc["eta"][fl], c[fl]), deposits=float(c[fl].sum())))
        P3[arm] = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    fixed = {}
    for arm, P in (("abf", PA), ("fr", PF)):
        tab = np.full((nb, 6), np.nan)
        for b in range(nb):
            for k, cb in enumerate((4, 5, 6, 11, 12, 13)):
                sl = slice(cb * (NG // NC), (cb + 1) * (NG // NC))
                c_ = P[0, b, sl, 0]
                if c_.sum() > 1000:
                    tab[b, k] = P[0, b, sl, 1].sum() / c_.sum() - (c_ * Gr[sl]).sum() / c_.sum()
        fixed[arm] = tab
    P3["cage_dev_by_band_at_bins_4_5_6_11_12_13"] = fixed
    out["P3"] = P3
    # ---------------- P5: sub-bin alternative ----------------
    dG = (np.roll(Gr, -1) - np.roll(Gr, 1)) / (2 * 2 * np.pi / NG)           # Gamma'_ref per fine bin (periodic)
    P5 = {}
    for arm, P in (("abf", PA), ("fr", PF)):
        for p, pn in enumerate(PROV[:2]):
            cnt_f = P[p, :, :, 0].sum(0)
            off = np.where(cnt_f > 0, P[p, :, :, Q["dphi"]].sum(0) / np.maximum(cnt_f, 1), 0.0)
            s_f = dG * off                                             # predicted sub-bin bias per fine bin
            d_f = np.where(cnt_f > 0, P[p, :, :, 1].sum(0) / np.maximum(cnt_f, 1), 0.0) - Gr
            Cc = cnt_f.reshape(NC, -1)
            s_c = (s_f.reshape(NC, -1) * Cc).sum(1) / np.maximum(Cc.sum(1), 1e-12)
            P5[f"{arm}_{pn}"] = dict(subbin_pred_coarse=s_c, mean_offset_flank=float(np.average(off[FLANK_F], weights=np.maximum(cnt_f[FLANK_F], 1e-12))),
                                     subbin_rms_flank=wrms(s_f[FLANK_F], cnt_f[FLANK_F]), dev_rms_flank=wrms(d_f[FLANK_F], cnt_f[FLANK_F]),
                                     resid_rms_flank=wrms((d_f - s_f)[FLANK_F], cnt_f[FLANK_F]))
    out["P5"] = P5
    # ---------------- P6: entry side into the coarse bin (flank), cage side vs window side, by residence band ----------------
    P6 = {}
    for arm, Ee in (("abf", EA), ("fr", EF)):
        # side 0 = from lower; on the left flank (phi < 0) that is the cage side, on the right flank the window side
        cs = np.zeros((nb, NG, 2))
        ws = np.zeros((nb, NG, 2))
        for j in range(NG):
            lower_is_cage = GRID[j] < 0
            cs[:, j] = Ee[0 if lower_is_cage else 1, :, j, :2]
            ws[:, j] = Ee[1 if lower_is_cage else 0, :, j, :2]
        rows = {}
        for lab, A in (("cage_side", cs), ("window_side", ws)):
            tot_c = A[..., 0].sum(0).reshape(NC, -1)
            with np.errstate(invalid="ignore", divide="ignore"):
                d = A[..., 1].sum(0).reshape(NC, -1).sum(1) / tot_c.sum(1) - (tot_c * Gr.reshape(NC, -1)).sum(1) / tot_c.sum(1)
            byband = []
            for b in range(nb):
                cb_ = A[b, :, 0].reshape(NC, -1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    db = A[b, :, 1].reshape(NC, -1).sum(1) / cb_.sum(1) - (cb_ * Gr.reshape(NC, -1)).sum(1) / cb_.sum(1)
                byband.append(dict(dev_left=float(np.average(np.nan_to_num(db[left]), weights=np.maximum(cb_.sum(1)[left], 1e-12))),
                                   dev_right=float(np.average(np.nan_to_num(db[right]), weights=np.maximum(cb_.sum(1)[right], 1e-12))),
                                   share=float(cb_.sum(1)[fl].sum())))
            rows[lab] = dict(dev_coarse=d, dev_left=float(np.average(np.nan_to_num(d[left]), weights=np.maximum(tot_c.sum(1)[left], 1e-12))),
                             dev_right=float(np.average(np.nan_to_num(d[right]), weights=np.maximum(tot_c.sum(1)[right], 1e-12))),
                             weight_flank=float(tot_c.sum(1)[fl].sum()), by_band={k: np.array([r[k] for r in byband]) for k in byband[0]})
        tot_all = Ee[..., 0].sum((0, 1)).reshape(NC, -1).sum(1)[fl].sum()
        for lab in rows:
            rows[lab]["weight_flank"] = rows[lab]["weight_flank"] / tot_all
        P6[arm] = rows
    out["P6"] = P6
    out["band_steps"] = fr[0]["arr_band_steps"]
    out["dt"] = float(fr[0]["cfg"]["dt"])
    return out


def jackknife(blk, Fr, Gr, n=16):
    full = stats(blk, list(range(n)), Fr, Gr)
    reps = [stats(blk, [i for i in range(n) if i != d], Fr, Gr) for d in range(n)]

    def walk(f, rs):
        if isinstance(f, dict):
            return {k: walk(f[k], [r[k] for r in rs]) for k in f}
        if isinstance(f, (list, tuple)):
            a = np.array(rs, float)
            return np.sqrt((n - 1) / n * ((a - a.mean(0)) ** 2).sum(0))
        if isinstance(f, (float, int, np.floating, np.integer)) and not isinstance(f, bool):
            a = np.array(rs, float)
            return float(np.sqrt((n - 1) / n * ((a - a.mean()) ** 2).sum()))
        if isinstance(f, np.ndarray):
            a = np.array(rs, float)
            return np.sqrt((n - 1) / n * ((a - a.mean(0)) ** 2).sum(0))
        return None
    return full, walk(full, reps)


def tojson(x):
    if isinstance(x, dict):
        return {str(k): tojson(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tojson(v) for v in x]
    if isinstance(x, np.ndarray):
        return np.where(np.isfinite(x), x, None).tolist() if x.dtype.kind == "f" else x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        v = x.item()
        return v if np.isfinite(v) else None
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, range):
        return list(x)
    return x


def block(name):
    B = BLOCKS[name]
    mt.set_reference_base(S.dpath(B["refbase"]))
    Fr, Gr = mt.reference_for_grid(NG, B["variant"])
    blk = load_block(name)
    ident = {arm: all(blk[f"identity_{arm}"].values()) for arm in ("fr", "abf")}
    if not all(ident.values()):
        raise RuntimeError(f"{name}: identity gate FAILED {blk['identity_fr']} {blk['identity_abf']}")
    full, se = jackknife(blk, Fr, Gr)
    return dict(block=name, label=B["label"], identity_bitwise={arm: dict(n=16, all_equal=ident[arm]) for arm in ident},
                late_from_t=float((int(blk["fr"][0]["save_steps"][-1]) // 2) * full["dt"]),
                band_edges_tu=(full["band_steps"] * full["dt"]).tolist(), flank_coarse_bins=FLANK_C.tolist(), full=full, jackknife_se=se)


def fmt(v, s):
    return f"{v:+.3f}±{s:.3f}"


def report(o):
    F, E_ = o["full"], o["jackknife_se"]
    fl = o["flank_coarse_bins"]
    print(f"== {o['block']}: {o['label']} (late half from t = {o['late_from_t']:g}; identity gate fr {o['identity_bitwise']['fr']['all_equal']} abf {o['identity_bitwise']['abf']['all_equal']})")
    P, S_ = F["P12"], E_["P12"]
    print("  P1/P2 per-provenance late deviation <f>_p - Gamma_ref per coarse bin (flank bins 3-7 | 10-14), kJ/mol/rad:")
    for k in ("abf_cage", "abf_window", "abf_all", "fr_cage", "fr_window", "fr_all"):
        d, s = np.array(P["dev"][k], float), np.array(S_["dev"][k], float)
        print(f"     {k:10s}: " + " ".join(f"{d[c]:+.2f}" for c in fl[:5]) + " | " + " ".join(f"{d[c]:+.2f}" for c in fl[5:]) +
              f"   (s.e. {np.nanmean(s[fl]):.2f})")
    print("     weights w_cage (share of the bin's late deposits with cage provenance), ABF then FR:")
    for k in ("abf_cage", "fr_cage", "abf_window", "fr_window"):
        w = np.array(P["weight"][k], float)
        print(f"     {k:10s}: " + " ".join(f"{w[c]:.3f}" for c in fl[:5]) + " | " + " ".join(f"{w[c]:.3f}" for c in fl[5:]))
    pr, prs = np.array(P["pred_fr_error"], float), np.array(S_["pred_fr_error"], float)
    print("     pred (w^FR . d^ABF): " + " ".join(f"{pr[c]:+.2f}" for c in fl[:5]) + " | " + " ".join(f"{pr[c]:+.2f}" for c in fl[5:]) + f"   (s.e. {np.nanmean(prs[fl]):.2f})")
    f_ = P["flank"]
    s_ = S_["flank"]
    sd_ = P["flank_sides_left_right"]
    print(f"     EXACT decomposition of FR's flank excess over ABF (RMS): weight term {f_['weight_term_rms']:.3f}±{s_['weight_term_rms']:.3f} + state term {f_['state_term_rms']:.3f}±{s_['state_term_rms']:.3f} = excess {f_['obs_excess_rms']:.3f} (identity residual {f_['identity_check']:.1e}); "
          f"signed left/right: weight {sd_['weight_term'][0]:+.3f}/{sd_['weight_term'][1]:+.3f}, state {sd_['state_term'][0]:+.3f}/{sd_['state_term'][1]:+.3f}, excess {P['dev']['fr_all'][fl[0]] * 0 + np.average(np.nan_to_num(np.array(P['excess'], float))[fl[:5]], weights=np.array(P['deposits']['fr_all'], float)[fl[:5]]):+.3f}/{np.average(np.nan_to_num(np.array(P['excess'], float))[fl[5:]], weights=np.array(P['deposits']['fr_all'], float)[fl[5:]]):+.3f}")
    print(f"     flank summary: ABF balance {f_['abf_balance']:+.4f}±{s_['abf_balance']:.4f}; |d_cage| ABF {f_['abf_dev_cage_rms']:.3f}±{s_['abf_dev_cage_rms']:.3f} FR {f_['fr_dev_cage_rms']:.3f}±{s_['fr_dev_cage_rms']:.3f}; "
          f"|d_window| ABF {f_['abf_dev_window_rms']:.3f}±{s_['abf_dev_window_rms']:.3f} FR {f_['fr_dev_window_rms']:.3f}±{s_['fr_dev_window_rms']:.3f}")
    print(f"     (a) FR-ABF deviation difference RMS: cage {f_['dev_diff_cage_rms']:.3f}±{s_['dev_diff_cage_rms']:.3f}, window {f_['dev_diff_window_rms']:.3f}±{s_['dev_diff_window_rms']:.3f}")
    print(f"     (b) w_cage ABF {f_['w_cage_abf']:.4f}±{s_['w_cage_abf']:.4f} -> FR {f_['w_cage_fr']:.4f}±{s_['w_cage_fr']:.4f}; w_window ABF {f_['w_window_abf']:.4f} -> FR {f_['w_window_fr']:.4f}")
    print(f"     (c) observed FR flank error RMS {f_['obs_fr_rms']:.3f}±{s_['obs_fr_rms']:.3f} (ABF {f_['obs_abf_rms']:.3f}); predicted {f_['pred_rms']:.3f}±{s_['pred_rms']:.3f}; residual obs-pred {f_['resid_rms']:.3f}±{s_['resid_rms']:.3f}; "
          f"sign agreement {f_['sign_agreement']:.2f}; excess over ABF: obs {f_['obs_excess_rms']:.3f} residual {f_['resid_excess_rms']:.3f}")
    sd = P["flank_sides_left_right"]
    print("     signed flank means left(phi<0)/right: " + "; ".join(f"{k} {v[0]:+.3f}/{v[1]:+.3f}" for k, v in sd.items()))
    print("  P4 conditionals by provenance in ABF (cage - window, flank RMS, sd units where available) and FR - ABF displacement (flank RMS; share of flank weight displaced toward the cage law):")
    for c, v in F["P4"].items():
        s = E_["P4"][c]
        print(f"     {c:4s}{' (sd units)' if v['standardised'] else ' (natural)'}: cage-window {v['flank_diff_rms']:.4f}±{s['flank_diff_rms']:.4f}; FR-ABF {v['flank_disp_rms']:.4f}±{s['flank_disp_rms']:.4f}; toward cage {v['flank_toward_cage_share']:.2f}")
    print(f"  P3 cage-provenance flank deviation by time since leaving the cage (bands {o['band_edges_tu']} t.u.):")
    for arm in ("abf", "fr"):
        r, s = F["P3"][arm], E_["P3"][arm]
        print(f"     {arm}: |d| " + " ".join(f"{a:.3f}±{b:.3f}" for a, b in zip(r["dev_rms"], s["dev_rms"])) + " | left " + " ".join(f"{a:+.2f}" for a in r["dev_left"]) +
              " | right " + " ".join(f"{a:+.2f}" for a in r["dev_right"]) + " | eta(sd) " + " ".join(f"{a:.3f}" for a in r["eta_rms_std"]) + " | share " + " ".join(f"{a / sum(r['deposits']):.3f}" for a in r["deposits"]))
    for arm in ("abf", "fr"):
        t = np.array(F["P3"]["cage_dev_by_band_at_bins_4_5_6_11_12_13"][arm], float)
        print(f"     {arm} cage-provenance deviation by band at fixed coarse bins 4,5,6 | 11,12,13 (rows: bands): " +
              "; ".join(" ".join(f"{v:+.1f}" if np.isfinite(v) else "  ." for v in row) for row in t))
    print("  P5 sub-bin alternative (flank, fine bins): mean offset, predicted |Gamma' x offset| RMS vs observed |d_p| RMS, residual:")
    for k, v in F["P5"].items():
        s = E_["P5"][k]
        print(f"     {k:11s}: offset {v['mean_offset_flank']:+.5f} rad; sub-bin pred {v['subbin_rms_flank']:.4f}±{s['subbin_rms_flank']:.4f}; |d_p| {v['dev_rms_flank']:.3f}±{s['dev_rms_flank']:.3f}; resid {v['resid_rms_flank']:.3f}")
    print("  P6 entry side into the coarse bin (flank): cage-side vs window-side entries, signed left/right deviation and flank weight; by residence band:")
    for arm in ("abf", "fr"):
        for lab in ("cage_side", "window_side"):
            v = F["P6"][arm][lab]
            s = E_["P6"][arm][lab]
            print(f"     {arm} {lab:11s}: {v['dev_left']:+.3f}±{s['dev_left']:.3f} / {v['dev_right']:+.3f}±{s['dev_right']:.3f} weight {v['weight_flank']:.3f}±{s['weight_flank']:.3f} | by band left "
                  + " ".join(f"{a:+.2f}" for a in v["by_band"]["dev_left"]) + " right " + " ".join(f"{a:+.2f}" for a in v["by_band"]["dev_right"]))


def main():
    names = sys.argv[1:] or list(BLOCKS)
    res = {}
    for n in names:
        o = block(n)
        res[n] = o
        report(o)
    p = os.path.join(S.ROOT, "results", "adaptive_fr_campaign", "mechanism", "arrival_split.json")
    json.dump(tojson(res), open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
