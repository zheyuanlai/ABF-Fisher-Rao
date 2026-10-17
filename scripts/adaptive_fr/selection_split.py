"""Experiment D4 (EXPLORATORY; docs/adaptive_fr_campaign/MECHANISM_SELECTION_PREREG.md): a direct test of the selection
hypothesis of MECHANISM.md Sec. 2.3 reading 4, from the read-only selection diagnostic (engine ``diag_sel``) on bitwise
re-runs of BOTH arms of the two D2 blocks (ABF vs the same ABF + FR; no other arm).

Per block:
* identity gate: every re-run reproduces its historical arm bitwise (sv_C, sv_M, sv_h, sv_i, sv_u, sv_cond, q_final);
* P0 carrier: late-half population conditional displacement FR vs ABF per fibre coordinate (eta, u_x^2, bond, u_z^2),
  standardised by the ABF within-bin sd; the ITE carrier is the coordinate that exceeds 3 s.e. and ABF's split-half floor;
* P1 selection at birth / death: the quantile of the parent's (killed walker's) fibre value within the ABF late-half
  conditional law at its bin (u, v) and within the FR arm's own law (u', v'); tail share and mean vs the exact null
  expectation under random selection from the same bins;
* P2 ancestor classes: late-half (t > T/2) deposits labelled by the quartile of u (carrier coordinate) of their
  ancestor's birth event (founders separate): mean-force wRMS and F RMS vs the exact-MC reference, signed error per
  region and per coarse bin; a variance-matched ABF control (disjoint seed quartets / halves);
* P3 descendants' conditional vs time since birth, tail vs central classes, standardised deviation from ABF's;
* P4 locality: late error by birth region, by death region, and for deposits inside vs outside the birth coarse bin;
* P5 late error by time-since-birth band, all / tail / central.
Seed-jackknife s.e. (drop one of 16; every pooled law and class assignment is recomputed per replicate).

    ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao python scripts/adaptive_fr/selection_split.py
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

NG, NC, NFH = 180, 18, 40
COORDS = ("eta", "ux2", "bond", "uz2")
IDENT_KEYS = ("sv_C", "sv_M", "sv_h", "sv_i", "sv_u", "sv_cond", "q_final")
GRID = (np.arange(NG) + 0.5) / NG * 2 * np.pi - np.pi
REG = {"window": np.abs(GRID) < 0.5, "flank": (np.abs(GRID) >= 0.5) & (np.abs(GRID) < 2.0), "cage": np.abs(GRID) >= 2.0}
CGRID = (np.arange(NC) + 0.5) / NC * 2 * np.pi - np.pi
CREG = {"window": np.abs(CGRID) < 0.5, "flank": (np.abs(CGRID) >= 0.5) & (np.abs(CGRID) < 2.0), "cage": np.abs(CGRID) >= 2.0}
BLOCKS = {
    "ERI256": dict(dir="results/adaptive_fr_campaign/D3_eri600_K256_sel", hist="results/convergence_fr_campaign/ERI_K256",
                   K=256, refbase="results/eri_campaign/reference", variant="gard_tait", seeds=range(1_111_000, 1_111_016),
                   label="ERI 600 K, K256, rate 2, continuous FR"),
    "ITE300_K256": dict(dir="results/adaptive_fr_campaign/D3_ite300_K256_sel", hist="results/convergence_fr_campaign/M300_256_rate1",
                        K=256, refbase="results/overnight/ite/reference", variant="ite_itq3", seeds=range(1_121_000, 1_121_016),
                        label="ITE 300 K, K256, rate 1, continuous FR"),
}
CLASSES = ("founder", "Q1", "Q2", "Q3", "Q4", "tail", "central", "descendants", "all_fr", "abf")


def wrms(D, w):
    w = np.asarray(w, float)
    m = w > 0
    return float(np.sqrt((w[m] * D[m] ** 2).sum() / w[m].sum())) if m.any() else float("nan")


def region_of(bins):
    b = np.asarray(bins)
    out = np.full(b.shape, 1, dtype=np.int64)              # 0 window, 1 flank, 2 cage
    out[REG["window"][b]] = 0
    out[REG["cage"][b]] = 2
    return out


def load_block(name):
    B = BLOCKS[name]
    out = {}
    for arm in ("fr", "abf"):
        new = [mt.load_run(f) for f in S.glob_both(B["dir"], f"{arm}_N{B['K']}_s*.npz")]
        old = {int(re.search(r"_s(\d+)\.npz$", f).group(1)): f for f in S.glob_both(B["hist"], f"{arm}_N{B['K']}_s*.npz")}
        seeds = [int(r["cfg"]["seed"]) for r in new]
        assert seeds == list(B["seeds"]), (name, arm, seeds)
        ident = {}
        for r in new:
            h = np.load(old[int(r["cfg"]["seed"])], allow_pickle=False)
            ident[int(r["cfg"]["seed"])] = all(bool(np.array_equal(r[k], h[k])) for k in IDENT_KEYS)
            assert int(r["sel_overflow_events"]) == 0 and int(r["sel_untracked_deposits"]) == 0
            if arm == "fr":
                assert int(r["sel_n_events"]) == int(r["ist"][2]) == int(r["sel_ev_i"].shape[0])
        out[arm] = new
        out[f"identity_{arm}"] = ident
    return out


def quantiles(H, coord, bins, x, ranges):
    """Mid-rank quantile of x (events) within the histogram law H[coord, bins] (NFH bins on ranges[coord])."""
    lo, hi = ranges[coord]
    b = np.clip(np.floor((x - lo) / (hi - lo) * NFH).astype(np.int64), 0, NFH - 1)
    h = H[coord]                                              # (NG, NFH)
    cum = np.cumsum(h, axis=1)
    tot = cum[:, -1]
    below = cum[bins, b] - h[bins, b]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = (below + 0.5 * h[bins, b]) / tot[bins]
    return u


def null_expectation(H, coord, bins):
    """Under random selection from the law H at each event's bin: E[tail share], E[mean quantile] and the expected decile
    histogram of the quantile (exact for the discretised mid-rank quantile)."""
    h = H[coord]
    cum = np.cumsum(h, axis=1)
    tot = np.maximum(cum[:, -1], 1e-300)
    umid = (cum - 0.5 * h) / tot[:, None]                    # (NG, NFH) mid-rank quantile of each histogram bin
    p = h / tot[:, None]
    tail = ((umid < 0.25) | (umid >= 0.75)) * p
    dec = np.clip(np.floor(umid * 10).astype(int), 0, 9)
    nh = np.zeros(10)
    for d in range(10):
        nh[d] = ((dec == d) * p).sum(1)[bins].mean()
    return float(tail.sum(1)[bins].mean()), float((umid * p).sum(1)[bins].mean()), nh


SD_FLOOR_FRAC = 0.01                      # standardisation floor: 1 % of the coordinate's range


def sd_floor(coord, ranges):
    lo, hi = ranges[coord]
    return SD_FLOOR_FRAC * (hi - lo)


def degenerate(H, coord, ranges):
    """A coordinate is degenerate when more than half of the ABF deposit weight sits in coarse bins whose within-bin sd is
    below the standardisation floor (ITE: eta = cos(2 pi Z / a) is ~1 wherever the molecule sits on the channel axis)."""
    m, sd, tot = hist_moments(H, coord, ranges)
    return bool(tot[sd < sd_floor(coord, ranges)].sum() / tot.sum() > 0.5)


def hist_moments(H, coord, ranges, coarse=True):
    """Mean and sd of the law H[coord] per (coarse) bin from the histogram (bin midpoints)."""
    lo, hi = ranges[coord]
    mid = lo + (np.arange(NFH) + 0.5) / NFH * (hi - lo)
    h = H[coord].reshape(NC, -1, NFH).sum(1) if coarse else H[coord]
    tot = np.maximum(h.sum(1), 1e-300)
    m = (h * mid).sum(1) / tot
    v = (h * mid ** 2).sum(1) / tot - m ** 2
    return m, np.sqrt(np.maximum(v, 0.0)), tot


def stats(blk, idx, Fr, Gr, ref, carrier_for_classes=None):
    """All pooled statistics for the seeds in idx (list of indices into the 16 runs)."""
    fr = [blk["fr"][i] for i in idx]
    ab = [blk["abf"][i] for i in idx]
    ranges = fr[0]["sel_fh_ranges"]
    nsteps = int(fr[0]["save_steps"][-1])
    ih = int(np.searchsorted(fr[0]["save_steps"], nsteps // 2))
    assert int(fr[0]["save_steps"][ih]) == nsteps // 2 == int(fr[0]["sel_split_step"])
    HA = sum(r["sel_fh"][1] for r in ab)                      # ABF late-half law (4, NG, NFH)
    HF = sum(r["sel_fh"][1] for r in fr)                      # FR late-half law
    out = {}
    # ---------------- late-half estimators of the arms (checks) ----------------
    def late(r):
        return r["sv_M"][-1] - r["sv_M"][ih], r["sv_C"][-1] - r["sv_C"][ih]
    Mab = sum(late(r)[0] for r in ab)
    Cab = sum(late(r)[1] for r in ab)
    Mfr = sum(late(r)[0] for r in fr)
    Cfr = sum(late(r)[1] for r in fr)
    for r in fr:                                              # the record partitions the deposits exactly
        Ml, Cl = late(r)
        assert np.array_equal(r["sel_evC"][:, 1].sum(0) + r["sel_fdC"][1], Cl)
        assert np.allclose(r["sel_evM"][:, 1].sum(0) + r["sel_fdM"][1], Ml, rtol=0, atol=1e-8 * np.abs(Ml).max())
    for r in ab:
        assert np.array_equal(r["sel_fdC"][1], late(r)[1]) and r["sel_n_events"] == 0

    def est(M, C):
        G = mt.mean_force(M, C)
        D = G - Gr
        Cc = C.reshape(NC, -1)
        coarse = (D.reshape(NC, -1) * Cc).sum(1) / np.maximum(Cc.sum(1), 1e-12)
        return dict(wrms=wrms(D, C), f_rms=float(mt.centred_rms(mt.histogram_pmf(G) - Fr)) if (C > 0).all() else float("nan"),
                    deposits=float(C.sum()),
                    region={k: float(np.average(D[m], weights=np.maximum(C[m], 1e-12))) for k, m in REG.items()},
                    coarse=coarse)
    # ---------------- P0: population conditional displacement, late half ----------------
    P0 = {}
    fdB_a = sum(r["sel_fdB"][1] for r in ab)                  # ABF late (NC, 6)
    popF = sum(r["sel_evB"][:, 1].sum((0, 1)) + r["sel_fdB"][1] for r in fr)   # FR late, all deposits (NC, 6)
    cc = ref["counts"].reshape(NC, -1)
    for ci, coord in enumerate(COORDS):
        mA, sdA, _ = hist_moments(HA, ci, ranges)
        m_abf = fdB_a[:, ci + 2] / np.maximum(fdB_a[:, 0], 1)
        m_fr = popF[:, ci + 2] / np.maximum(popF[:, 0], 1)
        rkey = {"eta": "eta", "ux2": "ux2", "bond": "bond", "uz2": "uz2"}[coord]
        m_ref = (ref[rkey].reshape(NC, -1) * cc).sum(1) / cc.sum(1)
        w = popF[:, 0] / popF[:, 0].sum()
        dg = degenerate(HA, ci, ranges)
        z = (m_fr - m_abf) / np.maximum(sdA, sd_floor(ci, ranges))
        P0[coord] = dict(fr_minus_abf_standardised=z, rms_standardised=float(np.sqrt((w * z ** 2).sum())), degenerate=dg,
                         rms_natural=float(np.sqrt((w * (m_fr - m_abf) ** 2).sum())),
                         fr_minus_ref_rms=float(np.sqrt((w * (m_fr - m_ref) ** 2).sum())),
                         abf_minus_ref_rms=float(np.sqrt((w * (m_abf - m_ref) ** 2).sum())),
                         abf_sd_mean=float(sdA.mean()), abf_mean=m_abf, fr_mean=m_fr)
    out["P0"] = P0
    # ---------------- P1: selection at birth and death ----------------
    P1 = {}
    EV_I = np.concatenate([r["sel_ev_i"] for r in fr])
    EV_F = np.concatenate([r["sel_ev_f"] for r in fr])
    pb, kb = EV_I[:, 3], EV_I[:, 4]
    U = {}
    for ci, coord in enumerate(COORDS):
        u_abf = quantiles(HA, ci, pb, EV_F[:, ci], ranges)
        u_fr = quantiles(HF, ci, pb, EV_F[:, ci], ranges)
        v_abf = quantiles(HA, ci, kb, EV_F[:, 4 + ci], ranges)
        v_fr = quantiles(HF, ci, kb, EV_F[:, 4 + ci], ranges)
        U[coord] = u_abf
        tn_b, mn_b, nh_b = null_expectation(HF, ci, pb)
        tn_d, mn_d, nh_d = null_expectation(HF, ci, kb)
        tna_b, mna_b, nha_b = null_expectation(HA, ci, pb)
        tail = lambda u: float(np.mean((u < 0.25) | (u >= 0.75)))
        P1[coord] = dict(n_events=int(len(u_abf)),
                         degenerate=degenerate(HA, ci, ranges),
                         birth_vs_abf=dict(tail=tail(u_abf), mean=float(u_abf.mean()), null_tail=tna_b, null_mean=mna_b,
                                           hist=np.histogram(u_abf, bins=10, range=(0, 1))[0] / len(u_abf), null_hist=nha_b),
                         birth_vs_fr=dict(tail=tail(u_fr), mean=float(u_fr.mean()), null_tail=tn_b, null_mean=mn_b,
                                          hist=np.histogram(u_fr, bins=10, range=(0, 1))[0] / len(u_fr), null_hist=nh_b,
                                          late_births_tail=tail(u_fr[EV_I[:, 0] > nsteps // 2]),
                                          late_births_mean=float(u_fr[EV_I[:, 0] > nsteps // 2].mean())),
                         death_vs_abf=dict(tail=tail(v_abf), mean=float(v_abf.mean())),
                         death_vs_fr=dict(tail=tail(v_fr), mean=float(v_fr.mean()), null_tail=tn_d, null_mean=mn_d,
                                          hist=np.histogram(v_fr, bins=10, range=(0, 1))[0] / len(v_fr), null_hist=nh_d))
        # net fibre selection of the birth-death exchange (births add at u', deaths remove at v'): > 0 pushes the
        # population toward high values; per birth region
        rb_, rd_ = region_of(pb), region_of(kb)
        P1[coord]["net_exchange_mean"] = float(u_fr.mean() - v_fr.mean())
        P1[coord]["birth_vs_fr"]["mean_by_region"] = np.array([float(u_fr[rb_ == g].mean()) for g in range(3)])
        P1[coord]["death_vs_fr"]["mean_by_region"] = np.array([float(v_fr[rd_ == g].mean()) for g in range(3)])
    # birth / death region shares (W, F, C), whole run and late half; late net source per region per t.u. per walker
    rb, rd = region_of(pb), region_of(kb)
    late_ev = EV_I[:, 0] > nsteps // 2
    T_late = (nsteps - nsteps // 2) * float(fr[0]["cfg"]["dt"]) * len(fr)
    K = int(fr[0]["cfg"]["N"])
    P1["birth_region_share"] = np.bincount(rb, minlength=3) / len(rb)
    P1["death_region_share"] = np.bincount(rd, minlength=3) / len(rd)
    P1["late_births_per_tu_per_walker_by_region"] = np.bincount(rb[late_ev], minlength=3) / T_late / K
    P1["late_deaths_per_tu_per_walker_by_region"] = np.bincount(rd[late_ev], minlength=3) / T_late / K
    P1["late_net_source_by_region"] = P1["late_births_per_tu_per_walker_by_region"] - P1["late_deaths_per_tu_per_walker_by_region"]
    P1["late_events_per_tu_per_run"] = float(late_ev.sum() / ((nsteps - nsteps // 2) * float(fr[0]["cfg"]["dt"])) / len(fr))
    out["P1"] = P1
    # ---------------- P2: ancestor classes (per carrier coordinate), late half ----------------
    P2, P4, P5, P3 = {}, {}, {}, {}
    nb = fr[0]["sel_evB"].shape[2]
    for ci, coord in enumerate(COORDS):
        if carrier_for_classes is not None and coord not in carrier_for_classes:
            continue
        acc = {k: [np.zeros(NG), np.zeros(NG)] for k in ("Q1", "Q2", "Q3", "Q4")}
        accP = {k: [np.zeros(NG), np.zeros(NG)] for k in ("Q1", "Q2", "Q3", "Q4")}       # classes vs the FR population law
        accT = {k: [np.zeros(NG), np.zeros(NG)] for k in ("born_early", "born_late")}
        accB = {k: np.zeros((nb, NC, 6)) for k in ("tail", "central")}
        accB_all = np.zeros((nb, NC, 6))
        accB_early = {k: np.zeros((nb, NC, 6)) for k in ("tail", "central")}
        loc = {k: [np.zeros(NG), np.zeros(NG)] for k in ("local", "nonlocal")}
        rxc = np.zeros((3, 4, 2, NG))                          # POST HOC (P6): birth region x class -> late C, M
        breg = {k: [np.zeros(NG), np.zeros(NG)] for k in ("window", "flank", "cage")}
        dreg = {k: [np.zeros(NG), np.zeros(NG)] for k in ("window", "flank", "cage")}
        fdC = np.zeros(NG)
        fdM = np.zeros(NG)
        for r in fr:
            ev_i, ev_f = r["sel_ev_i"], r["sel_ev_f"]
            u = quantiles(HA, ci, ev_i[:, 3], ev_f[:, ci], ranges)
            q = np.digitize(u, [0.25, 0.5, 0.75])            # 0..3
            qp = np.digitize(quantiles(HF, ci, ev_i[:, 3], ev_f[:, ci], ranges), [0.25, 0.5, 0.75])
            C1, M1 = r["sel_evC"][:, 1], r["sel_evM"][:, 1]
            for k, kk in enumerate(("Q1", "Q2", "Q3", "Q4")):
                m = q == k
                acc[kk][0] += C1[m].sum(0)
                acc[kk][1] += M1[m].sum(0)
                mp = qp == k
                accP[kk][0] += C1[mp].sum(0)
                accP[kk][1] += M1[mp].sum(0)
            early = ev_i[:, 0] <= nsteps // 2
            accT["born_early"][0] += C1[early].sum(0)
            accT["born_early"][1] += M1[early].sum(0)
            accT["born_late"][0] += C1[~early].sum(0)
            accT["born_late"][1] += M1[~early].sum(0)
            tailm = (q == 0) | (q == 3)
            accB["tail"] += r["sel_evB"][tailm, 1].sum(0)
            accB["central"] += r["sel_evB"][~tailm, 1].sum(0)
            accB_early["tail"] += r["sel_evB"][tailm, 0].sum(0)
            accB_early["central"] += r["sel_evB"][~tailm, 0].sum(0)
            accB_all += r["sel_evB"][:, 1].sum(0)
            rb_ = region_of(ev_i[:, 3])
            for g in range(3):
                for k in range(4):
                    mm = (rb_ == g) & (q == k)
                    rxc[g, k, 0] += C1[mm].sum(0)
                    rxc[g, k, 1] += M1[mm].sum(0)
            cb = ev_i[:, 3] * NC // NG
            lm = (np.arange(NG) // (NG // NC))[None, :] == cb[:, None]
            loc["local"][0] += (C1 * lm).sum(0)
            loc["local"][1] += (M1 * lm).sum(0)
            loc["nonlocal"][0] += (C1 * ~lm).sum(0)
            loc["nonlocal"][1] += (M1 * ~lm).sum(0)
            for g, name in enumerate(("window", "flank", "cage")):
                mb = region_of(ev_i[:, 3]) == g
                breg[name][0] += C1[mb].sum(0)
                breg[name][1] += M1[mb].sum(0)
                md = region_of(ev_i[:, 4]) == g
                dreg[name][0] += C1[md].sum(0)
                dreg[name][1] += M1[md].sum(0)
            fdC += r["sel_fdC"][1]
            fdM += r["sel_fdM"][1]
        groups = {"founder": (fdM, fdC)}
        for kk in ("Q1", "Q2", "Q3", "Q4"):
            groups[kk] = (acc[kk][1], acc[kk][0])
        groups["tail"] = (acc["Q1"][1] + acc["Q4"][1], acc["Q1"][0] + acc["Q4"][0])
        groups["central"] = (acc["Q2"][1] + acc["Q3"][1], acc["Q2"][0] + acc["Q3"][0])
        groups["descendants"] = (sum(acc[k][1] for k in acc), sum(acc[k][0] for k in acc))
        groups["all_fr"] = (Mfr, Cfr)
        groups["abf"] = (Mab, Cab)
        assert np.array_equal(groups["descendants"][1] + fdC, Cfr)
        tab = {g: est(M, C) for g, (M, C) in groups.items()}
        for g in tab:
            tab[g]["share_of_late_fr_deposits"] = tab[g]["deposits"] / float(Cfr.sum())
        gp = {kk: (accP[kk][1], accP[kk][0]) for kk in accP}
        gp["tail"] = (accP["Q1"][1] + accP["Q4"][1], accP["Q1"][0] + accP["Q4"][0])
        gp["central"] = (accP["Q2"][1] + accP["Q3"][1], accP["Q2"][0] + accP["Q3"][0])
        tabp = {g: est(M, C) for g, (M, C) in gp.items()}
        for g in tabp:
            tabp[g]["share_of_late_fr_deposits"] = tabp[g]["deposits"] / float(Cfr.sum())
        tab["classes_vs_fr_population_law"] = tabp
        tab["lineage_born"] = {k: dict(est(v[1], v[0]), share_of_late_fr_deposits=float(v[0].sum() / Cfr.sum())) for k, v in accT.items()}
        P2[coord] = tab
        # P6 (post hoc, not preregistered): birth region x fibre class at birth.  If the fibre state carries the error,
        # the class changes the error WITHIN a birth region; if it does not, the carrier is not this coordinate.
        P6c = {}
        for g, name in enumerate(("window", "flank", "cage")):
            P6c[name] = {kk: est(rxc[g, k, 1], rxc[g, k, 0]) for k, kk in enumerate(("Q1", "Q2", "Q3", "Q4"))}
            P6c[name]["tail"] = est(rxc[g, 0, 1] + rxc[g, 3, 1], rxc[g, 0, 0] + rxc[g, 3, 0])
            P6c[name]["central"] = est(rxc[g, 1, 1] + rxc[g, 2, 1], rxc[g, 1, 0] + rxc[g, 2, 0])
            P6c[name]["below_median"] = est(rxc[g, 0, 1] + rxc[g, 1, 1], rxc[g, 0, 0] + rxc[g, 1, 0])
            P6c[name]["above_median"] = est(rxc[g, 2, 1] + rxc[g, 3, 1], rxc[g, 2, 0] + rxc[g, 3, 0])
            for kk in P6c[name]:
                P6c[name][kk]["share_of_late_fr_deposits"] = P6c[name][kk]["deposits"] / float(Cfr.sum())
        P4[coord] = dict(by_birth_region={k: est(v[1], v[0]) for k, v in breg.items()}, region_x_class=P6c,
                         by_death_region={k: est(v[1], v[0]) for k, v in dreg.items()},
                         local=est(loc["local"][1], loc["local"][0]), non_local=est(loc["nonlocal"][1], loc["nonlocal"][0]))
        # P5 age bands (coarse level; reference coarse-averaged with its own counts)
        Gc_ref = (Gr.reshape(NC, -1) * cc).sum(1) / cc.sum(1)
        def band_tab(B):
            cnt, fs = B[..., 0], B[..., 1]
            G = fs / np.maximum(cnt, 1)
            D = G - Gc_ref[None, :]
            w = cnt / np.maximum(cnt.sum(1, keepdims=True), 1e-12)
            return dict(wrms_coarse=np.sqrt((w * D ** 2).sum(1)), deposits=cnt.sum(1), signed_coarse=D)
        def coarse_ref(M, C):
            Cc = C.reshape(NC, -1)
            Gc = (M.reshape(NC, -1)).sum(1) / np.maximum(Cc.sum(1), 1)
            w = Cc.sum(1) / Cc.sum()
            return float(np.sqrt((w * (Gc - Gc_ref) ** 2).sum()))
        P5[coord] = dict(all=band_tab(accB_all), tail=band_tab(accB["tail"]), central=band_tab(accB["central"]),
                         abf_coarse_wrms=coarse_ref(Mab, Cab), all_fr_coarse_wrms=coarse_ref(Mfr, Cfr))
        # P3 descendants' conditional vs age (late half; early half supplementary), standardised by the ABF law
        P3c = {}
        mA_fine, sdA_c, _ = hist_moments(HA, ci, ranges)
        m_abf = fdB_a[:, ci + 2] / np.maximum(fdB_a[:, 0], 1)
        for grp in ("tail", "central"):
            for half, B in (("late", accB[grp]), ("early", accB_early[grp])):
                cnt = B[..., 0]
                m = B[..., ci + 2] / np.maximum(cnt, 1)
                z = (m - m_abf[None, :]) / np.maximum(sdA_c, sd_floor(ci, ranges))[None, :]
                w = cnt / np.maximum(cnt.sum(1, keepdims=True), 1e-12)
                P3c[f"{grp}_{half}"] = dict(rms_standardised=np.sqrt((w * z ** 2).sum(1)),
                                            mean_standardised=(w * z).sum(1), deposits=cnt.sum(1))
        # birth- / death-state displacement (moment statistic, complementary to the rank statistic u'): the parent's
        # (killed walker's) exact fibre value minus the ABF arm's EXACT deposit-sum conditional mean, in units of the
        # ABF within-bin sd.  Coarse: the late-half mean per coarse bin (fdB); fine: the whole-run mean per fine bin
        # (Seta / Sux2 / Sbond / Suz2 over C).  Histogram-midpoint means are NOT used here (for a law piled against an
        # edge of its range they are biased by up to half a histogram bin).
        sdc = np.maximum(sdA_c, sd_floor(ci, ranges))
        key = ("Seta", "Sux2", "Sbond", "Suz2")[ci]
        mA_fine_exact = sum(r[key] for r in ab) / np.maximum(sum(r["sv_C"][-1] for r in ab), 1)
        cbp, cbk = pb * NC // NG, kb * NC // NG
        P3c["birth_state_mean_standardised"] = float(((EV_F[:, ci] - m_abf[cbp]) / sdc[cbp]).mean())
        P3c["birth_state_mean_standardised_fine_law"] = float(((EV_F[:, ci] - mA_fine_exact[pb]) / sdc[cbp]).mean())
        P3c["death_state_mean_standardised"] = float(((EV_F[:, 4 + ci] - m_abf[cbk]) / sdc[cbk]).mean())
        P3c["death_state_mean_standardised_fine_law"] = float(((EV_F[:, 4 + ci] - mA_fine_exact[kb]) / sdc[cbk]).mean())
        # the FR population's own late-half displacement at the birth bins, same units (what births would show if they
        # were an unbiased draw from the displaced population)
        m_fr_c = popF[:, ci + 2] / np.maximum(popF[:, 0], 1)
        P3c["population_displacement_at_birth_bins_standardised"] = float(((m_fr_c - m_abf)[cbp] / sdc[cbp]).mean())
        P3[coord] = P3c
    out.update(P2=P2, P3=P3, P4=P4, P5=P5, band_steps=fr[0]["sel_band_steps"], late_from_step=nsteps // 2,
               dt=float(fr[0]["cfg"]["dt"]))
    return out


def jackknife(blk, Fr, Gr, ref, n=16):
    full = stats(blk, list(range(n)), Fr, Gr, ref)
    reps = [stats(blk, [i for i in range(n) if i != d], Fr, Gr, ref) for d in range(n)]

    def walk(f, rs):
        if isinstance(f, dict):
            return {k: walk(f[k], [r[k] for r in rs]) for k in f}
        if isinstance(f, (float, int, np.floating, np.integer)) and not isinstance(f, bool):
            a = np.array(rs, float)
            return float(np.sqrt((n - 1) / n * ((a - a.mean()) ** 2).sum()))
        if isinstance(f, np.ndarray):
            a = np.array(rs, float)
            return np.sqrt((n - 1) / n * ((a - a.mean(0)) ** 2).sum(0))
        return None
    return full, walk(full, reps)


def variance_matched_abf(blk, Fr, Gr):
    """Late-half ABF wRMS from disjoint seed quartets (~1/4 of the deposits) and halves (~1/2): the pure sample-size effect."""
    ab = blk["abf"]
    nsteps = int(ab[0]["save_steps"][-1])
    ih = int(np.searchsorted(ab[0]["save_steps"], nsteps // 2))
    out = {}
    for name, size in (("quarter", 4), ("half", 8)):
        vals, fvals = [], []
        for s0 in range(0, 16, size):
            M = sum(ab[i]["sv_M"][-1] - ab[i]["sv_M"][ih] for i in range(s0, s0 + size))
            C = sum(ab[i]["sv_C"][-1] - ab[i]["sv_C"][ih] for i in range(s0, s0 + size))
            G = mt.mean_force(M, C)
            vals.append(wrms(G - Gr, C))
            fvals.append(float(mt.centred_rms(mt.histogram_pmf(G) - Fr)))
        out[name] = dict(wrms_mean=float(np.mean(vals)), wrms_values=vals, f_rms_mean=float(np.mean(fvals)), f_rms_values=fvals)
    return out


def abf_split_half_floor(blk):
    """ABF late-half conditional: standardised RMS difference between the two 8-seed halves / 2 (s.e. proxy of the 16-seed mean)."""
    ab = blk["abf"]
    ranges = ab[0]["sel_fh_ranges"]
    HA = sum(r["sel_fh"][1] for r in ab)
    A = sum(r["sel_fdB"][1] for r in ab[:8])
    B = sum(r["sel_fdB"][1] for r in ab[8:])
    out = {}
    for ci, coord in enumerate(COORDS):
        _, sd, _ = hist_moments(HA, ci, ranges)
        mA = A[:, ci + 2] / np.maximum(A[:, 0], 1)
        mB = B[:, ci + 2] / np.maximum(B[:, 0], 1)
        w = (A[:, 0] + B[:, 0]) / (A[:, 0] + B[:, 0]).sum()
        out[coord] = float(np.sqrt((w * ((mA - mB) / np.maximum(sd, sd_floor(ci, ranges))) ** 2).sum()) / 2)
    return out


def tojson(x):
    if isinstance(x, dict):
        return {str(k): tojson(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tojson(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, range):
        return list(x)
    return x


def decide_carrier(name, P0, se, floor):
    """P0 rule: the coordinate with the largest standardised late displacement, if > 3 s.e. and > the ABF split-half floor."""
    best, bval = None, -1.0
    for c in COORDS:
        v = P0[c]["rms_standardised"]
        if not P0[c]["degenerate"] and v > bval:
            best, bval = c, v
    ok = bval > 3 * se[best]["rms_standardised"] and bval > floor[best]
    return best, bool(ok)


def block(name):
    B = BLOCKS[name]
    mt.set_reference_base(S.dpath(B["refbase"]))
    Fr, Gr = mt.reference_for_grid(NG, B["variant"])
    ref = np.load(os.path.join(S.dpath(B["refbase"]), B["variant"], "reference_profile.npz"))
    blk = load_block(name)
    ident = {arm: all(blk[f"identity_{arm}"].values()) for arm in ("fr", "abf")}
    if not all(ident.values()):
        raise RuntimeError(f"{name}: identity gate FAILED {blk['identity_fr']} {blk['identity_abf']}")
    full, se = jackknife(blk, Fr, Gr, ref)
    floor = abf_split_half_floor(blk)
    carrier, found = decide_carrier(name, full["P0"], se["P0"], floor)
    out = dict(block=name, label=B["label"], identity_bitwise={arm: dict(n=16, all_equal=ident[arm]) for arm in ident},
               n_events_per_run=[int(r["sel_n_events"]) for r in blk["fr"]],
               replacements_per_run=[int(r["ist"][2]) for r in blk["fr"]],
               carrier=dict(coordinate=carrier, qualifies=found, prereg_eri_carrier="eta", abf_split_half_floor=floor),
               late_from_t=float(full["late_from_step"] * full["dt"]), band_edges_tu=(full["band_steps"] * full["dt"]).tolist(),
               full=full, jackknife_se=se, variance_matched_abf=variance_matched_abf(blk, Fr, Gr))
    return out


def report(o):
    n = o["block"]
    print(f"== {n}: {o['label']} (late half from t = {o['late_from_t']:g}; identity gate fr {o['identity_bitwise']['fr']['all_equal']} "
          f"abf {o['identity_bitwise']['abf']['all_equal']}; events/run median {np.median(o['n_events_per_run']):.0f})")
    F, E_ = o["full"], o["jackknife_se"]
    print("  P0 late-half population conditional, FR - ABF, standardised by the ABF within-bin sd (count-weighted RMS over 18 coarse bins):")
    for c in COORDS:
        p, s = F["P0"][c], E_["P0"][c]
        print(f"     {c:4s}{' (DEGENERATE)' if p['degenerate'] else ''}: {p['rms_standardised']:.4f} +- {s['rms_standardised']:.4f} (ABF split-half floor {o['carrier']['abf_split_half_floor'][c]:.4f}); "
              f"natural {p['rms_natural']:.5f}; vs MC ref: FR {p['fr_minus_ref_rms']:.5f} ABF {p['abf_minus_ref_rms']:.5f}")
    print(f"     -> carrier by the P0 rule: {o['carrier']['coordinate']} (qualifies: {o['carrier']['qualifies']})")
    print("  P1 selection: quantile of the parent's fibre value at birth within the FR population law (u') and ABF law (u); deaths v'")
    for c in COORDS:
        p, s = F["P1"][c], E_["P1"][c]
        bf, ba, df = p["birth_vs_fr"], p["birth_vs_abf"], p["death_vs_fr"]
        print(f"     {c:4s}: births u' tail {bf['tail']:.4f} +- {s['birth_vs_fr']['tail']:.4f} (null {bf['null_tail']:.4f}; late births {bf['late_births_tail']:.4f}) mean {bf['mean']:.4f} +- {s['birth_vs_fr']['mean']:.4f} (null {bf['null_mean']:.4f}) | "
              f"u (vs ABF) tail {ba['tail']:.4f} mean {ba['mean']:.4f} +- {s['birth_vs_abf']['mean']:.4f} | deaths v' tail {df['tail']:.4f} +- {s['death_vs_fr']['tail']:.4f} mean {df['mean']:.4f} +- {s['death_vs_fr']['mean']:.4f} (null {df['null_tail']:.4f})")
    print(f"     birth region share W/F/C {np.round(F['P1']['birth_region_share'], 3)}  death region share {np.round(F['P1']['death_region_share'], 3)}; "
          f"late net source (births - deaths) per t.u. per walker W/F/C {np.round(F['P1']['late_net_source_by_region'], 4)} +- {np.round(E_['P1']['late_net_source_by_region'], 4)}; "
          f"late events per t.u. per run {F['P1']['late_events_per_tu_per_run']:.1f}")
    for c in COORDS:
        p, s = F["P1"][c], E_["P1"][c]
        print(f"     {c:4s}: net exchange (births u' - deaths v') {p['net_exchange_mean']:+.4f} +- {s['net_exchange_mean']:.4f}; births u' mean by region W/F/C "
              f"{np.round(p['birth_vs_fr']['mean_by_region'], 4)} +- {np.round(s['birth_vs_fr']['mean_by_region'], 4)}; deaths v' {np.round(p['death_vs_fr']['mean_by_region'], 4)}")
    cc = o["carrier"]["coordinate"]
    for c in (cc,) if n != "ITE300_K256" else COORDS:
        print(f"  P2 late-half estimates by ancestor class, carrier = {c}:")
        for g in CLASSES:
            p, s = F["P2"][c][g], E_["P2"][c][g]
            print(f"     {g:11s}: share {p['share_of_late_fr_deposits']:.3f} | mean-force wRMS {p['wrms']:.4f} +- {s['wrms']:.4f} | F RMS {p['f_rms']:.4f} +- {s['f_rms']:.4f} | "
                  f"signed W/flank/cage " + " ".join(f"{x:+.3f}" for x in p["region"].values()))
        for g in ("tail", "central"):
            p, s = F["P2"][c]["classes_vs_fr_population_law"][g], E_["P2"][c]["classes_vs_fr_population_law"][g]
            print(f"     {g + ' (vs FR law)':20s}: share {p['share_of_late_fr_deposits']:.3f} | wRMS {p['wrms']:.4f} +- {s['wrms']:.4f} | F RMS {p['f_rms']:.4f} +- {s['f_rms']:.4f}")
        for g in ("born_early", "born_late"):
            p, s = F["P2"][c]["lineage_born"][g], E_["P2"][c]["lineage_born"][g]
            print(f"     lineage {g:11s}: share {p['share_of_late_fr_deposits']:.3f} | wRMS {p['wrms']:.4f} +- {s['wrms']:.4f} | F RMS {p['f_rms']:.4f} +- {s['f_rms']:.4f}")
    vm = o["variance_matched_abf"]
    print(f"  variance-matched ABF: quarter pools wRMS {vm['quarter']['wrms_mean']:.4f} F RMS {vm['quarter']['f_rms_mean']:.4f}; half pools wRMS {vm['half']['wrms_mean']:.4f} F RMS {vm['half']['f_rms_mean']:.4f}")
    c = cc
    print(f"  P3 descendants' {c} conditional vs age (late half), standardised RMS deviation from ABF (bands {o['band_edges_tu']} t.u.):")
    for g in ("tail_late", "central_late"):
        p, s = F["P3"][c][g], E_["P3"][c][g]
        print(f"     {g:13s}: " + " ".join(f"{a:.3f}+-{b:.3f}" for a, b in zip(p["rms_standardised"], s["rms_standardised"])) +
              "  (deposits " + " ".join(f"{d:.2e}" for d in p["deposits"]) + ")")
    for c3 in COORDS:
        P3c, s3 = F["P3"][c3], E_["P3"][c3]
        print(f"     {c3:4s} moment statistics (exact ABF means, sd units): births {P3c['birth_state_mean_standardised']:+.4f} +- {s3['birth_state_mean_standardised']:.4f} "
              f"(fine law {P3c['birth_state_mean_standardised_fine_law']:+.4f}); deaths {P3c['death_state_mean_standardised']:+.4f} +- {s3['death_state_mean_standardised']:.4f} "
              f"(fine {P3c['death_state_mean_standardised_fine_law']:+.4f}); FR population at the birth bins {P3c['population_displacement_at_birth_bins_standardised']:+.4f} +- {s3['population_displacement_at_birth_bins_standardised']:.4f}")
    print(f"  P4 locality ({c} classes irrelevant): wRMS local (deposit in the birth coarse bin) {F['P4'][c]['local']['wrms']:.4f} +- {E_['P4'][c]['local']['wrms']:.4f} "
          f"vs non-local {F["P4"][c]["non_local"]["wrms"]:.4f} +- {E_["P4"][c]["non_local"]["wrms"]:.4f}")
    for k in ("by_birth_region", "by_death_region"):
        print(f"     {k}: " + " | ".join(f"{r}: share {F['P4'][c][k][r]['deposits'] / F['P2'][c]['all_fr']['deposits']:.3f} wRMS {F['P4'][c][k][r]['wrms']:.4f}+-{E_['P4'][c][k][r]['wrms']:.4f} "
                                          f"signed W/F/C " + " ".join(f"{x:+.3f}" for x in F['P4'][c][k][r]['region'].values()) for r in ("window", "flank", "cage")))
    print(f"  P6 (post hoc) late error by birth region x {c} class at birth (wRMS, F RMS, share; coarse bins 3-6 / 11-14 means):")
    for r in ("window", "flank", "cage"):
        for kk in ("Q1", "Q2", "Q3", "Q4", "below_median", "above_median"):
            p, s = F["P4"][c]["region_x_class"][r][kk], E_["P4"][c]["region_x_class"][r][kk]
            co = np.array(p["coarse"])
            print(f"     {r:6s} {kk:12s}: share {p['share_of_late_fr_deposits']:.3f} wRMS {p['wrms']:.3f}+-{s['wrms']:.3f} F {p['f_rms']:.3f}+-{s['f_rms']:.3f} "
                  f"| coarse 3-6 {co[3:7].mean():+.2f}  11-14 {co[11:15].mean():+.2f}")
    print(f"  P5 late-half coarse wRMS by time-since-birth band (reference rows: ABF {F['P5'][c]['abf_coarse_wrms']:.4f}, all FR {F['P5'][c]['all_fr_coarse_wrms']:.4f}):")
    for g in ("all", "tail", "central"):
        p, s = F["P5"][c][g], E_["P5"][c][g]
        print(f"     {g:8s}: " + " ".join(f"{a:.3f}+-{b:.3f}" for a, b in zip(p["wrms_coarse"], s["wrms_coarse"])) +
              "  (share " + " ".join(f"{d / F['P2'][c]['all_fr']['deposits']:.3f}" for d in p["deposits"]) + ")")


def main():
    names = sys.argv[1:] or list(BLOCKS)
    res = {}
    for n in names:
        o = block(n)
        res[n] = o
        report(o)
    p = os.path.join(S.ROOT, "results", "adaptive_fr_campaign", "mechanism", "selection_split.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(tojson(res), open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
