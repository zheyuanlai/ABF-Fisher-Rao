import sys, json, numpy as np
sys.path.insert(0, "/home/zheyuanlai/ABF-Fisher-Rao/src")
import lta_ladder_numba as E
ROOT = "/home/zheyuanlai/ABF-Fisher-Rao"
for T in (300, 150):
    ref = np.load(f"{ROOT}/results/uniform_campaign/lta/reference/reference_T{T}.npz", allow_pickle=True)
    Fref = ref["F"]; gref = ref["grid_phi"]
    prod = {m: np.load(f"{ROOT}/results/lta_histogram/production_T{T}/{m}.npz", allow_pickle=True) for m in ("abf", "fr_uniform")}
    for m, pm in (("abf", "abf"), ("fr", "fr_uniform")):
        try:
            r = E.load_result(f"T{T}_{m}_s1.npz")
        except FileNotFoundError:
            print(T, m, "missing"); continue
        use_prod = r["C_prod"].sum(-1) > 0
        mf = np.where(use_prod[:, None], E.mean_force(r["M_prod"], r["C_prod"]), E.mean_force(r["M_all"], r["C_all"]))
        pmf = E.histogram_pmf(mf)
        d = pmf - Fref[None, :]; d = d - d.mean(-1, keepdims=True)
        e = np.sqrt((d ** 2).mean(-1))
        IF = np.trapezoid(e, r["save_t"])
        P = prod[pm]
        dp = P["pmf"] - Fref[None, None, :]; dp = dp - dp.mean(-1, keepdims=True)
        ep = np.sqrt((dp ** 2).mean(-1))          # (101, 16)
        IFp = np.trapezoid(ep, P["times"], axis=0)
        ev_p = P["total_replacement_events"]; cx_p = P["n_cage_crossings"] / 1024
        print(f"T {T} {m}: numba seed1 e_F(T) {e[-1]:.4f} I_F {IF:.2f} events {int(r['total_replacement_events'])} "
              f"cage-cross/rep {r['n_cage_crossings_lineage'][-1]/1024:.3f} transloc {int(r['events_translocations'][-1])} "
              f"wcross {int(r['events_window_crossings'][-1])} minESS {np.nanmin(r['gen_ess'][r['save_step']>=20000]) if m=='fr' else float('nan'):.3f} wall {r['wall_s']:.0f}s | "
              f"production 16 seeds: e_F(T) median {np.median(ep[-1]):.4f} [{ep[-1].min():.4f},{ep[-1].max():.4f}] "
              f"I_F median {np.median(IFp):.2f} [{IFp.min():.2f},{IFp.max():.2f}] events median {np.median(ev_p):.0f} [{ev_p.min()},{ev_p.max()}] "
              f"cage-cross/rep median {np.median(cx_p):.3f} [{cx_p.min():.3f},{cx_p.max():.3f}]")
