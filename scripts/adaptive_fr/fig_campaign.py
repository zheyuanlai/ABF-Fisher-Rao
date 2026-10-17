"""Figures of the adaptive-FR campaign (ABF vs ABF + FR only), from the analysis JSON / curves in
results/adaptive_fr_campaign/{analysis,mechanism}.  Bands: light 25-75 % of seeds, dark 95 % bootstrap CI of the
median.  Every panel states its status (confirmatory / exploratory / descriptive).

    ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao python scripts/adaptive_fr/fig_campaign.py [names...]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "convergence_fr"))
sys.path.insert(0, os.path.join(HERE, "..", "eri_campaign"))
import atlas_lib as L  # noqa: E402
from figstyle import ABF, FR, INK2, MUTED, SERIES, setup  # noqa: E402

plt = setup()
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
AN = os.path.join(ROOT, "results", "adaptive_fr_campaign", "analysis")
ME = os.path.join(ROOT, "results", "adaptive_fr_campaign", "mechanism")
FIG = os.path.join(ROOT, "figures", "adaptive_fr")


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def bands(ax, t, X, col, lab, lw=1.6):
    med, lo, hi = L.boot_median_curve(X)
    ax.fill_between(t, np.nanquantile(X, 0.25, 1), np.nanquantile(X, 0.75, 1), color=col, alpha=0.12, lw=0)
    ax.fill_between(t, lo, hi, color=col, alpha=0.32, lw=0)
    ax.plot(t, med, color=col, lw=lw, label=lab)


def load(block):
    R = json.load(open(os.path.join(AN, f"{block}.json")))
    z = np.load(os.path.join(AN, f"{block}_curves.npz"))
    return R, z


def fig_A():
    """1. ABF vs early-stopping FR at K256, T200 (Experiment A)."""
    R, z = load("A")
    t = z["t"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    ax = axs[0]
    bands(ax, t, z["eF_abf"], ABF, "ABF (EB runs, reused)")
    bands(ax, t, z["eF_fr"], FR, "ABF + FR stopped at 10 t.u.")
    ax.set_yscale("log")
    ax.axvline(1, color=MUTED, ls=":", lw=1)
    ax.axvline(10, color=FR, ls="--", lw=1)
    ax.set_xlabel("t (t.u.)")
    ax.set_ylabel("e_F(t) (kJ/mol, RMS vs exact MC)")
    ax.legend(fontsize=7.5)
    ax = axs[1]
    rel = 100 * (z["eF_fr"] / z["eF_abf"] - 1)
    med, lo, hi = L.boot_median_curve(rel)
    ax.fill_between(t, lo, hi, color=FR, alpha=0.25, lw=0)
    ax.plot(t, med, color=FR, lw=1.5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(10, color=FR, ls="--", lw=1)
    ax.set_xlabel("t (t.u.)")
    ax.set_ylabel("paired e_F(FR)/e_F(ABF) − 1 (%)")
    ax.set_ylim(-60, 60)
    ax = axs[2]
    rows = R["time_to_accuracy"]["sustained"]
    x = np.arange(len(rows))
    y = [r["S_median"] for r in rows]
    ax.errorbar(x, y, yerr=[[a - r["S_ci95"][0] for a, r in zip(y, rows)], [r["S_ci95"][1] - a for a, r in zip(y, rows)]],
                fmt="o", color=FR, capsize=3)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['eps_kJmol']:.2f}" for r in rows])
    ax.set_xlabel("tolerance ε (kJ/mol)")
    ax.set_ylabel("sustained speedup τ_ABF/τ_FR (median, 95 % CI)")
    p = R["primary"]
    ax.set_title(f"primary S_geo {p['median']:.2f} [{p['ci95'][0]:.2f}, {p['ci95'][1]:.2f}] — UNRESOLVED", fontsize=9, loc="left")
    fig.suptitle("Experiment A (confirmatory): ITE 300 K · K 256 · T 200 t.u. · 180 bins · FR rate 1.0 from 1 t.u., stopped at 10 t.u. · "
                 "16 paired seeds (ABF arm = historical EB runs)", x=0.01, ha="left", fontsize=9.5, y=1.02)
    fig.tight_layout()
    save(fig, "A_ite300_K256_T200_stop10")


def fig_equal_budget():
    """3. Equal-budget comparison, two algorithms x allocations (A.5 + E.3 if present)."""
    E = json.load(open(os.path.join(AN, "A_equal_budget4.json")))
    z = np.load(os.path.join(AN, "A_equal_budget4_curves.npz"))
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4))
    ax = axs[0]
    style = {"ABF_K1024": (ABF, ":", "ABF K1024 × 50"), "FR_K1024": (FR, ":", "FR-stop-10 K1024 × 50"),
             "ABF_K256": (ABF, "-", "ABF K256 × 200"), "FR_K256": (FR, "-", "FR-stop-10 K256 × 200")}
    K = {"ABF_K1024": 1024, "FR_K1024": 1024, "ABF_K256": 256, "FR_K256": 256}
    for k, (col, ls, lab) in style.items():
        B = K[k] * z[f"steps_{k}"] / 1e8
        X = z[f"eF_{k}"]
        ax.plot(B, np.median(X, 1), color=col, ls=ls, lw=1.6, label=lab)
        ax.fill_between(B, np.quantile(X, 0.25, 1), np.quantile(X, 0.75, 1), color=col, alpha=0.08, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 2)
    ax.set_xlabel("production molecular force evaluations (10⁸)")
    ax.set_ylabel("e_F (kJ/mol): median, 25–75 %")
    ax.legend(fontsize=7.5)
    ax = axs[1]
    r = E["production"]
    ks = list(style)
    v = [r["geo"][k]["median_1e8"] for k in ks]
    rng = np.random.default_rng(2026)
    ci = []
    for k in ks:
        ps = np.array(r["geo"][k]["per_seed"])
        b = np.median(ps[rng.integers(0, len(ps), (4000, len(ps)))], 1)
        ci.append((np.quantile(b, 0.025), np.quantile(b, 0.975)))
    cols = [style[k][0] for k in ks]
    ax.bar(range(len(ks)), v, color=cols, alpha=0.8, width=0.6)
    ax.errorbar(range(len(ks)), v, yerr=[[a - c[0] for a, c in zip(v, ci)], [c[1] - a for a, c in zip(v, ci)]], fmt="none", ecolor=INK2, capsize=3)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([style[k][2] for k in ks], fontsize=7.5, rotation=12)
    ax.set_ylabel("median per-seed geometric-mean sustained τ (10⁸ force evals)")
    q = r["ratios_geo_tauB"]["FR_K256 / ABF_K256"]
    ax.set_title(f"best FR / best ABF (both K256): {q['ratio']:.2f} [{q['ci95'][0]:.2f}, {q['ci95'][1]:.2f}] — unresolved", fontsize=9, loc="left")
    fig.suptitle("ITE 300 K at an equal budget of 5.12×10⁸ production force evaluations (two algorithms × two allocations; "
                 "cross-K comparisons unpaired)", x=0.01, ha="left", fontsize=9.5, y=1.02)
    fig.tight_layout()
    save(fig, "equal_budget_ite300")


CELLS = [("C1", "ITE 300 K (in-sample)"), ("C2", "ITE 150 K (in-sample)"), ("C3", "LTA 150 K (held out, +)"),
         ("C4", "LTA 300 K (held out)"), ("C5", "ERI 600 K (held out, −)")]


def fig_cross_system():
    """5. Cross-system convergence atlas under R1, and 2. speedup vs tolerance per cell."""
    fig, axs = plt.subplots(2, 5, figsize=(19, 7))
    for j, (b, lab) in enumerate(CELLS):
        R, z = load(b)
        t = z["t"]
        ax = axs[0, j]
        bands(ax, t, z["eF_abf"], ABF, "ABF")
        bands(ax, t, z["eF_fr"], FR, "ABF + FR (online stop R1)")
        st = [s for s in R["fr_schedule"]["stop_t"] if s is not None]
        ax.axvspan(min(st), max(st), color=FR, alpha=0.08, lw=0)
        ax.axvline(np.median(st), color=FR, ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_title(f"{b}: {lab}\nK {R['K']} · T {R['T']:g} t.u.", fontsize=9, loc="left")
        ax.set_xlabel("t (t.u.)")
        if j == 0:
            ax.set_ylabel("e_F(t) (kJ/mol)")
            ax.legend(fontsize=7)
        ax = axs[1, j]
        rows = R["time_to_accuracy"]["sustained"]
        x = np.arange(len(rows))
        y = np.array([r["S_median"] for r in rows])
        ax.errorbar(x, y, yerr=[y - [r["S_ci95"][0] for r in rows], [r["S_ci95"][1] for r in rows] - y], fmt="o", color=FR, capsize=3)
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['eps_kJmol']:.2f}" for r in rows], fontsize=7)
        ax.set_xlabel("tolerance ε (kJ/mol)")
        p = R["primary"]
        ax.set_title(f"S_geo {p['median']:.2f} [{p['ci95'][0]:.2f}, {p['ci95'][1]:.2f}] → {p['verdict_bonferroni']}", fontsize=8.5, loc="left")
        if j == 0:
            ax.set_ylabel("sustained speedup τ_ABF/τ_FR")
    fig.suptitle("Experiment C (confirmatory, 16 fresh paired seeds per cell): ABF vs ABF + FR stopped by the frozen online rule R1. "
                 "Shaded/dashed: range/median of the online stop times. Verdict by the Bonferroni 99 % CI.", x=0.01, ha="left", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, "cross_system_atlas")


def fig_stop_times():
    """4. Online stopping times and the controller's directedness trajectories."""
    import glob
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import lta_ladder_numba as LE
    dirs = {"C1": "C1_ite300_K1024_R1", "C2": "C2_ite150_K256_R1", "C3": "C3_lta150_K1024_R1", "C4": "C4_lta300_K1024_R1",
            "C5": "C5_eri600_K1024_R1"}
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax = axs[0]
    for j, (b, lab) in enumerate(CELLS):
        fs = sorted(glob.glob(os.path.join(ROOT, "results", "adaptive_fr_campaign", dirs[b], "fr_N*_s*.npz")))
        for i, f in enumerate(fs):
            z = LE.load_result(f) if "lta" in dirs[b] else dict(np.load(f, allow_pickle=False))
            log = [r for r in json.loads(str(z["fr_controller_log"])) if "j" in r]
            tt = np.array([r["t"] for r in log])
            jj = np.array([r["j"] / r["j_max"] if r["j_max"] > 0 else 0 for r in log])
            ax.plot(tt, jj, color=SERIES[j], lw=0.6, alpha=0.35, label=lab if i == 0 else None)
    ax.axhline(0.25, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("t (t.u.)")
    ax.set_ylabel("directedness j(t) / running max j_max")
    ax.set_xlim(0, 30)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("online statistic (one line per FR run; dashed: θ = 0.25)", fontsize=9, loc="left")
    ax = axs[1]
    for j, (b, lab) in enumerate(CELLS):
        R = json.load(open(os.path.join(AN, f"{b}.json")))
        st = np.array([s for s in R["fr_schedule"]["stop_t"] if s is not None])
        ax.scatter(st, np.full(st.size, j) + np.linspace(-0.2, 0.2, st.size), s=14, color=SERIES[j])
        ax.text(31, j, f"S_geo {R['primary']['median']:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(CELLS)))
    ax.set_yticklabels([f"{b}: {lab}" for b, lab in CELLS], fontsize=7.5)
    ax.set_xlabel("online FR stop time t_stop = 2 t1 (t.u.)")
    ax.set_xlim(0, 36)
    ax.set_title("where R1 stopped FR (16 runs per cell)", fontsize=9, loc="left")
    fig.suptitle("Experiment C: the frozen online rule R1 (reference-free) — controller statistic and realised stop times",
                 x=0.01, ha="left", fontsize=9.5, y=1.02)
    fig.tight_layout()
    save(fig, "online_stop_times")


def fig_bias_variance():
    """6. Bias-variance decomposition over time."""
    z = np.load(os.path.join(ME, "bias_variance_curves.npz"))
    blocks = [("X1", "ITE 300 K K1024 continuous FR (expl.)"), ("E1", "ITE 300 K K1024 FR stop 10"),
              ("A", "ITE 300 K K256 T200 FR stop 10"), ("ERI1024", "ERI 600 K K1024 continuous FR")]
    fig, axs = plt.subplots(2, 4, figsize=(17, 6.6))
    for j, (b, lab) in enumerate(blocks):
        for i, (key, ttl) in enumerate((("", "cumulative estimate"), ("_late", "late-only deposits"))):
            ax = axs[i, j]
            for arm, col in (("abf", ABF), ("fr", FR)):
                t = z[f"{b}_{arm}_t"] if not key else z[f"{b}_{arm}_t_late"]
                b2 = z[f"{b}_{arm}{key}_bias2" if key else f"{b}_{arm}_bias2"]
                v = z[f"{b}_{arm}{key}_var" if key else f"{b}_{arm}_var"]
                ax.plot(t, np.sqrt(np.maximum(b2, 0)), color=col, lw=1.6, label=f"{arm.upper()} |bias|")
                ax.plot(t, np.sqrt(v), color=col, lw=1.0, ls="--", label=f"{arm.upper()} seed s.d.")
            ax.set_yscale("log")
            ax.set_title(f"{lab}\n{ttl}", fontsize=8.5, loc="left")
            ax.set_xlabel("t (t.u.)")
            if j == 0:
                ax.set_ylabel("kJ/mol (free energy, centred)")
                ax.legend(fontsize=6.5)
    fig.suptitle("Bias–variance decomposition of the free-energy estimate across 16 seeds (squared bias corrected for finite seeds and "
                 "reference noise). Late-only: deposits after the FR stop (stopped schedules) or after T/2.", x=0.01, ha="left", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, "bias_variance")


def fig_clone_age():
    """7. Conditional mean-force and genealogy diagnostics (D2)."""
    D = json.load(open(os.path.join(ME, "clone_age.json")))
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.9))
    ax = axs[0]
    names = list(D)
    keys = ["young", "old", "all", "abf"]
    cols = [SERIES[3], SERIES[2], FR, ABF]
    w = 0.18
    for i, k in enumerate(keys):
        y = [D[n]["late_half"][k]["free_energy_rms_vs_ref"] for n in names]
        ax.bar(np.arange(len(names)) + (i - 1.5) * w, y, width=w, color=cols[i],
               label={"young": "FR, deposits of walkers cloned < 1 t.u. ago", "old": "FR, all other deposits",
                      "all": "FR, all deposits", "abf": "ABF"}[k])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("late-half pooled free-energy error vs MC (kJ/mol)")
    ax.legend(fontsize=6.8)
    ax.set_title("does the late FR bias live in recent clones?", fontsize=9, loc="left")
    ax = axs[1]
    e = D["ERI256"]["eta_conditional_minus_ref"]
    x = np.arange(18)
    for k, col, lab in (("young", SERIES[3], "FR young"), ("old", SERIES[2], "FR old"), ("abf", ABF, "ABF")):
        ax.plot(x, e[k], "o-", color=col, ms=3, lw=1.2, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("coarse CV bin (cage at the ends, window in the middle)")
    ax.set_ylabel("⟨η | φ⟩ − exact MC")
    ax.legend(fontsize=7)
    ax.set_title("ERI 600 K: cage-axis conditional by clone age", fontsize=9, loc="left")
    ax = axs[2]
    bv = json.load(open(os.path.join(ME, "bias_variance.json")))
    blocks = ["X1", "M300_256", "M150_256", "X2", "ERI256", "ERI1024", "E1", "E2", "A"]
    yb = [bv[b]["fr"]["late_only_final"]["bias"] for b in blocks]
    ya = [bv[b]["abf"]["late_only_final"]["bias"] for b in blocks]
    ax.scatter(range(len(blocks)), ya, color=ABF, s=22, label="ABF")
    ax.scatter(range(len(blocks)), yb, color=FR, s=22, label="FR")
    ax.axvline(5.5, color=MUTED, lw=0.8, ls=":")
    ax.text(2.5, max(yb) * 0.95, "continuous FR", ha="center", fontsize=7.5, color=INK2)
    ax.text(7, max(yb) * 0.95, "stopped FR", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels(blocks, rotation=30, fontsize=7.5)
    ax.set_ylabel("|bias| of the late-only estimate (kJ/mol)")
    ax.legend(fontsize=7)
    ax.set_title("late-deposit bias: continuous vs stopped FR", fontsize=9, loc="left")
    fig.suptitle("Experiment D (exploratory mechanism diagnostics; D2 re-runs are bitwise the historical continuous-FR arms)",
                 x=0.01, ha="left", fontsize=9.5, y=1.02)
    fig.tight_layout()
    save(fig, "mechanism_clone_age_bias")


def fig_meanforce_genealogy():
    """7b. Conditional mean-force errors by clone age (D2) and FR genealogy over time in the C cells."""
    import glob
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import lta_ladder_numba as LE
    D = json.load(open(os.path.join(ME, "clone_age.json")))
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.2))
    cols = {"young": SERIES[3], "old": SERIES[2], "all": FR, "abf": ABF}
    labs = {"young": "FR, walkers cloned < 1 t.u. ago (few deposits)", "old": "FR, all other walkers", "all": "FR, all deposits",
            "abf": "ABF"}
    for ax, (blk, ttl) in zip(axs[:2], (("ERI256", "ERI 600 K, K256, continuous FR"), ("ITE300_K256", "ITE 300 K, K256, continuous FR"))):
        x = np.arange(18)
        for k in ("young", "old", "all", "abf"):
            y = D[blk]["late_half"][k]["signed_mean_force_error_by_coarse_bin"]
            ax.plot(x, y, color=cols[k], lw=1.0 if k == "young" else 1.6, alpha=0.6 if k == "young" else 1.0,
                    marker="o", ms=3, label=labs[k])
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("coarse CV bin (cage at the ends, window in the middle)")
        ax.set_title(f"{ttl}:\nlate-half (t > T/2) mean force − exact MC, pooled over 16 seeds", fontsize=8.5, loc="left")
        if blk == "ERI256":
            ax.set_ylabel("count-weighted Γ − Γ_ref per coarse bin")
            ax.legend(fontsize=6.8, loc="lower left")
    dirs = {"C1": "C1_ite300_K1024_R1", "C2": "C2_ite150_K256_R1", "C3": "C3_lta150_K1024_R1", "C4": "C4_lta300_K1024_R1",
            "C5": "C5_eri600_K1024_R1"}
    for j, (b, lab) in enumerate(CELLS):
        fs = sorted(glob.glob(os.path.join(ROOT, "results", "adaptive_fr_campaign", dirs[b], "fr_N*_s*.npz")))
        E_, W_ = [], []
        for f in fs:
            if "lta" in dirs[b]:
                z = LE.load_result(f)
                t, e, w = np.asarray(z["save_t"], float), np.asarray(z["gen_ess_win"], float), np.asarray(z["gen_max_frac_win"], float)
            else:
                z = np.load(f, allow_pickle=False)
                t, e, w = z["times"], z["sv_u"][:, 2], z["sv_u"][:, 3]
            E_.append(e), W_.append(w)
        n = min(len(e) for e in E_)
        R = json.load(open(os.path.join(AN, f"{b}.json")))
        st = np.median([s_ for s_ in R["fr_schedule"]["stop_t"] if s_ is not None])
        for ax, X in ((axs[2], E_), (axs[3], W_)):
            X = np.array([x[:n] for x in X])
            ax.plot(t[:n], np.median(X, 0), color=SERIES[j], lw=1.4, label=f"{b}: {lab}")
            ax.axvline(st, color=SERIES[j], lw=0.8, ls=":")
    axs[2].axhline(0.15, color=INK2, lw=0.8, ls="--")
    axs[2].text(29.5, 0.16, "R1 safety backstop 0.15", fontsize=7, color=INK2, ha="right")
    axs[2].set_ylabel("windowed ancestor ESS / K (median over seeds)")
    axs[2].set_ylim(0, 1.02)
    axs[3].axhline(0.10, color=INK2, lw=0.8, ls="--")
    axs[3].text(0.5, 0.102, "R1 lineage backstop 0.10", fontsize=7, color=INK2, ha="left", va="bottom")
    axs[3].set_ylabel("windowed max lineage share (median over seeds)")
    axs[3].legend(fontsize=6.8, loc="center right")
    for ax in axs[2:]:
        ax.set_xlim(0, 30)
        ax.set_xlabel("t (t.u.); dotted: median R1 stop")
    axs[2].set_title("FR genealogy in the Experiment C cells (R1 runs):\ndiversity of ancestors", fontsize=8.5, loc="left")
    axs[3].set_title("concentration on one lineage", fontsize=8.5, loc="left")
    fig.suptitle("Experiment D (exploratory): conditional mean-force error by clone age, and FR genealogy over time",
                 x=0.01, ha="left", fontsize=9.5, y=1.04)
    fig.tight_layout()
    save(fig, "mechanism_meanforce_genealogy")


def fig_selection():
    """9. Experiment D4 (exploratory): the selection test. Late-half per-coarse-bin mean-force error by the ancestor's birth
    region and by the ancestor's fibre class at birth (vs figure 7b); birth quantiles vs the exact null; descendants'
    conditional vs time since birth."""
    D = json.load(open(os.path.join(ME, "selection_split.json")))
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.6))
    x = np.arange(18)
    regcol = {"window": SERIES[6], "flank": SERIES[2], "cage": SERIES[7]}
    for row, blk in enumerate(("ERI256", "ITE300_K256")):
        o = D[blk]
        F, E_ = o["full"], o["jackknife_se"]
        c = "eta" if blk == "ERI256" else o["carrier"]["coordinate"]
        P2, P4 = F["P2"][c], F["P4"][c]
        # (a) by birth region of the ancestor
        ax = axs[row, 0]
        for r in ("window", "flank", "cage"):
            y = np.array(P4["by_birth_region"][r]["coarse"])
            se = np.array(E_["P4"][c]["by_birth_region"][r]["coarse"])
            sh = P4["by_birth_region"][r]["deposits"] / P2["all_fr"]["deposits"]
            ax.errorbar(x, y, yerr=se, color=regcol[r], lw=1.3, marker="o", ms=3, capsize=0, label=f"ancestor born in the {r} ({sh:.0%} of late FR deposits)")
        ax.plot(x, P2["all_fr"]["coarse"], color=FR, lw=2.2, label="FR, all deposits")
        ax.plot(x, P2["abf"]["coarse"], color=ABF, lw=2.2, label="ABF")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylabel("late-half Γ − Γ_ref per coarse bin (kJ/mol/rad)")
        ax.set_title(f"{o['label']}:\nlate error by the ancestor's BIRTH REGION (P4)", fontsize=8.5, loc="left")
        ax.legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=1)
        # (b) by fibre class at birth (carrier coordinate), vs the ABF law at the birth bin
        ax = axs[row, 1]
        qcol = {"Q1": SERIES[6], "Q2": SERIES[0], "Q3": SERIES[5], "Q4": SERIES[7]}
        for q in ("Q1", "Q2", "Q3", "Q4"):
            ax.plot(x, np.array(P2[q]["coarse"]), color=qcol[q], lw=1.2, marker="o", ms=2.5, alpha=0.9,
                    label=f"{c} at birth in {q} of the ABF law at the birth bin ({P2[q]['share_of_late_fr_deposits']:.0%})")
        ax.plot(x, P2["all_fr"]["coarse"], color=FR, lw=2.2, label="FR, all deposits")
        ax.plot(x, P2["abf"]["coarse"], color=ABF, lw=2.2, label="ABF")
        ax.axhline(0, color=INK2, lw=0.8)
        how = "preregistered carrier" if blk == "ERI256" else f"P0 rule; displacement {F['P0'][c]['rms_standardised']:.4f} sd, marginal"
        ax.set_title(f"late error by the ancestor's FIBRE CLASS at birth (P2)\ncoordinate {c} ({how})", fontsize=8.5, loc="left")
        ax.legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=1)
        for a in (axs[row, 0], axs[row, 1]):
            a.set_xlabel("coarse CV bin (cage at the ends, window in the middle)")
        # (c) selection at birth: u' deciles vs the exact null
        ax = axs[row, 2]
        p1 = F["P1"][c]["birth_vs_fr"]
        ax.bar(np.arange(10) + 0.5, p1["hist"], width=0.8, color=SERIES[3], alpha=0.75, label="births (parents FR cloned)")
        ax.step(np.arange(11), list(p1["null_hist"]) + [p1["null_hist"][-1]], where="post", color=INK2, lw=1.3,
                label="exact null: random draw from the\npopulation at the same bins")
        ax.set_xlim(0, 10)
        ax.set_xticks(range(0, 11, 2))
        ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.2)])
        ax.set_xlabel(f"quantile u' of the parent's {c} within the FR population law at its bin")
        ax.set_ylabel("share of births")
        ns = abs(p1["mean"] - 0.5) / max(E_["P1"][c]["birth_vs_fr"]["mean"], 1e-12)
        ax.set_title(f"P1: are the cloned walkers fibre-selected? mean u' {p1['mean']:.4f} ± {E_['P1'][c]['birth_vs_fr']['mean']:.4f}"
                     f"\n(null 0.5; {ns:.1f} s.e.) — {'no' if ns < 3 else 'yes'}", fontsize=8.5, loc="left")
        ax.legend(fontsize=6.6, loc="upper left")
        # (d) descendants' conditional vs time since birth
        ax = axs[row, 3]
        bands_tu = o["band_edges_tu"]
        mids = [bands_tu[0] / 2] + [np.sqrt(bands_tu[i] * bands_tu[i + 1]) for i in range(len(bands_tu) - 1)] + [bands_tu[-1] * 1.8]
        for g, col, lab in (("tail_late", SERIES[7], "tail classes Q1 ∪ Q4"), ("central_late", SERIES[0], "central classes Q2 ∪ Q3")):
            y = np.array(F["P3"][c][g]["rms_standardised"])
            se = np.array(E_["P3"][c][g]["rms_standardised"])
            ax.errorbar(mids, y, yerr=se, color=col, lw=1.3, marker="o", ms=3.5, capsize=0, label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("time since birth (t.u.), bands 0–0.1, 0.1–0.3, 0.3–1, 1–3, 3–10, ≥ 10")
        ax.set_ylabel(f"|⟨{c} | bin⟩ − ABF| / ABF within-bin sd (count-weighted RMS)")
        ax.set_title(f"P3: descendants' {c} conditional relaxes toward ABF's with age", fontsize=8.5, loc="left")
        ax.legend(fontsize=6.6, loc="upper right")
    fig.suptitle("Experiment D4 (exploratory): is the late FR bias carried by fibre-selected clones? Both arms re-run bitwise with the read-only "
                 "selection record; late half = t > T/2; pooled over 16 seeds; seed-jackknife s.e.", x=0.01, ha="left", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, "mechanism_selection")

def fig_arrival():
    """10. Experiment D5 (exploratory): the arrival-law test. Per-provenance late deviations and weights in both arms, the
    arrival-law prediction of FR's late error from ABF's deviations and FR's weights, and the decay with time since leaving
    the cage."""
    D = json.load(open(os.path.join(ME, "arrival_split.json")))
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.0))
    for row, blk in enumerate(("ERI256", "ITE300_K256")):
        o = D[blk]
        F, E_ = o["full"], o["jackknife_se"]
        fl = np.array(o["flank_coarse_bins"])
        P, S_ = F["P12"], E_["P12"]
        x = np.arange(18)
        arr = lambda d: np.array([np.nan if v is None else v for v in d], float)
        # (a) per-provenance deviations, ABF vs FR
        ax = axs[row, 0]
        for k, col, ls, lab in (("abf_cage", SERIES[7], "-", "ABF, cage provenance"), ("abf_window", SERIES[6], "-", "ABF, window provenance"),
                                ("fr_cage", SERIES[7], "--", "FR, cage provenance"), ("fr_window", SERIES[6], "--", "FR, window provenance")):
            ax.errorbar(x[fl], arr(P["dev"][k])[fl], yerr=arr(S_["dev"][k])[fl], color=col, ls=ls, lw=1.3, marker="o", ms=3, capsize=0, label=lab)
        ax.plot(x, arr(P["dev"]["abf_all"]), color=ABF, lw=2.0, label="ABF, all deposits")
        ax.plot(x, arr(P["dev"]["fr_all"]), color=FR, lw=2.0, label="FR, all deposits")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylabel("late-half ⟨f⟩_p − Γ_ref per coarse bin (kJ/mol/rad)")
        ax.set_xlabel("coarse CV bin (flank bins only for the provenance classes)")
        ax.set_title(f"{o['label']}:\nP1/P2(a): deviation by where the walker came from", fontsize=8.5, loc="left")
        ax.legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2)
        # (b) weights
        ax = axs[row, 1]
        for k, col, ls, lab in (("abf_cage", SERIES[7], "-", "ABF: cage provenance"), ("fr_cage", SERIES[7], "--", "FR: cage provenance"),
                                ("abf_window", SERIES[6], "-", "ABF: window provenance"), ("fr_window", SERIES[6], "--", "FR: window provenance")):
            ax.errorbar(x[fl], arr(P["weight"][k])[fl], yerr=arr(S_["weight"][k])[fl], color=col, ls=ls, lw=1.3, marker="o", ms=3, capsize=0, label=lab)
        ax.set_ylabel("share of the bin's late deposits")
        ax.set_xlabel("coarse CV bin")
        ax.set_title("P2(b): provenance weights, ABF vs FR", fontsize=8.5, loc="left")
        ax.legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2)
        # (c) prediction vs observation
        ax = axs[row, 2]
        ax.errorbar(x[fl], arr(P["dev"]["fr_all"])[fl], yerr=arr(S_["dev"]["fr_all"])[fl], color=FR, lw=2.0, marker="o", ms=3, capsize=0, label="FR, observed late error")
        ax.errorbar(x[fl], arr(P["pred_fr_error"])[fl], yerr=arr(S_["pred_fr_error"])[fl], color=INK2, lw=1.4, ls="--", marker="s", ms=3, capsize=0,
                    label="arrival-law prediction Σ_p w_p^FR d_p^ABF")
        ax.plot(x[fl], arr(P["dev"]["abf_all"])[fl], color=ABF, lw=2.0, label="ABF, observed")
        ax.bar(x[fl] - 0.2, arr(P["weight_term"])[fl], width=0.4, color=SERIES[3], alpha=0.8, label="weight term Σ_p Δw_p d_p^ABF (post hoc)")
        ax.bar(x[fl] + 0.2, arr(P["state_term"])[fl], width=0.4, color=SERIES[2], alpha=0.8, label="state term Σ_p w_p^FR Δd_p (post hoc)")
        ax.axhline(0, color=INK2, lw=0.8)
        f_ = P["flank"]
        ax.set_title(f"P2(c): FR's late flank error from ABF's deviations and FR's weights; exact split of the excess\n"
                     f"RMS obs {f_['obs_fr_rms']:.3f}, pred {f_['pred_rms']:.3f}; excess {f_['obs_excess_rms']:.3f} = weight {f_['weight_term_rms']:.3f} + state {f_['state_term_rms']:.3f}", fontsize=8.5, loc="left")
        ax.set_xlabel("coarse CV bin (flank)")
        ax.set_ylabel("late-half Γ − Γ_ref (kJ/mol/rad)")
        ax.legend(fontsize=6.6, loc="lower left" if row == 0 else "upper right")
        # (d) decay with time since leaving the cage
        ax = axs[row, 3]
        b = o["band_edges_tu"]
        mids = [b[0] / 2] + [np.sqrt(b[i] * b[i + 1]) for i in range(len(b) - 1)] + [b[-1] * 1.8]
        for arm, col in (("abf", ABF), ("fr", FR)):
            r, s = F["P3"][arm], E_["P3"][arm]
            ax.errorbar(mids, arr(r["dev_left"]), yerr=arr(s["dev_left"]), color=col, lw=1.3, marker="o", ms=3.5, capsize=0, label=f"{arm.upper()}: left flank (φ < 0)")
            ax.errorbar(mids, arr(r["dev_right"]), yerr=arr(s["dev_right"]), color=col, lw=1.3, ls="--", marker="s", ms=3.5, capsize=0, label=f"{arm.upper()}: right flank (φ > 0)")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("time since leaving the cage (t.u.), band geometric midpoints")
        ax.set_ylabel("cage-provenance flank deviation (kJ/mol/rad)")
        ax.set_title("P3: the cage-provenance deviation decays with time since leaving", fontsize=8.5, loc="left")
        ax.legend(fontsize=6.6, loc="best")
    fig.suptitle("Experiment D5 (exploratory): the arrival-law mechanism — deposits labelled by the region the walker last visited; both arms re-run "
                 "bitwise; late half = t > T/2; pooled over 16 seeds; seed-jackknife s.e.", x=0.01, ha="left", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, "mechanism_arrival")


FRONTIER_K = [1024, 256, 64, 16]
FRONTIER_BLOCK = {1024: "C1", 256: "EA_R1", 64: "EB64", 16: "EB16"}


def boot_med_ci(ps, n=4000, seed=2026):
    rng = np.random.default_rng(seed)
    ps = np.asarray(ps, float)
    b = np.median(ps[rng.integers(0, len(ps), (n, len(ps)))], 1)
    return np.quantile(b, 0.025), np.quantile(b, 0.975)


def fig_frontier():
    """3b. Equal-budget frontier: ABF vs ABF + FR(R1) at K in {1024, 256, 64, 16}, B = 5.12e8 (E.3 + E-A2)."""
    E = json.load(open(os.path.join(AN, "E_frontier.json")))
    z = np.load(os.path.join(AN, "E_frontier_curves.npz"))
    Ks = [K for K in FRONTIER_K if f"ABF_K{K}" in E["configs"]]
    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, len(Ks) * 2)
    for j, K in enumerate(Ks):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        for alg, col, lab in (("ABF", ABF, "ABF"), ("FR", FR, "ABF + FR (R1)")):
            k = f"{alg}_K{K}"
            B = K * z[f"steps_{k}"] / 1e8
            X = z[f"eF_{k}"]
            ax.plot(B, np.median(X, 1), color=col, lw=1.5, label=lab)
            ax.fill_between(B, np.quantile(X, 0.25, 1), np.quantile(X, 0.75, 1), color=col, alpha=0.12, lw=0)
        ax.set_yscale("log")
        ax.set_ylim(0.015, 3)
        ax.set_xlim(0, 5.12)
        T = E["production"]["budget"] / K / 1e4
        ax.set_title(f"K{K} × T{T:g} t.u.", fontsize=9, loc="left")
        ax.set_xlabel("production force evaluations (10⁸)")
        if j == 0:
            ax.set_ylabel("e_F (kJ/mol): median, 25–75 %")
            ax.legend(fontsize=7.5)
    r = E["production"]
    ax = fig.add_subplot(gs[1, 0:len(Ks)])
    for alg, col, lab, mk in (("ABF", ABF, "ABF", "o"), ("FR", FR, "ABF + FR (R1)", "s")):
        v, lo, hi = [], [], []
        for K in Ks:
            ps = r["geo"][f"{alg}_K{K}"]["per_seed"]
            m = r["geo"][f"{alg}_K{K}"]["median_1e8"]
            a, b = boot_med_ci(ps)
            v.append(m), lo.append(a), hi.append(b)
        x = np.array(Ks, float) * (0.93 if alg == "ABF" else 1.07)
        v, lo, hi = map(np.array, (v, lo, hi))
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt=mk + "-", color=col, ms=6, capsize=3, lw=1.4, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([f"K{K}" for K in Ks])
    ax.minorticks_off()
    ax.set_xlabel("replica allocation K (T = B / K)")
    ax.set_ylabel("median per-seed geometric-mean\nsustained τ (10⁸ force evals)")
    ax.legend(fontsize=7.5, loc="upper left")
    best = r["best_allocation"]
    q = r["ratios_geo_tauB"][f"{best['FR-stop']} / {best['ABF']}"]
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
    ax.set_title(f"frontier (unpaired across K): best ABF {best['ABF']}, best FR {best['FR-stop']}\nbest FR / best ABF "
                 f"{q['ratio']:.2f} [{q['ci95'][0]:.2f}, {q['ci95'][1]:.2f}] → {r['question_B_verdict']}", fontsize=8.5, loc="left")
    ax = fig.add_subplot(gs[1, len(Ks):])
    for i, K in enumerate(Ks):
        pth = os.path.join(AN, f"{FRONTIER_BLOCK[K]}.json")
        if not os.path.exists(pth):
            continue
        p = json.load(open(pth))["primary"]
        ax.errorbar([i], [p["median"]], yerr=[[p["median"] - p["ci95"][0]], [p["ci95"][1] - p["median"]]], fmt="o", color=FR, capsize=3, ms=6)
        ax.text(i + 0.12, p["median"], f"{p['median']:.2f}  ({p['fr_faster_seeds']}/16)", fontsize=7.5, color=INK2, va="center")
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([f"K{K}\n({FRONTIER_BLOCK[K]})" for K in Ks], fontsize=8)
    ax.set_xlim(-0.4, len(Ks) - 0.2)
    ax.set_ylabel("paired S_geo = τ_ABF / τ_FR (95 % CI)")
    ax.set_title("same-K paired speedup (fixed K; 16 paired seeds each;\nFR-faster seed count in brackets)", fontsize=8.5, loc="left")
    fig.suptitle("Experiment E (ITE 300 K): equal budget B = 5.12×10⁸ production force evaluations, ABF vs ABF + FR stopped online "
                 "by R1, at four replica allocations", x=0.01, ha="left", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, "equal_budget_frontier")


def end_labels(ax, items, x, min_ratio=1.18):
    """Direct labels at the right end of log-y lines, nudged apart so none overlap (items: (y, text))."""
    items = sorted(items)
    ys = [np.log(y) for y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + np.log(min_ratio))
    for (_, t), ly in zip(items, ys):
        ax.text(x, np.exp(ly), t, fontsize=7.5, color=INK2, va="center")


def fig_wallclock():
    """8. Measured throughput vs threads, and wall-clock time to accuracy (E.4 + E-A2)."""
    W = json.load(open(os.path.join(AN, "E_wallclock_tta.json")))
    src = "E_wallclock_bench_run2.json" if W["throughput_source"] == "run2" else "E_wallclock_bench.json"
    bn = json.load(open(os.path.join(AN, src)))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axs[0]
    Ks = sorted({r["K"] for r in bn["rows"]}, reverse=True)
    labs = []
    for K in Ks:
        ends = []
        for m, col, ls in (("abf", ABF, "-"), ("fr", FR, "--")):
            rr = sorted([r for r in bn["rows"] if r["K"] == K and r["method"] == m], key=lambda r: r["threads"])
            P = [r["threads"] for r in rr]
            y = [1e6 * r["s_per_step"] for r in rr]
            ax.plot(P, y, ls, color=col, marker="o", ms=3.5, lw=1.3, label=("ABF" if m == "abf" else "ABF + FR (R1)") if K == Ks[0] else None)
            ends.append(y[-1])
        labs.append((float(np.sqrt(np.prod(ends))), f"K{K}"))
    end_labels(ax, labs, P[-1] * 1.12)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(0.8, 60)
    ax.set_xticks([1, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "4", "8", "16", "32"])
    ax.set_xlabel("numba threads P")
    ax.set_ylabel("measured wall-clock per MD step (µs)")
    ax.legend(fontsize=7.5)
    ax.set_title(f"throughput on {bn['host']} ({bn['cpu']}, shared;\n1-min load average {bn['load_before'][0]:.0f} before, {bn['load_after'][0]:.0f} after)", fontsize=8, loc="left")
    for col_ax, mode, ttl in ((axs[1], "production", "production"), (axs[2], "all_in_common", "all-in (incl. equilibration)")):
        ax = col_ax
        per = W[mode]["per_P"]
        Ps = [p for p in per if p != "best"]
        labs = []
        for K in Ks:
            ends = []
            for alg, col, ls in (("ABF", ABF, "-"), ("FR", FR, "--")):
                k = f"{alg}_K{K}"
                if k not in per[Ps[0]]["configs"]:
                    continue
                x = [int(p) for p in Ps]
                y = np.array([per[p]["configs"][k]["median_wall_s"] for p in Ps])
                lo = np.array([per[p]["configs"][k]["ci95"][0] for p in Ps])
                hi = np.array([per[p]["configs"][k]["ci95"][1] for p in Ps])
                ax.plot(x, y, ls, color=col, marker="o", ms=3.5, lw=1.3)
                ax.fill_between(x, lo, hi, color=col, alpha=0.08, lw=0)
                ends.append(y[-1])
            labs.append((float(np.sqrt(np.prod(ends))), f"K{K}"))
        end_labels(ax, labs, x[-1] * 1.12)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(0.8, 60)
        ax.set_xticks([1, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "4", "8", "16", "32"])
        ax.set_xlabel("numba threads P")
        ax.set_ylabel("wall-clock time to accuracy (s): median, 95 % CI")
        b = per["best"]
        q = [v for kk, v in b["ratios"].items() if kk.startswith("best")][0]
        ax.set_title(f"{ttl}; best thread count per config: best ABF {b['best_allocation']['ABF']}, best FR "
                     f"{b['best_allocation']['FR-stop']}\nbest FR / best ABF {q['ratio']:.2f} [{q['ci95'][0]:.2f}, {q['ci95'][1]:.2f}] → {b['verdict']}",
                     fontsize=8, loc="left")
    fig.suptitle("Experiment E.4: wall-clock = measured s/step at P threads × measured per-seed geometric-mean sustained τ (steps); "
                 "not a re-run at every P", x=0.01, ha="left", fontsize=9.5, y=1.03)
    fig.tight_layout()
    save(fig, "wallclock_ite300")


def main():
    which = sys.argv[1:] or ["A", "equal", "frontier", "cross", "stops", "bv", "clone", "mfgen", "wallclock", "selection", "arrival"]
    f = {"A": fig_A, "equal": fig_equal_budget, "frontier": fig_frontier, "cross": fig_cross_system, "stops": fig_stop_times,
         "bv": fig_bias_variance, "clone": fig_clone_age, "mfgen": fig_meanforce_genealogy, "wallclock": fig_wallclock,
         "selection": fig_selection, "arrival": fig_arrival}
    for w in which:
        f[w]()


if __name__ == "__main__":
    main()
