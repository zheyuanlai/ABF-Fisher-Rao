#!/usr/bin/env python
"""Experiment B figures B1-B7 (prereg section 2.8) -> figures/final_campaign/expB/ (+ MANIFEST.json with the
legibility check of every figure: scripts/equal_budget/fig_legibility.check_figure_safe).

    python scripts/final_campaign/expB_figures.py [--systems gateway lta150]

  B1 throughput vs threads per (system, N, method), with the serial engine and the F1 / F1b lines
  B2 wall time per walker-step vs N (best implementation, serial engine, parallel t = 1)
  B3 W_mid per arm: seed points, medians with bootstrap 95 % CIs, censored runs marked
  B4 e_F vs wall-clock: median and IQR across seeds per arm, thresholds marked
  B5 bootstrap distributions of rho(f, a) = median W_f / median W_a with the decision CIs and rho = 1
  B6 cost to the mid threshold: force evaluations, used and allocated core-hours (medians, 95 % CIs)
  B7 overhead split from the in-engine timers (force/move, ABF deposit + bias read, synchronisation, FR, saves)
A figure whose inputs do not exist yet is skipped and listed as such in the manifest.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import expB_common as C  # noqa: E402
import expB_stats as S  # noqa: E402

SYSTEM_LABEL = {"gateway": "entropic gateway", "lta150": "ethane / LTA 150 K"}
# Amendment 3 (A3.3 a): every gateway Experiment B result carries the exploratory label
EXPLORATORY_TAG = {"gateway": "\n[exploratory (after a frozen V-B4 FAIL; Amendment 3)]"}
EXPLORATORY_TAG_SHORT = {"gateway": "\n[exploratory; Amendment 3]"}      # for narrow panels (same meaning)


def tag(s, short=False):
    return (EXPLORATORY_TAG_SHORT if short else EXPLORATORY_TAG).get(s, "")
METHOD_LABEL = {"abf": "ABF", "fr": "ABF+FR"}
TIMER_LABEL = {"t_force_move_s": "force / move", "t_abf_s": "ABF deposit + bias read", "t_sync_s": "synchronisation",
               "t_fr_s": "FR", "t_save_s": "saves"}
THR_COLOR = {"strict": "0.25", "mid": "0.45", "loose": "0.65"}


def setup_style():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 110, "savefig.dpi": 160})


def arm_color(arm):
    m, N = C.parse_arm(arm)
    cm = plt.get_cmap("Reds" if m == "fr" else "Blues")
    return cm(0.45 + 0.5 * math.log2(max(N, 1)) / 9.0)


def save(fig, fig_dir, tag, manifest, inputs):
    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(fig_dir, f"{tag}.{ext}")
        fig.savefig(p)
        paths.append(os.path.relpath(p, C.ROOT))
    leg = None
    try:
        import fig_legibility as LG
        leg = LG.check_figure_safe(fig)
    except Exception as e:  # noqa: BLE001
        leg = dict(status="not checked", reason=f"{type(e).__name__}: {e}")
    plt.close(fig)
    manifest[tag] = dict(files=paths, inputs=inputs, legibility=leg)


def skip(manifest, tag, why):
    manifest[tag] = dict(status="skipped", reason=why)


def _load(path):
    return C.read_json(path) if os.path.exists(path) else None


def _num(v):
    return C.num(v) if v is not None else math.nan


# ================================================================================================ B1, B2
def fig_b1(systems, out_root, fig_dir, manifest):
    data = {s: _load(os.path.join(out_root, "bench", s, "summary.json")) for s in systems}
    dec = {s: _load(os.path.join(out_root, "bench", s, "decision.json")) for s in systems}
    have = [s for s in systems if data[s]]
    if not have:
        return skip(manifest, "B1_throughput_vs_threads", "no benchmark summary")
    fig, axes = plt.subplots(2 * len(have), 1, figsize=(7.2, 3.0 * 2 * len(have)), squeeze=False)
    xs = {"serial": 0, 1: 1, 2: 2, 4: 3, 8: 4, 16: 5, 32: 6}
    for i, s in enumerate(have):
        summ = data[s]["summary"]
        for j, m in enumerate(("abf", "fr")):
            ax = axes[2 * i + j][0]
            Ns = sorted({v["N"] for v in summ.values() if v["method"] == m})
            for N in Ns:
                col = arm_color(C.arm_name(m, N))
                par = sorted((v["threads"], v["throughput"]) for v in summ.values()
                             if v["method"] == m and v["N"] == N and v["impl"] == "parallel" and v["n"])
                main = [(xs[t], y) for t, y in par if t <= 16]
                if main:
                    ax.plot(*zip(*main), "-o", color=col, ms=3.5, lw=1.2, label=f"N = {N}")
                d32 = [(xs[t], y) for t, y in par if t == 32]
                if d32 and main:
                    ax.plot([main[-1][0], d32[0][0]], [main[-1][1], d32[0][1]], ":", color=col, lw=1.0)
                    ax.plot(*d32[0], "o", mfc="white", color=col, ms=4)
                ser = summ.get(f"{m}{N}_serial")
                if ser and ser["n"]:
                    ax.plot(xs["serial"], ser["throughput"], "s", color=col, ms=4)
            g = (dec[s] or {}).get("gate", {})
            f1 = g.get("F1", {}).get(m, {})
            if f1.get("serial_throughput") and math.isfinite(_num(f1.get("serial_throughput"))):
                ax.axhline(6.0 * _num(f1["serial_throughput"]), color="k", ls="--", lw=0.9,
                           label="F1: 6x serial (N 512)")
            if m == "fr":
                t1 = _num(g.get("F1b", {}).get("parallel_t1_throughput"))
                if math.isfinite(t1):
                    ax.axhline(4.0 * t1, color="tab:green", ls="-.", lw=0.9, label="F1b: 4x parallel t=1 (FR 512)")
            ax.plot([], [], "s", color="0.3", ms=4, label="square: validated serial engine")
            has32 = any(v["threads"] == 32 and v["n"] for v in summ.values())
            if has32:
                ax.plot([], [], "o", mfc="white", color="0.3", ms=4, label="open: t = 32 (descriptive)")
            ticks = ["serial", 1, 2, 4, 8, 16] + ([32] if has32 else [])
            ax.set_xticks([xs[t] for t in ticks])
            ax.set_xticklabels(["serial", "1", "2", "4", "8", "16"] + (["32*"] if has32 else []))
            ax.set_yscale("log")
            ax.set_xlabel("threads (32*: descriptive)" if has32 else "threads (t = 32 not measured; Amendment 4)")
            ax.set_ylabel("throughput (walker-steps / s)")
            ax.set_title(f"{SYSTEM_LABEL.get(s, s)}: {METHOD_LABEL[m]}{tag(s)}", fontsize=9)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.subplots_adjust(left=0.12, right=0.66, hspace=0.6, top=0.95, bottom=0.08)
    save(fig, fig_dir, "B1_throughput_vs_threads", manifest, [f"bench/{s}/summary.json" for s in have])


def fig_b2(systems, out_root, fig_dir, manifest):
    data = {s: _load(os.path.join(out_root, "bench", s, "summary.json")) for s in systems}
    have = [s for s in systems if data[s]]
    if not have:
        return skip(manifest, "B2_wall_per_walker_step_vs_N", "no benchmark summary")
    fig, axes = plt.subplots(1, len(have), figsize=(4.6 * len(have), 3.6), squeeze=False)
    for i, s in enumerate(have):
        ax = axes[0][i]
        summ = data[s]["summary"]
        for m, mk in (("abf", "o"), ("fr", "s")):
            Ns = sorted({v["N"] for v in summ.values() if v["method"] == m})
            best, ser, t1 = [], [], []
            for N in Ns:
                c = [v for v in summ.values() if v["method"] == m and v["N"] == N and v["n"] and not v["descriptive"]]
                if c:
                    best.append((N, min(v["us_per_walker_step"] for v in c)))
                sv = summ.get(f"{m}{N}_serial")
                if sv and sv["n"]:
                    ser.append((N, sv["us_per_walker_step"]))
                pv = summ.get(f"{m}{N}_parallel_t1")
                if pv and pv["n"]:
                    t1.append((N, pv["us_per_walker_step"]))
            col = "tab:red" if m == "fr" else "tab:blue"
            if best:
                ax.plot(*zip(*best), "-" + mk, color=col, ms=4, label=f"{METHOD_LABEL[m]} best (t <= 16)")
            if ser:
                ax.plot(*zip(*ser), "--" + mk, color=col, mfc="white", ms=4, label=f"{METHOD_LABEL[m]} serial engine")
            if t1:
                ax.plot(*zip(*t1), ":", color=col, lw=1.0, label=f"{METHOD_LABEL[m]} parallel t = 1")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N (walkers)")
        ax.set_ylabel("wall time per walker-step (us)")
        ax.set_title(SYSTEM_LABEL.get(s, s) + tag(s), fontsize=9)
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save(fig, fig_dir, "B2_wall_per_walker_step_vs_N", manifest, [f"bench/{s}/summary.json" for s in have])


# ================================================================================================ B3-B7 (production)
def _analysis(out_root, s):
    return _load(os.path.join(out_root, "analysis", s, "summary.json"))


def _seed_rows(out_root, s):
    p = os.path.join(out_root, "tables", f"{s}_seeds.csv")
    if not os.path.exists(p):
        return []
    import csv
    with open(p) as fh:
        return list(csv.DictReader(fh))


def fig_b3(systems, out_root, fig_dir, manifest):
    have = [s for s in systems if _analysis(out_root, s) and _seed_rows(out_root, s)]
    if not have:
        return skip(manifest, "B3_W_mid_per_arm", "no production analysis")
    fig, axes = plt.subplots(1, len(have), figsize=(4.8 * len(have), 3.8), squeeze=False)
    rng = np.random.default_rng(0)
    for i, s in enumerate(have):
        ax = axes[0][i]
        A = _analysis(out_root, s)
        rows = [r for r in _seed_rows(out_root, s) if r["W_mid_primary"] == "True"]
        arms = A["arms"]
        fin = [float(r["W_mid"]) for r in rows if r["W_mid"] != "inf"]
        top = (max(fin) if fin else 1.0) * 2.0
        for k, a in enumerate(arms):
            w = [float(r["W_mid"]) for r in rows if r["arm"] == a]
            f = [x for x in w if math.isfinite(x)]
            nc = sum(1 for x in w if math.isinf(x))
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(f)), f, s=9, color=arm_color(a), alpha=0.7, lw=0)
            if nc:
                ax.scatter([k], [top], marker="^", s=30, color=arm_color(a))
                ax.annotate(f"{nc} cens.", (k, top), textcoords="offset points", xytext=(0, 6), ha="center",
                            fontsize=7)
            pa = A["results"]["W_mid"].get("per_arm", {}).get(a) if isinstance(A["results"]["W_mid"], dict) else None
            if pa:
                med, lo, hi = _num(pa["median"]), _num(pa["ci95"][0]), _num(pa["ci95"][1])
                if math.isfinite(med):
                    ax.errorbar([k + 0.3], [med], yerr=[[med - lo], [min(hi, top) - med]] if math.isfinite(lo) else None,
                                fmt="D", color="k", ms=4, capsize=3, lw=1)
        ax.set_yscale("log")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([f"{METHOD_LABEL[C.parse_arm(a)[0]]}\nN = {C.parse_arm(a)[1]}" for a in arms])
        ax.set_xlabel("arm")
        ax.set_ylabel("W_mid (wall s to persistent mid accuracy)")
        dec_txt = (f"exploratory decision: {A.get('decision_primary')}" if A.get("exploratory")
                   else f"decision: {A.get('decision_primary')}")
        ax.set_title(f"{SYSTEM_LABEL.get(s, s)}: {dec_txt}{tag(s)}", fontsize=9)
    fig.tight_layout()
    save(fig, fig_dir, "B3_W_mid_per_arm", manifest, [f"analysis/{s}/summary.json" for s in have])


def fig_b4(systems, out_root, fig_dir, manifest):
    have = [s for s in systems if os.path.exists(os.path.join(out_root, "analysis", s, "curves.npz"))]
    if not have:
        return skip(manifest, "B4_eF_vs_wall", "no production curves")
    fig, axes = plt.subplots(1, len(have), figsize=(6.4 * len(have), 3.9), squeeze=False)
    for i, s in enumerate(have):
        ax = axes[0][i]
        A = _analysis(out_root, s)
        with np.load(os.path.join(out_root, "analysis", s, "curves.npz")) as z:
            g = z["w_grid"]
            for a in A["arms"]:
                if f"{a}_median" not in z:
                    continue
                col = arm_color(a)
                ax.plot(g, z[f"{a}_median"], color=col, lw=1.4, label=f"{METHOD_LABEL[C.parse_arm(a)[0]]} "
                                                                       f"N = {C.parse_arm(a)[1]}")
                ax.fill_between(g, z[f"{a}_q25"], z[f"{a}_q75"], color=col, alpha=0.18, lw=0)
        for name, thr in A["thresholds"].items():
            ax.axhline(thr, color=THR_COLOR[name], ls="--", lw=0.8)
            ax.annotate(name, (g[-1], thr), textcoords="offset points", xytext=(-2, 2), fontsize=7, ha="right",
                        color=THR_COLOR[name])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("wall-clock time since timed start (s)")
        ax.set_ylabel("e_F (median, IQR)")
        ax.set_title(SYSTEM_LABEL.get(s, s) + tag(s), fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    save(fig, fig_dir, "B4_eF_vs_wall", manifest, [f"analysis/{s}/curves.npz" for s in have])


def fig_b5(systems, out_root, fig_dir, manifest):
    have = [s for s in systems if os.path.exists(os.path.join(out_root, "analysis", s, "boot.npz"))]
    if not have:
        return skip(manifest, "B5_rho_bootstrap", "no bootstrap arrays")
    panels = []
    for s in have:
        A = _analysis(out_root, s)
        dec = A["results"]["W_mid"].get("decision")
        if not dec:
            continue
        with np.load(os.path.join(out_root, "analysis", s, "boot.npz")) as z:
            boot = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("W_mid__")}
        for key, c in dec["contrasts"].items():
            f, a = key.split("|")
            if f in boot and a in boot:
                panels.append((s, key, S.ratio(boot[f], boot[a]), c))
    if not panels:
        return skip(manifest, "B5_rho_bootstrap", "no contrasts")
    nc = min(4, len(panels))
    nr = int(math.ceil(len(panels) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.7 * nr), squeeze=False)
    for k, (s, key, r, c) in enumerate(panels):
        ax = axes[k // nc][k % nc]
        d = r[~np.isnan(r)]
        fin = d[np.isfinite(d) & (d > 0)]
        if fin.size:
            ax.hist(np.log10(fin), bins=40, color="0.55")
        ax.axvline(0.0, color="k", lw=1.0, label="rho = 1")
        lo, hi = _num(c.get("lower_ABF_level")), _num(c.get("upper_FR_level"))
        for v, ls, lab in ((lo, ":", "ABF-side lower end (1 - 0.05/|A| CI)"),
                           (hi, "--", "FR-side upper end (1 - 0.05/|F| CI)")):
            if math.isfinite(v) and v > 0:
                ax.axvline(math.log10(v), color="tab:purple", ls=ls, lw=1.0, label=lab)
        extra = []
        if np.any(d == 0):
            extra.append(f"rho=0: {int(np.sum(d == 0))}")
        if np.any(np.isinf(d)):
            extra.append(f"rho=inf: {int(np.sum(np.isinf(d)))}")
        if np.any(np.isnan(r)):
            extra.append(f"undef.: {int(np.sum(np.isnan(r)))}")
        ax.set_title(f"{s}: {key.replace('|', ' / ')}" + ("\n" + ", ".join(extra) if extra else "") + tag(s, short=True),
                     fontsize=7.5)
        ax.set_xlabel("log10 rho (W_mid ratio)")
        ax.set_ylabel("resamples")
    for k in range(len(panels), nr * nc):
        axes[k // nc][k % nc].set_visible(False)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, fig_dir, "B5_rho_bootstrap", manifest, [f"analysis/{s}/boot.npz" for s in have])


def fig_b6(systems, out_root, fig_dir, manifest):
    have = [s for s in systems if _analysis(out_root, s)]
    if not have:
        return skip(manifest, "B6_cost_to_accuracy", "no production analysis")
    keys = (("force_evals_mid", "force evaluations to mid"), ("used_core_h_mid", "used core-h to mid"),
            ("alloc_core_h_mid", "allocated core-h (16 x W) to mid"))
    fig, axes = plt.subplots(len(have), 3, figsize=(10.5, 3.2 * len(have)), squeeze=False)
    for i, s in enumerate(have):
        A = _analysis(out_root, s)
        pa = A["results"].get("per_arm", {})
        arms = [a for a in A["arms"] if a in pa]
        for j, (k, lab) in enumerate(keys):
            ax = axes[i][j]
            for x, a in enumerate(arms):
                v = pa[a].get(k)
                if not v:
                    continue
                med, lo, hi = (_num(t) for t in v)
                if math.isfinite(med):
                    ax.errorbar([x], [med], yerr=[[med - lo], [hi - med]] if math.isfinite(hi) else None, fmt="o",
                                color=arm_color(a), capsize=3)
                else:
                    ax.annotate("censored", (x, 1), xycoords=("data", "axes fraction"), ha="center", va="top",
                                fontsize=7)
            ax.set_yscale("log")
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels([f"{METHOD_LABEL[C.parse_arm(a)[0]]}\n{C.parse_arm(a)[1]}" for a in arms], fontsize=7)
            ax.set_xlabel("arm (method, N)")
            ax.set_ylabel(lab)
            ax.set_title(SYSTEM_LABEL.get(s, s) + tag(s, short=True), fontsize=8.5)
    fig.tight_layout()
    save(fig, fig_dir, "B6_cost_to_accuracy", manifest, [f"analysis/{s}/summary.json" for s in have])


def fig_b7(systems, out_root, fig_dir, manifest):
    have = [s for s in systems if _analysis(out_root, s)]
    src = "production timers"
    if not have:
        have = [s for s in systems if _load(os.path.join(out_root, "bench", s, "summary.json"))]
        src = "benchmark timers"
    if not have:
        return skip(manifest, "B7_overhead_split", "no timers")
    fig, axes = plt.subplots(1, len(have), figsize=(4.8 * len(have), 3.8), squeeze=False)
    cols = plt.get_cmap("tab10")
    for i, s in enumerate(have):
        ax = axes[0][i]
        bars = {}
        if src == "production timers":
            A = _analysis(out_root, s)
            for a, d in A["results"].get("per_arm", {}).items():
                bars[a] = {k: (_num(d["overheads"][k]["fraction"][0]) if d["overheads"].get(k, {}).get("fraction")
                               else math.nan) for k in C.TIMER_KEYS}
        else:
            summ = _load(os.path.join(out_root, "bench", s, "summary.json"))["summary"]
            for cfg, v in summ.items():
                if v["impl"] == "parallel" and v["N"] == 512 and v["threads"] == 16:
                    bars[f"{v['method']}512"] = {k: _num(v["timer_fractions"].get(k)) for k in C.TIMER_KEYS}
        names = list(bars)
        seen = set()
        for x, a in enumerate(names):
            if not any(math.isfinite(bars[a][k]) for k in C.TIMER_KEYS):     # serial engine: no in-engine timers
                ax.annotate("no in-engine timers\n(serial engine)", (x, 0.5), ha="center", va="center", fontsize=6.5,
                            rotation=90)
                continue
            bottom = 0.0
            for j, k in enumerate(C.TIMER_KEYS):
                v = bars[a][k]
                if not math.isfinite(v):
                    continue
                ax.bar(x, v, bottom=bottom, color=cols(j), width=0.6, label=None if k in seen else TIMER_LABEL[k])
                seen.add(k)
                bottom += v
            if bottom < 1.0:
                ax.bar(x, 1.0 - bottom, bottom=bottom, color="0.85", width=0.6,
                       label=None if "other" in seen else "other")
                seen.add("other")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"{METHOD_LABEL[C.parse_arm(a)[0]]}\nN = {C.parse_arm(a)[1]}" for a in names], fontsize=7)
        ax.set_xlabel("arm")
        ax.set_ylabel("fraction of wall time")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{SYSTEM_LABEL.get(s, s)} ({src}){tag(s)}", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, fig_dir, "B7_overhead_split", manifest, [s for s in have])


def make_all(systems=("gateway", "lta150"), out_root=C.OUT_ROOT, fig_dir=C.FIG_ROOT):
    setup_style()
    manifest = {}
    for f in (fig_b1, fig_b2, fig_b3, fig_b4, fig_b5, fig_b6, fig_b7):
        try:
            f(list(systems), out_root, fig_dir, manifest)
        except Exception as e:  # noqa: BLE001
            manifest[f.__name__] = dict(status="error", reason=f"{type(e).__name__}: {e}")
    os.makedirs(fig_dir, exist_ok=True)
    C.write_json(os.path.join(fig_dir, "MANIFEST.json"), manifest)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Experiment B figures")
    ap.add_argument("--systems", nargs="*", default=["gateway", "lta150"])
    ap.add_argument("--out-root", default=C.OUT_ROOT)
    ap.add_argument("--fig-dir", default=C.FIG_ROOT)
    a = ap.parse_args(argv)
    m = make_all(a.systems, a.out_root, a.fig_dir)
    print(json.dumps({k: (v.get("status") or (v.get("legibility") or {}).get("status")) for k, v in m.items()},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
