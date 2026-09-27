#!/usr/bin/env python
"""Cross-system figures for the histogram-ABF campaign: the effect-size figure (FR relative
improvement in I_F and final e_F, kernel vs histogram, both systems) and the efficiency
table/figure (estimator cost from the profiling runs, end-to-end wall time from the campaign).

Reads results/histogram_abf/{entropic_bottleneck,wca}/confirmation/summary.json and
results/histogram_abf/profiling/profile_*.json; writes results/histogram_abf/figures/ and
results/histogram_abf/efficiency.json / efficiency.md.
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results", "histogram_abf")
COL = {"kernel": "#1f77b4", "hist": "#d62728"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
    out_dir = os.path.join(RES, "figures"); os.makedirs(out_dir, exist_ok=True)
    systems = [("entropic_bottleneck", "Entropic bottleneck"), ("wca", "WCA dimer"), ("gateway", "Entropic gateway")]
    summ = {}
    for key, _ in systems:
        p = os.path.join(RES, key, "confirmation", "summary.json")
        if os.path.exists(p):
            summ[key] = json.load(open(p))

    # ---- effect-size figure
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2), sharey=False)
    for ax, (key, title) in zip(axes, systems):
        if key not in summ:
            ax.set_title(f"{title}: no confirmation yet"); continue
        eff = summ[key]["fr_effect"]
        for i, (k, lab) in enumerate((("int_l2_f", "$\\Delta I_F$"), ("final_l2_f", "$\\Delta e_F(T)$"))):
            for est in ("kernel", "hist"):
                e = eff[est][k]; xx = i + (-0.17 if est == "kernel" else 0.17)
                ax.errorbar([xx], [e["median"]], yerr=[[e["median"] - e["ci95"][0]], [e["ci95"][1] - e["median"]]], fmt="o", color=COL[est], capsize=3,
                            label=({"kernel": "kernel ABF estimator", "hist": "histogram ABF estimator"}[est] if i == 0 else None))
                ax.annotate(f"{e['wins']}/{e['n']}", (xx, e["ci95"][1]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
        ax.axhline(0, color="k", lw=0.8); ax.set_xticks([0, 1]); ax.set_xticklabels(["$I_F$", "$e_F(T)$"])
        ax.set_title(f"{title} (n = {len(summ[key].get('seeds', summ[key].get('rows', [])))})"); ax.set_ylabel("uniform-FR effect vs ABF (%)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Fisher-Rao effect under the kernel and the histogram mean-force estimator (median, bootstrap 95 % CI, seed wins)", fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"effect_size_both_systems.{ext}"), bbox_inches="tight")

    # ---- efficiency
    prof = {}
    for dev in ("cuda", "cpu"):
        p = os.path.join(RES, "profiling", f"profile_{dev}.json")
        if os.path.exists(p):
            prof[dev] = json.load(open(p))
    rows = []
    for dev, pr in prof.items():
        for key, title in systems:
            if key not in pr:
                continue
            e = pr[key]
            rows.append(dict(device=dev, system=title, force_us=1e6 * e["physical_force_s"], fr_kde_us=1e6 * e["fr_kde_score_s"],
                             kernel_est_us=1e6 * e["kernel"]["estimator_per_step_s"], hist_est_us=1e6 * e["histogram"]["estimator_per_step_s"],
                             kernel_total_us=1e6 * e["kernel"]["total_step_s"], hist_total_us=1e6 * e["histogram"]["total_step_s"],
                             est_speedup=e["estimator_speedup"], total_speedup=e["total_step_speedup"],
                             kernel_mem=e["kernel"].get("peak_mem_bytes"), hist_mem=e["histogram"].get("peak_mem_bytes")))
    campaign = {}
    if "entropic_bottleneck" in summ:
        ef = summ["entropic_bottleneck"]["efficiency"]
        campaign["entropic_bottleneck"] = dict(kernel_wall_s=ef["kernel_wall_s"], hist_wall_s=ef["hist_wall_s"], speedup=ef["speedup"], R=ef["R"],
                                              note="one batch of 20 seeds x [abf, fr_uniform], 40000 steps, end to end")
    if "gateway" in summ:
        ef = summ["gateway"]["efficiency"]
        campaign["gateway"] = dict(kernel_wall_s=ef["kernel_wall_s"], hist_wall_s=ef["hist_wall_s"], speedup=ef["speedup"], R=ef["R"],
                                   note="one batch of 32 (init, seed) rows x [abf, fr_uniform], 100000 steps, end to end")
    if "wca" in summ:
        ef = summ["wca"]["efficiency"]
        campaign["wca"] = {m: ef[m]["runtime_s"]["median"] for m in ef}
        campaign["wca"]["note"] = "median sampler seconds per 120000-step run; kernel arms carry the read-out bank (accepted scoring), histogram arms none"
    json.dump(dict(profiling_rows=rows, campaign_wall=campaign), open(os.path.join(RES, "efficiency.json"), "w"), indent=2, default=float)
    with open(os.path.join(RES, "efficiency.md"), "w") as fh:
        fh.write("| device | system | phys. force (us) | FR KDE+score (us) | kernel estimator (us/step) | histogram estimator (us/step) | estimator speedup | kernel total (us/step) | histogram total (us/step) | total speedup | peak mem kernel / hist (MB) |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            mem = (f"{r['kernel_mem'] / 1e6:.0f} / {r['hist_mem'] / 1e6:.0f}" if r["kernel_mem"] else "n/a")
            fh.write(f"| {r['device']} | {r['system']} | {r['force_us']:.0f} | {r['fr_kde_us']:.0f} | {r['kernel_est_us']:.0f} | {r['hist_est_us']:.0f} | {r['est_speedup']:.2f}x | "
                     f"{r['kernel_total_us']:.0f} | {r['hist_total_us']:.0f} | {r['total_speedup']:.2f}x | {mem} |\n")
        fh.write("\nCampaign end-to-end wall time:\n\n```\n" + json.dumps(campaign, indent=1, default=float) + "\n```\n")
    if rows:
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        labels = [f"{r['system']}\n({r['device']})" for r in rows]
        x = np.arange(len(rows)); w = 0.2
        ax.bar(x - 1.5 * w, [r["kernel_est_us"] for r in rows], w, color=COL["kernel"], label="kernel estimator / step")
        ax.bar(x - 0.5 * w, [r["hist_est_us"] for r in rows], w, color=COL["hist"], label="histogram estimator / step")
        ax.bar(x + 0.5 * w, [r["kernel_total_us"] for r in rows], w, color=COL["kernel"], alpha=0.4, label="kernel total / step")
        ax.bar(x + 1.5 * w, [r["hist_total_us"] for r in rows], w, color=COL["hist"], alpha=0.4, label="histogram total / step")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_yscale("log"); ax.set_ylabel("microseconds per MD step"); ax.legend(fontsize=7)
        ax.set_title("Estimator cost vs total step cost (profiling runs, synchronised timing)", fontsize=9)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"efficiency.{ext}"), bbox_inches="tight")
    print(f"wrote {os.path.relpath(out_dir, ROOT)}/effect_size_both_systems.*, efficiency.*, {os.path.relpath(RES, ROOT)}/efficiency.json|md")


if __name__ == "__main__":
    main()
