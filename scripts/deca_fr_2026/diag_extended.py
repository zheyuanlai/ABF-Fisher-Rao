"""Post-protocol diagnostic: which conditional mixture above 2.9 nm?

Compares the ABF mean force F'(R) and the second-half structural-class mixture p(y | R) on the
extended region between (i) production ABF from the helix (K = 64 x 2 ns, 16 seeds), (ii) ABF on
[2.6, 3.6] nm from a restrained PPII start (ext_ppii_K64), optionally (iii) the full-domain
collapsed-PPII / mixed init cells, and the two reference F' estimators.  Writes a JSON table and a
figure; prints the table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                                          # noqa: E402

INTERVALS = [(2.7, 2.9), (2.9, 3.1), (3.1, 3.3), (3.3, 3.5)]


def _final_mean_force(z):
    mf = z["mean_force"]
    assert mf.ndim == 3 and mf.shape[-1] == z["grid"].size, mf.shape
    return mf[-1]                                                            # (R, n_grid)


def _y_mix(z, lo, hi, n_lab, second_half=True):
    xi, y, st = z["label_xi"], z["label_y"].astype(int), z["label_steps"]
    sel = st >= st.max() / 2 if second_half else np.ones(st.shape, bool)
    out = []
    for r in range(xi.shape[1]):
        x = xi[sel, r].ravel(); yy = y[sel, r].ravel()
        m = (x >= lo) & (x < hi)
        out.append(np.bincount(yy[m], minlength=n_lab)[:n_lab] / max(m.sum(), 1) if m.sum() >= 50 else np.full(n_lab, np.nan))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="results/deca_fr_2026/reference/reference.npz")
    ap.add_argument("--helix", default="results/deca_fr_2026/production/K64_T2ns/abf.npz")
    ap.add_argument("--ext", default="results/deca_fr_2026/diagnostics/ext_ppii_K64/abf.npz")
    ap.add_argument("--extra", nargs="*", default=["results/deca_fr_2026/diagnostics/init_ppii_K64/abf.npz",
                                                   "results/deca_fr_2026/diagnostics/init_mixed_K64/abf.npz"])
    ap.add_argument("--out", default="results/deca_fr_2026/diagnostics/analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ref = np.load(os.path.join(C.ROOT, args.reference))
    g_ref, Fp_cond = ref["grid"], ref["Fprime_ref"]
    Fp_grad = np.gradient(ref["F_ref"], g_ref)
    Jy = ref["joint_y"]; n_lab = Jy.shape[0]

    sources = {"helix history (prod K64 ABF)": args.helix, "PPII history, extended region": args.ext}
    for p in args.extra:
        if os.path.exists(os.path.join(C.ROOT, p)):
            sources[os.path.basename(os.path.dirname(p)) + " (full domain)"] = p
    table, fig_data = {}, {}
    for name, p in sources.items():
        p = os.path.join(C.ROOT, p)
        if not os.path.exists(p):
            print(f"[skip] {name}: {p} missing"); continue
        z = np.load(p); grid = z["grid"]; mf = _final_mean_force(z)
        row = {}
        for lo, hi in INTERVALS:
            m = (grid >= lo) & (grid < hi)
            per_seed = mf[:, m].mean(1)
            mr = (g_ref >= lo) & (g_ref < hi)
            row[f"{lo}-{hi}"] = dict(Fp_median=float(np.median(per_seed)), Fp_q25=float(np.percentile(per_seed, 25)),
                                     Fp_q75=float(np.percentile(per_seed, 75)),
                                     ref_cond=float(Fp_cond[mr].mean()), ref_grad=float(Fp_grad[mr].mean()),
                                     y_mix=np.nanmedian(_y_mix(z, lo, hi, n_lab), axis=0).round(3).tolist())
        table[name] = row
        fig_data[name] = (grid, np.median(mf, 0), np.percentile(mf, 25, 0), np.percentile(mf, 75, 0))
    ref_rows = {}
    for lo, hi in INTERVALS:
        mr = (g_ref >= lo) & (g_ref < hi)
        p = Jy[:, mr].sum(1); p /= max(p.sum(), 1e-300)
        ref_rows[f"{lo}-{hi}"] = dict(ref_cond=float(Fp_cond[mr].mean()), ref_grad=float(Fp_grad[mr].mean()), y_mix=p.round(3).tolist())
    table["reference"] = ref_rows

    lines = ["| source | " + " | ".join(f"F' [{lo},{hi}) med [q25,q75]" for lo, hi in INTERVALS) + " |", "|---|" + "---|" * len(INTERVALS)]
    for name, row in table.items():
        if name == "reference":
            lines.append("| reference F'(cond) | " + " | ".join(f"{row[k]['ref_cond']:.1f}" for k in row) + " |")
            lines.append("| reference F'(grad) | " + " | ".join(f"{row[k]['ref_grad']:.1f}" for k in row) + " |")
        else:
            lines.append(f"| {name} | " + " | ".join(f"{row[k]['Fp_median']:.1f} [{row[k]['Fp_q25']:.1f}, {row[k]['Fp_q75']:.1f}]" for k in row) + " |")
    lines += ["", "| source | " + " | ".join(f"p(y) [{lo},{hi}) 2nd half" for lo, hi in INTERVALS) + " |", "|---|" + "---|" * len(INTERVALS)]
    for name, row in table.items():
        lines.append(f"| {name} | " + " | ".join(str([round(v, 2) for v in row[k]['y_mix']]) for k in row) + " |")
    md = "\n".join(lines)
    print(md)
    with open(os.path.join(C.ROOT, args.out, "extended_region.md"), "w") as fh:
        fh.write(md + "\n")
    with open(os.path.join(C.ROOT, args.out, "extended_region.json"), "w") as fh:
        json.dump(table, fh, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(g_ref, Fp_cond, "k-", lw=1.4, label="reference F' (conditional)")
    ax.plot(g_ref, Fp_grad, "k--", lw=1.0, label="reference F' (grad MBAR F)")
    for name, (grid, med, q25, q75) in fig_data.items():
        l, = ax.plot(grid, med, lw=1.6, label=name)
        ax.fill_between(grid, q25, q75, color=l.get_color(), alpha=0.2, lw=0)
    ax.set_xlim(2.55, 3.55); ax.set_ylim(-20, 220)
    ax.set_xlabel("R (nm)"); ax.set_ylabel("F' (kJ/mol/nm)")
    ax.axvspan(3.5, 3.55, color="0.85"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Mean force on the extended region by sampling history")
    fig.tight_layout(); fig.savefig(os.path.join(C.ROOT, args.out, "fig_extended_region.png"), dpi=160)
    print("wrote", os.path.join(args.out, "fig_extended_region.png"))


if __name__ == "__main__":
    main()
