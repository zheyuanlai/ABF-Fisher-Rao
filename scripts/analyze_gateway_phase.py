#!/usr/bin/env python
"""Classify every cell of the entropic-gateway screen, and freeze the result.

Reads the ABF-only raw artifact and applies the classification rule that was fixed in
``gateway_core`` before any of it ran.  Writes the **whole** map -- every cell, including
the ones where mFR could not possibly help -- plus a frozen JSON that the mFR stage reads
to find its anchor.  The anchor is selected by a rule stated here in full, not by looking
at which cell mFR wins.

    python scripts/analyze_gateway_phase.py
    python scripts/analyze_gateway_phase.py --dir results/gateway_phase/production
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_core as gw  # noqa: E402

REGIMES = ("ABF-sufficient", "intermediate", "establishment-limited", "discovery-limited")


def load(dirpath):
    d = np.load(os.path.join(dirpath, "raw.npz"), allow_pickle=True)
    prov = json.load(open(os.path.join(dirpath, "provenance.json")))
    return d, prov


def cell_rows(d):
    """Rebuild per-run metric dicts from the flat arrays, keyed by cell."""
    t = d["t"][0]
    P, Q = d["P_regions"], d["Q_regions"]
    cells = {}
    for i in range(len(P)):
        key = (float(d["config_beta"][i]) if "config_beta" in d else
               json.loads(str(d["config_json"][i]))["beta"],
               float(d["s"][i]), float(d["r_ratio"][i]), str(d["init"][i]))
        m = gw.hit_and_establish(P[i][:, 2], Q[i][:, 2], t)
        m.update(seed=int(d["seed"][i]), final_l2_f=float(d["final_l2_f"][i]),
                 int_l2_f=float(d["int_l2_f"][i]), final_l2_fp=float(d["final_l2_fp"][i]),
                 int_l2_fp=float(d["int_l2_fp"][i]),
                 barrier_kT=float(d["barrier_kT"][i]))
        cells.setdefault(key, []).append(m)
    return cells, t


def med(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    v = np.where(np.isfinite(v), v, np.nan)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results/gateway_phase/production"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--refreeze", action="store_true",
                    help="overwrite an existing frozen classification (breaks the "
                         "before-any-FR-arm provenance chain; use deliberately)")
    a = ap.parse_args()
    out = a.out or a.dir
    d, prov = load(a.dir)
    cells, times = cell_rows(d)
    T = float(times[-1])
    print(f"{len(cells)} (beta, s, r, init) cells from {len(d['seed'])} runs, T = {T:g}\n")

    table = []
    for key in sorted(cells):
        beta, s, r, init = key
        rows = cells[key]
        reg = gw.classify(rows)
        n_seeds = len(rows)
        late = np.array([(not np.isfinite(x["T_hit_frac"])) or
                         x["T_hit_frac"] >= gw.DISCOVERY_FRAC for x in rows])
        table.append(dict(
            beta=beta, s=s, r=r, init=init, regime=reg, n_seeds=n_seeds,
            barrier_kT=rows[0]["barrier_kT"],
            T_hit_frac=med(rows, "T_hit_frac"), T_est_frac=med(rows, "T_est_frac"),
            est_gap_frac=med(rows, "est_gap_frac"),
            below_half_frac=med(rows, "below_half_frac"),
            seeds_late_discovery=int(late.sum()),
            seeds_never_established=int(sum(1 for x in rows
                                            if not np.isfinite(x["T_est"]))),
            final_occupancy=med(rows, "final_occupancy"),
            final_target=med(rows, "final_target"),
            occ_over_target=med(rows, "final_occupancy") / max(med(rows, "final_target"), 1e-12),
            integrated_deficit=med(rows, "integrated_deficit"),
            max_rel_deficit=med(rows, "max_rel_deficit"),
            final_l2_f=med(rows, "final_l2_f"), int_l2_f=med(rows, "int_l2_f"),
            int_l2_fp=med(rows, "int_l2_fp")))

    csv_path = os.path.join(out, "phase_table.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f"wrote {csv_path}")

    # ------------------------------------------------------------- the map
    betas = sorted({t["beta"] for t in table})
    ss = sorted({t["s"] for t in table})
    rs = sorted({t["r"] for t in table})
    short = {"ABF-sufficient": "A", "intermediate": "i",
             "establishment-limited": "E", "discovery-limited": "D"}
    for init in sorted({t["init"] for t in table}):
        print(f"\n=== init = {init} "
              f"{'(headline arm)' if init == 'left' else '(mechanism control)'} ===")
        for beta in betas:
            print(f"\n  beta = {beta:g}   (barrier {table[0]['barrier_kT']:.1f}"
                  f"-{max(t['barrier_kT'] for t in table):.1f} kT)")
            print("        " + "".join(f"r={r:<10g}" for r in rs))
            for s in ss:
                line = f"  s={s:<4g}"
                for r in rs:
                    c = next(t for t in table if (t["beta"], t["s"], t["r"], t["init"])
                             == (beta, s, r, init))
                    line += f"{short[c['regime']]} gap={c['est_gap_frac']:.2f}  "
                print(line)
    print("\n  A = ABF-sufficient   i = intermediate   "
          "E = establishment-limited   D = discovery-limited")

    counts = {k: sum(1 for t in table if t["regime"] == k) for k in REGIMES}
    print("\ncell counts over the whole map: "
          + ", ".join(f"{k} {v}" for k, v in counts.items()))

    # -------------------------------------------------------------- anchor
    anchor, anchor_note = select_anchor(table)
    frozen = dict(
        frozen_at=_dt.datetime.now().isoformat(timespec="seconds"),
        source_dir=os.path.relpath(a.dir, ROOT),
        raw_sha256=hashlib.sha256(
            open(os.path.join(a.dir, "raw.npz"), "rb").read()).hexdigest()[:16],
        provenance=prov, rule=dict(
            discovery_frac=gw.DISCOVERY_FRAC,
            est_gap_sufficient=gw.EST_GAP_SUFFICIENT,
            est_gap_limited=gw.EST_GAP_LIMITED,
            below_half_frac=gw.BELOW_HALF_FRAC,
            discovery_seed_frac=gw.DISCOVERY_SEED_FRAC,
            est_band=list(gw.EST_BAND), hold_frac=gw.HOLD_FRAC, x_basin=gw.X_BASIN,
            priority="discovery-limited > establishment-limited > ABF-sufficient > "
                     "intermediate"),
        regime_counts=counts, table=table, anchor=anchor, anchor_rule=anchor_note)
    # A file called "frozen" must not silently re-freeze.  The anchor run records the
    # frozen file's timestamp and hash as its provenance, so rewriting it -- even with
    # identical content and a new timestamp -- breaks the chain that proves the cell was
    # chosen before any Fisher-Rao arm ran.  Re-running for the figure alone is the common
    # case and must not have that side effect.
    fpath = os.path.join(out, "phase_classification.frozen.json")
    if os.path.exists(fpath) and not a.refreeze:
        old = json.load(open(fpath))
        same = (old["raw_sha256"] == frozen["raw_sha256"]
                and old["regime_counts"] == frozen["regime_counts"]
                and old["anchor"] == frozen["anchor"])
        print(f"\n{fpath} exists and is NOT rewritten "
              f"(frozen {old['frozen_at']}).")
        print(f"  re-derived classification {'MATCHES' if same else 'DIFFERS FROM'} the "
              f"frozen one." + ("" if same else "  Pass --refreeze deliberately if the "
                                "inputs really changed."))
        if not same:
            raise SystemExit(2)
    else:
        with open(fpath, "w") as fh:
            json.dump(frozen, fh, indent=2, default=float)
        print(f"\nwrote {fpath}")
    print(f"\nANCHOR (preregistered): beta={anchor['beta']:g}, s={anchor['s']:g}, "
          f"r={anchor['r']:g}  [{anchor['regime']}]")
    print(f"  {anchor_note}")

    make_figure(os.path.join(out, "gateway_phase_diagram.pdf"), table, betas, ss, rs,
                anchor)


def select_anchor(table):
    """Pick the mFR anchor by a rule stated before any FR arm runs.

    Requirements, in order:
      * ``init == 'left'`` -- the headline arm, where discovery is not handed over;
      * regime ``establishment-limited`` in the frozen classification;
      * **interior, not knife-edge**: every one of its neighbours along s, r and beta that
        exists in the map must also be establishment-limited.  A cell on the boundary would
        make the whole comparison a test of the classifier's threshold rather than of mFR;
      * among the survivors, the largest median integrated deficit -- the most population
        actually missing, for the longest, which is the quantity mFR claims to repair.
    """
    betas = sorted({t["beta"] for t in table})
    ss = sorted({t["s"] for t in table})
    rs = sorted({t["r"] for t in table})
    idx = {(t["beta"], t["s"], t["r"]): t for t in table if t["init"] == "left"}

    def neighbours(b, s, r):
        for axis, vals, cur in ((0, betas, b), (1, ss, s), (2, rs, r)):
            i = vals.index(cur)
            for j in (i - 1, i + 1):
                if 0 <= j < len(vals):
                    k = [b, s, r]
                    k[axis] = vals[j]
                    yield tuple(k)

    limited = [t for t in idx.values() if t["regime"] == "establishment-limited"]
    interior = [t for t in limited
                if all(idx[n]["regime"] == "establishment-limited"
                       for n in neighbours(t["beta"], t["s"], t["r"]) if n in idx)]
    pool = interior or limited
    note = ("interior establishment-limited cell (all existing neighbours along s, r and "
            "beta share its regime), largest median integrated deficit")
    if not interior:
        note = ("NO interior establishment-limited cell exists; fell back to the "
                "establishment-limited cell with the largest median integrated deficit. "
                "The anchor is therefore on a regime boundary and the comparison must say "
                "so.")
    if not pool:
        return None, "no establishment-limited cell anywhere in the map"
    best = max(pool, key=lambda t: t["integrated_deficit"])
    return best, note


def make_figure(path, table, betas, ss, rs, anchor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    colors = ["#4C72B0", "#C7C7C7", "#DD8452", "#C44E52"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    inits = sorted({t["init"] for t in table})
    fig, axes = plt.subplots(len(inits), len(betas),
                             figsize=(3.2 * len(betas), 3.3 * len(inits)), squeeze=False)
    for ri, init in enumerate(inits):
        for ci, beta in enumerate(betas):
            ax = axes[ri][ci]
            M = np.full((len(ss), len(rs)), np.nan)
            for i, s in enumerate(ss):
                for j, r in enumerate(rs):
                    c = next((t for t in table
                              if (t["beta"], t["s"], t["r"], t["init"])
                              == (beta, s, r, init)), None)
                    if c is not None:
                        M[i, j] = REGIMES.index(c["regime"])
                        ax.text(j, i, f"{c['est_gap_frac']:.2f}", ha="center",
                                va="center", fontsize=7,
                                color="white" if M[i, j] in (0, 3) else "black")
                        if anchor and (beta, s, r, init) == (anchor["beta"], anchor["s"],
                                                             anchor["r"], "left"):
                            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                                       edgecolor="black", lw=3))
            ax.imshow(M, cmap=cmap, norm=norm, aspect="auto", origin="upper")
            ax.set_xticks(range(len(rs))); ax.set_xticklabels([f"{r:g}" for r in rs])
            ax.set_yticks(range(len(ss))); ax.set_yticklabels([f"{s:g}" for s in ss])
            ax.set_xlabel("$r=\\omega_{in}/\\omega_{out}$")
            if ci == 0:
                ax.set_ylabel(f"{init}\n$s$ (gateway width)")
            ax.set_title(f"$\\beta$ = {beta:g}", fontsize=10)
    handles = [Patch(facecolor=c, label=lab) for c, lab in zip(colors, REGIMES)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Entropic gateway: ABF-only regime map (cell text = "
                 "$(T_{est}-T_{hit})/T$)\nblack box = preregistered mFR anchor",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path, format="pdf", bbox_inches="tight")
    # PNG as well: every other figure in the report is a PNG, and mixing raster
    # and vector includes has bitten this build before.
    fig.savefig(path.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
