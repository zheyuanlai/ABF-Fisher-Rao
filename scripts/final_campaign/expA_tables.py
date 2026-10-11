#!/usr/bin/env python
"""Seed-level tables of Experiment A (docs/final_campaign/PREREGISTRATION.md section 1.8).

    python scripts/final_campaign/expA_tables.py [--cells-dir configs/final_campaign/cells/expA]
        [--out-dir results/final_campaign/expA/tables] [--n-boot 10000] [--no-verify-runs] [--provisional]
        [--force] [--recompute] [--dry-run]

From each cell's analysis summary (scripts/equal_budget/analyze_ladder.py; nothing is re-scored), per seed, cell
(lambda), N and arm: Ibar_F, final e_F, the persistent tau at the mid e_F threshold (0.0055; budget fraction u and
t.u.; 'inf' = censored), and the pair-level quantities G(Ibar_F) = (FR - ABF)/ABF, G(final e_F) and
L = log(Ibar_F^FR / Ibar_F^ABF) (repeated on both arm rows of the pair; empty when the pair is incomplete).
Per (lambda, N) the cell-level companions of the paired G (PREREGISTRATION 1.6): median G (the summary's paired
block) and the mean-based companions mean G and mean L with percentile seed-bootstrap 95 % CIs, for Ibar_F and final
e_F (in seed_level.md and tables_meta.json 'cell_summaries').
Writes <out-dir>/seed_level.csv (long format: one row per seed, lambda, N, arm), seed_level.md (one table per
(lambda, N), one row per seed with both arms) and tables_meta.json (input_signature over the summaries, code, n_boot
and the completeness basis; complete; completeness; provenance).  Order: a placeholder tables_meta.json (state
'writing', no signature, complete false), then the CSV and markdown, then the final tables_meta.json.

Before writing (and before skipping an up-to-date output) every summary must belong to its CURRENT config / scoring
code / reference and be fresh against the run tree (cross_cell.check_summary / check_freshness; --no-verify-runs skips
freshness and forbids complete).  complete = expA_analysis_common.completeness() (incl. every cell's completeness
audit PASS on the current summary, every summary at n_boot 10 000, no --provisional).  Output policy as
expA_analysis_common.decide_write (decided on tables_meta.json).
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import expA_analysis_common as C  # noqa: E402

SCHEMA = "final_expA_tables/2"
COLUMNS = ["lambda", "cell", "N", "seed", "arm", "Ibar_F", "final_e_F", "tau_e_F_mid_u", "tau_e_F_mid_t",
           "G_Ibar_F", "G_final_e_F", "L_Ibar_F", "complete"]
CODE_FILES = [os.path.abspath(__file__), os.path.join(HERE, "expA_analysis_common.py")]


def _g(a, f):
    a, f = C.fnum(a), C.fnum(f)
    if not (math.isfinite(a) and math.isfinite(f)) or a == 0:
        return math.nan
    return (f - a) / a


def _l(a, f):
    a, f = C.fnum(a), C.fnum(f)
    if not (math.isfinite(a) and math.isfinite(f)) or a <= 0 or f <= 0:
        return math.nan
    return math.log(f / a)


def rows_of(cells):
    """Long-format rows (dicts) for every planned (lambda, N, seed, arm)."""
    out = []
    for lam, rec in sorted(cells.items()):
        S = rec["S"]
        for N in rec["P"]["N_ladder"]:
            vals = {m: {k: C.per_seed(S, N, m, k) if S else {}
                        for k in ("Ibar_F", "final_e_F", "tau_e_F_mid_u", "tau_e_F_mid_t")} for m in ("abf", "fr")}
            comp = {m: set(C.complete_seeds(S, N, m)) if S else set() for m in ("abf", "fr")}
            for s in [int(x) for x in rec["P"]["seeds"]]:
                pair = s in comp["abf"] and s in comp["fr"]
                ia, if_ = vals["abf"]["Ibar_F"].get(s), vals["fr"]["Ibar_F"].get(s)
                fa, ff = vals["abf"]["final_e_F"].get(s), vals["fr"]["final_e_F"].get(s)
                G1 = _g(ia, if_) if pair else math.nan
                G2 = _g(fa, ff) if pair else math.nan
                Lv = _l(ia, if_) if pair else math.nan
                for m in ("abf", "fr"):
                    v = vals[m]
                    out.append(dict(**{"lambda": C.lam_key(lam)}, cell=rec["name"], N=int(N), seed=s, arm=m,
                                    Ibar_F=v["Ibar_F"].get(s, math.nan), final_e_F=v["final_e_F"].get(s, math.nan),
                                    tau_e_F_mid_u=v["tau_e_F_mid_u"].get(s, math.nan),
                                    tau_e_F_mid_t=v["tau_e_F_mid_t"].get(s, math.nan),
                                    G_Ibar_F=G1, G_final_e_F=G2, L_Ibar_F=Lv, complete=s in comp[m]))
    return out


def _cell(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        if math.isinf(v):
            return "inf"
        return repr(v)
    return str(v)


def to_csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _cell(r[k]) for k in COLUMNS})
    return buf.getvalue()


def cell_summaries(cells, n_boot=C.N_BOOT):
    """{lambda: {N: {metric: dict(median_G, G_ci95, wins, n, mean=mean_companions)}}} for Ibar_F and final e_F."""
    out = {}
    for lam, rec in sorted(cells.items()):
        S = rec["S"]
        if S is None:
            continue
        for N in rec["P"]["N_ladder"]:
            for metric in ("Ibar_F", "final_e_F"):
                g = C.summary_contrast(S, N, metric) or {}
                out.setdefault(C.lam_key(lam), {}).setdefault(str(N), {})[metric] = dict(
                    median_G=C.fnum(g.get("G_median")), G_ci95=[C.fnum(x) for x in (g.get("G_ci95") or [None, None])],
                    wins=g.get("wins"), n=g.get("n"),
                    mean=C.mean_companions(C.per_seed(S, N, "abf", metric), C.per_seed(S, N, "fr", metric), n_boot))
    return out


def _pct_ci(d):
    ci = d.get("ci95") or [None, None]
    return (f"{100 * C.fnum(d.get('mean')):+.2f} % [{100 * C.fnum(ci[0]):+.2f}, {100 * C.fnum(ci[1]):+.2f}]"
            if d and d.get("n") else "--")


def to_md(rows, fixture, summaries=None):
    f = C.fmt
    L = ["# Experiment A: seed-level results (PREREGISTRATION.md 1.8)", "",
         "Per seed: Ibar_F, final e_F and the persistent tau at e_F <= 0.0055 (budget fraction u; 'inf' = censored) of "
         "each arm; G = (FR - ABF)/ABF of the pair.  Source: the cells' analysis summaries.", ""]
    if fixture:
        L += ["**FIXTURE INPUT: synthetic tiny-budget data, never production evidence.**", ""]
    keys = sorted({(r["lambda"], r["N"]) for r in rows}, key=lambda k: (float(k[0]), -k[1]))
    for lk, N in keys:
        sel = [r for r in rows if r["lambda"] == lk and r["N"] == N]
        L += [f"## lambda = {lk}, N = {N}", ""]
        cs = ((summaries or {}).get(lk) or {}).get(str(N)) or {}
        for metric in ("Ibar_F", "final_e_F"):
            c = cs.get(metric)
            if not c:
                continue
            ml = c["mean"]["L"]
            L.append(f"* {metric}: median G {100 * C.fnum(c['median_G']):+.2f} % {C.fmt_ci([100 * C.fnum(x) for x in c['G_ci95']])} "
                     f"(wins {c['wins']}/{c['n']}); mean G {_pct_ci(c['mean']['G'])}; mean L {f(ml.get('mean'), 4)} "
                     f"{C.fmt_ci(ml.get('ci95'), 4)} (n {c['mean']['G'].get('n')})")
        L += ["",
              "| seed | ABF Ibar_F | FR Ibar_F | G(Ibar_F) | ABF e_F(1) | FR e_F(1) | G(e_F(1)) | ABF tau mid (u) | "
              "FR tau mid (u) |", "|---|---|---|---|---|---|---|---|---|"]
        for s in sorted({r["seed"] for r in sel}):
            a = next(r for r in sel if r["seed"] == s and r["arm"] == "abf")
            b = next(r for r in sel if r["seed"] == s and r["arm"] == "fr")
            g1 = f"{100 * a['G_Ibar_F']:+.2f} %" if math.isfinite(C.fnum(a["G_Ibar_F"])) else "--"
            g2 = f"{100 * a['G_final_e_F']:+.2f} %" if math.isfinite(C.fnum(a["G_final_e_F"])) else "--"
            L.append(f"| {s} | {f(a['Ibar_F'], 5)} | {f(b['Ibar_F'], 5)} | {g1} | {f(a['final_e_F'], 5)} | "
                     f"{f(b['final_e_F'], 5)} | {g2} | {f(a['tau_e_F_mid_u'], 4)} | {f(b['tau_e_F_mid_u'], 4)} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", default=C.CELLS_DIR)
    ap.add_argument("--out-dir", default=C.TABLES_DIR)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT, help="resamples of the mean-based companion CIs")
    ap.add_argument("--no-verify-runs", action="store_true",
                    help="skip the summaries' run-tree freshness check (forbids complete)")
    ap.add_argument("--provisional", action="store_true",
                    help="force complete = false (the synthesis runs over a failed completeness audit)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cells = C.load_cells(a.cells_dir)
    if not cells:
        raise C.ExpAError(f"no cell config under {a.cells_dir}")
    out = os.path.abspath(a.out_dir)
    meta_p = os.path.join(out, "tables_meta.json")
    verify = not a.no_verify_runs
    ins = {C.lam_key(l): dict(config=C.sha256_file(r["config"]),
                              summary=C.sha256_file(r["summary_path"]) if r["S"] is not None else None)
           for l, r in sorted(cells.items())}
    comp = C.completeness(cells, a.n_boot, verify, a.provisional)
    sig = C.digest(dict(schema=SCHEMA, inputs=ins, code=C.code_digests(CODE_FILES), n_boot=int(a.n_boot),
                        verify_runs=verify, audits=C.audit_signature(comp["audits"]), provisional=bool(a.provisional)))
    if a.dry_run:
        sp = C.summary_problems(cells, verify)
        try:
            decision = C.decide_write(meta_p, sig, a.force, a.recompute)
        except C.ExpAError as e:
            decision = f"REFUSE ({e})"
        print(f"complete-eligible {comp['complete']}; would {decision} {out}/seed_level.{{csv,md}} (signature "
              f"{sig[:16]})" + (f"; summaries not current for lambda {[f'{l:g}' for l in sp]}: would stop" if sp else ""))
        return 0
    C.require_current_summaries(cells, verify)          # also before skipping
    decision = C.decide_write(meta_p, sig, a.force, a.recompute)
    if decision == "skip":
        print(f"{out} is up to date (signature {sig[:16]}; summaries verified current)")
        return 0
    production, fixture, probs = C.design_status(cells)
    rows = rows_of(cells)
    summ = cell_summaries(cells, a.n_boot)
    prev = C.previous_signature(meta_p)
    C.begin_output(meta_p, SCHEMA, prev)
    C.write_text(os.path.join(out, "seed_level.csv"), to_csv(rows))
    C.write_text(os.path.join(out, "seed_level.md"), to_md(rows, fixture, summ))
    meta = dict(schema=SCHEMA, input_signature=sig, provenance=C.provenance(CODE_FILES), inputs=ins,
                complete=comp["complete"], completeness=comp, provisional=bool(a.provisional), freshness_verified=verify,
                n_boot=int(a.n_boot), fixture=fixture, design_problems=probs, n_rows=len(rows), columns=COLUMNS,
                files=dict(csv="seed_level.csv", md="seed_level.md"),
                file_sha256=dict(csv=C.sha256_file(os.path.join(out, "seed_level.csv")),
                                 md=C.sha256_file(os.path.join(out, "seed_level.md"))),
                cell_summaries=summ,
                note="G and L are pair-level (repeated on both arm rows); empty = pair incomplete; inf = censored tau")
    if prev and prev != sig:
        meta["replaced_signature"] = prev
    C.write_json(meta_p, meta)
    print(f"wrote {len(rows)} rows to {out}/seed_level.csv and seed_level.md (complete {meta['complete']})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except C.ExpAError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
