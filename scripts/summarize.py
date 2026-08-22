"""Aggregate every stored result into the campaign's summary tables (no simulation)."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from rcwfr.campaign import paired_bootstrap, rel_change

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def jl(p):
    with open(p) as f:
        return json.load(f)


def confirm_table(sysname):
    p = RES / "confirm" / f"{sysname}.json"
    if not p.exists():
        return None
    d = jl(p)
    fl, arms = d["floor"], d["arms"]
    rows = []
    for k, v in arms.items():
        IF = np.asarray(v["I_F"], float)
        eF = np.asarray(v["e_F_final"], float)
        m, lo, hi = paired_bootstrap(IF)
        rows.append(dict(arm=k, I_F=m, I_F_lo=lo, I_F_hi=hi,
                         e_F=float(np.median(eF)), ratio=float(np.median(eF)) / fl,
                         chan=float(np.median(np.asarray(v["chan"], float))),
                         cov=float(np.median(np.asarray(v["cov"], float))),
                         acc=v.get("ex_accept")))
    rows.sort(key=lambda r: r["I_F"])
    print(f"\n### {sysname}   (floor {fl:.5f})")
    print(f"{'arm':14s} {'I_F':>9s} {'95% CI':>20s} {'e_F':>9s} {'/fl':>6s} "
          f"{'chan':>7s} {'cov':>5s}")
    for r in rows:
        print(f"{r['arm']:14s} {r['I_F']:9.5f} [{r['I_F_lo']:8.5f},{r['I_F_hi']:8.5f}] "
              f"{r['e_F']:9.5f} {r['ratio']:6.1f} {r['chan']:7.4f} {r['cov']:5.2f}")
    print("\n  paired median rel. change in I_F (negative = row better):")
    base = [b for b in ("ti_cold", "reti_cold", "abf") if b in arms]
    print(f"  {'arm':14s} " + " ".join(f"{'vs ' + b:>26s}" for b in base))
    for r in rows:
        cells = []
        for b in base:
            mm, lo, hi = paired_bootstrap(rel_change(arms[r["arm"]]["I_F"], arms[b]["I_F"]))
            tag = "*" if hi < 0 else (" " if lo > 0 else "~")
            cells.append(f"{100*mm:+7.1f}% [{100*lo:+6.1f},{100*hi:+6.1f}]{tag}")
        print(f"  {r['arm']:14s} " + " ".join(f"{c:>26s}" for c in cells))
    return rows


def torsion_table():
    p = RES / "torsion" / "torsion_scaling.json"
    if not p.exists():
        return
    d = jl(p)
    print("\n### TORSION domain-size scaling (P1) - paired median rel. change in I_F")
    print(f"{'L':>5} {'wells':>6} {'floor':>8} | {'RC-WFR vs ABF':>26} | {'RC-WFR vs fixed TI':>26}")
    for L in sorted(d, key=float):
        r = d[L]
        m1, lo1, hi1, *_ = r["wfr_vs_abf"]; m2, lo2, hi2, *_ = r["wfr_vs_ti"]
        print(f"{float(L):>5.0f} {r['n_wells']:>6} {r['floor']:>8.5f} | "
              f"{100*m1:+7.1f}% [{100*lo1:+6.1f},{100*hi1:+6.1f}] | "
              f"{100*m2:+7.1f}% [{100*lo2:+6.1f},{100*hi2:+6.1f}]")
    print("\n  best I_F per family:")
    print(f"{'L':>5} " + " ".join(f"{k:>11s}" for k in ("wfr", "abf", "ti_cold", "reti_cold")))
    for L in sorted(d, key=float):
        arms = d[L]["arms"]
        vals = []
        for pre in ("wfr", "abf", "ti_cold", "reti_cold"):
            cand = [np.median(v["I_F"]) for k, v in arms.items() if k.startswith(pre)]
            vals.append(min(cand) if cand else np.nan)
        print(f"{float(L):>5.0f} " + " ".join(f"{v:>11.5f}" for v in vals))


def mspec_table():
    p = RES / "mspec" / "CHANNEL_mspec.json"
    if not p.exists():
        return
    d = jl(p)
    print("\n### Fiber-size scaling (P2) - RE acceptance vs RC-WFR's lift bias")
    print(f"{'m_spec':>7} {'|F|rms':>8} {'RE acc':>7} | {'wfr':>9} {'ti_cold':>9} "
          f"{'reti':>9}   (I_F / |F|rms)   {'wfr chan':>9} {'reti chan':>10}")
    for m in sorted(d, key=int):
        r = d[m]; a = r["arms"]
        g = lambda k: float(np.median(np.asarray(a[k]["I_F_rel"], float)))
        c = lambda k: float(np.median(np.asarray(a[k]["chan"], float)))
        print(f"{int(m):>7} {r['F_rms']:>8.3f} {a['reti_cold_M256']['ex_accept']:>7.3f} | "
              f"{g('wfr'):>9.4f} {g('ti_cold'):>9.4f} {g('reti_cold_M256'):>9.4f}"
              f"{'':>18} {c('wfr'):>9.4f} {c('reti_cold_M256'):>10.4f}")


if __name__ == "__main__":
    for s in ("EB", "CHANNEL"):
        confirm_table(s)
    torsion_table()
    mspec_table()
