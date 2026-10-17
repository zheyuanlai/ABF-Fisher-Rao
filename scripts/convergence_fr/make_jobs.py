"""Job files for the convergence-FR campaign blocks (PREREGISTRATION.md sections 3-5).  Every config is the frozen
validated one of its cell, copied from the historical job files, with only the fields named here changed.

    python scripts/convergence_fr/make_jobs.py E1            # ITE 300 K K1024, ABF vs ABF + FR stopped at 10 t.u.
    python scripts/convergence_fr/make_jobs.py ERIcal --rate 2.0
    python scripts/convergence_fr/make_jobs.py ERI --rate R
    python scripts/convergence_fr/make_jobs.py ITEcal --rate 2.0
    python scripts/convergence_fr/make_jobs.py M300_256|M150_256|M150_1024 --rate R
    python scripts/convergence_fr/make_jobs.py E1x           # exploratory t_stop ladder on X1 seeds

Runs are single-threaded (bitwise identical to the threaded historical runs; the replica loop is order-free).
"""
import argparse
import json
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = "results/convergence_fr_campaign"
CFG = os.path.join(ROOT, "configs", "convergence_fr")


def base(path):
    j = json.load(open(os.path.join(ROOT, path)))
    c = dict(j["jobs"][0]["cfg"])
    c.pop("seed")
    return c


def ite(temp):
    c = base("configs/overnight/ite/jobs_prod.json")                 # X1: 300 K, K1024, T50, 180 bins, rate 2.0
    if temp == 150:
        c150 = base("configs/overnight/ite150/jobs_prod.json")       # X2: same model at 150 K
        c["T_K"] = c150["T_K"]
        c["ff"] = c150["ff"]
    return c


def eri():
    c = base("configs/eri_campaign/jobs/screen_dilute.json")         # ERI 600 K, dt 2e-4, T40, 180 bins (ABF-only)
    c["fr_start_steps"] = c["warmup_steps"]                          # FR starts at the end of the bias ramp (1 t.u.)
    return c


def jobs_for(c, seeds, arms, outdir, tag=""):
    js = []
    for s in seeds:
        for method, over in arms:
            cc = dict(c, seed=s, **over)
            name = f"{method}{tag}_N{cc['N']}_s{s}.npz" if method == "abf" else f"fr{over.get('_tag', '')}_N{cc['N']}_s{s}.npz"
            cc.pop("_tag", None)
            js.append(dict(cfg=cc, method=method, out=f"{OUT}/{outdir}/{name}", threads=1))
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("block")
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--K", type=int, default=None, help="ERI: one replica count only (the calibrated rate may differ by K)")
    a = ap.parse_args()
    b = a.block
    if b == "E1":
        c = ite(300)                                                 # rate 2.0, fr_start 1 t.u. (X1 settings)
        stop = 100_000                                               # t_stop = 10 t.u. (prereg section 3.2)
        js = jobs_for(c, range(1_100_000, 1_100_016), [("abf", {}), ("fr", dict(fr_stop_steps=stop))], "E1_ite300_stop10")
    elif b == "E1x":
        c = ite(300)
        arms = [("fr", dict(fr_stop_steps=s, _tag=f"_stop{s // 10000}")) for s in (50_000, 100_000, 200_000)]
        js = jobs_for(c, range(980_000, 980_004), arms, "E1x_ite300_ladder")
    elif b == "ERIcal":
        js = []
        for K, ge in ((256, None), (1024, None)):
            c = dict(eri(), N=K)
            for s in range(1_110_000, 1_110_003):
                cc = dict(c, seed=s, fr_rate=a.rate)
                js.append(dict(cfg=cc, method="fr", out=f"{OUT}/ERIcal/fr_N{K}_rate{a.rate:g}_s{s}.npz", threads=1))
    elif b == "ERI":
        js = []
        for K in ((1024, 256) if a.K is None else (a.K,)):
            c = dict(eri(), N=K, fr_rate=a.rate)
            js += jobs_for(c, range(1_111_000, 1_111_016), [("abf", {}), ("fr", {})], f"ERI_K{K}")
    elif b == "ITEcal":
        js = []
        for temp in (300, 150):
            c = dict(ite(temp), N=256)
            for s in range(1_120_000, 1_120_003):
                cc = dict(c, seed=s, fr_rate=a.rate)
                js.append(dict(cfg=cc, method="fr", out=f"{OUT}/ITEcal_T{temp}/fr_N256_rate{a.rate:g}_s{s}.npz", threads=1))
    elif b in ("M300_256", "M150_256", "M150_1024", "M300_1024"):
        temp, K = int(b[1:4]), int(b.split("_")[1])
        c = dict(ite(temp), N=K, fr_rate=a.rate)
        s0 = {"M300_256": 1_121_000, "M150_256": 1_122_000, "M150_1024": 1_123_000, "M300_1024": 1_124_000}[b]
        js = jobs_for(c, range(s0, s0 + 16), [("abf", {}), ("fr", {})], f"{b}_rate{a.rate:g}")
    else:
        raise SystemExit(f"unknown block {b}")
    os.makedirs(CFG, exist_ok=True)
    tag = b if a.rate is None else f"{b}_rate{a.rate:g}"
    if a.K is not None:
        tag += f"_K{a.K}"
    path = os.path.join(CFG, f"jobs_{tag}.json")
    json.dump(dict(stage=tag, jobs=js), open(path, "w"), indent=1)
    print(path, len(js), "jobs")


if __name__ == "__main__":
    main()
