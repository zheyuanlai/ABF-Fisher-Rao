"""Job lists for candidate A' (ethane / rigid Si-ITE / Dubbeldam 2004 / 300 K), PREREGISTRATION.md sections 4-6.

    python scripts/overnight/ite_jobs.py STAGE --dt 2e-4 [--T 200 --ramp 1 --eq 5 --ng 180]

Stages: horizon, warm, bins, screen.  Times are in t.u. and converted with dt.  Two arms only; every job here is the
ABF arm (method "abf") except the FR stages written by later amendments.
"""
import argparse
import json
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FF = {"T_K": 300.0, "eps_O_K": 93.0, "sig_O": 3.48, "eps_Si_K": 0.0, "sig_Si": 1.0, "eps_CC_K": 108.0,
      "sig_CC": 3.76, "rc": 12.0, "shift": True, "k_bond_K": 96500.0, "r0": 1.54}


TEMP = 300.0          # set by --temp (A'-150: 150 K); seed bases are shifted by --seedshift (Amendment 3)
SEEDSHIFT = 0


def cfg(N, seed, dt, T, ramp, eq, ng, save_tu=None):
    seed = seed + SEEDSHIFT
    n = int(round(T / dt))
    save_every = int(round((save_tu if save_tu else T / 400.0) / dt))
    return dict(variant="ite_itq3", T_K=TEMP, ff=dict(FF, T_K=TEMP), N=N, M=1, seed=seed, dt=dt, n_steps=n, save_every=save_every,
                n_grid=ng, warmup_steps=int(round(ramp / dt)), trust_count=200.0, fr_start_steps=10 ** 12,
                fr_every=max(1, int(round(0.001 / dt))), genealogy_window_steps=int(round(0.05 * T / dt)),
                equil_steps=int(round(eq / dt)), window_phi=0.5, cage_phi=2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--T", type=float, default=200.0)
    ap.add_argument("--ramp", type=float, default=1.0)
    ap.add_argument("--eq", type=float, default=5.0)
    ap.add_argument("--ng", type=int, default=180)
    ap.add_argument("--K", type=int, default=1024, help="eligible replica count (FR stages)")
    ap.add_argument("--rate", type=float, default=None, help="frozen FR rate (production)")
    ap.add_argument("--block", default="explore", help="production block: explore (8 seeds) | confirm (16 seeds)")
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--seedshift", type=int, default=0)
    ap.add_argument("--cell", default="ite")
    a = ap.parse_args()
    global TEMP, SEEDSHIFT
    TEMP, SEEDSHIFT = a.temp, a.seedshift
    jobs = []
    out = f"results/overnight/{a.cell}"
    if a.stage == "horizon":          # 4a: T up to 400 t.u. (rule compares T and 2T), default ramp 1 / pre-eq 5
        for N in (1024, 64):
            for s in range(3):
                jobs.append(dict(cfg=cfg(N, 940000 + s, a.dt, 400.0, 1.0, 5.0, 180, save_tu=0.5), method="abf",
                                 out=f"{out}/calib_horizon/abf_N{N}_s{s}.npz"))
    elif a.stage == "warm":           # 4b
        for N in (1024, 64):
            for s in range(3):
                for ramp in (1.0, 4.0):
                    for eq in (0.0, 5.0):
                        jobs.append(dict(cfg=cfg(N, 941000 + s, a.dt, a.T, ramp, eq, 180), method="abf",
                                         out=f"{out}/calib_warm/abf_N{N}_r{ramp:g}_e{eq:g}_s{s}.npz"))
    elif a.stage == "bins":           # 4c
        for N in (1024, 64):
            for s in range(3):
                for ng in (90, 180, 360):
                    jobs.append(dict(cfg=cfg(N, 942000 + s, a.dt, a.T, a.ramp, a.eq, ng), method="abf",
                                     out=f"{out}/calib_bins/abf_N{N}_ng{ng}_s{s}.npz"))
    elif a.stage == "screen":         # section 5
        for N in (1024, 256, 64):
            for s in range(8):
                jobs.append(dict(cfg=cfg(N, 950000 + s, a.dt, a.T, a.ramp, a.eq, a.ng), method="abf",
                                 out=f"{out}/screen/abf_N{N}_s{s}.npz"))
    elif a.stage == "frcal":          # section 6: safety-only rate ladder, FR arm only, 3 seeds (960 000+)
        for rate in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            for s in range(3):
                c = cfg(a.K, 960000 + s, a.dt, a.T, a.ramp, a.eq, a.ng)
                c.update(fr_rate=rate, fr_start_steps=int(round(a.ramp / a.dt)))
                jobs.append(dict(cfg=c, method="fr", out=f"{out}/frcal/fr_N{a.K}_rate{rate:g}_s{s}.npz"))
    elif a.stage == "prod":           # section 6: paired two-arm blocks on fresh seeds
        if a.rate is None:
            raise SystemExit("--rate (frozen by the safety calibration) is required")
        base, n = (970000, 8) if a.block == "explore" else (980000, 16)
        for s in range(n):
            for method in ("abf", "fr"):
                c = cfg(a.K, base + s, a.dt, a.T, a.ramp, a.eq, a.ng)
                c.update(fr_rate=a.rate, fr_start_steps=int(round(a.ramp / a.dt)))
                jobs.append(dict(cfg=c, method=method, out=f"{out}/prod_{a.block}/{method}_N{a.K}_s{s}.npz"))
    else:
        raise SystemExit(f"unknown stage {a.stage}")
    for j in jobs:                    # threads: the replica force loop is parallel; small K gains nothing from threads
        j["threads"] = 8 if j["cfg"]["N"] >= 512 else (2 if j["cfg"]["N"] >= 256 else 1)
    os.makedirs(os.path.join(ROOT, "configs", "overnight", a.cell), exist_ok=True)
    path = os.path.join(ROOT, "configs", "overnight", a.cell, f"jobs_{a.stage}.json")
    json.dump(dict(stage=f"{a.cell} {a.stage} ({a.temp} K, dt {a.dt}, T {a.T}, ramp {a.ramp}, eq {a.eq}, ng {a.ng})",
                   jobs=jobs),
              open(path, "w"), indent=1)
    print(path, len(jobs))


if __name__ == "__main__":
    main()
