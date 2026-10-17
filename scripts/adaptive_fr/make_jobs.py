"""Job files for the adaptive-FR campaign (docs/adaptive_fr_campaign/PREREGISTRATION.md).  Every config is a frozen
validated config of an earlier block with only the named fields changed.

    python scripts/adaptive_fr/make_jobs.py A        # ITE 300 K, K256, T200: FR arm (rate 1.0, stop 10) on EB seeds
    python scripts/adaptive_fr/make_jobs.py EB8_rate1   # E-A3: K8 x T6400 both arms, saves every 1 t.u.
    python scripts/adaptive_fr/make_jobs.py EB4_abf     # E-A3: K4 x T12800 ABF only;  EB1_abf: K1 x T51200 ABF only
    python scripts/adaptive_fr/make_jobs.py Ecal_K8_rate1   # E-A3 blind calibration rungs (also Ecal_K4_rate*)
"""
import argparse
import json
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = "results/adaptive_fr_campaign"
CFG = os.path.join(ROOT, "configs", "adaptive_fr")


def eb_jobs():
    return json.load(open(os.path.join(ROOT, "configs", "convergence_fr", "jobs_EB.json")))["jobs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("block")
    a = ap.parse_args()
    if a.block == "A":
        js = []
        for j in eb_jobs():
            c = dict(j["cfg"], fr_rate=1.0, fr_stop_steps=100_000)
            js.append(dict(cfg=c, method="fr", out=f"{OUT}/A_ite300_K256_T200_stop10/fr_N256_s{c['seed']}.npz", threads=1))
    elif a.block in ("C1", "C2", "C5"):                   # ITE / ERI engine cells with the frozen online rule R1
        import sys
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from eri.fr_controller import R1
        if a.block == "C1":        # E1 configuration: ITE 300 K, K1024, T50, rate 2.0
            src = json.load(open(os.path.join(ROOT, "configs/convergence_fr/jobs_E1.json")))["jobs"][0]["cfg"]
            seeds, sub = range(1_300_000, 1_300_016), "C1_ite300_K1024_R1"
        elif a.block == "C2":      # E2 configuration: ITE 150 K, K256, T100, rate 0.5
            src = json.load(open(os.path.join(ROOT, "configs/convergence_fr/jobs_E2.json")))["jobs"][0]["cfg"]
            seeds, sub = range(1_310_000, 1_310_016), "C2_ite150_K256_R1"
        else:                      # ERI K1024 configuration: 600 K, T40, rate 2.0
            src = json.load(open(os.path.join(ROOT, "configs/convergence_fr/jobs_ERI_rate2_K1024.json")))["jobs"][0]["cfg"]
            seeds, sub = range(1_340_000, 1_340_016), "C5_eri600_K1024_R1"
        base = {k: v for k, v in src.items() if k not in ("seed", "fr_stop_steps")}
        js = []
        for s in seeds:
            js.append(dict(cfg=dict(base, seed=s), method="abf", out=f"{OUT}/{sub}/abf_N{base['N']}_s{s}.npz", threads=1))
            js.append(dict(cfg=dict(base, seed=s, fr_controller=R1), method="fr", out=f"{OUT}/{sub}/fr_N{base['N']}_s{s}.npz", threads=1))
    elif a.block in ("C3", "C4"):                         # LTA numba engine, historical histogram production config
        import sys
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from eri.fr_controller import R1
        T_K, s0, sub = (150.0, 1_320_000, "C3_lta150_K1024_R1") if a.block == "C3" else (300.0, 1_330_000, "C4_lta300_K1024_R1")
        js = []
        for s in range(s0, s0 + 16):
            for method, over in (("abf", {}), ("fr", dict(fr_controller=R1))):
                js.append(dict(T_K=T_K, N=1024, seed=s, cfg=over, method=method, save_every=500,
                               out=f"{OUT}/{sub}/{method}_N1024_s{s}.npz"))
    elif a.block == "D2":                                  # EXPLORATORY mechanism: re-run continuous-FR arms with the age split
        js = []
        for jf, sub, age in (("configs/convergence_fr/jobs_ERI_rate2_K256.json", "D2_eri600_K256_age", 5000),
                             ("configs/convergence_fr/jobs_M300_256_rate1.json", "D2_ite300_K256_age", 10000)):
            for j in json.load(open(os.path.join(ROOT, jf)))["jobs"]:
                if j["method"] != "fr":
                    continue
                c = dict(j["cfg"], diag_age_steps=age)
                js.append(dict(cfg=c, method="fr", out=f"{OUT}/{sub}/{os.path.basename(j['out'])}", threads=1,
                               historical=j["out"]))
    elif a.block == "D3":                                  # EXPLORATORY selection test (MECHANISM_SELECTION_PREREG.md): BOTH arms
        js = []                                            # of the D2 blocks re-run bitwise with the read-only selection diagnostic
        for jf, sub in (("configs/convergence_fr/jobs_ERI_rate2_K256.json", "D3_eri600_K256_sel"),
                        ("configs/convergence_fr/jobs_M300_256_rate1.json", "D3_ite300_K256_sel")):
            for j in json.load(open(os.path.join(ROOT, jf)))["jobs"]:
                c = dict(j["cfg"], diag_sel=1)
                js.append(dict(cfg=c, method=j["method"], out=f"{OUT}/{sub}/{os.path.basename(j['out'])}", threads=1,
                               historical=j["out"]))
    elif a.block == "D5":                                  # EXPLORATORY arrival-law test (MECHANISM_ARRIVAL_PREREG.md): BOTH arms
        js = []                                            # of the D2 blocks re-run bitwise with the read-only provenance record
        for jf, sub in (("configs/convergence_fr/jobs_ERI_rate2_K256.json", "D5_eri600_K256_arr"),
                        ("configs/convergence_fr/jobs_M300_256_rate1.json", "D5_ite300_K256_arr")):
            for j in json.load(open(os.path.join(ROOT, jf)))["jobs"]:
                c = dict(j["cfg"], diag_arr=1)
                js.append(dict(cfg=c, method=j["method"], out=f"{OUT}/{sub}/{os.path.basename(j['out'])}", threads=1,
                               historical=j["out"]))
    elif a.block.startswith("Ecal_K"):                    # blind safety calibration at small K (ITE 300 K, T 50, continuous FR)
        Kc = int(a.block.split("_K")[1].split("_")[0])
        rate = float(a.block.split("rate")[1])
        base = {k: v for k, v in eb_jobs()[0]["cfg"].items() if k != "seed"}
        base.update(N=Kc, n_steps=500_000, save_every=1250, fr_rate=rate)
        s0 = {64: 1_360_000, 16: 1_380_000, 8: 1_430_000, 4: 1_440_000}[Kc]      # K8 / K4: amendment E-A3
        js = [dict(cfg=dict(base, seed=s), method="fr", out=f"{OUT}/Ecal_K{Kc}/fr_N{Kc}_rate{rate:g}_s{s}.npz", threads=1)
              for s in range(s0, s0 + 3)]
    elif a.block == "EA_R1":                               # R1 at K256 x T200 (FR arm only; ABF arm = EB, reused)
        import sys
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from eri.fr_controller import R1
        js = []
        for j in eb_jobs():
            c = dict(j["cfg"], fr_rate=1.0, fr_controller=R1)
            js.append(dict(cfg=c, method="fr", out=f"{OUT}/EA_ite300_K256_T200_R1/fr_N256_s{c['seed']}.npz", threads=1))
    elif a.block.startswith("EB64_rate"):                  # K64 x T800 frontier cell, ABF vs ABF + FR(R1)
        import sys
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from eri.fr_controller import R1
        rate = float(a.block.split("rate")[1])
        base = {k: v for k, v in eb_jobs()[0]["cfg"].items() if k != "seed"}
        base.update(N=64, n_steps=8_000_000, save_every=20_000, fr_rate=rate)
        js = []
        for s in range(1_350_000, 1_350_016):
            js.append(dict(cfg=dict(base, seed=s), method="abf", out=f"{OUT}/EB64_ite300_K64_T800_R1/abf_N64_s{s}.npz", threads=1))
            js.append(dict(cfg=dict(base, seed=s, fr_controller=R1), method="fr", out=f"{OUT}/EB64_ite300_K64_T800_R1/fr_N64_s{s}.npz", threads=1))
    elif a.block.startswith("EB16_rate"):                  # K16 x T3200 frontier cell (amendment E-A2), ABF vs ABF + FR(R1)
        import sys
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from eri.fr_controller import R1
        rate = float(a.block.split("rate")[1])
        base = {k: v for k, v in eb_jobs()[0]["cfg"].items() if k != "seed"}
        base.update(N=16, n_steps=32_000_000, save_every=80_000, fr_rate=rate)
        js = []
        for s in range(1_370_000, 1_370_016):
            js.append(dict(cfg=dict(base, seed=s), method="abf", out=f"{OUT}/EB16_ite300_K16_T3200_R1/abf_N16_s{s}.npz", threads=1))
            js.append(dict(cfg=dict(base, seed=s, fr_controller=R1), method="fr", out=f"{OUT}/EB16_ite300_K16_T3200_R1/fr_N16_s{s}.npz", threads=1))
    elif a.block.startswith("EB8_rate"):                   # K8 x T6400 frontier cell (amendment E-A3), ABF vs ABF + FR(R1)
        import sys
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from eri.fr_controller import R1
        rate = float(a.block.split("rate")[1])
        base = {k: v for k, v in eb_jobs()[0]["cfg"].items() if k != "seed"}
        base.update(N=8, n_steps=64_000_000, save_every=10_000, fr_rate=rate)   # saves every 1 t.u. (E-A3)
        js = []
        for s in range(1_400_000, 1_400_016):
            js.append(dict(cfg=dict(base, seed=s), method="abf", out=f"{OUT}/EB8_ite300_K8_T6400_R1/abf_N8_s{s}.npz", threads=1))
            js.append(dict(cfg=dict(base, seed=s, fr_controller=R1), method="fr", out=f"{OUT}/EB8_ite300_K8_T6400_R1/fr_N8_s{s}.npz", threads=1))
    elif a.block in ("EB4_abf", "EB1_abf"):                # K4 x T12800 and K1 x T51200 frontier cells (E-A3), ABF only
        Kc, ns, s0, sub = (4, 128_000_000, 1_410_000, "EB4_ite300_K4_T12800_abf") if a.block == "EB4_abf" else \
                          (1, 512_000_000, 1_420_000, "EB1_ite300_K1_T51200_abf")
        base = {k: v for k, v in eb_jobs()[0]["cfg"].items() if k != "seed"}
        base.update(N=Kc, n_steps=ns, save_every=10_000)                           # saves every 1 t.u. (E-A3)
        js = [dict(cfg=dict(base, seed=s), method="abf", out=f"{OUT}/{sub}/abf_N{Kc}_s{s}.npz", threads=1)
              for s in range(s0, s0 + 16)]
    elif a.block.startswith("R2"):                         # R2 campaign (docs/r2_campaign/PREREGISTRATION.md): frozen rule R2
        import sys
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from eri.fr_controller import R2
        OUT2 = "results/r2_campaign/raw"
        js = []
        if a.block in ("R2C1", "R2C2", "R2C5", "R2N256"):    # ITE / ERI engine cells
            if a.block == "R2C1":      # E1 configuration: ITE 300 K, K1024, T50, rate 2.0 (in-sample)
                src = json.load(open(os.path.join(ROOT, "configs/convergence_fr/jobs_E1.json")))["jobs"][0]["cfg"]
                seeds, sub = range(1_400_000, 1_400_016), "R2C1_ite300_K1024_R2"
            elif a.block == "R2C2":    # E2 configuration: ITE 150 K, K256, T100, rate 0.5 (in-sample)
                src = json.load(open(os.path.join(ROOT, "configs/convergence_fr/jobs_E2.json")))["jobs"][0]["cfg"]
                seeds, sub = range(1_410_000, 1_410_016), "R2C2_ite150_K256_R2"
            elif a.block == "R2C5":    # ERI K1024 configuration: 600 K, T40, rate 2.0 (in-sample; equivalence test)
                src = json.load(open(os.path.join(ROOT, "configs/convergence_fr/jobs_ERI_rate2_K1024.json")))["jobs"][0]["cfg"]
                seeds, sub = range(1_440_000, 1_440_016), "R2C5_eri600_K1024_R2"
            else:                      # ERI K256 configuration: 600 K, T40, rate 2.0 (held-out allocation; equivalence test)
                src = json.load(open(os.path.join(ROOT, "configs/convergence_fr/jobs_ERI_rate2_K256.json")))["jobs"][0]["cfg"]
                seeds, sub = range(1_470_000, 1_470_016), "R2N256_eri600_K256_R2"
            base = {k: v for k, v in src.items() if k not in ("seed", "fr_stop_steps")}
            for s in seeds:
                js.append(dict(cfg=dict(base, seed=s), method="abf", out=f"{OUT2}/{sub}/abf_N{base['N']}_s{s}.npz", threads=1))
                js.append(dict(cfg=dict(base, seed=s, fr_controller=R2), method="fr", out=f"{OUT2}/{sub}/fr_N{base['N']}_s{s}.npz", threads=1))
        elif a.block in ("R2C3", "R2C4", "R2H80", "R2H350"):   # LTA numba engine, historical histogram production config
            T_K, s0, sub, rate = {"R2C3": (150.0, 1_420_000, "R2C3_lta150_K1024_R2", 0.2),
                                  "R2C4": (300.0, 1_430_000, "R2C4_lta300_K1024_R2", 0.2),
                                  "R2H80": (80.0, 1_450_000, "R2H80_lta80_K1024_R2", 0.1),     # the sweep's 80 K rate
                                  "R2H350": (350.0, 1_460_000, "R2H350_lta350_K1024_R2", 0.2)}[a.block]
            for s in range(s0, s0 + 16):
                for method, over in (("abf", dict(fr_rate=rate)), ("fr", dict(fr_rate=rate, fr_controller=R2))):
                    js.append(dict(T_K=T_K, N=1024, seed=s, cfg=over, method=method, save_every=500,
                                   out=f"{OUT2}/{sub}/{method}_N1024_s{s}.npz"))
        else:
            raise SystemExit(f"unknown R2 block {a.block}")
    else:
        raise SystemExit(f"unknown block {a.block}")
    os.makedirs(CFG, exist_ok=True)
    p = os.path.join(CFG, f"jobs_{a.block}.json")
    json.dump(dict(stage=a.block, jobs=js), open(p, "w"), indent=1)
    print(p, len(js), "jobs")


if __name__ == "__main__":
    main()
