"""Job files for the adaptive-FR campaign (docs/adaptive_fr_campaign/PREREGISTRATION.md).  Every config is a frozen
validated config of an earlier block with only the named fields changed.

    python scripts/adaptive_fr/make_jobs.py A        # ITE 300 K, K256, T200: FR arm (rate 1.0, stop 10) on EB seeds
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
    else:
        raise SystemExit(f"unknown block {a.block}")
    os.makedirs(CFG, exist_ok=True)
    p = os.path.join(CFG, f"jobs_{a.block}.json")
    json.dump(dict(stage=a.block, jobs=js), open(p, "w"), indent=1)
    print(p, len(js), "jobs")


if __name__ == "__main__":
    main()
