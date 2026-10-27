"""Job lists of the NSCC A100 study (JSONL, one job per line; run by scripts/nscc_a100/runner.py).

    python scripts/nscc_a100/plan.py launch                       # freeze configs/nscc_a100/gpu_launch.json
    python scripts/nscc_a100/plan.py bench --out-dir results/nscc_a100/jobs
    python scripts/nscc_a100/plan.py law --out-dir results/nscc_a100/jobs
    python scripts/nscc_a100/plan.py prod --out-dir results/nscc_a100/jobs     (needs results/nscc_a100/routing/)
    python scripts/nscc_a100/plan.py confirm --out-dir results/nscc_a100/jobs  (needs results/nscc_a100/selection/)

Each command writes <kind>_gpu.jsonl and <kind>_cpu.jsonl (GPU jobs run in an A100 job, CPU jobs on a CPU node).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nscc_common as NC  # noqa: E402

H = NC.H
ORDER_SEED = 20261026


def seeds_of(rng):
    return list(range(int(rng[0]), int(rng[1]) + 1))


def write(path, jobs):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{path}: {len(jobs)} jobs")


def cmd_launch(a):
    """configs/nscc_a100/gpu_launch.json from the tuning sweeps: per system kind and N, the launch of minimum ABF
    c_ss (the rule of campaign.json launch_tuning)."""
    out = {"_rule": "per N: argmin ABF c_ss over the candidate launches on the 0.25-scaled benchmark window "
                    "(scripts/nscc_a100/bench_gpu.py --sweep); FR uses the same launch",
           "gateway": {}, "lta": {}}
    for system, kind, fn in (("gateway", "gateway", "tune_gateway.jsonl"), ("lta150", "lta", "tune_lta150.jsonl")):
        path = os.path.join(NC.OUT_ROOT, "bench", fn)
        rows = [json.loads(ln) for ln in open(path) if ln.strip()]
        rows = [r for r in rows if "error" not in r and r["method"] == "abf"]
        byN = {}
        for r in rows:
            byN.setdefault(r["N"], []).append(r)
        for N, rs in sorted(byN.items()):
            b = min(rs, key=lambda r: r["c_ss_ns"])
            L = b["launch"]
            e = dict(blocks=L[0], threads=L[1], wpt=L[2], helpers=(L[3] if len(L) > 3 else 0)) \
                if kind == "gateway" else dict(G=L[0], WB=L[1])
            e.update(c_ss_ns=round(b["c_ss_ns"], 3), candidates=len(rs), source=fn)
            out[kind][str(N)] = e
    # rungs between tuned N: nearest tuned N on a log scale (same launch shape scaled to N for the gateway)
    for kind, ladder in (("gateway", [2 ** k for k in range(12)]), ("lta", [2 ** k for k in range(11)])):
        tuned = sorted(int(k) for k in out[kind])
        for N in ladder:
            if str(N) in out[kind]:
                continue
            near = min(tuned, key=lambda t: abs(np.log2(t) - np.log2(N)))
            e = dict(out[kind][str(near)])
            if kind == "gateway":
                B = int(e["blocks"])
                wpb = -(-N // B)
                hp = int(e.get("helpers", 0))
                thr = max(32, -(-(-(-wpb // e["wpt"])) // 32) * 32) * (2 if hp else 1)
                if thr > 1024 // e["wpt"]:
                    B = -(-N // (256 * e["wpt"]))
                    thr = 256 * (2 if hp else 1)
                e.update(blocks=B, threads=thr)
            else:
                G = int(e["G"])
                WB = int(min(e["WB"], max(1, N)))
                WB = max(WB, -(-N // 108))                     # at most 108 co-resident blocks
                while (G * WB) % 32:
                    WB += 1
                if G * WB + 32 > 512:
                    raise SystemExit(f"no valid LTA launch interpolated for N = {N} from N = {near}")
                e.update(G=G, WB=WB)
            e["source"] = f"interpolated from N = {near}"
            e.pop("c_ss_ns", None)
            out[kind][str(N)] = e
    path = os.path.join(NC.ROOT, "configs", "nscc_a100", "gpu_launch.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(path)


def cmd_bench(a):
    cfg = NC.load_config()
    gpu, cpu = [], []
    for system in NC.SYSTEMS:
        spec = H.system_spec(system)
        for r in range(int(cfg["benchmark"]["repeats"])):
            seed = spec["bench_seeds"][r]
            for N in sorted(spec["N_ladder"]):
                for method in H.methods_for(N):
                    gpu.append(NC.make_job("bench", system, method, N, seed, "G1", r=r))
                    cpu.append(NC.make_job("bench", system, method, N, seed, "S1", r=r))
                    for t in NC.CPU_THREADS:
                        if t <= N:
                            cpu.append(NC.make_job("bench", system, method, N, seed, f"P{t}", r=r))
    write(os.path.join(a.out_dir, "bench_gpu.jsonl"), gpu)
    write(os.path.join(a.out_dir, "bench_cpu.jsonl"), cpu)


def cmd_law(a):
    cfg = NC.load_config()
    gpu, cpu = [], []
    for system in NC.SYSTEMS:
        sx = cfg["systems"][system]
        for seed in seeds_of(sx["law_seeds"]):
            for N in sx["law_N"]:
                for method in ("abf", "fr"):
                    gpu.append(NC.make_job("law", system, method, N, seed, "G1"))
                    cpu.append(NC.make_job("law", system, method, N, seed, f"P{min(8, N)}"))
    write(os.path.join(a.out_dir, "law_gpu.jsonl"), gpu)
    write(os.path.join(a.out_dir, "law_cpu.jsonl"), cpu)


def seed_major(jobs_by_seed):
    out = []
    for seed in sorted(jobs_by_seed):
        js = jobs_by_seed[seed]
        perm = np.random.default_rng(ORDER_SEED + seed).permutation(len(js))
        out.extend(js[i] for i in perm)
    return out


def cmd_prod(a):
    cfg = NC.load_config()
    with open(os.path.join(NC.OUT_ROOT, "routing", "routing.json")) as f:
        R = json.load(f)
    gpu, cpu = {}, {}
    for system in NC.SYSTEMS:
        spec = H.system_spec(system)
        for seed in seeds_of(cfg["systems"][system]["prod_seeds"]):
            for N in sorted(spec["N_ladder"]):
                be = R["route"][system][str(N)]["backend"]
                for method in H.methods_for(N):
                    j = NC.make_job("prod", system, method, N, seed, be)
                    (gpu if be == "G1" else cpu).setdefault(seed, []).append(j)
    write(os.path.join(a.out_dir, "prod_gpu.jsonl"), seed_major(gpu))
    write(os.path.join(a.out_dir, "prod_cpu.jsonl"), seed_major(cpu))


def cmd_confirm(a):
    cfg = NC.load_config()
    with open(os.path.join(NC.OUT_ROOT, "selection", "selection.json")) as f:
        S = json.load(f)
    gpu, cpu = {}, {}
    for system in NC.SYSTEMS:
        sel = S["systems"][system]
        for seed in seeds_of(cfg["systems"][system]["confirm_seeds"]):
            for method in ("abf", "fr"):
                arm = sel[method]
                if arm is None:
                    continue
                j = NC.make_job("confirm", system, method, arm["N"], seed, arm["backend"])
                (gpu if arm["backend"] == "G1" else cpu).setdefault(seed, []).append(j)
    write(os.path.join(a.out_dir, "confirm_gpu.jsonl"), seed_major(gpu))
    write(os.path.join(a.out_dir, "confirm_cpu.jsonl"), seed_major(cpu))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("launch", "bench", "law", "prod", "confirm"))
    ap.add_argument("--out-dir", default=os.path.join(NC.OUT_ROOT, "jobs"))
    a = ap.parse_args(argv)
    {"launch": cmd_launch, "bench": cmd_bench, "law": cmd_law, "prod": cmd_prod, "confirm": cmd_confirm}[a.cmd](a)


if __name__ == "__main__":
    main()
