"""Run a list of NSCC A100 study jobs inside one PBS allocation, each in a fresh pinned process.

    python scripts/nscc_a100/runner.py --jobs <file.jsonl> --mode gpu
    python scripts/nscc_a100/runner.py --jobs <file.jsonl> --mode cpu [--max-ccds 14]

gpu: jobs run ONE AT A TIME (latency measurement: never two runs on the GPU, and no CPU job of this study on the node
     meanwhile); the child is pinned to two of the job's cores (its host driver + the FR-uniform producer thread).
cpu: CCD-exclusive packing on a whole CPU node: a run with t <= 8 threads holds one CCD (8 physical cores), t = 16 two
     CCDs of one socket; the child is pinned to the first t physical cores of its set (SMT siblings left idle).
Completed jobs (record status 'complete' with a matching signature) are skipped, so a resubmitted job list resumes.
Every finished child prints 'DONE ...' (or 'CHILD_FAILED ...'); the runner ends with 'RUNNER_DONE n_ok n_failed'.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def physical_cores():
    """Allocated physical cores (lowest logical id per core) grouped by CCD (shared L3), from sysfs."""
    cpus = sorted(os.sched_getaffinity(0))
    seen = {}
    for c in cpus:
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
                sib = f.read().strip()
            first = int(sib.replace("-", ",").split(",")[0])
        except OSError:
            first = c
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/cache/index3/shared_cpu_list") as f:
                l3 = f.read().strip()
        except OSError:
            l3 = "?"
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id") as f:
                pkg = int(f.read().strip())
        except OSError:
            pkg = 0
        if first == c:
            seen[c] = (pkg, l3)
    ccds = {}
    for c, key in seen.items():
        ccds.setdefault(key, []).append(c)
    groups = [sorted(v) for _, v in sorted(ccds.items(), key=lambda kv: min(kv[1]))]
    return groups, {g[0]: seen[g[0]][0] for g in groups}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--mode", required=True, choices=("gpu", "cpu"))
    ap.add_argument("--max-ccds", type=int, default=14)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--systems", default=None, help="comma list: only these systems")
    ap.add_argument("--configs", default=None, help="comma list of substrings: only configs containing one")
    ap.add_argument("--shard", default=None, help="k/n: only jobs with seed %% n == k")
    ap.add_argument("--pin-offset", type=int, default=0, help="gpu mode: pin children to physical cores [k, k+2)")
    a = ap.parse_args(argv)
    import nscc_common as NC
    out_root = a.out_root or NC.OUT_ROOT
    with open(a.jobs) as f:
        jobs = [json.loads(ln) for ln in f if ln.strip()]
    if a.systems:
        keep = set(a.systems.split(","))
        jobs = [j for j in jobs if j["system"] in keep]
    if a.configs:
        subs = a.configs.split(",")
        jobs = [j for j in jobs if any(x in j["config"] for x in subs)]
    if a.shard:
        k, n = (int(v) for v in a.shard.split("/"))
        jobs = [j for j in jobs if int(j["seed"]) % n == k]
    specs = {}
    todo = []
    for j in jobs:
        sp = specs.setdefault(j["system"], NC.H.system_spec(j["system"]))
        if NC.record_status(j, out_root, sp) == "complete":
            continue
        todo.append(j)
    print(f"RUNNER {a.mode} jobs {len(jobs)} to run {len(todo)} host {os.uname().nodename}", flush=True)
    child = [sys.executable, os.path.join(HERE, "child.py")]
    n_ok = n_fail = 0
    logdir = os.path.join(out_root, "logs")
    os.makedirs(logdir, exist_ok=True)
    pbs = (os.environ.get("PBS_JOBID") or "local").split(".")[0]
    if a.mode == "gpu":
        groups, _ = physical_cores()
        flat = [c for g in groups for c in g]
        pin = ",".join(str(c) for c in flat[a.pin_offset:a.pin_offset + 2])
        for j in todo:
            if j["impl"] != "gpu":
                print(f"SKIP non-GPU job in gpu mode: {j['config']}", flush=True)
                continue
            cmd = ["taskset", "-c", pin] + child + [json.dumps(j)] + (["--out-root", out_root] if a.out_root else [])
            t = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            ok = r.returncode == 0
            n_ok += ok
            n_fail += (not ok)
            lines = [ln for ln in r.stdout.splitlines() if ln.startswith(("DONE", "CHILD_FAILED"))]
            print((lines[-1] if lines else f"CHILD_FAILED rc {r.returncode} {j['config']} s{j['seed']}") +
                  f"  [{time.time() - t:.1f}s]", flush=True)
            if not ok:
                with open(os.path.join(logdir, f"fail_{pbs}_{j['config']}_s{j['seed']}.log"), "w") as fh:
                    fh.write(r.stdout + "\n" + r.stderr)
    else:
        groups, pkg = physical_cores()
        groups = [g for g in groups if len(g) >= 8]
        free = list(range(min(len(groups), a.max_ccds)))
        print(f"RUNNER cpu: {len(groups)} CCDs of >= 8 cores, using {len(free)}", flush=True)
        # longest first (better packing): cost ~ walker-steps / threads
        todo.sort(key=lambda j: -(j["N"] * j["n_steps"]) / max(1, j["threads"]))
        running = []
        while todo or running:
            launched = True
            while todo and launched:
                launched = False
                for idx, j in enumerate(todo):
                    need = 1 if j["threads"] <= 8 else 2
                    if need == 1 and free:
                        sets = [free.pop(0)]
                    elif need == 2:
                        pair = None
                        for x in free:
                            for y in free:
                                if x < y and pkg[groups[x][0]] == pkg[groups[y][0]]:
                                    pair = (x, y)
                                    break
                            if pair:
                                break
                        if not pair:
                            continue
                        free.remove(pair[0])
                        free.remove(pair[1])
                        sets = list(pair)
                    else:
                        continue
                    cores = [c for s in sets for c in groups[s]][: j["threads"]]
                    cmd = (["taskset", "-c", ",".join(map(str, cores))] + child + [json.dumps(j)] +
                           (["--out-root", out_root] if a.out_root else []))
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    running.append((p, j, sets, time.time()))
                    todo.pop(idx)
                    launched = True
                    break
            time.sleep(0.2)
            still = []
            for p, j, sets, t in running:
                if p.poll() is None:
                    if time.time() - t > a.timeout:
                        p.kill()
                    still.append((p, j, sets, t))
                    continue
                out, err = p.communicate()
                ok = p.returncode == 0
                n_ok += ok
                n_fail += (not ok)
                lines = [ln for ln in out.splitlines() if ln.startswith(("DONE", "CHILD_FAILED"))]
                print((lines[-1] if lines else f"CHILD_FAILED rc {p.returncode} {j['config']} s{j['seed']}") +
                      f"  [{time.time() - t:.1f}s]", flush=True)
                if not ok:
                    with open(os.path.join(logdir, f"fail_{pbs}_{j['config']}_s{j['seed']}.log"), "w") as fh:
                        fh.write(out + "\n" + err)
                free.extend(sets)
                free.sort()
            running = still
    print(f"RUNNER_DONE {n_ok} {n_fail}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
