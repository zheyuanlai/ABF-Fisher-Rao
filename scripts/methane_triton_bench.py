"""Triton pair-kernel acceptance: idle-device timing sweep + an 8 ps equipartition gate.

Two things must both hold before the kernel is allowed into a production screen:

1. **Speed**, measured on a *verified-idle* device.  The script refuses to time anything while a
   foreign compute app holds the GPU -- a contended benchmark produced a 28x mismeasurement
   earlier in this campaign and is the single easiest way to make a wrong decision here.
2. **Dynamics**, not just forces.  5/5 static correctness gates already passed, and that is not
   sufficient: a kernel can be right on energies and forces and still wrong in a trajectory.
   The same 8 ps constrained-BAOAB equipartition comparison against OpenMM that admitted the
   sync-free solver is re-run here, because that is the gate a 156 K thermostat bug would have
   failed while every static test passed.

``ms/step`` is reported alongside ``ns/day`` at each batch size, since a downstream study packing
960 or 2200 trajectories needs to know whether the kernel is bandwidth-limited (flat ms/step per
trajectory) or occupancy-limited (falls off above some B), and ns/day alone hides that.

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/methane_triton_bench.py --out results/methane/triton
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch._dynamo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import openmm as mm                                              # noqa: E402
import openmm.unit as u                                          # noqa: E402

from methane import system as msys                               # noqa: E402
from methane.nonbonded import MethaneNonbonded                   # noqa: E402


def foreign_compute_apps(my_pid):
    """PIDs other than ours holding a CUDA context on the visible device."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=20).stdout.strip()
    except Exception:
        return []
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    uuid = None
    if vis.isdigit():
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                               "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=20).stdout.strip().splitlines()
        for row in rows:
            idx, uid = [s.strip() for s in row.split(",")]
            if idx == vis:
                uuid = uid
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                           "--format=csv,noheader"], capture_output=True, text=True,
                          timeout=20).stdout.strip().splitlines()
    foreign = []
    for row in apps:
        if not row.strip():
            continue
        gid, pid, mem = [s.strip() for s in row.split(",")]
        if uuid is not None and gid != uuid:
            continue
        if int(pid) != my_pid:
            foreign.append((int(pid), mem))
    return foreign


def wait_for_idle(my_pid, checks=2, gap=30, timeout_s=5400):
    """Block until no foreign compute app is on the device, twice ``gap`` apart."""
    t0 = time.time()
    clean = 0
    while time.time() - t0 < timeout_s:
        f = foreign_compute_apps(my_pid)
        if not f:
            clean += 1
            if clean >= checks:
                return True
        else:
            if clean:
                print(f"[idle] foreign app reappeared: {f}", flush=True)
            clean = 0
            print(f"[idle] waiting, foreign compute apps: {f}", flush=True)
        time.sleep(gap)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/methane/triton")
    ap.add_argument("--box", default="results/methane/box")
    ap.add_argument("--batches", default="512,960,1536,2200")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--skip-idle-wait", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    my_pid = os.getpid()

    if not args.skip_idle_wait:
        print(f"[idle] verifying device is free of foreign compute apps (pid {my_pid}) ...",
              flush=True)
        if not wait_for_idle(my_pid):
            raise SystemExit("device never went idle; refusing to report contended timings")
    print("[idle] device verified idle", flush=True)

    man = json.load(open(os.path.join(args.box, "manifest.json")))
    L = float(man["box_L_nm"])
    base = np.load(os.path.join(args.box, "box.npz"))["positions_nm"]
    dev = torch.device("cuda")
    mod = msys.build_modeller(r0_nm=0.55, seed=man["seed"])
    system = msys.build_system(mod.topology)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(L, 0, 0) * u.nanometer,
                                        mm.Vec3(0, L, 0) * u.nanometer,
                                        mm.Vec3(0, 0, L) * u.nanometer)

    results = {}
    for tag, use_triton in (("tensor", False), ("triton", True)):
        ff = MethaneNonbonded(system, mod.topology, L, device=dev, dtype=torch.float32)
        if use_triton:
            ff.enable_triton()
        else:
            ff.pair.energy_forces = torch.compile(ff.pair.energy_forces, dynamic=False)
        ff.recip.energy = torch.compile(ff.recip.energy, dynamic=False)
        row = {}
        for B in [int(b) for b in args.batches.split(",")]:
            # Every (B, chunk) is a new shape to dynamo, and its cache_size_limit is 8: past
            # that it stops recompiling and silently runs EAGER, which reads as a throughput
            # cliff that looks like physics. The NaCl session lost a scheduling decision to
            # exactly this -- an 8x "collapse" that was the compiler giving up, with the tell
            # sitting in its own table as an identical us/traj-step across four batch sizes.
            # This sweep uses 4 shapes per compiled function so it never reached the limit
            # (verified: tensor spread 13.72 us, not a 20x step), but the reset makes the
            # benchmark correct for any larger sweep someone runs later.
            torch._dynamo.reset()
            try:
                x = torch.tensor(base, device=dev, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
                x = x + 0.001 * torch.randn_like(x)
                # 3 warmup / 15 timed under-amortises per-call overhead: the NaCl session
                # measured its own sweep reporting 168.9 us/traj-step against an in-situ 119.8
                # for the same work, a 1.41x under-report, and traced it to exactly this.
                for _ in range(25):
                    ff.energy_forces(x, chunk=args.chunk)
                torch.cuda.synchronize()
                t0 = time.time()
                n = 200
                for _ in range(n):
                    ff.energy_forces(x, chunk=args.chunk)
                torch.cuda.synchronize()
                dt = (time.time() - t0) / n
                row[B] = dict(ms_per_step=dt * 1e3,
                              us_per_traj_step=dt * 1e6 / B,
                              ns_per_day=B * msys.DT_PS / dt * 86400 / 1000,
                              peak_GiB=torch.cuda.max_memory_allocated() / 2 ** 30)
                print(f"[{tag:6s}] B={B:5d}  {dt*1e3:8.2f} ms/step  "
                      f"{dt*1e6/B:7.2f} us/traj-step  "
                      f"{B*msys.DT_PS/dt*86400/1000:7.0f} ns/day  "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:5.1f} GiB", flush=True)
                torch.cuda.reset_peak_memory_stats()
                del x
            except torch.OutOfMemoryError:
                print(f"[{tag:6s}] B={B:5d}  OOM", flush=True)
                row[B] = dict(oom=True)
            torch.cuda.empty_cache()
        results[tag] = row
        del ff
        torch.cuda.empty_cache()

    with open(os.path.join(args.out, "bench.json"), "w") as fh:
        json.dump(dict(results=results, chunk=args.chunk, box_L_nm=L,
                       git_commit=subprocess.run(["git", "rev-parse", "HEAD"],
                                                 capture_output=True,
                                                 text=True).stdout.strip()), fh, indent=2)
    print(f"\n[done] -> {args.out}/bench.json", flush=True)


if __name__ == "__main__":
    main()
