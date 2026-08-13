"""Engine throughput and Triton correctness for the NaCl system (SPEC §9: measure, then report).

Two modes, deliberately separable because they have different hardware requirements:

  --correctness   validates the fused Triton pair kernel on THIS system against the
                  parity-gated tensor path (float32) and float64 ground truth.  Contention
                  does not affect correctness, so this may run on a shared device.

  --timing        ms/step and ns/day against batch size / chunk / kernel.  **Requires a
                  verified-idle device**: a contended GPU reads ~28x slow and is
                  indistinguishable from a code defect (the campaign's recorded trap).  The
                  script refuses to run unless the device is idle, unless --allow-contended.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/nacl_benchmark.py --correctness
    CUDA_VISIBLE_DEVICES=2 python scripts/nacl_benchmark.py --timing --out results/nacl/stage1
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from methane.dynamics import BAOAB, RigidWaterConstraints                    # noqa: E402
from nacl import system as nsys                                             # noqa: E402
from nacl.nonbonded import NaClNonbonded                                    # noqa: E402


def device_is_idle(threshold_mib=500):
    """Idle check for THIS device only.

    The first version summed compute-app memory across the whole node, so any process on any
    GPU -- the methane screen on GPU 3, other groups on 0/1 -- read as contention here and the
    timing path could never run. Scope to the UUID of the device this process is pinned to.
    """
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                               "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip().splitlines()
        uuid = next(u.strip() for r in rows
                    for i, u in [r.split(",", 1)] if i.strip() == visible)
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip()
        me = os.getpid()
        used = [int(m) for line in out.splitlines() if line.strip()
                for g, pid, m in [[x.strip() for x in line.split(",")]]
                if g == uuid and int(pid) != me]
        return (sum(used) < threshold_mib), used
    except Exception as exc:                                  # pragma: no cover
        return False, [str(exc)]


def load_start():
    box = json.load(open(nsys.REPO / "results/nacl/box/box_manifest.json"))
    L = float(box["L_nm"])
    x = dict(np.load(nsys.REPO / "results/nacl/box/npt_final_state.npz"))["positions_nm"]
    return L, x


def correctness(L, x0):
    dev = "cuda"
    x = torch.tensor(np.repeat(x0[None], 4, 0), device=dev, dtype=torch.float32)

    ff64 = NaClNonbonded(L, device=dev, dtype=torch.float64)
    e64, f64 = ff64.pair.energy_forces(x.double())

    ff32 = NaClNonbonded(L, device=dev, dtype=torch.float32)
    e32, f32 = ff32.pair.energy_forces(x)

    fft = NaClNonbonded(L, device=dev, dtype=torch.float32).enable_triton()
    from methane.triton_pair import pair_energy_forces_triton
    et, ft = pair_energy_forces_triton(fft.pair, x, fft._mol_id)

    def rel(a, b):
        return float((a - b).abs().max() / b.abs().max())

    out = dict(
        e_triton_vs_f64=float((et.double() - e64).abs().max() / e64.abs().max()),
        f_triton_vs_f64=rel(ft.double(), f64),
        e_torch32_vs_f64=float((e32.double() - e64).abs().max() / e64.abs().max()),
        f_torch32_vs_f64=rel(f32.double(), f64),
        f_triton_vs_torch32=rel(ft, f32),
        batched_consistency=float((ft[0] - ft[3]).abs().max()),
    )
    # full engine (pair + exclusion + reciprocal + self) through both paths
    e_a, f_a = ff32.energy_forces(x)
    e_b, f_b = fft.energy_forces(x)
    out["engine_e_rel"] = float((e_b - e_a).abs().max() / e_a.abs().max())
    out["engine_f_rel"] = rel(f_b, f_a)
    print(json.dumps(out, indent=2))
    # The criterion is RELATIVE to this system's measured float32 floor, not an absolute
    # transplanted from methane: the first version required f_triton_vs_f64 < 2e-5, but NaCl's
    # gated tensor float32 path itself sits ~3.6e-5 from float64 (every site charged and
    # LJ-active), so the absolute threshold failed BOTH kernels' shared representation, not the
    # Triton kernel. Triton is accepted when its disagreement with the tensor path is well
    # under the float32-vs-float64 floor -- i.e. pure reassociation -- on the same system.
    floor = out["f_torch32_vs_f64"]
    ok = (out["f_triton_vs_torch32"] <= 0.5 * floor
          and out["engine_f_rel"] <= 0.5 * floor
          and out["batched_consistency"] < 1e-3)
    out["float32_floor_vs_f64"] = floor
    out["criterion"] = "triton-vs-tensor and engine-vs-engine <= 0.5 * float32 floor; batch exact"
    print(f"TRITON STATIC: {'PASS' if ok else 'FAIL'} "
          f"(triton-vs-tensor {out['f_triton_vs_torch32']:.2e} vs floor {floor:.2e})")
    return out, ok


def trajectory_gate(L, x0, dt, walkers=8, warm_ps=2.0, measure_ps=8.0):
    """The methane session's standard: a kernel can be right on forces and wrong in dynamics.

    Same starts, same seed streams, tensor vs Triton path, 8 ps of BAOAB: kinetic temperatures
    must agree within 3 sigma (blocked SEMs) and constraint violations must be same-order.
    """
    import torch
    from methane.dynamics import BAOAB, RigidWaterConstraints
    dev = "cuda"
    res = {}
    for use_triton in (False, True):
        ff = NaClNonbonded(L, device=dev, dtype=torch.float32)
        if use_triton:
            ff.enable_triton()
        cons = RigidWaterConstraints(ff.params["waters"], nsys.rigid_water_lengths(),
                                     ff.params["mass"], device=dev, dtype=torch.float32)
        integ = BAOAB(lambda q: ff.energy_forces(q), ff.params["mass"], cons, dt,
                      nsys.TEMPERATURE_K, nsys.GAMMA_PS, device=dev, dtype=torch.float32)
        gen = torch.Generator(device=dev).manual_seed(881)
        x = torch.tensor(np.repeat(x0[None], walkers, 0), device=dev, dtype=torch.float32)
        v = integ.maxwell_velocities(x, generator=gen)
        _, f = ff.energy_forces(x)
        temps, viol = [], 0.0
        n_warm, n_meas = int(warm_ps / dt), int(measure_ps / dt)
        for step_i in range(n_warm + n_meas):
            _, f = integ.step(x, v, f, generator=gen)
            if step_i >= n_warm and step_i % max(1, int(0.1 / dt)) == 0:
                temps.append(float(integ.temperature(v).mean()))
                viol = max(viol, cons.max_violation(x))
        t = np.asarray(temps)
        per = max(2, int(round(2.5 / 0.1)))
        nb = len(t) // per
        sem = (float(np.std(t[:nb * per].reshape(nb, per).mean(1), ddof=1) / np.sqrt(nb))
               if nb >= 3 else float(np.std(t) / np.sqrt(len(t))))
        res["triton" if use_triton else "tensor"] = dict(
            T_mean=float(t.mean()), T_sem=sem, max_violation_nm=viol)
        del ff
        torch.cuda.empty_cache()
    dT = abs(res["triton"]["T_mean"] - res["tensor"]["T_mean"])
    sigma = float(np.hypot(res["triton"]["T_sem"], res["tensor"]["T_sem"]))
    viol_ratio = res["triton"]["max_violation_nm"] / max(res["tensor"]["max_violation_nm"], 1e-12)
    ok = (dT <= 3.0 * sigma) and (0.1 <= viol_ratio <= 10.0)
    res.update(dT_K=dT, sigma_K=sigma, viol_ratio=viol_ratio, PASS=bool(ok))
    print(f"TRITON TRAJECTORY: {'PASS' if ok else 'FAIL'} "
          f"(dT {dT:.2f} +- {sigma:.2f} K = {dT/max(sigma,1e-9):.1f} sigma, "
          f"viol ratio {viol_ratio:.2f})")
    return res, ok


def timing(L, x0, batches, chunks, dt, steps=30):
    dev = "cuda"
    rows = []
    for use_triton in (False, True):
        for B in batches:
            for chunk in (chunks if not use_triton else [0]):
                try:
                    ff = NaClNonbonded(L, device=dev, dtype=torch.float32)
                    if use_triton:
                        ff.enable_triton()
                    ff.pair.energy_forces = torch.compile(ff.pair.energy_forces, dynamic=False)
                    ff.recip.energy = torch.compile(ff.recip.energy, dynamic=False)
                    cons = RigidWaterConstraints(ff.params["waters"],
                                                 nsys.rigid_water_lengths(),
                                                 ff.params["mass"], device=dev,
                                                 dtype=torch.float32)
                    integ = BAOAB(lambda q: ff.energy_forces(q, chunk=max(chunk, 1)),
                                  ff.params["mass"], cons, dt, nsys.TEMPERATURE_K,
                                  nsys.GAMMA_PS, device=dev, dtype=torch.float32)
                    x = torch.tensor(np.repeat(x0[None], B, 0), device=dev, dtype=torch.float32)
                    gen = torch.Generator(device=dev).manual_seed(1)
                    v = integ.maxwell_velocities(x, generator=gen)
                    _, f = ff.energy_forces(x, chunk=max(chunk, 1))
                    for _ in range(5):                      # warm / compile
                        _, f = integ.step(x, v, f, generator=gen)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(steps):
                        _, f = integ.step(x, v, f, generator=gen)
                    torch.cuda.synchronize()
                    el = time.perf_counter() - t0
                    ms = el / steps * 1e3
                    ns_day = B * steps * dt * 1e-3 / el * 86400.0
                    rows.append(dict(kernel="triton" if use_triton else "tensor", B=B,
                                     chunk=chunk, ms_per_step=ms, ns_per_day=ns_day,
                                     peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9))
                    print(f"  {'triton' if use_triton else 'tensor':6s} B={B:5d} "
                          f"chunk={chunk:4d}  {ms:8.2f} ms/step  {ns_day:9.0f} ns/day  "
                          f"{rows[-1]['peak_mem_GB']:5.1f} GB", flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"  {'triton' if use_triton else 'tensor':6s} B={B:5d} "
                          f"chunk={chunk:4d}  OOM", flush=True)
                finally:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correctness", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--allow-contended", action="store_true")
    ap.add_argument("--batches", default="64,256,512,1024,2048")
    ap.add_argument("--chunks", default="64,128,256")
    ap.add_argument("--out", default="results/nacl/stage1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    L, x0 = load_start()
    idle, used = device_is_idle()
    print(f"[device] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')} "
          f"idle={idle} (compute-app memory: {used} MiB)", flush=True)

    report = dict(L_nm=L, gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                  device_idle=idle, device_used_mib=used)
    if args.correctness:
        report["correctness"], ok_static = correctness(L, x0)
        ok_traj, res_traj = True, None
        if ok_static:
            gate_path = nsys.REPO / "results/nacl/stage1/dynamics_gate.json"
            dtq = (json.load(open(gate_path))["dt_chosen_ps"] if gate_path.exists()
                   else nsys.DT_PS)
            res_traj, ok_traj = trajectory_gate(L, x0, dtq)
        report["trajectory_gate"] = res_traj
        report["triton_correctness_pass"] = bool(ok_static and ok_traj)
    if args.timing:
        if not idle and not args.allow_contended:
            raise SystemExit("device is NOT idle; timing numbers here are worthless "
                             "(the 28x trap). Re-run when idle or pass --allow-contended.")
        gate_path = nsys.REPO / "results/nacl/stage1/dynamics_gate.json"
        dt = (json.load(open(gate_path))["dt_chosen_ps"] if gate_path.exists() else nsys.DT_PS)
        report["dt_ps"] = dt
        report["timing"] = timing(L, x0, [int(b) for b in args.batches.split(",")],
                                  [int(c) for c in args.chunks.split(",")], dt)
        best = max(report["timing"], key=lambda r: r["ns_per_day"])
        report["best"] = best
        print(f"\n[best] {best}", flush=True)
    path = os.path.join(args.out, "benchmark.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(report)
    with open(path, "w") as fh:
        json.dump(prev, fh, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
