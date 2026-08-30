#!/usr/bin/env python
"""Throughput / memory benchmark for the flexible ZIF-8 engine.

Chooses the production replica count N, the force-evaluation chunk and the
dtype on COMPUTE AND STABILITY ONLY -- no error metric is computed and the
reference is never loaded (the prereg's rule for this stage).

    CUDA_VISIBLE_DEVICES=3 python -u scripts/zif8_throughput.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from zif8.core_zif8 import ZIF8SimConfig, ZIF8System  # noqa: E402


def bench(system, B, n_steps=25, warmup=6):
    dev, dt = system.device, system.dtype
    q = system.pos0_frame[None].repeat(B, 1, 1).clone()
    if system.with_guest:
        q = torch.cat([q, torch.zeros(B, system.n_guest, 3, device=dev, dtype=dt)], 1)
        com = system.center[None, :] + 3.0 * system.normal[None, :]
        q[:, system.n_frame + 0] = com - 0.77 * system.normal
        q[:, system.n_frame + 1] = com + 0.77 * system.normal
    g = torch.Generator(device=dev).manual_seed(0)
    q = q + 0.05 * torch.randn(q.shape, generator=g, device=dev, dtype=dt)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        system.forces(q)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        system.forces(q)
    torch.cuda.synchronize()
    el = (time.perf_counter() - t0) / n_steps
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    return el, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="128,256,512,1024,2048,4096")
    ap.add_argument("--chunks", default="128,256,512,1024")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results/uniform_campaign/zif8/stage0/throughput.json"))
    a = ap.parse_args()
    assert torch.cuda.is_available(), "this benchmark needs the GPU"
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB")

    rows = []
    for dtype, name in ((torch.float64, "f64"), (torch.float32, "f32")):
        for chunk in [int(x) for x in a.chunks.split(",")]:
            s = ZIF8System(300.0, dev, dtype=dtype, root=ROOT, chunk=chunk)
            for B in [int(x) for x in a.sizes.split(",")]:
                if B < chunk:
                    continue
                try:
                    el, peak = bench(s, B)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  {name} chunk={chunk:5d} B={B:5d}: OOM")
                    continue
                sps = 1.0 / el
                # ns/day of AGGREGATE sampling at dt = 0.5 fs
                ns_day = B * 0.0005 * sps * 86400 / 1000.0
                rows.append(dict(dtype=name, chunk=chunk, B=B, sec_per_step=el,
                                 steps_per_s=sps, peak_gib=peak, ns_per_day=ns_day))
                print(f"  {name} chunk={chunk:5d} B={B:5d}: {1e3*el:8.2f} ms/step  "
                      f"{sps:7.2f} steps/s  peak {peak:6.2f} GiB  "
                      f"{ns_day:8.1f} ns/day aggregate", flush=True)
                torch.cuda.empty_cache()
            del s
            torch.cuda.empty_cache()

    # f32 vs f64 parity on forces (decides whether f32 is admissible at all)
    s64 = ZIF8System(300.0, dev, dtype=torch.float64, root=ROOT, chunk=64, compile=False)
    s32 = ZIF8System(300.0, dev, dtype=torch.float32, root=ROOT, chunk=64, compile=False)
    g = torch.Generator(device=dev).manual_seed(3)
    q = s64.pos0_frame[None].repeat(8, 1, 1).clone()
    q = torch.cat([q, torch.zeros(8, 2, 3, device=dev, dtype=torch.float64)], 1)
    com = s64.center[None, :] + 3.0 * s64.normal[None, :]
    q[:, -2] = com - 0.77 * s64.normal
    q[:, -1] = com + 0.77 * s64.normal
    q = q + 0.08 * torch.randn(q.shape, generator=g, device=dev, dtype=torch.float64)
    F64 = s64.forces(q)
    F32 = s32.forces(q.to(torch.float32)).to(torch.float64)
    rel = ((F64 - F32).norm(dim=-1) / F64.norm(dim=-1).clamp_min(1.0)).max()
    print(f"\nf32 vs f64 force parity: max relative error {float(rel):.2e}")

    best = max((r for r in rows if r["dtype"] == "f64"), key=lambda r: r["ns_per_day"])
    print(f"best f64: chunk={best['chunk']} B={best['B']} "
          f"{best['ns_per_day']:.0f} ns/day aggregate, peak {best['peak_gib']:.1f} GiB")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(rows=rows, f32_force_rel_err=float(rel), best_f64=best,
                       device=torch.cuda.get_device_name(0),
                       n_atoms=s64.n_atoms, rc=s64.rc), fh, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
