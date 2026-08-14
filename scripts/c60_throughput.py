"""Idle-device throughput for the batched C60 torch engine -- SPEC §11, measured not assumed.

Measures ms/step for the fixed-cage (reference-window) configuration across batch sizes and
chunk sizes, float32 with and without torch.compile, on the pinned GPU.  Writes
results/c60/parity/throughput.json.  The reference design consumes these numbers; if the
frozen budget is infeasible at the measured cost, that is an amendment, not a quiet reduction.

Usage:  CUDA_VISIBLE_DEVICES=3 python scripts/c60_throughput.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from c60 import system as csys  # noqa: E402
from c60.dynamics import C60Dynamics  # noqa: E402
from c60.nonbonded import C60Nonbonded  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "c60", "parity")


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if dev != "3":
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={dev!r}; SPEC §11 pins this study to GPU 3")
    assert torch.cuda.device_count() == 1, "exactly one visible device (GPU policy)"

    fz = np.load(os.path.join(OUT, "..", "box", "frozen_box.npz"))
    lx, lz = float(fz["lx_nm"]), float(fz["lz_nm"])
    base = np.asarray(fz["positions"], dtype=np.float64)

    mod = csys.build_modeller()
    import openmm as mm
    import openmm.unit as u
    box = [mm.Vec3(lx, 0, 0), mm.Vec3(0, lx, 0), mm.Vec3(0, 0, lz)] * u.nanometer
    system = csys.build_system(mod.topology, box_vectors=box, pme_params=csys.pme_params())
    alpha, nx, ny, nz = csys.pme_params()

    results = {}
    dt_ps = csys.DT_PS
    combos = [(544, 256, False)]                       # one eager baseline
    combos += [(B, c, True) for B in (272, 544, 816) for c in (128, 256)]
    for B, chunk, compiled in combos:
        if True:
                torch.cuda.empty_cache()
                eng = C60Nonbonded(system, mod.topology, (lx, lx, lz), alpha,
                                   (nx, ny, nz), device="cuda", dtype=torch.float32)

                ef = eng.energy_forces
                if compiled:
                    try:
                        ef = torch.compile(eng.energy_forces, dynamic=False)
                    except Exception as err:
                        results[f"B{B}_c{chunk}_compiled"] = f"compile failed: {err}"
                        continue

                def force(q, _ef=ef, _chunk=chunk):
                    e, f_raw = _ef(q, chunk=_chunk)
                    return e, f_raw

                dyn = C60Dynamics(eng, dt_ps, device="cuda", dtype=torch.float32,
                                  force_fn=force)
                x = torch.as_tensor(base, device="cuda", dtype=torch.float32)[None] \
                    .repeat(B, 1, 1).contiguous()
                eng.compute_vsites(x)
                gen = torch.Generator(device="cuda").manual_seed(1)
                v = dyn.maxwell_velocities(x, generator=gen)

                try:
                    _, f_raw = force(x)
                    f = eng.redistribute(f_raw)
                    # warm-up (compile happens here)
                    for _ in range(3):
                        e, f_raw = force(x)
                        f = eng.redistribute(f_raw)
                        _ = dyn.step(x, v, f)
                    torch.cuda.synchronize()
                    n_steps = 20
                    t0 = time.perf_counter()
                    for _ in range(n_steps):
                        e, f = dyn.step(x, v, f)
                    torch.cuda.synchronize()
                    el = (time.perf_counter() - t0) / n_steps
                    agg_ns_day = B * dt_ps * 1e-3 * 86400.0 / el
                    key = f"B{B}_c{chunk}_" + ("compiled" if compiled else "eager")
                    results[key] = dict(ms_per_step=el * 1e3,
                                        aggregate_ns_per_day=agg_ns_day,
                                        mem_gb=torch.cuda.max_memory_allocated() / 2**30)
                    print(f"{key:24s} {el*1e3:8.1f} ms/step  {agg_ns_day:9.1f} ns/day agg  "
                          f"mem {results[key]['mem_gb']:.1f} GB", flush=True)
                except torch.cuda.OutOfMemoryError:
                    results[f"B{B}_c{chunk}"] = "OOM"
                    print(f"B{B}_c{chunk}: OOM", flush=True)
                del eng, dyn, x
                torch.cuda.reset_peak_memory_stats()

    with open(os.path.join(OUT, "throughput.json"), "w") as fh:
        json.dump(dict(results=results, dt_ps=dt_ps, device=dev,
                       torch=torch.__version__), fh, indent=1)


if __name__ == "__main__":
    main()
