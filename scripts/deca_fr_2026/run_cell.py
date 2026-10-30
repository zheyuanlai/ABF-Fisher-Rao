"""Run one arm of one cell of the 2026-10 deca-alanine study.

    python scripts/deca_fr_2026/run_cell.py --config configs/deca_fr_2026/<stage>.yaml \
        --cell <name> --method abf|fr_uniform [--prepare-only] [--device cuda]

A cell is ``(K, T, estimator, seeds, FR settings)``; its two arms load the SAME initial
population file (``<out>/<cell>/init.npz``, built once by ``--prepare-only`` or by the first
arm) and the same ``rng_seed``, so they share coordinates, velocities and the dynamics noise
stream until the first birth--death event.  **This script never loads the reference.**
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                                          # noqa: E402
from deca.core import run_sampler_deca                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--method", default=None, choices=list(C.STUDY_METHODS))
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--n-steps", type=int, default=None, help="smoke override")
    ap.add_argument("--out", default=None, help="override the config's out directory")
    ap.add_argument("--force", action="store_true", help="re-run even if a complete result exists")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    cfg = C.load_yaml(args.config)
    cell = cfg["cells"][args.cell]
    out_dir = os.path.join(args.out or cfg["out"], args.cell)
    os.makedirs(out_dir, exist_ok=True)
    over = {}
    if args.n_steps is not None:
        over["n_steps"] = int(args.n_steps)
    # cell-level keys (n_walkers, n_steps, rng_seed) are sim fields too; then the cell's sim: block
    cell_top = {k: cell[k] for k in ("n_walkers", "n_steps", "rng_seed") if k in cell}
    sim = C.make_sim(cfg.get("defaults"), cell_top, cell.get("sim"), over)
    seeds = list(map(int, cell["seeds"]))
    dtype = torch.float64

    init_path = os.path.join(out_dir, "init.npz")
    engine = C.engine_for(args.device, compiled=not args.no_compile, dtype=dtype)
    if not os.path.exists(init_path):
        ip = {**cfg.get("init", {}), **cell.get("init", {})}     # cell-level init overrides
        print(f"[{args.cell}] building the initial population: {len(seeds)} seeds x "
              f"{sim.n_walkers} walkers, init_mode="
              f"{cell.get('init_mode', ip.get('init_mode', 'helix'))}", flush=True)
        init, rep = C.build_initial_population(
            engine, len(seeds), sim.n_walkers, sim.rng_seed, args.device, dtype,
            dt=sim.dt, gamma=sim.gamma, temperature=sim.temperature,
            equil_ps=float(ip.get("equil_ps", 20.0)), jitter_nm=float(ip.get("jitter_nm", 0.006)),
            init_mode=str(cell.get("init_mode", ip.get("init_mode", "helix"))),
            restrain_k=float(ip.get("restrain_k", 0.0)), restrain_R0=ip.get("restrain_R0"))
        # np.savez appends ".npz" unless the name already ends with it -- keep the suffix
        tmp = init_path[:-4] + f".tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, positions=init, seeds=np.asarray(seeds),
                            report=np.array(json.dumps(rep)))
        os.replace(tmp, init_path)
    if args.prepare_only:
        print(f"[{args.cell}] init ready: {init_path}")
        return
    if args.method is None:
        raise SystemExit("--method is required unless --prepare-only")

    z = np.load(init_path, allow_pickle=True)
    init = z["positions"]
    assert init.shape[:2] == (len(seeds), sim.n_walkers), (init.shape, len(seeds), sim.n_walkers)
    init_report = json.loads(str(z["report"]))

    res_npz = os.path.join(out_dir, f"{args.method}.npz")
    res_json = os.path.splitext(res_npz)[0] + ".json"
    if os.path.exists(res_json) and not args.force:
        with open(res_json) as fh:
            if json.load(fh).get("complete"):
                print(f"[{args.cell}/{args.method}] complete result exists; skipping")
                return
    partial = os.path.join(out_dir, f"{args.method}.partial.npz")

    print(f"[{args.cell}/{args.method}] K={sim.n_walkers} T={sim.n_steps*sim.dt/1000:.3f} ns "
          f"seeds={len(seeds)} estimator={sim.abf_estimator} fr_rate={sim.fr_rate} "
          f"controller={'yes' if sim.fr_controller else 'no'} hash={sim.config_hash()}", flush=True)
    out = run_sampler_deca(args.method, engine, sim, seeds, init, device=args.device, dtype=dtype,
                           progress_every=max(50_000, sim.n_steps // 20), partial_path=partial)
    out["provenance"] = C.provenance(sim, engine, args.device, dict(
        stage=cfg.get("stage"), cell=args.cell, method=args.method, init=init_report,
        config_file=os.path.relpath(args.config, C.ROOT)))
    C.save_result(res_npz, out)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"[{args.cell}/{args.method}] DONE {out['runtime_seconds']/60:.1f} min -> {res_npz}",
          flush=True)


if __name__ == "__main__":
    main()
