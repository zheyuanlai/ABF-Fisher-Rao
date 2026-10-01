#!/usr/bin/env python
"""Runner for the 1-D (incomplete-CV) alanine study: ABF vs ABF + uniform FR on phi OR psi alone.

Same GPU policy, init cache, CUDA-graph replay and artifact layout as scripts/run_alanine_study.py.
  CUDA_VISIBLE_DEVICES=3 python -u scripts/run_alanine_1d_study.py --config configs/alanine_1d/alanine_1d.yaml \
      --stage <stage> --init-cache <npz> --cuda-graph
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_alanine_study import (A_ATOMS, enforce_gpu_policy, git_provenance, init_c7eq,   # noqa: E402
                               run_id, run_is_valid, save_atomic)
from alanine.basins import from_reference                                                # noqa: E402
from alanine.core1d_ala import Ala1DSimConfig, BackboneCV1D, run_sampler_ala1d            # noqa: E402
from alanine.dynamics import KB, SeedFailure                                             # noqa: E402
from alanine.forcefield import TorchFF, extract_parameters, parameter_hash               # noqa: E402
from alanine.graphed import GraphedCV, GraphedForces                                     # noqa: E402
from alanine.system import reference_minimum                                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--init-cache", default=None)
    ap.add_argument("--cuda-graph", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    stage_name = a.stage or cfg["stage"]
    st = cfg["stages"][stage_name]
    base = dict(cfg.get("base", {})); base.update(st.get("overrides", {}))
    out_root = os.path.join(cfg["output_root"], stage_name)
    os.makedirs(os.path.join(out_root, "raw", "_failures"), exist_ok=True)
    dtype = torch.float64
    N = int(base["n_replicas"]); seeds = list(st["seeds"]); R = len(seeds)
    est_peak = 1.35e-3 * R * N
    if a.cpu:
        device, vis, free_gib = "cpu", "", 0.0
    else:
        vis, free_gib = enforce_gpu_policy(est_peak); device = "cuda"
    ref_path = cfg.get("reference", "results/alanine/reference/reference.npz")
    bm, ref_meta = from_reference(ref_path)
    system, X0 = reference_minimum()
    P = extract_parameters(system); phash = parameter_hash(P)
    if phash != ref_meta["param_hash"]:
        raise SystemExit(f"physics hash mismatch: run {phash} vs reference {ref_meta['param_hash']}")
    tff = TorchFF(P, device=device, dtype=dtype)
    labels = bm.label_tensor(device=device)
    keys = {f.name for f in Ala1DSimConfig.__dataclass_fields__.values()}
    sim = Ala1DSimConfig(**{k: v for k, v in base.items() if k in keys})
    cv = BackboneCV1D(sim.cv)
    methods = list(st["methods"]); init_mode = st.get("init", "c7eq"); prov = git_provenance()
    print(f"stage={stage_name} cv={sim.cv} methods={methods} init={init_mode} N={N} seeds={seeds} steps={sim.n_steps} "
          f"device={device} CUDA_VISIBLE_DEVICES={vis!r} free={free_gib:.1f} GiB", flush=True)
    print(f"  config_hash={sim.config_hash()} param_hash={phash} git={prov['commit'][:8]}{'+dirty' if prov['dirty'] else ''}", flush=True)
    if a.dry_run:
        for m in methods:
            spec = dict(stage=stage_name, method=m, init=init_mode, n_replicas=N, n_steps=sim.n_steps, seeds=seeds, cfg=sim.config_hash())
            print(f"  would run -> {run_id(spec)}.npz")
        return
    init = None
    if a.init_cache and os.path.exists(a.init_cache):
        cached = np.load(a.init_cache, allow_pickle=True); cmeta = json.loads(str(cached["meta"]))
        want = dict(init=init_mode, init_seed=int(st.get("init_seed", 4242)), R=R, N=N, seeds=[int(x) for x in seeds],
                    init_equil_ps=float(base.get("init_equil_ps", 20.0)), dt=sim.dt, gamma=sim.gamma,
                    temperature=sim.temperature, param_hash=phash)
        got = {k: cmeta.get(k) for k in want}
        if got != want:
            raise SystemExit(f"init cache {a.init_cache} was built for {got}, this stage needs {want}")
        init = torch.as_tensor(cached["init"], device=device, dtype=dtype)
        print(f"  init[{init_mode}] loaded from {a.init_cache}", flush=True)
    if init is None:
        if init_mode != "c7eq":
            raise SystemExit("only the c7eq init is supported here (use the cache for anything else)")
        init = init_c7eq(X0, tff, R, N, base.get("init_equil_ps", 20.0), sim.dt, sim.gamma, sim.temperature,
                         device, dtype, seed=st.get("init_seed", 4242))
    force_fn, cv_run = None, cv
    if a.cuda_graph:
        if a.cpu:
            raise SystemExit("--cuda-graph needs a CUDA device")
        t_cap = time.perf_counter()
        force_fn = GraphedForces(tff, batch=R * N, device=device, dtype=dtype)
        cv_run = GraphedCV(cv, batch=R * N, beta=1.0 / (KB * sim.temperature), n_atoms=A_ATOMS, device=device, dtype=dtype)
        print(f"  cuda graphs captured in {time.perf_counter() - t_cap:.1f}s", flush=True)
    manifest = []
    for m in methods:
        spec = dict(stage=stage_name, method=m, init=init_mode, n_replicas=N, n_steps=sim.n_steps, seeds=seeds, cfg=sim.config_hash())
        rid = run_id(spec); path = os.path.join(out_root, "raw", rid + ".npz")
        if run_is_valid(path) and not a.overwrite:
            print(f"  skip (valid) {rid}", flush=True); manifest.append(dict(spec, run_id=rid, status="skipped", path=path)); continue
        try:
            t0 = time.perf_counter()
            out = run_sampler_ala1d(m, tff, cv_run, sim, seeds, init, labels, device, dtype=dtype,
                                    dump_dir=os.path.join(out_root, "raw", "_failures"), force_fn=force_fn)
            payload = {k: v for k, v in out.items() if isinstance(v, (np.ndarray, np.generic))}
            payload["meta"] = json.dumps(dict(spec, run_id=rid, param_hash=phash, config_hash=sim.config_hash(), reference=ref_path,
                                              cuda_visible_devices=vis, device=device, dtype="float64", basin_names=bm.names,
                                              basin_centres_deg=bm.centres_deg, git=prov, wall_seconds=out["wall_seconds"],
                                              ms_per_step=out["ms_per_step"], peak_cuda_gib=out["peak_cuda_gib"],
                                              clip_fraction=out["clip_fraction"], force_evaluations=out["force_evaluations"],
                                              aggregate_simulated_ps=out["aggregate_simulated_ps"], init_cache=a.init_cache,
                                              cuda_graph=bool(a.cuda_graph), **{k: getattr(sim, k) for k in keys}), default=float)
            save_atomic(path, **payload)
            manifest.append(dict(spec, run_id=rid, status="ok", path=path, wall_seconds=time.perf_counter() - t0, ms_per_step=out["ms_per_step"]))
        except SeedFailure as e:
            json.dump(dict(spec, run_id=rid, error=str(e), seed_index=e.seed_index, step=e.step, dump=e.dump_path),
                      open(os.path.join(out_root, "raw", "_failures", rid + ".json"), "w"), indent=2)
            print(f"  FAILED {rid}: {e}", flush=True); manifest.append(dict(spec, run_id=rid, status="failed", error=str(e)))
        except Exception as e:                                  # noqa: BLE001
            json.dump(dict(spec, run_id=rid, error=f"{type(e).__name__}: {e}"), open(os.path.join(out_root, "raw", "_failures", rid + ".json"), "w"), indent=2)
            print(f"  ERROR {rid}: {type(e).__name__}: {e}", flush=True); manifest.append(dict(spec, run_id=rid, status="error", error=str(e)))
    mp = os.path.join(out_root, "run_manifest.json")
    json.dump(manifest, open(mp, "w"), indent=2, default=str); print(f"wrote {mp}", flush=True)


if __name__ == "__main__":
    main()
