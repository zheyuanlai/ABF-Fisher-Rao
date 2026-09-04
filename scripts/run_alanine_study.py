"""Runner for the corrected 2-D alanine ABF vs oracle-mFR study.

GPU policy is enforced here, not merely documented: only one of GPUs 0-3 may be used, exactly one
device must be visible, ``CUDA_VISIBLE_DEVICES`` must be set explicitly to an absolute allowed
index, and free memory must exceed 1.5x the estimated peak.  The original absolute value is
recorded in every artifact.

Usage:
  CUDA_VISIBLE_DEVICES=7 python -u scripts/run_alanine_study.py --config configs/alanine/pilot.yaml
  ... --dry-run --only-method abf --only-seed 0 --overwrite
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alanine.basins import from_reference                                     # noqa: E402
from alanine.core2d_ala import AlaSimConfig, run_sampler_ala                  # noqa: E402
from alanine.cv2d import BackboneCV2D                                         # noqa: E402
from alanine.dynamics import BAOAB, KB, SeedFailure, make_seed_streams        # noqa: E402
from alanine.forcefield import TorchFF, extract_parameters, parameter_hash    # noqa: E402
from alanine.graphed import GraphedCV, GraphedForces                          # noqa: E402
from alanine.system import (PHI_ATOMS, PSI_ATOMS, reference_minimum,          # noqa: E402
                            relax_seeds, seed_umbrella_lattice)

#: The node was re-partitioned on 2026-08-02.  It used to be shared between two groups,
#: which is why only 4 of the 8 devices were ours; the split gave this group its own
#: four, renumbered 0-3.  They are still shared WITHIN the group, so the rule is now
#: "any of 0-3, but EXACTLY ONE at a time" -- and it is the device_count check below,
#: not this set, that actually enforces the "one" half of it.
ALLOWED_GPUS = {"0", "1", "2", "3"}
TWO_PI = 2.0 * math.pi
A_ATOMS = 22                      # Ace-Ala-Nme


# --------------------------------------------------------------------------- safety
def enforce_gpu_policy(est_peak_gib):
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is None:
        raise SystemExit("CUDA_VISIBLE_DEVICES must be set explicitly (absolute index, 0-3)")
    vis = vis.strip()
    if vis not in ALLOWED_GPUS:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={vis!r} is not an allowed absolute index "
                         f"(only {sorted(ALLOWED_GPUS)}); exactly one device may be used")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"expected exactly 1 visible GPU, saw {torch.cuda.device_count()}")
    free, total = torch.cuda.mem_get_info()
    free_gib = free / 2 ** 30
    if free_gib < 1.5 * est_peak_gib:
        raise SystemExit(f"only {free_gib:.1f} GiB free on GPU {vis}; need >= "
                         f"{1.5 * est_peak_gib:.1f} GiB (1.5x estimated peak {est_peak_gib:.1f})")
    return vis, free_gib


def git_provenance():
    def sh(*a):
        try:
            return subprocess.check_output(a, text=True).strip()
        except Exception:                                     # noqa: BLE001
            return ""
    return dict(commit=sh("git", "rev-parse", "HEAD"),
                branch=sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(sh("git", "status", "--porcelain")))


def run_id(spec):
    h = hashlib.md5(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return (f"{spec['stage']}__{spec['method']}__{spec['init']}__N{spec['n_replicas']}"
            f"__T{spec['n_steps']}__ns{len(spec['seeds'])}__{h}")


def save_atomic(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def run_is_valid(path):
    if not os.path.exists(path):
        return False
    try:
        d = np.load(path, allow_pickle=True)
        return "final_pmf" in d.files and bool(np.isfinite(d["final_pmf"]).all())
    except Exception:                                          # noqa: BLE001
        return False


# --------------------------------------------------------------------------- initialisation
def init_c7eq(X0, tff, R, N, equil_ps, dt, gamma, T, device, dtype, seed):
    """Independently thermalised walkers in the dominant C7eq basin.

    Every walker gets its own Maxwell velocities and its own Langevin noise for ``equil_ps``, so
    the ensemble is NOT R*N copies of one microscopic configuration.  Unbiased dynamics keep
    them in C7eq: the C7eq<->C7ax barrier is 15.8 kT, so no crossing is expected on this
    timescale.
    """
    B = R * N
    x = torch.as_tensor(np.repeat(X0[None], B, 0), device=device, dtype=dtype).contiguous()
    integ = BAOAB(tff.masses.cpu().numpy(), dt, gamma, T, lambda z: tff.forces(z),
                  device=device, dtype=dtype)
    g = torch.Generator(device=device).manual_seed(int(seed))
    v = integ.maxwell((B, x.shape[1], 3), g, device, dtype)
    f = tff.forces(x)
    for _ in range(int(equil_ps / dt)):
        x, v, f = integ.step(x, v, f, g)
    return x.reshape(R, N, x.shape[-2], 3)


def init_reference_equilibrium(X0, tff, F_ref, kT, R, N, equil_ps, dt, gamma, T,
                               device, dtype, seed, n_grid=97, relax_steps=300):
    """Crossed control: walkers drawn from the reference Boltzmann distribution on the torus.

    (phi, psi) are sampled from ``exp(-F_ref/kT)`` on the reference grid, realised by rigid
    dihedral rotation of the verified minimum, relieved of steric strain by restrained descent,
    then thermalised independently.  Generated from the reference only -- never from a method.
    """
    rng = np.random.default_rng(seed)
    finite = np.isfinite(F_ref)
    P = np.where(finite, np.exp(-(np.where(finite, F_ref, np.inf)
                                  - np.nanmin(F_ref[finite])) / kT), 0.0)
    P = (P / P.sum()).ravel()
    cells = rng.choice(P.size, size=R * N, p=P)
    dz = TWO_PI / n_grid
    ii, jj = np.unravel_index(cells, (n_grid, n_grid))
    centres = np.stack([-math.pi + (ii + 0.5) * dz, -math.pi + (jj + 0.5) * dz], -1)
    # relax in manageable chunks (restrained descent is CPU-side)
    seeds = seed_umbrella_lattice(X0, centres)
    tcpu = TorchFF(extract_parameters(reference_minimum()[0]), device="cpu", dtype=dtype)
    out = []
    for s in range(0, len(seeds), 2048):
        blk = torch.as_tensor(seeds[s:s + 2048])
        out.append(relax_seeds(tcpu, blk, centres[s:s + 2048], kappa=200.0,
                               n_steps=relax_steps).numpy())
    x = torch.as_tensor(np.concatenate(out), device=device, dtype=dtype).contiguous()
    integ = BAOAB(tff.masses.cpu().numpy(), dt, gamma, T, lambda z: tff.forces(z),
                  device=device, dtype=dtype)
    g = torch.Generator(device=device).manual_seed(int(seed) + 77)
    v = integ.maxwell((x.shape[0], x.shape[1], 3), g, device, dtype)
    f = tff.forces(x)
    for _ in range(int(equil_ps / dt)):
        x, v, f = integ.step(x, v, f, g)
    return x.reshape(R, N, x.shape[-2], 3)


# --------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-method", default=None)
    ap.add_argument("--only-seed", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="CPU-only (tests/smoke; never production)")
    ap.add_argument("--init-cache", default=None,
                    help="path of an .npz holding the initial ensemble: loaded if present, else "
                         "built exactly as without the flag and saved.  Lets several arms in "
                         "SEPARATE processes start from the bitwise-identical ensemble one "
                         "process built (the in-process force path is not deterministic across "
                         "processes), and skips the 20 ps equilibration for every arm but the first.")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="replay the physical force and the CV local mean force through CUDA graphs "
                         "(alanine.graphed): the same eager kernels in the same order, bitwise-"
                         "identical outputs, launch overhead removed (~4x per step at R*N=32768)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    stage_name = a.stage or cfg["stage"]
    st = cfg["stages"][stage_name]
    base = dict(cfg.get("base", {}))
    base.update(st.get("overrides", {}))
    out_root = os.path.join(cfg["output_root"], stage_name)
    os.makedirs(os.path.join(out_root, "raw"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "raw", "_failures"), exist_ok=True)

    dtype = torch.float64
    N = int(base["n_replicas"])
    seeds = [s for s in st["seeds"] if a.only_seed is None or s == a.only_seed]
    R = len(seeds)
    est_peak = 1.35e-3 * R * N          # GiB; measured 9.32 GiB at B=16384 (dense Hessian path)

    if a.cpu:
        device, vis, free_gib = "cpu", "", 0.0
    else:
        vis, free_gib = enforce_gpu_policy(est_peak)
        device = "cuda"

    ref_path = cfg.get("reference", "results/alanine/reference/reference.npz")
    bm, ref_meta = from_reference(ref_path)
    refd = np.load(ref_path, allow_pickle=True)
    F_ref = refd["F"]
    kT = ref_meta["kT_kJ"]

    system, X0 = reference_minimum()
    P = extract_parameters(system)
    phash = parameter_hash(P)
    if phash != ref_meta["param_hash"]:
        raise SystemExit(f"physics hash mismatch: run {phash} vs reference "
                         f"{ref_meta['param_hash']} -- the arms and the reference must share "
                         "force field, masses and parameters exactly")
    tff = TorchFF(P, device=device, dtype=dtype)
    cv = BackboneCV2D(PHI_ATOMS, PSI_ATOMS, n_atoms=22)
    labels = bm.label_tensor(device=device)

    sim_keys = {f.name for f in AlaSimConfig.__dataclass_fields__.values()}
    sim = AlaSimConfig(**{k: v for k, v in base.items() if k in sim_keys})

    methods = [m for m in st["methods"] if a.only_method is None or m == a.only_method]
    init_mode = st.get("init", "c7eq")
    prov = git_provenance()

    print(f"stage={stage_name} methods={methods} init={init_mode} N={N} seeds={seeds} "
          f"steps={sim.n_steps} device={device} CUDA_VISIBLE_DEVICES={vis!r} "
          f"free={free_gib:.1f} GiB est_peak={est_peak:.1f} GiB", flush=True)
    print(f"  config_hash={sim.config_hash()} param_hash={phash} "
          f"git={prov['commit'][:8]}{'+dirty' if prov['dirty'] else ''}", flush=True)
    if a.dry_run:
        for m in methods:
            spec = dict(stage=stage_name, method=m, init=init_mode, n_replicas=N,
                        n_steps=sim.n_steps, seeds=seeds, cfg=sim.config_hash())
            print(f"  would run -> {run_id(spec)}.npz")
        return

    # identical initial ensemble for every arm within a stage
    t_init = time.perf_counter()
    init = None
    if a.init_cache and os.path.exists(a.init_cache):
        cached = np.load(a.init_cache, allow_pickle=True)
        cmeta = json.loads(str(cached["meta"]))
        want = dict(init=init_mode, init_seed=int(st.get("init_seed", 4242)), R=R, N=N,
                    seeds=[int(x) for x in seeds], init_equil_ps=float(base.get("init_equil_ps", 20.0)),
                    dt=sim.dt, gamma=sim.gamma, temperature=sim.temperature, param_hash=phash)
        got = {k: cmeta.get(k) for k in want}
        if got != want:
            raise SystemExit(f"init cache {a.init_cache} was built for {got}, this stage needs {want}")
        init = torch.as_tensor(cached["init"], device=device, dtype=dtype)
        print(f"  init[{init_mode}] loaded from {a.init_cache} (built {cmeta.get('built')}, "
              f"git {cmeta.get('git', {}).get('commit', '')[:8]})", flush=True)
    if init is not None:
        pass
    elif init_mode == "c7eq":
        init = init_c7eq(X0, tff, R, N, base.get("init_equil_ps", 20.0), sim.dt, sim.gamma,
                         sim.temperature, device, dtype, seed=st.get("init_seed", 4242))
    elif init_mode == "reference_equilibrium":
        init = init_reference_equilibrium(X0, tff, F_ref, kT, R, N,
                                          base.get("init_equil_ps", 20.0), sim.dt, sim.gamma,
                                          sim.temperature, device, dtype,
                                          seed=st.get("init_seed", 4242))
    else:
        raise SystemExit(f"unknown init {init_mode!r}")
    print(f"  init[{init_mode}] built in {time.perf_counter()-t_init:.0f}s", flush=True)
    if a.init_cache and not os.path.exists(a.init_cache):
        os.makedirs(os.path.dirname(os.path.abspath(a.init_cache)), exist_ok=True)
        save_atomic(a.init_cache, init=init.detach().cpu().numpy(), meta=json.dumps(dict(
            init=init_mode, init_seed=int(st.get("init_seed", 4242)), R=R, N=N,
            seeds=[int(x) for x in seeds], init_equil_ps=float(base.get("init_equil_ps", 20.0)),
            dt=sim.dt, gamma=sim.gamma, temperature=sim.temperature, param_hash=phash,
            built=time.strftime("%Y-%m-%dT%H:%M:%S"), git=prov, cuda_visible_devices=vis)))
        print(f"  init[{init_mode}] saved -> {a.init_cache}", flush=True)

    # Optional CUDA-graph replay of the two launch-bound kernels of the hot loop.  Same
    # kernels, same order, bitwise-identical outputs (tests/test_alanine_graphed.py); the
    # noise draws stay eager so the dynamical RNG stream is untouched.
    force_fn, cv_run = None, cv
    if a.cuda_graph:
        if a.cpu:
            raise SystemExit("--cuda-graph needs a CUDA device")
        t_cap = time.perf_counter()
        force_fn = GraphedForces(tff, batch=R * N, device=device, dtype=dtype)
        cv_run = GraphedCV(cv, batch=R * N, beta=1.0 / (KB * sim.temperature), n_atoms=A_ATOMS,
                           device=device, dtype=dtype)
        print(f"  cuda graphs captured in {time.perf_counter()-t_cap:.1f}s", flush=True)

    manifest = []
    for m in methods:
        spec = dict(stage=stage_name, method=m, init=init_mode, n_replicas=N,
                    n_steps=sim.n_steps, seeds=seeds, cfg=sim.config_hash())
        rid = run_id(spec)
        path = os.path.join(out_root, "raw", rid + ".npz")
        if run_is_valid(path) and not a.overwrite:
            print(f"  skip (valid) {rid}", flush=True)
            manifest.append(dict(spec, run_id=rid, status="skipped", path=path))
            continue
        try:
            t0 = time.perf_counter()
            out = run_sampler_ala(m, tff, cv_run, sim, seeds, init, labels, device, dtype=dtype,
                                  reference_F=(F_ref if m == "fr_oracle" else None),
                                  dump_dir=os.path.join(out_root, "raw", "_failures"),
                                  force_fn=force_fn)
            payload = {k: v for k, v in out.items() if isinstance(v, (np.ndarray, np.generic))}
            payload["meta"] = json.dumps(dict(
                spec, run_id=rid, param_hash=phash, config_hash=sim.config_hash(),
                reference=ref_path, reference_param_hash=ref_meta["param_hash"],
                cuda_visible_devices=vis, device=device, dtype="float64",
                basin_names=bm.names, basin_centres_deg=bm.centres_deg,
                git=prov, wall_seconds=out["wall_seconds"], ms_per_step=out["ms_per_step"],
                peak_cuda_gib=out["peak_cuda_gib"], clip_fraction=out["clip_fraction"],
                force_evaluations=out["force_evaluations"],
                aggregate_simulated_ps=out["aggregate_simulated_ps"],
                init_equil_ps=base.get("init_equil_ps", 20.0),
                init_cache=a.init_cache, cuda_graph=bool(a.cuda_graph),
                fr_start_steps=sim.fr_start_steps, fr_every=sim.fr_every,
                fr_rate=sim.fr_rate), default=float)
            save_atomic(path, **payload)
            manifest.append(dict(spec, run_id=rid, status="ok", path=path,
                                 wall_seconds=time.perf_counter() - t0,
                                 ms_per_step=out["ms_per_step"],
                                 clip_fraction=out["clip_fraction"]))
        except SeedFailure as e:
            fp = os.path.join(out_root, "raw", "_failures", rid + ".json")
            json.dump(dict(spec, run_id=rid, error=str(e), seed_index=e.seed_index,
                           step=e.step, dump=e.dump_path), open(fp, "w"), indent=2)
            print(f"  FAILED {rid}: {e}", flush=True)
            manifest.append(dict(spec, run_id=rid, status="failed", error=str(e)))
        except Exception as e:                                  # noqa: BLE001
            fp = os.path.join(out_root, "raw", "_failures", rid + ".json")
            json.dump(dict(spec, run_id=rid, error=f"{type(e).__name__}: {e}"),
                      open(fp, "w"), indent=2)
            print(f"  ERROR {rid}: {type(e).__name__}: {e}", flush=True)
            manifest.append(dict(spec, run_id=rid, status="error", error=str(e)))

    mp = os.path.join(out_root, "run_manifest.json")
    json.dump(manifest, open(mp, "w"), indent=2, default=str)
    print(f"wrote {mp}", flush=True)


if __name__ == "__main__":
    main()
