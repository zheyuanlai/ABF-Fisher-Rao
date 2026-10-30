"""Shared helpers of the 2026-10 deca-alanine ABF vs ABF+FR study (docs/DECA_FR_2026_PROTOCOL.md).

Everything here is deliberately boring: configuration loading, the frozen initial-population
protocol (Amendment 4 of the 2026-08 study: jittered alpha-helix, relaxed, 20 ps unbiased
equilibration outside the ABF budget), provenance and a save helper.  No analysis, no reference.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from alanine.dynamics import BAOAB                                          # noqa: E402
from deca import system as dsys                                             # noqa: E402
from deca.core import DecaSimConfig                                         # noqa: E402
from deca.engine import make_engine                                         # noqa: E402
from deca.umbrella import relax_pool                                        # noqa: E402
from alkanes.distance_cv import DistanceCV, dist_bias_force                 # noqa: E402

KB = 0.008314462618
STUDY_METHODS = ("abf", "fr_uniform")


def load_yaml(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:                                                        # noqa: BLE001
        return "unknown"


def make_sim(*layers):
    """``DecaSimConfig`` from a stack of override dicts (later wins).  Lists become tuples."""
    merged = {}
    for d in layers:
        if d:
            merged.update(d)
    for k in ("fr_window", "fr_support_interval"):
        if k in merged and isinstance(merged[k], list):
            merged[k] = tuple(merged[k])
    sim = DecaSimConfig(**merged)
    sim.validate()
    return sim


def engine_for(device, compiled=True, dtype=torch.float64):
    """The ff14SB torch engine; ``torch.compile`` only on CUDA (the CPU path is for smoke tests)."""
    engine, system, top = make_engine(10, device=device, dtype=dtype,
                                      compiled=bool(compiled and device == "cuda"))
    return engine


INIT_MODES = ("helix", "ppii", "mixed")


def build_initial_population(engine, n_seeds, n_walkers, rng_seed, device, dtype=torch.float64,
                             dt=0.001, gamma=1.0, temperature=300.0, equil_ps=20.0,
                             jitter_nm=0.006, verbose=True, init_mode="helix",
                             restrain_k=0.0, restrain_R0=None):
    """Amendment 4 protocol: helix start, jittered per walker, relaxed, 20 ps unbiased BAOAB.

    ``init_mode`` (post-protocol diagnostics only): ``helix`` (protocol), ``ppii`` (every walker
    starts as polyproline II, phi/psi = -75/145, i.e. from the extended side) or ``mixed`` (the
    first half of each seed's walkers helical, the second half PPII).  ``restrain_k`` > 0
    (kJ/mol/nm^2) equilibrates under a harmonic restraint on R towards ``restrain_R0`` (a number)
    or, if that is None, towards each walker's own R after relaxation; without it an extended
    start collapses to R ~ 2.1 nm within the 20 ps (unbiased, no walls).
    The equilibration accumulates nothing.  Returns ``(positions (R, N, A, 3) float64, report)``.
    Deterministic in ``rng_seed`` (numpy jitter, torch Maxwell + noise), so both arms of a cell
    load the same file and the file can be rebuilt bit for bit.
    """
    if init_mode not in INIT_MODES:
        raise ValueError(f"init_mode must be one of {INIT_MODES}, got {init_mode!r}")
    B = n_seeds * n_walkers
    rng = np.random.default_rng(rng_seed)
    x_helix = dsys.build_helix(-57.0, -47.0, n_res=dsys.N_RES)
    x_ppii = dsys.build_helix(-75.0, 145.0, n_res=dsys.N_RES)
    if init_mode == "helix":
        starts = [x_helix] * B
    elif init_mode == "ppii":
        starts = [x_ppii] * B
    else:
        half = n_walkers // 2
        starts = [x_helix if (i % n_walkers) < half else x_ppii for i in range(B)]
    X0 = np.stack([x + jitter_nm * rng.standard_normal(x.shape) for x in starts])
    q = torch.as_tensor(X0, device=device, dtype=dtype).contiguous()

    q, ok, rep = relax_pool(engine, q)
    n_bad = int((~ok).sum())
    if n_bad:
        good = np.flatnonzero(ok)
        q[torch.as_tensor(np.flatnonzero(~ok), device=device)] = \
            q[torch.as_tensor(good[rng.integers(0, good.size, n_bad)], device=device)].clone()
    if verbose:
        print(f"  seed relaxation: {B - n_bad}/{B} pass; {n_bad} replaced", flush=True)

    gen = torch.Generator(device=device).manual_seed(int(rng_seed) + 555)
    integ = BAOAB(engine.masses, dt=dt, gamma=gamma, temperature=temperature,
                  force_fn=engine.forces, device=device, dtype=dtype)
    v = integ.maxwell((B, engine.n_atoms, 3), gen, device, dtype)
    f = engine.forces(q)
    cv = DistanceCV(*dsys.terminal_carbonyls(dsys.N_RES))
    bias_fn = None
    if restrain_k and restrain_k > 0:
        R_init = cv.value(q)
        R0 = R_init if restrain_R0 is None else torch.full_like(R_init, float(restrain_R0))

        def bias_fn(qq):
            Rq, grad_full, _ = cv.geometry(qq)
            return dist_bias_force(grad_full, -float(restrain_k) * (Rq - R0))
    n_eq = int(round(equil_ps / dt))
    t0 = time.perf_counter()
    for _ in range(n_eq):
        q, v, f = integ.step(q, v, f, gen, bias_fn=bias_fn)
    R_end = cv.value(q).cpu().numpy()
    ok_t, rep_t = dsys.validate_thermal(q.cpu().numpy(), dsys.N_RES)
    T_kin = float(integ.kinetic_temperature(v))
    if verbose:
        print(f"  {equil_ps:.0f} ps unbiased equilibration ({time.perf_counter()-t0:.0f}s): "
              f"structural {int(ok_t.sum())}/{B} pass (cis {rep_t['n_fail_cis']}, "
              f"chirality {rep_t['n_fail_chirality']}), T_kin {T_kin:.1f} K; R after equilibration "
              f"min/median/max {R_end.min():.2f}/{np.median(R_end):.2f}/{R_end.max():.2f} nm"
              f"{' (restrained)' if bias_fn is not None else ''}", flush=True)
    if ok_t.sum() < 0.95 * B:
        raise RuntimeError(f"equilibration damaged {B - int(ok_t.sum())}/{B} walkers")
    return q.reshape(n_seeds, n_walkers, engine.n_atoms, 3).cpu().numpy(), dict(
        n_replaced_seeds=n_bad, n_structural_fail=int(B - ok_t.sum()), equil_ps=equil_ps,
        jitter_nm=jitter_nm, rng_seed=int(rng_seed), kinetic_temperature_K=T_kin,
        init_mode=init_mode, restrain_k=float(restrain_k or 0.0),
        restrain_R0=(None if restrain_R0 is None else float(restrain_R0)),
        R_after_equil_quantiles=[float(x) for x in np.percentile(R_end, [0, 5, 50, 95, 100])])


def save_result(path_npz, out):
    """Arrays into the npz; everything else (config, controller logs, scalars) into a JSON twin."""
    arrays = {k: v for k, v in out.items() if isinstance(v, (np.ndarray, np.generic))}
    rest = {}
    for k, v in out.items():
        if k in arrays:
            continue
        try:
            json.dumps(v)
            rest[k] = v
        except TypeError:
            rest[k] = str(v)
    np.savez_compressed(path_npz, **arrays)
    with open(os.path.splitext(path_npz)[0] + ".json", "w") as fh:
        json.dump(rest, fh, indent=1)


def provenance(sim, engine, device, extra=None):
    p = dict(config=asdict(sim), config_hash=sim.config_hash(),
             parameter_hash=engine.parameter_hash(), torch=torch.__version__,
             device=(torch.cuda.get_device_name(0) if device == "cuda" else "cpu"),
             git=git_commit(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if extra:
        p.update(extra)
    return p
