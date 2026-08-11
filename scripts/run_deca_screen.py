"""ABF-only screen for deca-alanine — the run that decides whether any mFR arm is licensed.

v2 preregistration §6.5, with initial conditions frozen by **Amendment 4**: all 16 walkers start
from the equilibrated alpha-helix, so Gate B is a real test rather than a formality. 8 ensembles
(seeds 3000-3007) x 16 walkers x 0.5 ns = the historical 8 ns aggregate per ensemble.

    python scripts/run_deca_screen.py --out results/deca/screen

**This script never loads the reference.** It produces a ``xi`` trace and a bias profile;
``scripts/analyze_deca_screen.py`` applies Gates B and C against ``F_ref`` afterwards. That
separation is the reason the classification cannot be steered by an mFR result -- at this point
no mFR arm exists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from alanine.dynamics import BAOAB                                          # noqa: E402
from deca import system as dsys                                             # noqa: E402
from deca.core import DecaSimConfig, run_sampler_deca                        # noqa: E402
from deca.engine import make_engine                                         # noqa: E402
from deca.umbrella import relax_pool                                        # noqa: E402

SCREEN_SEEDS = list(range(3000, 3008))          # §5, frozen
EQUIL_PS = 20.0                                 # Amendment 4: outside the ABF budget


def build_initial_population(engine, n_seeds, n_walkers, cfg, device, dtype, verbose=True):
    """Amendment 4: helix start, jittered per walker, relaxed, then unbiased equilibration.

    The equilibration is unbiased and accumulates nothing -- it thermalises the structure, it
    does not advance the free-energy estimate.
    """
    B = n_seeds * n_walkers
    rng = np.random.default_rng(cfg.rng_seed)
    x = dsys.build_helix(-57.0, -47.0, n_res=dsys.N_RES)
    X0 = np.stack([x + 0.006 * rng.standard_normal(x.shape) for _ in range(B)])
    q = torch.as_tensor(X0, device=device, dtype=dtype).contiguous()

    q, ok, rep = relax_pool(engine, q)
    n_bad = int((~ok).sum())
    if n_bad:
        good = np.flatnonzero(ok)
        q[torch.as_tensor(np.flatnonzero(~ok), device=device)] = \
            q[torch.as_tensor(good[rng.integers(0, good.size, n_bad)], device=device)].clone()
    if verbose:
        print(f"  seed relaxation: {B - n_bad}/{B} pass; {n_bad} replaced", flush=True)

    gen = torch.Generator(device=device).manual_seed(int(cfg.rng_seed) + 555)
    integ = BAOAB(engine.masses, dt=cfg.dt, gamma=cfg.gamma, temperature=cfg.temperature,
                  force_fn=engine.forces, device=device, dtype=dtype)
    v = integ.maxwell((B, engine.n_atoms, 3), gen, device, dtype)
    f = engine.forces(q)
    n_eq = int(EQUIL_PS / cfg.dt)
    t0 = time.perf_counter()
    for _ in range(n_eq):
        q, v, f = integ.step(q, v, f, gen)
    ok_t, rep_t = dsys.validate_thermal(q.cpu().numpy(), dsys.N_RES)
    if verbose:
        print(f"  {EQUIL_PS:.0f} ps unbiased equilibration ({time.perf_counter()-t0:.0f}s): "
              f"structural {int(ok_t.sum())}/{B} pass "
              f"(cis {rep_t['n_fail_cis']}, chirality {rep_t['n_fail_chirality']})", flush=True)
    if ok_t.sum() < 0.95 * B:
        raise RuntimeError(f"equilibration damaged {B - int(ok_t.sum())}/{B} walkers")
    return q.reshape(n_seeds, n_walkers, engine.n_atoms, 3).cpu().numpy(), dict(
        n_replaced_seeds=n_bad, n_structural_fail=int(B - ok_t.sum()),
        equil_ps=EQUIL_PS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/deca/screen")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-steps", type=int, default=None, help="smoke override")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--no-compile", action="store_true",
                    help="eager forces; for CPU smoke runs only")
    ap.add_argument("--save-every", type=int, default=None, help="smoke override")
    ap.add_argument("--xi-trace-every", type=int, default=None, help="smoke override")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible; set CUDA_VISIBLE_DEVICES to one idle GPU")

    seeds = args.seeds if args.seeds else SCREEN_SEEDS
    over = {k: v for k, v in dict(n_steps=args.n_steps, save_every=args.save_every,
                                  xi_trace_every=args.xi_trace_every).items() if v is not None}
    cfg = DecaSimConfig(**over)
    if over:
        print(f"!! NON-DEFAULT CONFIG (smoke): {over}", flush=True)
    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    dtype = torch.float64

    engine, system, top = make_engine(10, device=args.device, dtype=dtype,
                                      compiled=not args.no_compile)
    print(f"ABF-only screen: {len(seeds)} ensembles x {cfg.n_walkers} walkers x "
          f"{cfg.n_steps * cfg.dt:.0f} ps  (batch {len(seeds)*cfg.n_walkers})", flush=True)

    init, init_rep = build_initial_population(engine, len(seeds), cfg.n_walkers, cfg,
                                              args.device, dtype)

    out = run_sampler_deca("abf", engine, cfg, seeds, init, device=args.device, dtype=dtype,
                           progress_every=50_000)

    raw = os.path.join(args.out, "raw", f"deca_screen_abf__{cfg.config_hash()}.npz")
    np.savez_compressed(raw, **{k: v for k, v in out.items()
                                if isinstance(v, (np.ndarray, np.generic))})
    with open(os.path.join(args.out, "provenance.json"), "w") as fh:
        json.dump(dict(config=cfg.__dict__, config_hash=cfg.config_hash(),
                       parameter_hash=engine.parameter_hash(), seeds=list(map(int, seeds)),
                       initial_conditions=init_rep, torch=torch.__version__,
                       device=(torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"),
                       prereg="docs/V2_PREREGISTRATION.md §6.5 + Amendment 4",
                       note="ABF only. No reference was loaded. No mFR arm exists."),
                  fh, indent=2)

    print(f"\n  runtime {out['runtime_seconds']/60:.1f} min")
    print(f"  xi trace {out['xi_trace'].shape}  range "
          f"[{out['xi_trace'].min():.3f}, {out['xi_trace'].max():.3f}] nm")
    print(f"  final effective counts: min {out['final_eff_counts'].min():.1f}")
    print(f"  raw -> {raw}")
    print("\nNow run: python scripts/analyze_deca_screen.py "
          f"--screen {args.out} --reference results/deca/reference")


if __name__ == "__main__":
    main()
