"""Config-driven OPES closure runner for the two toy systems (meta + eb).

Sweeps barrier x pace x sigma x seeds for one toy+stage and writes ONE npz per run
into <output_root>/raw/, in the schema src/closure_metrics.compute_metrics reads --
so the SAME aggregate_closure.py + tune_closure.py pipeline drives all three systems.
Idempotent: skips runs whose valid npz already exists (unless --overwrite).

Usage:
  CUDA_VISIBLE_DEVICES=5 python scripts/run_closure_toy.py --config configs/opes_closure/toys_closure.yaml \
        --toy meta --stage tune_r1 --shard 0 --num-shards 6
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np, yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def _run_id(toy, stage, seed, b, p, s):
    key = f"{toy}|{stage}|seed{seed}|b{b}|p{p}|s{s}"
    hh = hashlib.md5(key.encode()).hexdigest()[:10]
    return f"{stage}__opes__{toy}__seed{seed}__b{b:g}_p{int(p)}_s{s:g}__{hh}"


def _valid(path):
    if not os.path.exists(path):
        return False
    try:
        d = np.load(path, allow_pickle=True)
        ok = "l2_f" in d.files and np.isfinite(float(d["l2_f"]))
        d.close(); return bool(ok)
    except Exception:
        return False


def _meta_run(phys, gamma, gfb, warmup_frac, n_steps, seed, b, p, s):
    import opes_meta as om
    return om.run_opes_meta(
        beta=phys["beta"], n_particles=int(phys["n_particles"]),
        n_grid=int(phys["n_grid"]), x_min=phys["x_min"], x_max=phys["x_max"],
        n_steps=n_steps, seed=seed, barrier=b, pace=int(p), sigma=s,
        gamma=gamma, gamma_from_barrier=gfb,
        warmup_steps=int(n_steps * warmup_frac), estimator="meanforce")


def _eb_run(phys, gamma, gfb, warmup_frac, clip, n_steps, seed, b, p, s):
    import eb_abffr_core as eb, opes_core as oc, opes_eb as oe
    cfg = eb.PhysConfig(beta=phys["beta"], H=phys["H"], omega_out=phys["omega_out"],
                        omega_in=phys["omega_in"], s=phys["s"], N=int(phys["N"]),
                        dt=phys["dt"], n_steps=n_steps, h=phys["h"], min_count=phys["min_count"])
    occ = oc.OPESConfig(z_min=eb.XMIN, z_max=eb.XMAX, n_grid=eb.N_GRID, beta=phys["beta"],
                        barrier=b, pace=int(p), sigma=s, gamma=gamma, gamma_from_barrier=gfb,
                        bias_force_clip=clip, warmup_steps=int(n_steps * warmup_frac), fill_edges=True)
    return oe.run_opes_eb(cfg, seed=seed, opes_cfg=occ, estimator="meanforce")


def _enrich(out, toy, stage, n_steps, gamma):
    """Add the identity/schema keys aggregate_closure groups on (physics is fixed
    per toy => single successive-halving cell; h/n_dim/a intentionally absent)."""
    g = "inf" if (gamma == float("inf")) else f"{gamma:g}"
    out = dict(out)
    out.update(study=f"opes_closure_{toy}", stage=stage, name="opes", method="opes",
               mode="sample", n_steps=int(n_steps), n_replicas=int(out.get("N", out.get("n_particles", 0)) or 0),
               opes_gamma=out.get("opes_gamma", g), core_version="opes_v1")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--toy", required=True, choices=["meta", "eb"])
    ap.add_argument("--stage", required=True); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--seeds", default=None)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))[args.toy]
    st = cfg["stages"][args.stage]
    phys = cfg["physics"]; gamma = float("inf") if str(cfg.get("gamma", ".inf")).lower() in (".inf", "inf") else float(cfg["gamma"])
    gfb = bool(cfg.get("gamma_from_barrier", True)); wf = float(cfg.get("warmup_frac", 0.1))
    clip = float(cfg.get("bias_force_clip", 200.0)); n_steps = int(st["n_steps"])
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else list(st["seeds"])
    raw_dir = os.path.join(cfg["output_root"], "raw"); os.makedirs(raw_dir, exist_ok=True)
    grid = [(b, p, s, seed) for b in st["barriers"] for p in st["paces"]
            for s in st["sigmas"] for seed in seeds]
    grid = [g for i, g in enumerate(grid) if i % args.num_shards == args.shard]
    todo = [(b, p, s, sd) for (b, p, s, sd) in grid
            if args.overwrite or not _valid(os.path.join(raw_dir, _run_id(args.toy, args.stage, sd, b, p, s) + ".npz"))]
    if args.dry_run:
        print(f"[dry] toy={args.toy} stage={args.stage} shard={args.shard}/{args.num_shards} "
              f"grid={len(grid)} todo={len(todo)} n_steps={n_steps} seeds={seeds}")
        print(f"      barriers={st['barriers']} paces={st['paces']} sigmas={st['sigmas']}")
        return 0
    n_done = n_fail = 0
    for k, (b, p, s, sd) in enumerate(grid):
        path = os.path.join(raw_dir, _run_id(args.toy, args.stage, sd, b, p, s) + ".npz")
        if not args.overwrite and _valid(path):
            continue
        try:
            if args.toy == "meta":
                out = _meta_run(phys, gamma, gfb, wf, n_steps, sd, b, p, s)
            else:
                out = _eb_run(phys, gamma, gfb, wf, clip, n_steps, sd, b, p, s)
            out = _enrich(out, args.toy, args.stage, n_steps, gamma)
            tmp = path + ".tmp.npz"; np.savez_compressed(tmp, **out); os.replace(tmp, path)
            n_done += 1
            print(f"  [{k+1}/{len(grid)}] {args.toy} seed{sd} b{b:g}p{int(p)}s{s:g}: "
                  f"L2F={out['l2_f']:.4f} L2Fp={out['l2_fp']:.4f} neff={out.get('opes_neff_frac',float('nan')):.2f}")
        except Exception as exc:
            n_fail += 1
            print(f"  [{k+1}/{len(grid)}] {args.toy} seed{sd} b{b}p{p}s{s} FAILED: {exc!r}")
    print(f"[run] {args.toy}/{args.stage} shard{args.shard}: done={n_done} fail={n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
