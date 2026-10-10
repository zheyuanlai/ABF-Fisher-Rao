"""Lean torch drivers of the frozen production algorithms, for the Experiment III wall-clock benchmark.

docs/mechanism/SCIENTIFIC_PLAN.md section 8; configs/mechanism/parallel_benchmark.json.  ONE simulation per
driver (R = 1 row, no batching of independent seeds): the benchmark measures walker-level parallelism of a single
N = 512 run.  Nothing in src/ is modified; every operation that touches the dynamics is a call of the existing torch
engines' own functions, in their own order.

GatewayTorchRun -- the frozen gateway production algorithm (configs/equal_budget_v2/gateway_production.json;
    validated numba engine src/gateway_ladder_numba.py) on the torch primitives of src/gateway_core.py /
    src/eb_abffr_core.py.  The step body is gateway_core.simulate_batch's loop for one row and one method, with
    estimator 'histogram' (eb.HistogramABFEstimator: 180 [left, right) bins with the 1e-9 nudge, Gamma_j =
    M_j / (C_j + min_count), own-bin bias read AFTER all of the step's deposits) and method 'abf' or 'fr_uniform'
    (binned KDE eta on the 181-node grid, uniform score S = log p(x) - log q - KL(p || q) clipped at +-score_clip,
    gateway_core.resample_indices: death / clone candidates 1 - exp(-+ g S dt_fr), proportional cap floor(0.08 N),
    pool law), FR every fr_every steps from step 0 on the post-move positions, rate g = gamma (1 - exp(-step /
    ramp_steps)).  LEFT OUT (they read state and change no bit of the dynamics; the bitwise test
    tests/test_parallel_benchmark.py::test_gateway_torch_bitwise_vs_simulate_batch proves it): the kernel-estimator
    grid accumulators Sf / C / Sf2, the per-step node profile and PMF (Bbias), the estimated-target EMA, the
    estimated / oracle target computation at FR steps, the windowed genealogy labels, max |bias|, the per-save
    error / region / target diagnostics.  Placement change only: the (always False) sham mask lives on the host so
    that resample_indices' ``bool(sham_mask.any())`` costs no device synchronisation; simulate_batch's
    ``bool(fr_mask.any())`` test is a Python bool here.

LTATorchRun -- the frozen LTA production algorithm (configs/equal_budget_v2/lta_{300,150}K_production.json;
    validated numba engine src/lta_ladder_numba.py, a port of this torch engine) on src/lta/core_lta.py's
    LTASystem and alkanes.core._fr_score / _birth_death.  The step body is core_lta.run_sampler's loop for one row,
    abf_estimator 'histogram', method 'abf' or 'fr_uniform'.  LEFT OUT (diagnostic only; bitwise test
    test_lta_torch_bitwise_vs_run_sampler): the potential-energy pass and U(z) accumulators (u_of_z; the numba
    production ran with accumulate_u = False), the per-step PMF A_hat / B_n (they only feed estimated / oracle
    targets), region / transition / crossing bookkeeping, the per-opportunity host copies of the score and event
    counts (score std / absmax, birth / death histograms, event_counts list), the per-save KDE / KL / genealogy
    diagnostics.  The FR target is the uniform density per.normalize_density(ones) computed once (core_lta
    recomputes the identical tensor at every opportunity).

Save convention (identical to the numba engines, so eqb_metrics' scorers apply unchanged):
    gateway  save at completed-step count k holds the deposits of the states 0..k-1 (taken after step k-1);
    LTA      save at completed-step count k is taken during evaluated step k, accumulators INCLUDING step k's
             deposit (all-steps M_all / C_all and production M_prod / C_prod).
Force evaluations: gateway N k at save k; LTA N (k + 1).

Randomness (torch generators; same LAW as the numba engines, not the same numbers):
    gateway  initial conditions = gateway_core.init_conditions (numpy default_rng(1000 + seed): the numba engine's
             init_left, identical x0 -- so a torch run of a production seed SHARES that production run's x0 / y0,
             which analyze_parallel_benchmark.equivalence accounts for); Langevin normals
             torch.Generator(device).manual_seed(noise_seed_base + seed), FR uniforms
             torch.Generator(device).manual_seed(fr_seed_base + seed) -- simulate_batch's two streams with
             batch_seed = seed.  ABF and FR arms of one seed share x0, y0 and the noise STREAM, but they are
             EFFECTIVELY UNPAIRED after step 0: gateway_core.resample_indices returns a random-key permutation of all
             N slots at EVERY FR opportunity, also when no event fires (step 0 already, where the ramp gives g = 0),
             so from step 1 on the shared normals drive different walkers in the two arms (law unchanged: walkers are
             exchangeable).  The numba production arms stay bitwise identical until the first realised event; its
             ABF-FR Ibar_F correlation across the 32 gateway N = 512 seeds is about -0.03, so nothing measurable is
             lost (tests/test_parallel_benchmark.py::test_gateway_torch_arms_unpaired_after_first_opportunity).
    LTA      gen_dyn = Generator(device).manual_seed(rng_seed) (initial conditions, then one randn(q.shape) per
             move), gen_fr = manual_seed(rng_seed + 987654321) -- run_sampler's two streams; rng_seed = seed.  The
             LTA birth-death copies in place (no slot permutation), so the two arms stay paired until the first
             realised event (production ABF-FR Ibar_F correlation 0.94 at 300 K, 0.77 at 150 K).  The initial
             conditions are torch draws, independent of the numba production's.
On CUDA, float64 scatter_add_ uses atomics: runs are not bitwise reproducible (statistical equivalence is what
Experiment III tests).

Timing: ``run(on_start, on_save, on_end)`` returns per-save wall-clock stamps (time.perf_counter after
torch.cuda.synchronize on CUDA), and, with ``fr_timing``, the device time inside the FR blocks (CUDA events harvested
at every save, or perf_counter on the CPU).  No timing code runs between saves except the FR events.  Hooks (all
optional, used by parallel_benchmark.py for telemetry): ``on_start()`` right before t0 (after the setup and a device
sync), ``on_save(wall_s)`` right after each save's stamp -- the device is idle there (synchronised) and the hook's
own duration is EXCLUDED from every later stamp and from wall_total (returned as monitor_s) -- and ``on_end()`` right
after the final sync.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

import eb_abffr_core as eb  # noqa: E402
import gateway_core as gc  # noqa: E402
from lta import core_lta as CL  # noqa: E402
from alkanes import periodic as per  # noqa: E402
from alkanes.core import _birth_death, _fr_score  # noqa: E402

DRIVER_VERSION = "bench_torch_engines/2"
ENGINE_FILES = {
    "gateway": ["src/gateway_core.py", "src/eb_abffr_core.py"],
    "lta": ["src/lta/core_lta.py", "src/alkanes/core.py", "src/alkanes/periodic.py"],
}


class EquivalenceError(ValueError):
    """The torch engine cannot run the frozen production algorithm with these knobs."""


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _FRClock:
    """Device time spent inside the FR blocks.  CUDA: an event pair per opportunity, harvested (and released) at
    every save, where the driver synchronises anyway; CPU: perf_counter around the block."""

    def __init__(self, device, enabled):
        self.cuda = device.type == "cuda"
        self.enabled = bool(enabled)
        self.pending = []
        self.total_s = 0.0
        self._t = None

    def start(self):
        if not self.enabled:
            return
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._t = e
        else:
            self._t = time.perf_counter()

    def stop(self):
        if not self.enabled:
            return
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.pending.append((self._t, e))
        else:
            self.total_s += time.perf_counter() - self._t

    def harvest(self):
        """Call after a device synchronisation: adds the finished event pairs."""
        if self.enabled and self.cuda and self.pending:
            self.total_s += sum(a.elapsed_time(b) for a, b in self.pending) * 1e-3
            self.pending = []
        return self.total_s if self.enabled else math.nan


def _check_saves(save_steps, lo, hi):
    s = np.asarray(save_steps, dtype=np.int64).ravel()
    if s.size == 0 or np.any(np.diff(s) <= 0) or s[0] < lo or s[-1] > hi:
        raise ValueError(f"save_steps must be strictly increasing completed-step counts in [{lo}, {hi}]")
    return s


# =================================================================================================== gateway
def gateway_knobs(cfg, N):
    """Frozen gateway knobs (gateway_ladder_numba conventions) -> the torch engine's step-count knobs.  Raises
    EquivalenceError where the torch engine's rule differs from the numba production rule."""
    h = float(cfg["h"])
    fr_every = int(round(float(cfg["fr_interval_t"]) / h))
    if fr_every < 1 or abs(fr_every * h - float(cfg["fr_interval_t"])) > 1e-9 * max(1.0, float(cfg["fr_interval_t"])):
        raise EquivalenceError(f"fr_interval_t {cfg['fr_interval_t']} is not an integer number of steps at h {h}")
    ramp_f = float(cfg["ramp_t"]) / h
    ramp_steps = int(round(ramp_f))
    if ramp_steps < 1 or abs(ramp_f - ramp_steps) > 1e-6:
        # simulate_batch's ramp is int(ramp_fraction n_steps) steps; numba uses ramp_t / h as a float
        raise EquivalenceError(f"ramp_t / h = {ramp_f} is not an integer: the torch ramp is an integer step count")
    cap_numba = max(int(cfg.get("cap_min", 1)), int(math.floor(float(cfg["max_event_fraction"]) * N)))
    cap_torch = int(math.floor(float(cfg["max_event_fraction"]) * N))
    if cap_numba != cap_torch:
        # simulate_batch's cap is floor(mef N) with no floor of cap_min (production: identical for N >= 13)
        raise EquivalenceError(f"cap differs: numba max(cap_min, floor(mef N)) = {cap_numba}, torch floor(mef N) = "
                               f"{cap_torch} at N = {N}")
    if int(cfg["nb"]) <= 0:
        raise EquivalenceError("histogram estimator needs nb > 0")
    return dict(fr_every=fr_every, ramp_steps=ramp_steps, cap=cap_torch, dt_fr=h * fr_every, h=h)


class GatewayTorchRun:
    kind = "gateway"

    def __init__(self, cfg, method, N, seed, n_steps, save_steps, device, dtype=torch.float64,
                 noise_seed_base=2000, fr_seed_base=3000, fr_timing=True):
        if method not in ("abf", "fr"):
            raise ValueError("method must be 'abf' or 'fr'")
        self.cfg, self.method, self.N, self.seed = dict(cfg), method, int(N), int(seed)
        self.n_steps = int(n_steps)
        self.device, self.dtype = torch.device(device), dtype
        self.is_fr = method == "fr"
        if self.is_fr and self.N < 2:
            raise ValueError("no FR at N = 1")
        self.kn = gateway_knobs(cfg, self.N)
        self.save_at = _check_saves(save_steps, 1, self.n_steps)
        self.noise_seed = int(noise_seed_base) + self.seed
        self.fr_seed = int(fr_seed_base) + self.seed
        self.fr_timing = bool(fr_timing) and self.is_fr

    def meta(self):
        return dict(driver=DRIVER_VERSION, system_kind="gateway", engine="src/gateway_core.py (simulate_batch loop, "
                    "estimator 'histogram', method 'abf' | 'fr_uniform')", method=self.method, N=self.N, seed=self.seed,
                    n_steps=self.n_steps, device=str(self.device), dtype=str(self.dtype), noise_seed=self.noise_seed,
                    fr_seed=self.fr_seed, init="gateway_core.init_conditions (numpy default_rng(1000 + seed), init "
                    "'left') == gateway_ladder_numba.init_left", shares_production_init=True,
                    arm_pairing="x0 / y0 and the noise stream shared; effectively unpaired after step 0 (resample_indices "
                    "permutes all slots at every FR opportunity, also without events)",
                    **{k: v for k, v in self.kn.items()},
                    accumulator_convention="pre-deposit: save k holds deposits of states 0..k-1 (taken after step k-1)",
                    force_eval_convention="N per completed step (N k at save k)")

    def run(self, on_start=None, on_save=None, on_end=None):
        dev, dt_ = self.device, self.dtype
        c, N, kn = self.cfg, self.N, self.kn
        h = kn["h"]
        x_grid, dx, eval_mask, idx0 = eb.build_grid(dev, dt_)
        k_eta, r_eta = eb.gaussian_kernel(float(c["eta"]), dx, dev, dt_)

        def col(v):          # simulate_batch's per-row (R, 1) parameter tensors, R = 1
            return torch.tensor([float(v)], device=dev, dtype=dt_).repeat_interleave(1).unsqueeze(1)
        beta_b = torch.tensor([float(c["beta"])], device=dev, dtype=dt_)
        oout_b = torch.tensor([float(c["omega_out"])], device=dev, dtype=dt_)
        oin_b = torch.tensor([float(c["omega_in"])], device=dev, dtype=dt_)
        s_b = torch.tensor([float(c["s"])], device=dev, dtype=dt_)
        beta, Hc, oout, oin, sw = col(c["beta"]), col(c["H"]), col(c["omega_out"]), col(c["omega_in"]), col(c["s"])
        clip_r = col(c["score_clip"])
        maxfrac_r = col(c["max_event_fraction"])
        cap_r = torch.floor(maxfrac_r * N).long()
        gamma_r = torch.tensor([[float(c["gamma"])]], device=dev, dtype=dt_).reshape(1, 1)
        noise_amp = torch.sqrt(2.0 * h / beta)
        fr_mask = torch.tensor([True], device=dev)
        sham_mask = torch.tensor([False])               # host tensor: no device sync in resample_indices
        partner = torch.tensor([0], device=dev, dtype=torch.long)

        X0, Y0 = gc.init_conditions([self.seed], N, beta_b, oout_b, oin_b, s_b, ["left"], dev, dt_)
        X = X0.repeat_interleave(1, dim=0).clone()
        Y = Y0.repeat_interleave(1, dim=0).clone()
        est = eb.HistogramABFEstimator(1, int(c["nb"]), float(c["min_count"]), x_grid, dev, dt_)
        gen_n = torch.Generator(device=dev)
        gen_n.manual_seed(self.noise_seed)
        gen_f = torch.Generator(device=dev)
        gen_f.manual_seed(self.fr_seed)
        qu = torch.ones((1, eb.N_GRID), device=dev, dtype=dt_)
        q_uni = (qu / torch.clamp(eb.trapz(qu, dx), min=eb.EPS)).expand(1, eb.N_GRID)

        S_n = self.save_at.size
        nb = int(c["nb"])
        o_M = torch.zeros((S_n, nb), device=dev, dtype=dt_)
        o_C = torch.zeros((S_n, nb), device=dev, dtype=dt_)
        o_die = torch.zeros(S_n, device=dev, dtype=dt_)
        o_clone = torch.zeros(S_n, device=dev, dtype=dt_)
        tot_die = torch.zeros(1, device=dev, dtype=dt_)
        tot_clone = torch.zeros(1, device=dev, dtype=dt_)
        wall = np.full(S_n, np.nan)
        fr_s = np.full(S_n, np.nan)
        frc = _FRClock(dev, self.fr_timing)
        fe, ramp, dt_fr = kn["fr_every"], kn["ramp_steps"], kn["dt_fr"]
        is_fr = self.is_fr
        n_fr = 0
        save_list = self.save_at.tolist()
        sp, next_save = 0, save_list[0]

        mon_s = 0.0
        _sync(dev)
        if on_start is not None:
            on_start()
        t0 = time.perf_counter()
        for step in range(self.n_steps):
            om = eb.omega_of(X, oout, oin, sw)
            dom = eb.domega_of(X, oout, oin, sw)
            fx = eb.dU_of(X, Hc) + om * dom * Y * Y
            fy = om * om * Y
            est.update(X, fx)
            Fp_bins = est.bin_mean_force()
            zx = torch.randn((1, N), device=dev, dtype=dt_, generator=gen_n)
            zy = torch.randn((1, N), device=dev, dtype=dt_, generator=gen_n)
            bias_force = est.evaluate(X, Fp_bins)
            Xp = eb.reflect_into(X + (-fx + bias_force) * h + noise_amp * zx, eb.XMIN, eb.XMAX)
            Yp = Y + (-fy) * h + noise_amp * zy
            if is_fr and step % fe == 0:
                frc.start()
                g = gamma_r * (1.0 - math.exp(-max(step / ramp, 0.0)))
                p = eb.binned_density(Xp, k_eta, r_eta, dx)
                q = q_uni
                kl = eb.trapz(p * (torch.log(torch.clamp(p, min=eb.EPS))
                                   - torch.log(torch.clamp(q, min=eb.EPS))), dx).unsqueeze(1)
                S = (torch.log(torch.clamp(eb.interp1d(Xp, p, dx), min=eb.EPS))
                     - torch.log(torch.clamp(eb.interp1d(Xp, q, dx), min=eb.EPS)) - kl)
                S = torch.clamp(S, -clip_r, clip_r)
                sel, die, clone = gc.resample_indices(S, fr_mask, sham_mask, partner, g, dt_fr, cap_r, gen_f)
                Xp = torch.gather(Xp, 1, sel)
                Yp = torch.gather(Yp, 1, sel)
                tot_die += die.sum(dim=1).to(dt_)
                tot_clone += clone.sum(dim=1).to(dt_)
                n_fr += 1
                frc.stop()
            X, Y = Xp, Yp
            if step + 1 == next_save:
                o_M[sp].copy_(est.M[0])
                o_C[sp].copy_(est.C[0])
                o_die[sp] = tot_die[0]
                o_clone[sp] = tot_clone[0]
                _sync(dev)
                wall[sp] = time.perf_counter() - t0
                fr_s[sp] = frc.harvest()
                if on_save is not None:            # excluded from the clock (device idle: synchronised)
                    tm = time.perf_counter()
                    on_save(float(wall[sp]))
                    dtm = time.perf_counter() - tm
                    t0 += dtm
                    mon_s += dtm
                sp += 1
                next_save = save_list[sp] if sp < S_n else -1
        _sync(dev)
        wall_total = time.perf_counter() - t0
        if on_end is not None:
            on_end()
        if sp != S_n:
            raise RuntimeError(f"only {sp} of {S_n} saves fired")
        k = self.save_at.astype(np.int64)
        return dict(save_step=k, save_t=k * h, save_u=k / float(self.n_steps), monitor_s=float(mon_s),
                    wall_s_at_save=wall, fr_time_s_at_save=fr_s, n_force_evals_at_save=N * k,
                    M_all=o_M.cpu().numpy(), C_all=o_C.cpu().numpy(),
                    fr_kd_cum=o_die.cpu().numpy().astype(np.int64), fr_kc_cum=o_clone.cpu().numpy().astype(np.int64),
                    X_final=X[0].cpu().numpy(), Y_final=Y[0].cpu().numpy(),
                    n_fr_opps=int(n_fr), wall_total_s=float(wall_total), n_force_evals=int(N * self.n_steps))


# =================================================================================================== LTA
def lta_sim_config(cfg, T_K, N, n_steps, rng_seed):
    """core_lta.LTASimConfig of the frozen LTA production knobs (lta_ladder_numba conventions).  Raises
    EquivalenceError where the torch rule differs from the numba production rule."""
    cap_numba = max(int(cfg.get("cap_min", 1)), int(float(cfg["max_event_fraction"]) * N))
    cap_torch = int(float(cfg["max_event_fraction"]) * N)
    if cap_numba != cap_torch:
        raise EquivalenceError(f"cap differs: numba max(cap_min, int(mef N)) = {cap_numba}, torch int(mef N) = "
                               f"{cap_torch} at N = {N} (the finite-N extension has no torch counterpart)")
    dc = cfg.get("deposit_clip")
    if dc is not None and float(dc) != 8.0 * float(cfg["abf_force_clip"]):
        raise EquivalenceError("torch clamps the deposited force at 8 x abf_force_clip; deposit_clip differs")
    if cfg.get("accumulate_u"):
        raise EquivalenceError("accumulate_u is a diagnostic left out of the lean torch driver")
    return CL.LTASimConfig(dt=float(cfg["h"]), n_steps=int(n_steps), n_replicas=int(N), save_every=max(1, int(n_steps)),
                           rng_seed=int(rng_seed), n_grid=int(cfg["n_grid"]), kde_bandwidth=float(cfg["kde_bandwidth"]),
                           abf_bias_scale=float(cfg.get("abf_bias_scale", 1.0)),
                           abf_warmup_steps=int(cfg["warmup_steps"]), abf_force_clip=float(cfg["abf_force_clip"]),
                           estimator_burn_in_steps=int(cfg["burn_in_steps"]), fr_rate=float(cfg["fr_rate"]),
                           score_clip=float(cfg["score_clip"]), fr_start_steps=int(cfg["fr_start_steps"]),
                           fr_every=int(cfg["fr_every"]), max_event_fraction=float(cfg["max_event_fraction"]),
                           window_half=float(cfg.get("window_half", 1.5)), cage_min=float(cfg.get("cage_min", 4.0)),
                           abf_estimator="histogram")


class LTATorchRun:
    kind = "lta"

    def __init__(self, cfg, T_K, method, N, seed, n_steps, save_steps, device, dtype=torch.float64,
                 rng_seed=None, fr_timing=True):
        if method not in ("abf", "fr"):
            raise ValueError("method must be 'abf' or 'fr'")
        self.cfg, self.T_K, self.method = dict(cfg), float(T_K), method
        self.N, self.seed, self.n_steps = int(N), int(seed), int(n_steps)
        self.device, self.dtype = torch.device(device), dtype
        self.is_fr = method == "fr"
        if self.is_fr and self.N < 2:
            raise ValueError("no FR at N = 1")
        self.rng_seed = int(self.seed if rng_seed is None else rng_seed)
        self.sim = lta_sim_config(cfg, self.T_K, self.N, self.n_steps, self.rng_seed)
        self.save_at = _check_saves(save_steps, 0, self.n_steps)
        self.fr_timing = bool(fr_timing) and self.is_fr

    def meta(self):
        s = self.sim
        return dict(driver=DRIVER_VERSION, system_kind="lta", engine="src/lta/core_lta.py (run_sampler loop, "
                    "abf_estimator 'histogram', method 'abf' | 'fr_uniform')", method=self.method, N=self.N,
                    seed=self.seed, T_K=self.T_K, n_steps=self.n_steps, device=str(self.device), dtype=str(self.dtype),
                    rng_seed=self.rng_seed, fr_rng_seed=self.rng_seed + 987654321, shares_production_init=False,
                    arm_pairing="initial conditions and noise shared; paired until the first realised event (in-place "
                    "copies)",
                    cap=int(s.max_event_fraction * self.N), fr_every=s.fr_every, fr_start_steps=s.fr_start_steps,
                    fr_rate=s.fr_rate, score_clip=s.score_clip, kde_bandwidth=s.kde_bandwidth,
                    warmup_steps=s.abf_warmup_steps, burn_in_steps=s.estimator_burn_in_steps,
                    abf_force_clip=s.abf_force_clip, deposit_clip=8.0 * s.abf_force_clip, h=s.dt,
                    accumulator_convention="save k during evaluated step k, accumulators include step k's deposit",
                    force_eval_convention="N per evaluated step 0..n_steps (N (k + 1) at save k)")

    def run(self, on_start=None, on_save=None, on_end=None):
        dev, dt_ = self.device, self.dtype
        sim, N = self.sim, self.N
        system = CL.LTASystem(CL.LTAParams(temperature=self.T_K), dev, dt_, root=ROOT)
        R = 1
        beta = system.p.beta
        ng = sim.n_grid
        grid, dphi = per.periodic_grid(ng, device=dev, dtype=dt_)
        K_kde = per.wrapped_gaussian_kernel_matrix(grid, sim.kde_bandwidth)
        gen_dyn = torch.Generator(device=dev).manual_seed(int(sim.rng_seed))
        gen_fr = torch.Generator(device=dev).manual_seed(int(sim.rng_seed) + 987654321)
        q = system.initial_conditions(R, N, gen_dyn)
        noise_scale = math.sqrt(2.0 * sim.dt / beta)
        fsum = torch.zeros(R, ng, device=dev, dtype=dt_)
        csum = torch.zeros(R, ng, device=dev, dtype=dt_)
        fsum_prod = torch.zeros(R, ng, device=dev, dtype=dt_)
        csum_prod = torch.zeros(R, ng, device=dev, dtype=dt_)
        ancestors = torch.arange(N, device=dev).expand(R, N).clone() if self.is_fr else None
        q_grid = per.normalize_density(torch.ones(R, ng, device=dev, dtype=dt_), dphi)
        total_repl = torch.zeros(R, dtype=torch.long, device=dev)
        S_n = self.save_at.size
        o = torch.zeros((S_n, 4, ng), device=dev, dtype=dt_)
        o_repl = torch.zeros(S_n, dtype=torch.long, device=dev)
        wall = np.full(S_n, np.nan)
        fr_s = np.full(S_n, np.nan)
        frc = _FRClock(dev, self.fr_timing)
        fclip, dclip = sim.abf_force_clip, sim.abf_force_clip * 8
        burn, warm = sim.estimator_burn_in_steps, max(sim.abf_warmup_steps, 1)
        fr_start, fr_every = sim.fr_start_steps, max(int(sim.fr_every), 1)
        is_fr = self.is_fr
        n_opp = 0
        save_list = self.save_at.tolist()
        sp, next_save = 0, save_list[0]

        mon_s = 0.0
        _sync(dev)
        if on_start is not None:
            on_start()
        t0 = time.perf_counter()
        for step in range(sim.n_steps + 1):
            qf = q.reshape(R * N, 2, 3)
            F = system.forces(qf)
            f_loc, phi_f, grad_f = system.cv_local_mean_force(qf, F)
            phi = phi_f.reshape(R, N)
            f_loc = torch.clamp(f_loc, -dclip, dclip).reshape(R, N)
            fsum += per.bin_sum(phi, f_loc, ng)
            csum += per.bin_counts(phi, ng)
            if step >= burn:
                fsum_prod += per.bin_sum(phi, f_loc, ng)
                csum_prod += per.bin_counts(phi, ng)
            mf_profile = CL.histogram_mean_force(fsum, csum)
            ramp = min(1.0, step / warm)
            abf_scale = sim.abf_bias_scale * ramp
            mf_at = torch.gather(mf_profile, -1, CL.bin_index(phi, ng))
            mf_at = mf_at.clamp(-fclip, fclip)
            bias_at = abf_scale * mf_at
            bias_force = (bias_at.reshape(R * N)[:, None, None] * grad_f).reshape(R, N, 2, 3)
            if step == next_save:
                o[sp, 0] = fsum[0]
                o[sp, 1] = csum[0]
                o[sp, 2] = fsum_prod[0]
                o[sp, 3] = csum_prod[0]
                o_repl[sp] = total_repl[0]
                _sync(dev)
                wall[sp] = time.perf_counter() - t0
                fr_s[sp] = frc.harvest()
                if on_save is not None:            # excluded from the clock (device idle: synchronised)
                    tm = time.perf_counter()
                    on_save(float(wall[sp]))
                    dtm = time.perf_counter() - tm
                    t0 += dtm
                    mon_s += dtm
                sp += 1
                next_save = save_list[sp] if sp < S_n else -1
            if step == sim.n_steps:
                break
            noise = torch.randn(q.shape, generator=gen_dyn, device=dev, dtype=dt_)
            q = q + sim.dt * (F.reshape(R, N, 2, 3) + bias_force) + noise_scale * noise
            if is_fr:
                nxt = step + 1
                if nxt >= fr_start and (nxt - fr_start) % fr_every == 0:
                    frc.start()
                    phi_new = system.cv_value(q.reshape(R * N, 2, 3)).reshape(R, N)
                    score, _p, _kl = _fr_score(phi_new, grid, dphi, K_kde, q_grid, sim.kde_bandwidth, sim.score_clip)
                    q, ancestors, n_repl, _d, _b = _birth_death(q, score, ancestors, sim, gen_fr)
                    total_repl += n_repl
                    n_opp += 1
                    frc.stop()
        _sync(dev)
        wall_total = time.perf_counter() - t0
        if on_end is not None:
            on_end()
        if sp != S_n:
            raise RuntimeError(f"only {sp} of {S_n} saves fired")
        k = self.save_at.astype(np.int64)
        oo = o.cpu().numpy()
        return dict(save_step=k, save_t=k * sim.dt, save_u=k / float(sim.n_steps), monitor_s=float(mon_s),
                    wall_s_at_save=wall, fr_time_s_at_save=fr_s, n_force_evals_at_save=N * (k + 1),
                    M_all=oo[:, 0], C_all=oo[:, 1], M_prod=oo[:, 2], C_prod=oo[:, 3],
                    cum_deaths=o_repl.cpu().numpy().astype(np.int64),
                    q_final=q[0].cpu().numpy(), n_fr_opps=int(n_opp), wall_total_s=float(wall_total),
                    n_force_evals=int(N * (sim.n_steps + 1)))
