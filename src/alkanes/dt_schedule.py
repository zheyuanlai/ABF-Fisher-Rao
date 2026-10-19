"""Physical-time schedule for the pentane R15 timestep campaign (docs/PENTANE_R15_DT_FR.md).

Every time-dependent knob of :class:`alkanes.core_dist.DistSimConfig` is an integer step count.  When the
Langevin timestep changes, each one must be rescaled so that the PHYSICAL time it represents is unchanged:

    knob                      physical value       steps at dt 5e-4 (frozen production)
    ABF bias ramp (warm-up)   2.5                  5 000
    estimator burn-in         3.0                  6 000
    FR start                  6.0                  12 000
    FR event interval         0.0025               5        (birth-death exposure dt_eff = dt * fr_every)
    checkpoint interval       1.0                  2 000

The FR rate is per unit physical time and the per-event death probability is ``1 - exp(-rate * s * dt * fr_every)``,
so keeping ``dt * fr_every`` fixed keeps the per-event probability fixed for an identical score.  Nothing else in
the sampler counts steps (the KDE/ABF bandwidths, the clips and the walls are not time-dependent).
"""
from __future__ import annotations

FROZEN = dict(dt=5.0e-4, T=40.0, warmup_t=2.5, burn_t=3.0, fr_start_t=6.0, fr_interval_t=0.0025, save_t=1.0)


def _steps(t, dt, what):
    n = int(round(t / dt))
    if abs(n * dt - t) > 1e-9 * max(1.0, abs(t)):
        raise ValueError(f"{what}: physical time {t} is not an integer number of steps of dt {dt}")
    return n


def physical_schedule(dt, T=FROZEN["T"], warmup_t=FROZEN["warmup_t"], burn_t=FROZEN["burn_t"], fr_start_t=FROZEN["fr_start_t"],
                      fr_interval_t=FROZEN["fr_interval_t"], save_t=FROZEN["save_t"], deposit_interval_t=None):
    """Integer step counts realising the physical schedule at timestep ``dt`` (exactness enforced)."""
    s = dict(dt=float(dt), n_steps=_steps(T, dt, "T"), abf_warmup_steps=_steps(warmup_t, dt, "warm-up"),
             estimator_burn_in_steps=_steps(burn_t, dt, "burn-in"), fr_start_steps=_steps(fr_start_t, dt, "FR start"),
             fr_every=_steps(fr_interval_t, dt, "FR interval"), save_every=_steps(save_t, dt, "save"))
    s["deposit_stride"] = 1 if deposit_interval_t is None else _steps(deposit_interval_t, dt, "deposit interval")
    s["fr_dt_eff"] = s["fr_every"] * float(dt)
    s["T"] = float(T)
    return s
