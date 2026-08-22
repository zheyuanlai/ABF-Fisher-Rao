"""Frozen system registry and the shared numerical convention.

FROZEN (docs/RESULTS_LOG.md, estimator-floor calibration 2026-08-22):
  domain [-1.8, 1.8] (or as stated), G = 361, eval window [-1.5, 1.5],
  bw_mf = 0.02, n_min = 1.0  =>  estimator floor e_F ~ 0.004.
"""
from __future__ import annotations

from .grid import Grid1D
from .systems.base import SepSystem, SysParams

G_DEFAULT = 361


def _grid(xhalf=1.8, evalhalf=1.5, n=G_DEFAULT, bc="reflect"):
    return Grid1D(-xhalf, xhalf, n, -evalhalf, evalhalf, bc=bc)


SYSTEMS = {
    # ---- A. easy: enthalpic barrier + fast unimodal fiber -------------------
    "EB": dict(grid=_grid(), params=SysParams(
        beta=8.0, H=2.5, omega_out=1.0, omega_in=25.0, s=0.25)),

    # ---- B. slow fiber: the lift has to pay for a soft, slowly relaxing y ---
    "SLOWFIB": dict(grid=_grid(), params=SysParams(
        beta=8.0, H=2.5, omega_out=0.25, omega_in=4.0, s=0.25)),

    # ---- C. hidden two-channel fiber with a switch region at x = 0 ----------
    #    correct P(y>0|x) runs 1 -> 0 across the domain; channels interconvert
    #    only near x_sw, so an immobile window started in one channel is stuck.
    "CHANNEL": dict(grid=_grid(), params=SysParams(
        beta=8.0, H=1.0, omega_out=1.0, omega_in=3.0, s=0.25,
        A_c=2.0, c=1.0, a_min=0.15, x_sw=0.0, s_sw=0.30,
        delta0=0.8, s_delta=0.5, y_max=4.0)),

    # ---- D. long periodic CV: transport distance is the knob ---------------
    "TORSION": dict(grid=_grid(xhalf=6.0, evalhalf=6.0, n=1201, bc="periodic"),
                    params=SysParams(beta=8.0, barrier="periodic", H=3.0,
                                     n_wells=6, omega_out=1.0, omega_in=6.0)),
    "TORSION_S": dict(grid=_grid(xhalf=1.5, evalhalf=1.5, n=301, bc="periodic"),
                      params=SysParams(beta=8.0, barrier="periodic", H=3.0,
                                       n_wells=2, omega_out=1.0, omega_in=6.0)),
}


def torsion(L: float, well_spacing: float = 1.5, H: float = 1.0,
            omega_in: float = 6.0, omega_out: float = 1.0, beta: float = 8.0,
            dx: float = 0.01, **kw) -> SepSystem:
    """Periodic CV of length L with L/well_spacing identical wells.

    The local landscape is L-independent by construction, so L is a pure
    TRANSPORT-DISTANCE knob: it is the axis on which prediction P1 (RC-WFR's
    advantage over ABF grows with L, because ABF equilibrates the CV diffusively
    in O(L^2) while W+FR fronts in O(L)) is tested.
    """
    n_wells = int(round(L / well_spacing))
    n = int(round(L / dx)) + 1
    g = Grid1D(-L / 2, L / 2, n, -L / 2, L / 2, bc="periodic")
    p = SysParams(beta=beta, barrier="periodic", H=H, n_wells=n_wells,
                  omega_in=omega_in, omega_out=omega_out, **kw)
    return SepSystem(p, g)


def build(name: str, **param_overrides) -> SepSystem:
    if name.startswith("TORSION_L"):
        return torsion(float(name[len("TORSION_L"):]), **param_overrides)
    spec = SYSTEMS[name]
    p = spec["params"]
    if param_overrides:
        from dataclasses import replace
        p = replace(p, **param_overrides)
    return SepSystem(p, spec["grid"])
