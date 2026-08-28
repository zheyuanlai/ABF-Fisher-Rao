"""Information-target pulse--release Fisher--Rao ABF.

The first campaign is deliberately narrower than the full finite-horizon
allocation theory.  It uses the conditional local-force variance, but no
online IAT, trigger, cooldown, or receding-horizon controller.  A target is
constructed once at the first post-burn-in FR opportunity, held fixed for the
short burst, and then FR is switched off permanently.

Particle reallocation is not implemented here.  The simulator passes this
module's centered ``log(p/q)`` score to :func:`abffr.fr_v3.bd_standard`, the
same fixed-population standard birth--death operator used by clean-v2.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

import numpy as np
import torch

from . import clean_v2 as cv2, fr_v3, oracle_short_burst as osb
from . import torch_utils as tu


EPS = 1e-300
CALIBRATION_KIND = "dose_calibration"
ORACLE_CAMPAIGN_KIND = "oracle_campaign"
KINDS = (CALIBRATION_KIND, ORACLE_CAMPAIGN_KIND)
TARGETS = ("none", "information_oracle", "information_estimated")


@dataclass(frozen=True)
class InformationTarget:
    """Validated switch and fixed target-construction choices."""

    n_cells: int
    report_min: float
    report_max: float
    min_expected_particles_per_cell: float


@dataclass(frozen=True)
class TargetConstruction:
    """A fully auditable information target on one profile grid."""

    edges: np.ndarray
    leverage: np.ndarray
    force_variance: np.ndarray
    masses: np.ndarray
    density: np.ndarray
    risk: float
    uniform_risk: float

    @property
    def risk_ratio(self) -> float:
        return float(self.risk / self.uniform_risk) if self.uniform_risk > 0 else 1.0


def _expected_methods(kind: str) -> set[str]:
    if kind == CALIBRATION_KIND:
        return {"abf_fr_information_oracle"}
    if kind == ORACLE_CAMPAIGN_KIND:
        return {"abf_only", "abf_fr_information_oracle"}
    raise ValueError(f"unknown information_target.kind {kind!r}")


def validate_config(cfg: Mapping) -> List[int]:
    """Validate v1's oracle-first campaign and return exact firing steps."""
    block = cfg.get("information_target", {}) or {}
    if not bool(block.get("enabled", False)):
        raise ValueError("information_target.enabled must be true")
    if bool((cfg.get("clean_v2", {}) or {}).get("enabled", False)):
        raise ValueError("information_target and clean_v2 are mutually exclusive")

    kind = str(block.get("kind", ""))
    if kind not in KINDS:
        raise ValueError(
            f"information_target.kind must be one of {list(KINDS)}; got {kind!r}")

    # Carry forward the clean pulse path's operator-integrity constraints.  We
    # repeat the checks rather than pretending information targets are a
    # clean-v2 physical target.
    fr = cfg.get("fr", {}) or {}
    for key, why in cv2.BANNED_FR_KEYS.items():
        if key in fr:
            raise ValueError(f"information_target forbids fr.{key}: {why}")
    for key in cv2.ZERO_FR_KEYS:
        if key in fr and float(fr[key]) != 0.0:
            raise ValueError(f"information_target requires fr.{key} == 0")
    if not bool(fr.get("interval_scaled_clock", True)):
        raise ValueError("information_target requires fr.interval_scaled_clock: true")
    if bool((cfg.get("selection", {}) or {}).get("write_generic_best", True)):
        raise ValueError("information_target requires selection.write_generic_best: false")
    for retired in ("v3", "v4"):
        if cfg.get(retired):
            raise ValueError(f"information_target forbids the retired {retired}: block")

    methods = set(cfg.get("methods", []))
    expected = _expected_methods(kind)
    if methods != expected:
        raise ValueError(f"{kind} requires methods {sorted(expected)}; got {sorted(methods)}")
    if list(fr.get("target_types", [])) != ["information_oracle"]:
        raise ValueError("v1 oracle-first campaign requires fr.target_types: [information_oracle]")

    sim = cfg.get("simulation", {}) or {}
    domain = cfg.get("domain", {}) or {}
    n_particles = int(sim["n_particles"])
    n_cells = int(block.get("n_cells", 32))
    min_expected = float(block.get("min_expected_particles_per_cell", 1.0))
    if n_cells < 2:
        raise ValueError("information_target.n_cells must be at least 2")
    if min_expected <= 0.0 or n_cells * min_expected > n_particles:
        raise ValueError("the per-cell coverage floor is infeasible for K and n_cells")
    report_min = float(block["report_min"])
    report_max = float(block["report_max"])
    if not (float(domain["x_min"]) <= report_min < report_max <= float(domain["x_max"])):
        raise ValueError("information_target reporting interval must lie in the x domain")

    update_every = max(1, int((cfg.get("abf", {}) or {}).get("update_every", 1)))
    every_values = [int(v) for v in fr.get("fr_every_values", [])]
    if len(every_values) != 1 or every_values[0] <= 0:
        raise ValueError("information_target requires one positive FR spacing")
    if every_values[0] % update_every:
        raise ValueError("FR spacing must be a multiple of abf.update_every")

    gammas = [float(v) for v in fr.get("gamma_values", [])]
    if not gammas or any(g <= 0.0 for g in gammas):
        raise ValueError("every information-target gamma must be positive")
    if kind == ORACLE_CAMPAIGN_KIND:
        if len(gammas) != 1:
            raise ValueError("oracle campaign requires exactly one calibrated gamma")
        calibrated = float(block.get("calibrated_gamma", float("nan")))
        if not np.isfinite(calibrated) or calibrated != gammas[0]:
            raise ValueError("campaign gamma must equal information_target.calibrated_gamma")

    n_steps = int(sim["n_steps"])
    burnins = [float(v) for v in fr.get("burnin_fractions", [])]
    durations = [float(v) for v in fr.get("duration_fractions", [])]
    if len(burnins) != 1 or len(durations) != 1:
        raise ValueError("information_target requires one burn-in and duration")
    burn, stop = burnins[0], burnins[0] + durations[0]
    if not 0.0 < burn < stop < 1.0:
        raise ValueError("information_target requires burn-in, a finite burst, and release")
    steps = cv2.firing_steps(n_steps, burn, stop, every_values[0])
    expected_count = 1 if kind == CALIBRATION_KIND else 3
    if len(steps) != expected_count:
        raise ValueError(f"{kind} requires exactly {expected_count} pulse(s); got {steps}")
    frozen_steps = [int(v) for v in block.get("expected_firing_steps", [])]
    if steps != frozen_steps:
        raise ValueError(f"firing steps {steps} do not match frozen steps {frozen_steps}")
    return steps


def from_config(cfg: Mapping) -> Optional[InformationTarget]:
    block = cfg.get("information_target", {}) or {}
    if not bool(block.get("enabled", False)):
        return None
    validate_config(cfg)
    return InformationTarget(
        n_cells=int(block.get("n_cells", 32)),
        report_min=float(block["report_min"]),
        report_max=float(block["report_max"]),
        min_expected_particles_per_cell=float(
            block.get("min_expected_particles_per_cell", 1.0)),
    )


def validate_dose_receipt(cfg: Mapping, root: str | Path) -> Path:
    """Require the campaign dose to match its mechanism-only receipt."""
    block = cfg.get("information_target", {}) or {}
    if str(block.get("kind", "")) != ORACLE_CAMPAIGN_KIND:
        raise ValueError("dose receipt is required only for the oracle campaign")
    raw = str(block.get("dose_receipt", ""))
    if not raw:
        raise ValueError("oracle campaign requires information_target.dose_receipt")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(root) / path
    if not path.is_file():
        raise ValueError(f"information-target dose receipt does not exist: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "SELECTED" or receipt.get("selected") is None:
        raise ValueError("dose receipt does not authorize a campaign")
    selected = float(receipt["selected"]["gamma"])
    configured = float((cfg.get("fr", {}) or {})["gamma_values"][0])
    if selected != configured:
        raise ValueError(f"campaign gamma {configured:g} does not match receipt {selected:g}")
    return path


# Dose calibration and PASS/FAIL logic are target-agnostic mechanism rules.
summarize_doses = osb.summarize_doses
select_dose = osb.select_dose
classify_campaign = osb.classify_stage_a


def cell_indices(x_grid: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign profile nodes to half-open cells, including the right endpoint."""
    idx = np.searchsorted(edges, np.asarray(x_grid, dtype=float), side="right") - 1
    return np.clip(idx, 0, len(edges) - 2).astype(int)


def integration_leverage(x_grid: np.ndarray, edges: np.ndarray,
                         report_min: float, report_max: float) -> np.ndarray:
    """Diagonal free-energy integration leverage for piecewise-constant cells.

    ``H`` is the exact cumulative-trapezoid map on ``x_grid``.  ``B`` expands a
    cell-wise mean-force error to the profile grid.  The reported free energy
    is centered by its quadrature-weighted mean on the predeclared interval;
    the returned vector is ``diag((PHB)' W (PHB))``.  No reference free energy
    or thermal-scope mask enters this calculation.
    """
    x = np.asarray(x_grid, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if x.ndim != 1 or len(x) < 3 or np.any(np.diff(x) <= 0):
        raise ValueError("x_grid must be a strictly increasing 1-D grid")
    if not (x[0] <= report_min < report_max <= x[-1]):
        raise ValueError("reporting interval must lie inside x_grid")

    g = len(x)
    h = np.zeros((g, g), dtype=float)
    for i in range(1, g):
        widths = np.diff(x[:i + 1])
        h[i, :i] += 0.5 * widths
        h[i, 1:i + 1] += 0.5 * widths

    idx = cell_indices(x, edges)
    b = np.zeros((g, len(edges) - 1), dtype=float)
    b[np.arange(g), idx] = 1.0
    t = h @ b

    mask = (x >= float(report_min)) & (x <= float(report_max))
    xm = x[mask]
    if len(xm) < 2:
        raise ValueError("reporting interval contains fewer than two grid nodes")
    wm = np.empty_like(xm)
    wm[0] = 0.5 * (xm[1] - xm[0])
    wm[-1] = 0.5 * (xm[-1] - xm[-2])
    if len(xm) > 2:
        wm[1:-1] = 0.5 * (xm[2:] - xm[:-2])
    w = np.zeros(g, dtype=float)
    w[mask] = wm
    mean = (w[:, None] * t).sum(axis=0) / w.sum()
    centered = t - mean[None, :]
    leverage = (w[:, None] * centered ** 2).sum(axis=0)
    return np.maximum(leverage, 0.0)


def aggregate_variance(x_grid: np.ndarray, variance_grid: np.ndarray,
                       edges: np.ndarray) -> np.ndarray:
    """Average a conditional-force variance profile into allocation cells."""
    x = np.asarray(x_grid, dtype=float)
    v = np.asarray(variance_grid, dtype=float)
    if v.shape != x.shape or np.any(~np.isfinite(v)):
        raise ValueError("force-variance profile must be finite on x_grid")
    v = np.maximum(v, 0.0)
    idx = cell_indices(x, edges)
    out = np.zeros(len(edges) - 1, dtype=float)
    for j in range(len(out)):
        values = v[idx == j]
        if values.size:
            out[j] = float(values.mean())
    return out


def lower_bounded_masses(weights: np.ndarray, n_particles: int,
                         min_expected_particles_per_cell: float = 1.0) -> np.ndarray:
    """Solve ``q_j=max(floor,c*w_j)`` with ``sum q=1`` by bisection."""
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if w.ndim != 1 or np.any(~np.isfinite(w)):
        raise ValueError("information weights must be a finite 1-D array")
    floor = float(min_expected_particles_per_cell) / int(n_particles)
    if floor <= 0.0 or len(w) * floor > 1.0:
        raise ValueError("infeasible information-target coverage floor")
    if not np.any(w > 0.0):
        return np.full(len(w), 1.0 / len(w))
    if len(w) * floor == 1.0:
        return np.full(len(w), floor)

    # Find a finite upper bracket by doubling from the unconstrained scale.
    lo, hi = 0.0, 1.0 / max(float(w.sum()), EPS)
    while np.maximum(floor, hi * w).sum() < 1.0:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.maximum(floor, mid * w).sum() < 1.0:
            lo = mid
        else:
            hi = mid
    q = np.maximum(floor, 0.5 * (lo + hi) * w)
    q /= q.sum()
    return q


def density_from_masses(x_grid: np.ndarray, edges: np.ndarray,
                        masses: np.ndarray) -> np.ndarray:
    """Render cell masses as a positive piecewise-constant grid density."""
    x = np.asarray(x_grid, dtype=float)
    edges = np.asarray(edges, dtype=float)
    masses = np.asarray(masses, dtype=float)
    widths = np.diff(edges)
    if len(masses) != len(widths) or np.any(widths <= 0):
        raise ValueError("cell masses and edges are inconsistent")
    density = masses[cell_indices(x, edges)] / widths[cell_indices(x, edges)]
    z = np.trapezoid(density, x)
    if not np.isfinite(z) or z <= 0.0:
        raise ValueError("information target cannot be normalized")
    return density / z


def build_target(x_grid: np.ndarray, variance_grid: np.ndarray, *,
                 x_min: float, x_max: float, n_cells: int,
                 report_min: float, report_max: float, n_particles: int,
                 min_expected_particles_per_cell: float = 1.0,
                 leverage: Optional[np.ndarray] = None) -> TargetConstruction:
    """Construct ``q_j=max(1/K,c*sqrt(a_j*sigma_j^2))`` once."""
    edges = np.linspace(float(x_min), float(x_max), int(n_cells) + 1)
    lev = (integration_leverage(x_grid, edges, report_min, report_max)
           if leverage is None else np.asarray(leverage, dtype=float))
    var = aggregate_variance(x_grid, variance_grid, edges)
    coeff = np.maximum(lev * var, 0.0)
    masses = lower_bounded_masses(
        np.sqrt(coeff), n_particles, min_expected_particles_per_cell)
    density = density_from_masses(x_grid, edges, masses)
    risk = float(np.sum(coeff / masses))
    uniform = np.full(len(masses), 1.0 / len(masses))
    uniform_risk = float(np.sum(coeff / uniform))
    return TargetConstruction(edges, lev, var, masses, density, risk, uniform_risk)


def score(p_hat: torch.Tensor, q_grid: torch.Tensor, X: torch.Tensor,
          x0: float, dx: float):
    """Centered Fisher--Rao ``log(p/q)`` score for a frozen density target."""
    p_at = tu.interp1d(p_hat, X, x0, dx)
    q_at = tu.interp1d(q_grid, X, x0, dx)
    floored = ((p_at <= EPS) | (q_at <= EPS)).to(p_at.dtype).mean(dim=1)
    log_p = torch.log(p_at.clamp_min(EPS))
    log_q = torch.log(q_at.clamp_min(EPS))
    r = log_p - log_q
    return r - r.mean(dim=1, keepdim=True), log_p, log_q, floored


def row_score(log_p_at: torch.Tensor, log_q_at: torch.Tensor) -> fr_v3.FRScore:
    return fr_v3.FRScore(log_p=log_p_at, log_q=log_q_at)


__all__ = [
    "CALIBRATION_KIND", "ORACLE_CAMPAIGN_KIND", "KINDS", "TARGETS",
    "InformationTarget", "TargetConstruction", "aggregate_variance",
    "build_target", "cell_indices", "classify_campaign",
    "density_from_masses", "from_config", "integration_leverage",
    "lower_bounded_masses", "row_score", "score", "select_dose",
    "summarize_doses", "validate_config", "validate_dose_receipt",
]
