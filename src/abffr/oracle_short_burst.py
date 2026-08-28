"""Protocol guards for the oracle physical-target short-burst experiment.

This module does not implement a new reallocation operator. The campaign uses
``clean_v2`` and its unchanged ``fr_v3.bd_standard`` path. The only new
object is a schedule: one oracle-target pulse for mechanism-only dose
calibration, or three oracle-target pulses followed by permanent release to
plain ABF.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np

from . import clean_v2


CALIBRATION_KIND = "dose_calibration"
CAMPAIGN_KIND = "stage_a_campaign"
KINDS = (CALIBRATION_KIND, CAMPAIGN_KIND)

# Mechanism-only calibration gates. None uses a free-energy error outcome.
MIN_EVENT_FRACTION = 0.01
MAX_EVENT_FRACTION = 0.05
MAX_MEDIAN_KL_RATIO = 0.99
MIN_KL_DECREASE_FRACTION = 0.625
MIN_SINGLE_PULSE_ESS = 0.95


@dataclass(frozen=True)
class DoseSummary:
    gamma: float
    dtau: float
    n_seeds: int
    median_event_fraction: float
    median_kl_ratio: float
    kl_decrease_fraction: float
    median_ess_after: float
    max_logp_floored_fraction: float
    eligible: bool

    def to_row(self) -> Dict:
        return asdict(self)


def _expected_methods(kind: str) -> set[str]:
    if kind == CALIBRATION_KIND:
        return {"abf_fr_physical_oracle"}
    if kind == CAMPAIGN_KIND:
        return {"abf_only", "abf_fr_physical_oracle"}
    raise ValueError(f"unknown oracle_short_burst.kind {kind!r}")


def validate_config(cfg: Mapping) -> List[int]:
    """Validate the campaign wrapper and return its exact firing steps."""
    block = cfg.get("oracle_short_burst", {}) or {}
    if not bool(block.get("enabled", False)):
        raise ValueError("oracle_short_burst.enabled must be true")
    kind = str(block.get("kind", ""))
    if kind not in KINDS:
        raise ValueError(
            f"oracle_short_burst.kind must be one of {list(KINDS)}; got {kind!r}")

    clean_v2.validate_config(dict(cfg))
    fr = cfg.get("fr", {}) or {}
    methods = set(cfg.get("methods", []))
    expected_methods = _expected_methods(kind)
    if methods != expected_methods:
        raise ValueError(
            f"{kind} requires methods {sorted(expected_methods)}; got "
            f"{sorted(methods)}")
    if list(fr.get("target_types", [])) != ["physical_oracle"]:
        raise ValueError(
            "oracle short-burst admits only fr.target_types: "
            "[physical_oracle]")

    gammas = [float(x) for x in fr.get("gamma_values", [])]
    if not gammas or any(g <= 0.0 for g in gammas):
        raise ValueError("every oracle short-burst gamma must be positive")
    if kind == CAMPAIGN_KIND and len(gammas) != 1:
        raise ValueError("Stage A requires exactly one calibrated gamma")
    if kind == CAMPAIGN_KIND:
        calibrated = float(block.get("calibrated_gamma", float("nan")))
        if not np.isfinite(calibrated) or calibrated != gammas[0]:
            raise ValueError(
                "Stage A gamma must equal oracle_short_burst.calibrated_gamma "
                "from the mechanism-only receipt")

    sim = cfg.get("simulation", {}) or {}
    n_steps = int(sim["n_steps"])
    burnins = [float(x) for x in fr.get("burnin_fractions", [])]
    durations = [float(x) for x in fr.get("duration_fractions", [])]
    every_values = [int(x) for x in fr.get("fr_every_values", [])]
    if len(burnins) != 1 or len(durations) != 1 or len(every_values) != 1:
        raise ValueError(
            "oracle short-burst requires one burn-in, duration, and FR spacing")
    burn = burnins[0]
    stop = burn + durations[0]
    steps = clean_v2.firing_steps(n_steps, burn, stop, every_values[0])
    expected_count = 1 if kind == CALIBRATION_KIND else 3
    if len(steps) != expected_count:
        raise ValueError(
            f"{kind} requires exactly {expected_count} pulse(s); got {steps}")

    expected_steps = [int(x) for x in block.get("expected_firing_steps", [])]
    if steps != expected_steps:
        raise ValueError(
            f"firing steps {steps} do not match frozen expected steps "
            f"{expected_steps}")
    return steps


def validate_dose_receipt(cfg: Mapping, root: str | Path) -> Path:
    """Require Stage A's gamma to match the durable calibration receipt."""
    block = cfg.get("oracle_short_burst", {}) or {}
    if str(block.get("kind", "")) != CAMPAIGN_KIND:
        raise ValueError("dose receipts are required only for Stage A")
    raw = str(block.get("dose_receipt", ""))
    if not raw:
        raise ValueError("Stage A requires oracle_short_burst.dose_receipt")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(root) / path
    if not path.is_file():
        raise ValueError(f"Stage A dose receipt does not exist: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "SELECTED" or receipt.get("selected") is None:
        raise ValueError("Stage A dose receipt does not authorize a campaign")
    selected = float(receipt["selected"]["gamma"])
    configured = float((cfg.get("fr", {}) or {})["gamma_values"][0])
    if selected != configured:
        raise ValueError(
            f"Stage A gamma {configured:g} does not match receipt {selected:g}")
    return path


def summarize_doses(rows: Iterable[Mapping]) -> List[DoseSummary]:
    """Summarize one-pulse calibration rows and apply mechanism-only gates."""
    grouped: Dict[float, List[Mapping]] = {}
    for row in rows:
        grouped.setdefault(float(row["gamma"]), []).append(row)
    summaries: List[DoseSummary] = []
    for gamma in sorted(grouped):
        group = grouped[gamma]
        seeds = {int(r["seed"]) for r in group}
        if len(group) != len(seeds):
            raise ValueError(
                f"gamma {gamma:g} has {len(group)} rows for {len(seeds)} seeds; "
                "dose calibration requires exactly one pulse per seed")
        event = np.asarray([float(r["event_fraction"]) for r in group])
        before = np.asarray([float(r["kl_before"]) for r in group])
        after = np.asarray([float(r["kl_after"]) for r in group])
        if np.any(~np.isfinite(before)) or np.any(before <= 0.0):
            raise ValueError(f"gamma {gamma:g} has invalid pre-pulse KL values")
        ratio = after / before
        ess = np.asarray([float(r["ess_anc_after"]) for r in group])
        floor = np.asarray(
            [float(r["logp_floored_fraction"]) for r in group])
        dtau = np.asarray([float(r["dtau"]) for r in group])
        med_event = float(np.median(event))
        med_ratio = float(np.median(ratio))
        decrease = float(np.mean(after < before))
        med_ess = float(np.median(ess))
        max_floor = float(np.max(floor))
        eligible = bool(
            MIN_EVENT_FRACTION <= med_event <= MAX_EVENT_FRACTION
            and med_ratio <= MAX_MEDIAN_KL_RATIO
            and decrease >= MIN_KL_DECREASE_FRACTION
            and med_ess >= MIN_SINGLE_PULSE_ESS
            and max_floor == 0.0
        )
        summaries.append(DoseSummary(
            gamma=float(gamma), dtau=float(np.median(dtau)),
            n_seeds=len(seeds), median_event_fraction=med_event,
            median_kl_ratio=med_ratio, kl_decrease_fraction=decrease,
            median_ess_after=med_ess,
            max_logp_floored_fraction=max_floor, eligible=eligible))
    return summaries


def select_dose(rows: Iterable[Mapping]) -> tuple[DoseSummary, List[DoseSummary]]:
    """Return the smallest mechanism-eligible dose; never inspect FEC error."""
    summaries = summarize_doses(rows)
    eligible = [s for s in summaries if s.eligible]
    if not eligible:
        raise ValueError("no mechanism-eligible oracle short-burst dose")
    return min(eligible, key=lambda s: s.gamma), summaries


def classify_stage_a(*, acceleration_pass: bool, endpoint_pass: bool,
                     genealogy_pass: bool, mechanism_pass: bool,
                     pulse_count_pass: bool, censoring_pass: bool) -> str:
    """Predeclared PASS/FAIL/INCONCLUSIVE logic for the oracle diagnostic."""
    identified = (
        genealogy_pass and mechanism_pass and pulse_count_pass and censoring_pass)
    if not identified:
        return "INCONCLUSIVE"
    return "PASS" if acceleration_pass and endpoint_pass else "FAIL"


__all__ = [
    "CALIBRATION_KIND", "CAMPAIGN_KIND", "DoseSummary", "KINDS",
    "classify_stage_a", "select_dose", "summarize_doses", "validate_config",
    "validate_dose_receipt",
]
