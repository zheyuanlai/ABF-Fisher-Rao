"""Amendment 3 (A3.3 b) hybrid projection: pure parts (W_hyb construction, agreement label, decision wiring)."""
import math
import os
import sys

import numpy as np

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", p) for p in ("scripts/final_campaign",
                                                                           "scripts/equal_budget", "scripts", "src")]
import expB_hybrid as H  # noqa: E402
import expB_stats as S  # noqa: E402


def test_hybrid_values_scales_tau_by_budget_and_measured_speed():
    tau = {"fr512": {1: 0.1, 2: math.inf}, "abf16": {1: 0.5, 2: 0.25}, "abf64": {}}
    vals, comp = H.hybrid_values(tau, {"fr512": 2e-9, "abf16": 1e-8, "abf64": 1e-9}, B=1e9)
    assert vals["fr512"] == {1: 0.1 * 1e9 * 2e-9, 2: math.inf}
    assert vals["abf16"] == {1: 5.0, 2: 2.5} and comp["abf16"] == {1: 10.0, 2: 10.0}
    assert "abf64" not in vals                                  # no validated tau -> omitted


def test_hybrid_values_skips_arm_without_measured_speed():
    vals, _ = H.hybrid_values({"fr512": {1: 0.1}}, {"fr512": None}, B=1.0)
    assert vals == {}


def test_agreement_labels():
    assert H.agreement("FR advantage", "FR advantage") == "robust to the backend's law"
    assert H.agreement("unresolved", "FR advantage") == "law-sensitive"
    assert H.agreement(None, "unresolved") == "not evaluable"
    assert H.agreement("INCOMPLETE", "FR advantage") == "not evaluable"      # an incomplete stage has no decision


def test_decision_on_hybrid_values_runs_through_compare_arms():
    rng = np.random.default_rng(0)
    tau_f = {s: float(u) for s, u in enumerate(rng.uniform(0.05, 0.1, 32))}
    tau_a = {s: float(u) for s, u in enumerate(rng.uniform(0.5, 0.9, 32))}
    vals, comp = H.hybrid_values({"fr512": tau_f, "abf16": tau_a}, {"fr512": 1e-9, "abf16": 1e-9}, B=1e9)
    r = S.compare_arms(vals, ["fr512"], ["abf16"], pairs=[], completion=comp, n_boot=2000, gehan_perm=200)
    assert r["decision"]["decision"].startswith("FR")
