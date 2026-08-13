"""Run the gate classifier end to end on synthetic screen output, before it classifies anything.

``nacl_gates.py`` consumes the accepted reference and the screen's cell files and emits the
regime verdict -- discovery-limited, establishment-limited, ABF-sufficient -- plus the
mechanical cell selection.  Its pieces have unit tests; the whole path had never run.

Three worlds are planted, each with a known correct answer:

  ABF-sufficient        every state found early, occupancy tracks the bias-aware target
  discovery-limited     the SSIP state is never visited
  establishment-limited found early, but occupancy sits far below the target in the 2nd half

and the classifier must return each one.  A classifier that cannot distinguish planted worlds
cannot be trusted to classify the real one.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from nacl import system as nsys                                  # noqa: E402

N_GRID = 61
R_LO, R_HI = 0.20, 1.40
CIP_HI, SSIP_HI = 0.36, 0.70


def write_reference(path):
    """A two-basin reference with the analysis report the classifier requires."""
    os.makedirs(path, exist_ok=True)
    kT = nsys.kT_kJ()
    r = np.round(np.linspace(R_LO, R_HI, N_GRID), 6)
    F = (-6.0 * kT * np.exp(-((r - 0.28) / 0.03) ** 2)
         - 3.0 * kT * np.exp(-((r - 0.50) / 0.04) ** 2))
    np.savez(os.path.join(path, "reference.npz"), r_nm=r, F_ref=F, W_ref=F, f_cons=np.gradient(F, r),
             endpoint_window=np.ones(N_GRID, dtype=bool))
    basins = [dict(index=0, label="CIP", r_lo_nm=R_LO, r_hi_nm=CIP_HI, r_min_nm=0.28),
              dict(index=1, label="SSIP", r_lo_nm=CIP_HI, r_hi_nm=R_HI, r_min_nm=0.50)]
    json.dump(dict(acceptance=dict(ACCEPTED=True, ratio=0.21),
                   completeness=dict(COMPLETE=True),
                   gate0=dict(COMPUTABLE=True, global_spread_ratio=0.04,
                              barrier_region_ratio=0.06, coverage={}),
                   gateA=dict(COMPUTABLE=True, PASS=True, max_TV=0.71),
                   basins=basins),
              open(os.path.join(path, "reference_report.json"), "w"))
    return r, F, basins


def write_cell(path, N, world, r_ref, F_ref, n_seeds=8, T_ps=1000.0, n_cp=21):
    """One screen cell whose traces encode a known regime."""
    os.makedirs(path, exist_ok=True)
    grid = r_ref.copy()
    dt = 0.002
    n_frames = 60
    steps = (np.arange(n_frames) * int(round(T_ps / dt / n_frames))).astype(int)

    xi = np.full((n_frames, n_seeds, N), 0.28)                    # everyone starts in CIP
    if world != "discovery-limited":
        xi[3:, :, : max(1, N // 2)] = 0.50                        # half the walkers reach SSIP fast

    occ = np.zeros((n_cp, n_seeds, len(grid)))
    cip_bin = int(np.argmin(np.abs(grid - 0.28)))
    ssip_bin = int(np.argmin(np.abs(grid - 0.50)))
    for c in range(n_cp):
        for s in range(n_seeds):
            if world == "ABF-sufficient":
                occ[c, s, cip_bin] = 55.0
                occ[c, s, ssip_bin] = 45.0                        # near the target split
            elif world == "discovery-limited":
                occ[c, s, cip_bin] = 100.0
            else:                                                 # establishment-limited
                occ[c, s, cip_bin] = 97.0
                occ[c, s, ssip_bin] = 3.0                         # present but far under target
    pmf = np.zeros((n_cp, n_seeds, len(grid)))                    # no learned bias yet
    np.savez(os.path.join(path, f"cell_N{N}.npz"),
             N=N, T_ns=T_ps / 1000.0, n_steps=int(T_ps / dt), dt_ps=dt,
             seed_labels=np.arange(4000, 4000 + n_seeds), box_L_nm=2.8927, R_hi_nm=R_HI,
             grid=grid, dz=float(grid[1] - grid[0]),
             mean_force=np.zeros((n_seeds, len(grid))), pmf=np.zeros((n_seeds, len(grid))),
             eff_counts=np.full((n_seeds, len(grid)), 1000.0),
             xi_trace=xi, xi_steps=steps,
             y_trace=np.zeros((5, n_seeds * N, 3)), y_steps=np.arange(5),
             diag_occupancy=occ, diag_pmf=pmf,
             diag_times=np.linspace(0, T_ps, n_cp),
             diag_steps=np.linspace(0, T_ps / dt, n_cp).astype(int))


def run_gates(screen, ref):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "nacl_gates.py"),
                        "--screen", screen, "--ref", ref],
                       capture_output=True, text=True, cwd=ROOT)
    return r, (json.load(open(os.path.join(screen, "gates_report.json")))
               if os.path.exists(os.path.join(screen, "gates_report.json")) else None)


def test_the_planted_worlds_are_unambiguously_in_their_regimes():
    """Assert the planting, not just the label.

    A parametrised test that only checks the classifier's answer will keep passing if the
    reference or the occupancies drift until a 'world' no longer sits in the regime it is
    named after -- at which point the suite is testing something other than what it says.
    So compute the bias-aware target here and require a decisive margin either side.
    """
    beta = nsys.beta_per_kJ()
    kT = nsys.kT_kJ()
    r = np.round(np.linspace(R_LO, R_HI, N_GRID), 6)
    F = (-6.0 * kT * np.exp(-((r - 0.28) / 0.03) ** 2)
         - 3.0 * kT * np.exp(-((r - 0.50) / 0.04) ** 2))
    w = np.exp(-beta * (F - F.min()))
    ssip = (r >= CIP_HI) & (r <= R_HI)
    Q_ssip = float(w[ssip].sum() / w.sum())
    thresh = 0.5 * Q_ssip

    P_sufficient = 45.0 / 100.0
    P_deficient = 3.0 / 100.0
    assert P_sufficient > 3.0 * thresh, "the 'ABF-sufficient' world is not clearly established"
    assert P_deficient < 0.5 * thresh, "the 'establishment-limited' world is not clearly deficient"


@pytest.mark.parametrize("world,expect", [
    ("ABF-sufficient", "ABF-sufficient"),
    ("discovery-limited", "discovery-limited"),
    ("establishment-limited", "establishment-limited"),
])
def test_classifier_recovers_each_planted_world(world, expect):
    with tempfile.TemporaryDirectory() as tmp:
        ref, screen = os.path.join(tmp, "ref"), os.path.join(tmp, "screen")
        r, F, _ = write_reference(ref)
        write_cell(screen, 64, world, r, F)
        proc, rep = run_gates(screen, ref)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        assert rep is not None
        verdict = rep["cells"]["N64"]["verdict"]
        assert expect in verdict, f"planted {world}, classifier said: {verdict}"


def test_smallest_passing_cell_is_selected_mechanically():
    """If several cells are eligible the rule is the SMALLEST N, never the largest error."""
    with tempfile.TemporaryDirectory() as tmp:
        ref, screen = os.path.join(tmp, "ref"), os.path.join(tmp, "screen")
        r, F, _ = write_reference(ref)
        for N in (16, 32, 64):
            write_cell(screen, N, "establishment-limited", r, F)
        proc, rep = run_gates(screen, ref)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        assert rep["selection"]["eligible_cells"] == [16, 32, 64]
        assert rep["selection"]["chosen_N"] == 16


def test_refuses_to_run_when_gate_A_is_not_computable():
    with tempfile.TemporaryDirectory() as tmp:
        ref, screen = os.path.join(tmp, "ref"), os.path.join(tmp, "screen")
        r, F, _ = write_reference(ref)
        rep = json.load(open(os.path.join(ref, "reference_report.json")))
        rep["gateA"] = dict(COMPUTABLE=False, PASS=None, basin_sample_counts=[3, 0])
        json.dump(rep, open(os.path.join(ref, "reference_report.json"), "w"))
        write_cell(screen, 64, "ABF-sufficient", r, F)
        proc, _ = run_gates(screen, ref)
        assert proc.returncode != 0
        assert "NOT COMPUTABLE" in proc.stdout + proc.stderr


def test_refuses_an_unaccepted_reference():
    with tempfile.TemporaryDirectory() as tmp:
        ref, screen = os.path.join(tmp, "ref"), os.path.join(tmp, "screen")
        r, F, _ = write_reference(ref)
        rep = json.load(open(os.path.join(ref, "reference_report.json")))
        rep["acceptance"]["ACCEPTED"] = False
        json.dump(rep, open(os.path.join(ref, "reference_report.json"), "w"))
        write_cell(screen, 64, "ABF-sufficient", r, F)
        proc, _ = run_gates(screen, ref)
        assert proc.returncode != 0
        assert "NOT accepted" in proc.stdout + proc.stderr
