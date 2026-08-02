"""ACCEPTED STAGE-0 CHECKPOINT for Ace-Val-Nme.  Frozen 2026-08-01.

Everything in this module is a *decision that has already been measured*, not a tunable.  It
lives in code rather than only in a handoff so that a later run cannot quietly disagree with
the checkpoint: the screening scripts import these constants and assert against them.

What Stage 0 established, and where
-----------------------------------
``results/valine/chi1_profiles``   gate V1 PASS -- chi1 barriers 11.3-17.9 kT at six clamped
                                  backbone points, dominant rotamer identity changes with the
                                  backbone (t at C7eq/alphaL/C7ax, g+ at C5, g- at alphaR/bridge)
``results/valine/psi_profiles``    sec.32 for (phi,chi1): hidden coordinate psi has ONE populated
                                  state at all six anchors -> admissible
``results/valine/phi_profiles``    sec.32 for (psi,chi1): hidden coordinate phi has TWO populated
                                  states at (psi=-30, chi1=g-), wells at -63 and +57 deg split by
                                  a 30.8 kT barrier, minor state 26 % -> REJECTED

What Stage 0 did NOT establish
------------------------------
Nothing above says mFR will help.  The decisive gate is V3: does ABF *discover* every relevant
state and then leave one persistently *under-established*?  Until V3 is measured, no reference
FES and no mFR arm should be built -- that ordering is the whole point of the checkpoint.
"""
from __future__ import annotations

from .system import CHI1_ATOMS, PHI_ATOMS, PSI_ATOMS

#: The selected collective variable.  ``(psi, chi1)`` is REJECTED (see module docstring); the
#: alanine CV ``(phi, psi)`` is retained only as a control and hides the 11-18 kT chi1 barrier.
SELECTED_CV = "phi_chi1"
SELECTED_CV_ATOMS = (PHI_ATOMS, CHI1_ATOMS)
REJECTED_CV = {"psi_chi1": "hidden coordinate phi carries two populated states (30.8 kT apart)"}

#: Unrestrained dynamics (ABF, mFR, exploration).  1 fs is MANDATORY, not chosen: the torch
#: integrator implements no SHAKE and ``extract_parameters`` refuses a constrained system.
DT_UNRESTRAINED_PS = 0.001

#: Restrained dynamics (umbrella windows).  0.5 fs.
#:
#: The ORIGINAL justification -- that the stiff dihedral clamp is under-integrated at 1 fs while
#: the unrestrained system is fine -- is REFUTED; see CORRECTIONS below and
#: results/valine/dt_bias/.  Measured at 2048 walkers per group (kinetic sampling sigma 1.02 K,
#: against the 5.79 K of the B=64 measurement that produced the original claim):
#:
#:     dt      T_kin - 300 K                    T_bond - 300 K        T_angle - 300 K
#:             unrestrained / clamp500 / clamp110
#:     1.0 fs  -7.01 / -6.63 / -6.88            +25.8 / +27.7 / +26.6   -59.9 / -60.1 / -59.7
#:     0.5 fs  -1.70 / -1.56 / -1.72            +24.9 / +26.2 / +25.7   -59.4 / -59.5 / -59.4
#:
#: The kinetic deficit is INDEPENDENT of the restraint and scales as O(dt^2) (7.01/1.70 = 4.1).
#: It is the integrator, not the clamp.  The equipartition estimators on bonds and angles are
#: essentially UNCHANGED between the two timesteps (0.9 K and 0.5 K of a 26 K and 60 K offset),
#: so their offsets are static properties of a curvilinear-coordinate estimator -- internal
#: coordinates are not independent normal modes -- and NOT a temperature error, which would have
#: scaled with dt exactly as T_kin does.  The CONFIGURATIONAL distribution is therefore not
#: corrupted by the timestep, and free energies at 1 fs are sound.
#:
#: 0.5 fs is kept anyway, for two reasons that are about discipline rather than physics: it is
#: the value the screening plan froze, and halving an already-small kinetic artifact is cheap
#: insurance on a reference MBAR cannot unwind.  Relaxing a frozen value mid-study to save
#: wall-clock is the wrong trade.
DT_RESTRAINED_PS = 0.0005

#: Periodic grid.  MUST be odd -- ``alanine.projection.require_odd_grid`` raises otherwise,
#: because the Nyquist row k = n/2 has no representable derivative and ``gB == grad B`` stops
#: holding exactly.  97 is the alanine value, kept for comparability.
N_GRID = 97

#: Frozen physical model, identical to alanine so that Ala -> Val is the only change.
PHYSICAL_MODEL = ("vacuum ff14SB, NoCutoff, no constraints, no HMR, BAOAB, "
                  "gamma=1/ps, T=300K, float64, IUPAC dihedrals")
PARAM_HASH = "86622b245bb0"          # differs from alanine's 6ffd00dc241f, as it must

#: Estimator settings, frozen at the accepted alanine values.  No retuning: the whole point of
#: reusing them is that a Val result is comparable with the alanine null.
ESTIMATOR = dict(abf_bandwidth=0.08, kde_bandwidth=0.15, abf_min_count=200.0,
                 abf_force_clip=200.0, project_every=50, estimator_stride=1)

#: Genealogy gates.  The alanine values, which are STRICTER than the Val plan's 0.2-0.25 / 0.1,
#: and which the alanine pilot passed at 0.956-0.966 and 0.0015-0.0034.  Keeping them costs
#: nothing and preserves comparability.
GENEALOGY_GATES = dict(ess_age_over_N_min=0.30, wmax_max=0.05)

#: Measured Stage-0 verdicts, so a later analysis cannot restate them differently.
STAGE0_RESULTS = {
    "gate_V1": "PASS -- chi1 barrier >= 2 kT.  The NUMBER 11.3-17.9 kT is a CLAMPED conditional "
               "barrier; see CORRECTIONS['chi1_barrier_is_clamped'] for the free-backbone value",
    "gate_sec32_phi_chi1": "PASS -- one populated state in hidden psi at all six anchors",
    "gate_sec32_psi_chi1": "FAIL -- two populated states in hidden phi at (psi=-30, chi1=g-); "
                           "corroborated by S1, which saw 4 phi crossings in 2581 ns",
    "gate_distinguishability": "PASS -- balanced accuracy 0.973 recovering the 3-D state from "
                               "(phi, chi1); worst footprint overlap 0.189",
    "gate_V3": "NOT YET MEASURED -- discovery vs establishment; decides the study",
}

#: Measurements that CONTRADICT a claim in `VALINE_STAGE0_HANDOFF.md`.  Kept here rather than
#: only in the handoff so that code reading the checkpoint sees the correction too.
CORRECTIONS = {
    "chi1_barrier_is_clamped": (
        "Handoff sec.3 argues that a ~10 kT chi1 barrier is 'essentially never crossed by "
        "unbiased dynamics', and concludes a chi1-hidden design would have been "
        "discovery-limited -- called there the single most important Stage-0 result.  The S1 "
        "exploration measures the opposite: 2.70 chi1 rotamer changes per walker per ns, FLAT "
        "from 30 to 300 ps and identical for walkers seeded in a well (2.70) and on a barrier "
        "(2.69), i.e. an equilibrium rate.  That implies 6-8 kT with the backbone FREE against "
        "11.3-17.9 kT measured with phi and psi clamped at kappa=500 kJ/mol/rad^2.  The clamp "
        "forbids the backbone relaxation that accompanies the rotation.  This also explains R1's "
        "'honest surprise' that the umbrella barriers exceeded their own MEP upper bound: both "
        "clamp the backbone, so they share the defect.  V1 still passes; the discovery-limited "
        "argument for putting chi1 in the CV does not survive."),
    "the_slow_coordinate_is_phi": (
        "S1 saw 4 persistent phi-megabasin crossings in 2581 ns of aggregate unbiased sampling, "
        "against 2.7 chi1 crossings per walker per ns.  The transition matrix is strictly "
        "block-diagonal in sign(phi).  phi, not chi1, is the coordinate ABF must flatten -- and "
        "phi is in the selected CV."),
    "kinetic_temperature_is_not_the_restraint": (
        "Handoff R4 attributes a 6.8 K kinetic-temperature deficit to the stiff dihedral clamp, "
        "from a B=64 measurement whose sampling sigma was 5.79 K -- too noisy to resolve the "
        "unrestrained deficit, which was therefore read as absent.  At B=2048 per group "
        "(sigma 1.02 K) the deficit at dt=1 fs is -7.01 K UNRESTRAINED and -6.63 K clamped: the "
        "clamp is marginally WARMER, not colder.  It halves-squared with dt and the "
        "configurational estimators do not move, so it is an integrator artifact and the "
        "configurational distribution is unaffected.  See DT_RESTRAINED_PS and "
        "results/valine/dt_bias/."),
}


def assert_accepted(cv_name=None, dt_ps=None, n_grid=None, restrained=False):
    """Raise unless a run agrees with the frozen checkpoint.

    Called by the screening scripts at start-up.  A mismatch here is nearly always a silent
    scientific error -- an even grid degrades the projection identity, and the wrong dt puts a
    2 % kinetic-temperature bias into a reference that MBAR cannot unwind.
    """
    if cv_name is not None and cv_name != SELECTED_CV:
        raise SystemExit(
            f"CV {cv_name!r} is not the accepted Stage-0 selection {SELECTED_CV!r}"
            + (f"; {REJECTED_CV[cv_name]}" if cv_name in REJECTED_CV else ""))
    if n_grid is not None:
        if int(n_grid) % 2 == 0:
            raise SystemExit(f"n_grid={n_grid} is even; the Nyquist row breaks gB == grad(B)")
        if int(n_grid) != N_GRID:
            raise SystemExit(f"n_grid={n_grid} != accepted {N_GRID} (comparability with alanine)")
    if dt_ps is not None:
        want = DT_RESTRAINED_PS if restrained else DT_UNRESTRAINED_PS
        if abs(float(dt_ps) - want) > 1e-12:
            raise SystemExit(
                f"dt={dt_ps} ps != accepted {want} ps for "
                f"{'restrained' if restrained else 'unrestrained'} dynamics")
    return True
