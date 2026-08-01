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

#: Restrained dynamics (umbrella windows).  0.5 fs, because the stiff dihedral clamp is
#: under-integrated at 1 fs: measured kinetic temperature 293.2 K (-6.8 K, ~10 sigma) at 1 fs,
#: recovering to 299.0 K at 0.5 fs and 299.8 K at 0.25 fs -- a clean O(dt^2) signature.  The
#: UNRESTRAINED system is within 0.6 sigma of 300 K at every step size, so this is a property of
#: the restraint, not of the C-H stretches.  A softer clamp may be substituted only if its
#: kinetic temperature is verified first.
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
    "gate_V1": "PASS -- chi1 barrier 11.3-17.9 kT vs a >=2 kT requirement",
    "gate_sec32_phi_chi1": "PASS -- one populated state in hidden psi at all six anchors",
    "gate_sec32_psi_chi1": "FAIL -- two populated states in hidden phi at (psi=-30, chi1=g-)",
    "gate_V3": "NOT YET MEASURED -- discovery vs establishment; decides the study",
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
