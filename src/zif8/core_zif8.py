"""Ethane through a flexible ZIF-8 six-ring gate: the first FLEXIBLE-framework
stage of the uniform-FR campaign.

Why this stage exists (vs the closed rigid CHA stage): ZIF-8's window is
narrower than ethane, and passage is enabled by thermal linker-swing
fluctuations -- the framework gate is a physical HIDDEN slow coordinate.  The
stage therefore separates three clocks the campaign's predictor needs kept
apart:  discovery (T_cover)  !=  marginal establishment (T_marg)  !=
conditional gate equilibration (T_gate).  Marginal FR toward the uniform
target can only repair the middle one; the R15 lesson says it cannot repair
the third.  Every replica carries its OWN full flexible framework; a clone
copies the entire configuration and redraws momenta from Maxwell.

Model (frozen in configs/uniform_campaign/zif8_prereg.json before production):

  * framework   flexible ZIF-8, KROKIDAS et al. (JPCC 2015, 119, 27028) force
                field -- the force field of the 2024 ethane/ZIF-8 anchor paper
                (Schmidt/Cnudde/Van Speybroeck/Vanduyfhuys, JPCC 2024, 128,
                18509) -- built by scripts/build_zif8_framework.py, whose
                topology is validated TERM BY TERM against the published
                2x2x2 GROMACS enumeration.  Functional forms:
                  bonds      E = 1/2 k (r-r0)^2
                  angles     E = 1/2 k (th-th0)^2
                  propers    E = k (1 + cos(n phi - delta))
                  impropers  E = 1/2 k (psi-psi0)^2   (all k=0 in this FF)
                  LJ 12-6    truncated+shifted at rc, Lorentz-Berthelot,
                             1-4 scaled by fudgeLJ=0.5
                  Coulomb    damped shifted force (Fennell-Gezelter), 1-4
                             scaled by fudgeQQ=0.8333.  The model is DEFINED
                             with DSF electrostatics (declared, not an
                             approximation to something else).
  * guest       TraPPE-UA ethane CH3-CH3 (eps/kB 98 K, sigma 3.75 A, m 15.035),
                zero charge; the rigid TraPPE bond is replaced by the repo's
                harmonic convention k=400 kJ/mol/A^2, r0=1.54 A (declared
                surrogate; bond-length fluctuation sqrt(kT/k) ~ 0.08 A).
                Host-guest LJ by Lorentz-Berthelot.
  * confinement radial flat-bottom tube of radius R_tube about the cage-cage
                axis, on the guest COM: E += k/2 relu(rho - R_tube)^2.  The
                builder measures that this captures 98.9% of the accessible
                cage volume and that every OTHER window out of the cage lies
                1.5 A+ outside the tube.  Identical in every arm and in the
                reference.  There is NO axial wall -- see the CV.
  * dynamics    BAOAB Langevin, real masses, dt/friction from the prereg;
                units A, ps, kJ/mol.  The framework's k=0 acoustic mode is
                projected out of the thermostat (an infinite crystal has no
                k=0 Langevin noise, and the CV anchors are lab-fixed).  This
                stage upgrades from the campaign's overdamped convention
                because the RELATIVE clock of gate relaxation vs guest motion
                is the object of study; equilibrium F/U/S are unaffected.
  * CV          phi = wrap(2 pi xi / L),  xi = (COM_guest - win_center) . n,
                L = |(a/2)(1,1,1)| = 14.7146 A.  That vector is a LATTICE
                TRANSLATION, so xi is EXACTLY periodic and the channel needs
                no axial walls: cage -> window -> cage repeats.  The CV is
                evaluated on UNWRAPPED guest coordinates (the tube is longer
                than the minimum-image cube; a min-imaged CV would be wrong by
                up to 4.9 A -- measured in the builder).  phi is linear in the
                guest coordinates, so the geometric term in the local mean
                force vanishes and |grad phi|^2 = (2 pi/L)^2 sum_i (m_i/M)^2.
  * hidden gate DIAGNOSTIC ONLY -- never biased, never seen by FR.  The 6-ring
                alternates: three linkers present their ring C-H edge to the
                window (those six H are the crystallographic bottleneck,
                radius 2.853 A -> 3.5 A free diameter vs the literature 3.4 A)
                and three present their methyl.  A_gate = mean radial distance
                of the six bottleneck H from the instantaneous Zn6 centroid;
                theta_gate = mean linker-plane tilt vs the gate axis.

The circular estimator and the FR score are IMPORTED from the closed alkanes
engine (`alkanes.periodic`, `alkanes.core`); so are birth-death and genealogy.
FR additionally redraws every clone's momenta from Maxwell (positions
unchanged) and hands the clone its parent's cached force (exact -- F is a
function of q alone).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from alkanes import periodic as per
from alkanes.core import (_ancestor_stats, _birth_death, _fr_score,
                          assert_no_reference_leakage)

# --------------------------------------------------------------------------
# DETERMINISM.  This campaign's primary endpoint is a PAIRED per-seed
# difference, so the two arms must follow the same trajectory until FR first
# intervenes.  Measured on the H200 (B = 512, forces repeated on identical
# input):
#     torch.compile f32, default            6.1e-05  -> arms diverge at once
#     torch.compile f64, default            1.1e-13
#     eager (any dtype)                     0        but 8.8x slower
#     torch.compile f32 + the flags below   0        and 1.47x FASTER than f64
# The WCA stage of this project lost its pairing to exactly this effect
# (documented as "estimator scatter-add nondeterminism"); the cost of not
# losing it here is 1.75x, and it is paid deliberately.
torch.use_deterministic_algorithms(True, warn_only=True)
try:
    torch._inductor.config.deterministic = True
except Exception:                                     # pragma: no cover
    pass

EPS = 1.0e-12
KB = 0.008314462618             # kJ/mol/K
COULOMB = 1389.35457644382      # kJ/mol * A / e^2
MASS_TO_INTERNAL = 0.01         # amu -> kJ/mol ps^2 / A^2  (exact)
TWO_PI = 2.0 * math.pi
FR_METHODS = ("fr_uniform",)
ALL_METHODS = ("abf",) + FR_METHODS

GUEST = dict(name="ethane_trappe_ua", eps_K=[98.0, 98.0], sigma=[3.75, 3.75],
             mass=[15.035, 15.035], charge=[0.0, 0.0],
             bonds=[(0, 1, 1.54)], k_bond=400.0)


@dataclass
class ZIF8SimConfig:
    """Physical knobs are in ANGSTROM / ps / K; the engine converts the
    CV-space ones to radians with the framework's period L."""
    dt: float = 0.0005                  # ps  (explicit H: 0.5 fs)
    gamma: float = 1.0                  # 1/ps
    n_steps: int = 600_000
    n_replicas: int = 256
    save_every: int = 3_000
    rng_seed: int = 20260830
    # estimator (all lengths in A; converted with L)
    n_grid: int = 144
    abf_bandwidth_A: float = 0.15
    kde_bandwidth_A: float = 0.25
    abf_bias_scale: float = 1.0
    abf_warmup_steps: int = 40_000
    abf_force_clip_A: float = 30.0      # kJ/mol/A on the CV force
    estimator_burn_in_steps: int = 40_000
    abf_min_count: float = 20.0
    # Fisher--Rao, UNIFORM target only (rate frozen by SAFETY-ONLY calibration)
    fr_rate: float = 0.05
    score_clip: float = 2.0
    fr_start_steps: int = 40_000        # = warmup end (the LTA sweep lesson)
    fr_every: int = 5
    max_event_fraction: float = 0.02
    # region bookkeeping / diagnostics (A)
    window_half_A: float = 1.5
    cage_half_A: float = 2.0
    gate_every: int = 25
    gate_band_A: float = 1.0            # |xi| < this defines "at the gate"
    gate_lo: float = 2.2                # A_gate histogram range (A, radius)
    gate_hi: float = 4.6                # 0.025 A bins = 0.43 sd of the Stage-0B
    n_gate_bins: int = 96               # aperture law (a 1.08 sd bin cannot
    n_gate_xi: int = 8                  # resolve the stage's own diagnostic)

    def config_hash(self):
        import hashlib, json
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


class ZIF8System:
    """Flexible ZIF-8 (one full framework per replica) + optional TraPPE ethane."""

    def __init__(self, temperature, device, dtype=torch.float64, root=".",
                 framework="cache/zif8/framework.npz", with_guest=True,
                 compile=None, chunk=256, force_dtype=None):
        """``force_dtype`` runs ONLY the force/energy kernel at a lower
        precision while positions, velocities, the CV and every estimator
        accumulator stay in ``dtype``.  Mixed f64/f32 is 2.7x faster on the
        H200 and its bias in the local mean force -- the quantity ABF actually
        integrates -- is measured at 1e-6 relative in Stage 0C.  Full f32
        positions would NOT be safe: the guest's unwrapped axial coordinate
        drifts to O(100 A) while a step moves it 2.5e-4 A."""
        z = np.load(os.path.join(root, framework), allow_pickle=True)
        self.temperature = float(temperature)
        self.beta = 1.0 / (KB * self.temperature)
        self.kT = KB * self.temperature
        self.device, self.dtype = device, dtype
        self.with_guest = bool(with_guest)
        self.chunk = int(chunk)
        t = lambda x: torch.as_tensor(np.asarray(x), device=device, dtype=dtype)
        ti = lambda x: torch.as_tensor(np.asarray(x), device=device, dtype=torch.long)

        self.box = t(z["box"])
        self.rc = float(z["rc"])
        self.dsf_alpha = float(z["dsf_alpha"])
        assert self.rc < 0.5 * float(self.box.min()) + 1e-9, "rc violates minimum image"
        pos_f = t(z["pos"])
        self.n_frame = pos_f.shape[0]
        mass_amu_f, charge_f = t(z["mass_amu"]), t(z["charge_e"])
        eps_f, sig_f = t(z["lj_eps_kj"]), t(z["lj_sig_A"])

        if self.with_guest:
            g = GUEST
            self.n_guest = len(g["mass"])
            mass_amu = torch.cat([mass_amu_f, t(g["mass"])])
            charge = torch.cat([charge_f, t(g["charge"])])
            eps = torch.cat([eps_f, t(g["eps_K"]) * KB])
            sig = torch.cat([sig_f, t(g["sigma"])])
        else:
            self.n_guest = 0
            mass_amu, charge, eps, sig = mass_amu_f, charge_f, eps_f, sig_f
        self.n_atoms = self.n_frame + self.n_guest
        A = self.n_atoms
        self.mass_amu = mass_amu
        self.mass = mass_amu * MASS_TO_INTERNAL          # kJ/mol ps^2/A^2
        self.pos0_frame = pos_f

        # ---- pair matrices (A, A): Lorentz-Berthelot + exclusion/1-4 scaling
        eps_mat = torch.sqrt(eps[:, None] * eps[None, :])
        sig_mat = 0.5 * (sig[:, None] + sig[None, :])
        lj_scale = torch.ones(A, A, device=device, dtype=dtype)
        coul_scale = torch.ones(A, A, device=device, dtype=dtype)
        nf = self.n_frame
        lj_scale[:nf, :nf] = t(z["lj_scale"])
        coul_scale[:nf, :nf] = t(z["coul_scale"])
        idx = torch.arange(A, device=device)
        lj_scale[idx, idx] = 0.0
        coul_scale[idx, idx] = 0.0
        if self.with_guest:                               # guest internal excluded
            lj_scale[nf:, nf:] = 0.0
            coul_scale[nf:, nf:] = 0.0
        self.eps_mat = eps_mat * lj_scale
        self.qq_mat = (COULOMB * charge[:, None] * charge[None, :]) * coul_scale
        self.sig2_mat = sig_mat ** 2
        sr6c = (sig_mat ** 2 / self.rc ** 2) ** 3
        self.vshift_mat = 4.0 * self.eps_mat * (sr6c * sr6c - sr6c)
        a_rc = self.dsf_alpha * self.rc
        self.dsf_e_rc = math.erfc(a_rc) / self.rc
        self.dsf_f_rc = (math.erfc(a_rc) / self.rc ** 2
                         + 2.0 * self.dsf_alpha / math.sqrt(math.pi)
                         * math.exp(-a_rc * a_rc) / self.rc)
        hg = torch.zeros(A, A, device=device, dtype=dtype)
        if self.with_guest:
            hg[:nf, nf:] = 1.0
            hg[nf:, :nf] = 1.0
        self.hg_mask = hg

        # ---- bonded terms ----
        self.bonds = ti(z["bonds"]).reshape(-1, 2)
        self.bond_k, self.bond_r0 = t(z["bond_k"]), t(z["bond_r0"])
        self.angles = ti(z["angles"]).reshape(-1, 3)
        self.angle_k, self.angle_th0 = t(z["angle_k"]), t(z["angle_th0"])
        self.dihedrals = ti(z["dihedrals"]).reshape(-1, 4)
        self.dih_k, self.dih_n, self.dih_delta = (t(z["dih_k"]), t(z["dih_n"]),
                                                  t(z["dih_delta"]))
        self.impropers = ti(z["impropers"]).reshape(-1, 4)
        self.impr_k, self.impr_psi0 = t(z["impr_k"]), t(z["impr_psi0"])
        self.n_frame_bonds = self.bonds.shape[0]
        if self.with_guest:
            gb = [(nf + i, nf + j) for (i, j, _) in GUEST["bonds"]]
            self.bonds = torch.cat([self.bonds, ti(gb).reshape(-1, 2)])
            self.bond_k = torch.cat([self.bond_k,
                                     t([GUEST["k_bond"]] * len(gb))])
            self.bond_r0 = torch.cat([self.bond_r0,
                                      t([r0 for (_, _, r0) in GUEST["bonds"]])])

        # ---- CV frame, tube, gate atoms (all frozen in the npz) ----
        self.cage_A, self.cage_B = t(z["cage_A"]), t(z["cage_B"])
        self.center, self.normal = t(z["win_center"]), t(z["win_normal"])
        self._center0 = self.center.clone()          # for the virial scan
        self.period = float(z["period"])                  # L (A)
        self.k_phi = TWO_PI / self.period                 # rad per A
        self.xi_A, self.xi_B = float(z["xi_A"]), float(z["xi_B"])
        self.R_tube, self.k_wall = float(z["R_tube"]), float(z["k_wall"])
        m = t(GUEST["mass"] if self.with_guest else [1.0])
        self.mass_w = m / m.sum()
        self.grad_phi_sq = float((self.mass_w ** 2).sum()) * self.k_phi ** 2
        self.gate_zn = ti(z["gate_zn_idx"])                # (6,)
        self.gate_h = ti(z["gate_aperture_h"])             # (6,)
        self.gate_methyl = ti(z["gate_methyl_c"])          # (3,)
        self.gate_tri = ti(z["gate_tri"])                  # (6, 3)
        self.gate_aperture_crystal = float(np.mean(z["gate_aperture_crystal"]))
        # ---- lower-precision mirrors of everything the kernel touches ------
        self.force_dtype = force_dtype or dtype
        fd = self.force_dtype
        for name in ("box", "eps_mat", "qq_mat", "sig2_mat", "vshift_mat",
                     "hg_mask", "bond_k", "bond_r0", "angle_k", "angle_th0",
                     "dih_k", "dih_n", "dih_delta", "impr_k", "impr_psi0",
                     "center", "normal", "mass_w"):
            setattr(self, name + "_f", getattr(self, name).to(fd))
        if compile is None:
            compile = (torch.device(device).type == "cuda")
        self._terms = (torch.compile(self._energy_terms, dynamic=False)
                       if compile else self._energy_terms)

    # ------------------------------------------------------------------ utils
    def _min_image(self, d):
        box = self.box if d.dtype == self.dtype else self.box_f
        return d - box * torch.round(d / box)

    def pin_frame_com(self, v):
        """Remove the framework's mass-weighted COM velocity (the k=0 acoustic
        mode).  An infinite crystal has no k=0 Langevin noise, and the CV/tube
        anchors are lab-fixed, so without this the cell random-walks away from
        them.  The guest keeps its full dynamics."""
        nf = self.n_frame
        mf = self.mass[:nf]
        vcom = (v[:, :nf] * mf[None, :, None]).sum(dim=1, keepdim=True) / mf.sum()
        v = v.clone()
        v[:, :nf] = v[:, :nf] - vcom
        return v

    def maxwell_velocities(self, shape_prefix, gen):
        v = torch.randn(*shape_prefix, self.n_atoms, 3, generator=gen,
                        device=self.device, dtype=self.dtype)
        return v * torch.sqrt(self.kT / self.mass)[None, :, None]

    # ---------------------------------------------------------------- energy
    def _energy_terms(self, q):
        """(E_frame_bonded+wall, E_nb_frame, E_hostguest, E_guest_bond), each (B,).

        ``q`` is (B, A, 3) UNWRAPPED; pair terms use the minimum image, the CV
        and the tube wall do not (see the module docstring)."""
        B = q.shape[0]
        d = self._min_image(q[:, self.bonds[:, 0]] - q[:, self.bonds[:, 1]])
        r = d.norm(dim=-1)
        eb_all = 0.5 * self.bond_k_f[None, :] * (r - self.bond_r0_f[None, :]) ** 2
        nfb = self.n_frame_bonds
        Eb = eb_all[:, :nfb].sum(-1)
        Eg = (eb_all[:, nfb:].sum(-1) if self.with_guest
              else torch.zeros(B, device=q.device, dtype=q.dtype))
        a = self._min_image(q[:, self.angles[:, 0]] - q[:, self.angles[:, 1]])
        b = self._min_image(q[:, self.angles[:, 2]] - q[:, self.angles[:, 1]])
        cth = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(EPS)
        th = torch.arccos(cth.clamp(-1 + 1e-12, 1 - 1e-12))
        Eb = Eb + (0.5 * self.angle_k_f[None, :]
                   * (th - self.angle_th0_f[None, :]) ** 2).sum(-1)
        if self.dihedrals.shape[0]:
            phi = self._dihedral_angle(q, self.dihedrals)
            Eb = Eb + (self.dih_k_f[None, :]
                       * (1.0 + torch.cos(self.dih_n_f[None, :] * phi
                                          - self.dih_delta_f[None, :]))).sum(-1)
        if self.impropers.shape[0]:
            psi = self._dihedral_angle(q, self.impropers)
            dp = psi - self.impr_psi0_f[None, :]
            dp = dp - TWO_PI * torch.round(dp / TWO_PI)
            Eb = Eb + (0.5 * self.impr_k_f[None, :] * dp ** 2).sum(-1)
        if self.with_guest:                                # radial tube, no axial wall
            rel = (q[:, self.n_frame:] * self.mass_w_f[None, :, None]).sum(1) - self.center_f
            xi = (rel * self.normal_f).sum(-1)
            rho = (rel - xi[:, None] * self.normal_f[None, :]).norm(dim=-1)
            Eb = Eb + 0.5 * self.k_wall * torch.relu(rho - self.R_tube) ** 2
        dv = self._min_image(q[:, :, None, :] - q[:, None, :, :])
        r2 = (dv * dv).sum(-1)
        mask = r2 < self.rc ** 2
        # floor r2 at (0.01 A)^2, not EPS: in f32 a 1/EPS diagonal makes
        # sr6^2 = (sig^2/EPS)^6 overflow to inf, and eps_mat = 0 there turns
        # 0*inf into NaN.  No real pair reaches 0.01 A.
        inv_r2 = torch.where(mask, 1.0 / r2.clamp_min(1.0e-4), torch.zeros_like(r2))
        sr6 = (self.sig2_mat_f[None] * inv_r2) ** 3
        v_lj = 4.0 * self.eps_mat_f[None] * (sr6 * sr6 - sr6) - self.vshift_mat_f[None]
        rr = torch.sqrt(r2.clamp_min(1.0e-4))
        v_c = self.qq_mat_f[None] * (torch.erfc(self.dsf_alpha * rr) / rr
                                   - self.dsf_e_rc + self.dsf_f_rc * (rr - self.rc))
        v = torch.where(mask, v_lj + v_c, torch.zeros_like(v_lj))
        Ehg = 0.5 * (v * self.hg_mask_f[None]).sum(dim=(1, 2))
        Enb = 0.5 * v.sum(dim=(1, 2)) - Ehg
        return Eb, Enb, Ehg, Eg

    def _dihedral_angle(self, q, quad):
        b1 = self._min_image(q[:, quad[:, 1]] - q[:, quad[:, 0]])
        b2 = self._min_image(q[:, quad[:, 2]] - q[:, quad[:, 1]])
        b3 = self._min_image(q[:, quad[:, 3]] - q[:, quad[:, 2]])
        n1, n2 = torch.cross(b1, b2, dim=-1), torch.cross(b2, b3, dim=-1)
        b2n = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(EPS)
        x = (n1 * n2).sum(-1)
        y = (torch.cross(n1, n2, dim=-1) * b2n).sum(-1)
        return torch.atan2(y, x)

    def _chunked(self, q, fn):
        B, c = q.shape[0], self.chunk
        if B <= c:
            return fn(q)
        outs = [fn(q[i:i + c]) for i in range(0, B, c)]
        return tuple(torch.cat([o[k] for o in outs]) for k in range(len(outs[0])))

    def potential_energy(self, q, split=False):
        with torch.no_grad():
            out = self._chunked(q.to(self.force_dtype), self._terms)
        Eb, Enb, Ehg, Eg = (x.to(self.dtype) for x in out)
        if split:
            return Eb + Enb + Ehg + Eg, Ehg, Eb + Enb
        return Eb + Enb + Ehg + Eg

    def forces(self, q):
        def one(qc):
            qg = qc.detach().requires_grad_(True)
            Eb, Enb, Ehg, Eg = self._terms(qg)
            (F,) = torch.autograd.grad((Eb + Enb + Ehg + Eg).sum(), qg)
            return (-F.detach(),)
        return self._chunked(q.to(self.force_dtype), one)[0].to(self.dtype)

    # ------------------------------------------------------------- pressure -
    def pressure(self, q, v, eps=1.0e-5):
        """Instantaneous pressure (bar) from the atomic virial.

        W = -dU/d(eps) under an AFFINE scaling of both the coordinates and the
        box by (1+eps), obtained by central finite difference -- exact for any
        potential without needing per-term virial expressions.
            P = (2 K + W) / (3 V)
        This is what makes a barostat-free equilibrium-lattice determination
        possible: scan the lattice constant and find where <P> = 1 bar.
        """
        V = float(self.box.prod())
        box0 = self.box.clone()
        cen = self.box * 0.5

        def u_scaled(s):
            self.box = box0 * s
            self.center = self._center0 * s
            out = self.potential_energy(cen[None, None, :]
                                        + (q - cen[None, None, :]) * s)
            return out
        try:
            Up = u_scaled(1.0 + eps)
            Um = u_scaled(1.0 - eps)
        finally:
            self.box = box0
            self.center = self._center0
        dU = (Up - Um) / (2.0 * eps)          # dU/d(eps)
        K = (0.5 * self.mass[None, :, None] * v ** 2).sum(dim=(1, 2))
        W = -dU
        # kJ/mol/A^3 -> bar : 1 kJ/mol/A^3 = 1e30/6.02214076e23 * 1e3 Pa / 1e5
        KJ_PER_MOL_A3_TO_BAR = 1.0e33 / 6.02214076e23 / 1.0e5
        return ((2.0 * K + W) / (3.0 * V)) * KJ_PER_MOL_A3_TO_BAR

    # ------------------------- CV: phi = wrap(2 pi xi / L), UNWRAPPED coords -
    def guest(self, q):
        return q[:, self.n_frame:]

    def xi_value(self, q):
        rel = (self.guest(q) * self.mass_w[None, :, None]).sum(dim=1) - self.center
        return (rel * self.normal).sum(-1)

    def cv_value(self, q):
        x = self.k_phi * self.xi_value(q)
        return x - TWO_PI * torch.round(x / TWO_PI)

    def cv_local_mean_force(self, q, F):
        """f_loc = -(F . grad phi)/|grad phi|^2; linear CV, no geometric term."""
        fdot = (self.guest(F) * self.mass_w[None, :, None]
                * self.normal[None, None, :]).sum(dim=(1, 2)) * self.k_phi
        return -fdot / self.grad_phi_sq, self.cv_value(q)

    def bias_cartesian(self, bias_gen, R, N):
        """Generalized CV force (phi units) -> cartesian force on the guest."""
        out = torch.zeros(R * N, self.n_atoms, 3, device=self.device, dtype=self.dtype)
        out[:, self.n_frame:] = (bias_gen.reshape(-1)[:, None, None] * self.k_phi
                                 * self.mass_w[None, :, None]
                                 * self.normal[None, None, :])
        return out

    # ------------------------- hidden-gate diagnostics (never biased) -------
    def gate_observables(self, q):
        """(A_gate (B,), theta_gate (B,)).

        A_gate = mean radial distance (from the instantaneous Zn6 centroid,
        perpendicular to the FIXED gate axis) of the six bottleneck ring-H.
        theta_gate = mean angle (deg) between each gate linker's ring plane
        and the gate axis."""
        # Min-image the ring against the FIXED window centre, never against a
        # ring atom: opposite gate Zn are 12.02 A apart, i.e. exactly a/2 in
        # each cartesian component, so referencing one of them puts every
        # other Zn on the min-image knife edge and any thermal jitter flips
        # the wrap (measured: A_gate jumps 2.85 -> 5.03 A).  Each Zn is 6.01 A
        # from the window centre, comfortably inside a/2 = 8.50 A.
        zn = q[:, self.gate_zn]
        ctr = self.center + self._min_image(zn - self.center).mean(dim=1)
        dh = self._min_image(q[:, self.gate_h] - ctr[:, None, :])
        ax = (dh * self.normal[None, None, :]).sum(-1)
        rho = (dh - ax[..., None] * self.normal[None, None, :]).norm(dim=-1)
        a_gate = rho.mean(dim=-1)
        tri = q[:, self.gate_tri.reshape(-1)].reshape(q.shape[0], 6, 3, 3)
        u = self._min_image(tri[:, :, 1] - tri[:, :, 0])
        w = self._min_image(tri[:, :, 2] - tri[:, :, 0])
        nrm = torch.cross(u, w, dim=-1)
        nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(EPS)
        cosang = (nrm * self.normal[None, None, :]).sum(-1).abs().clamp(0.0, 1.0)
        return a_gate, torch.rad2deg(torch.arccos(cosang)).mean(dim=-1)

    # ------------------------------------------------------------ minimizer -
    def minimize(self, q, n_steps=4000, step0=1.0e-5, f_tol=5.0, verbose=False):
        """Adaptive-step steepest descent to a local minimum of U.

        The published X-ray structure is NOT a minimum of the force field: its
        C-H distances are the usual X-ray ~0.95 A against the FF r0 of
        1.08-1.09 A, which alone puts ~500 kJ/mol/A on every H (the Zn-N-C
        skeleton is already at |F| ~ 1 kJ/mol/A).  Relaxing before any
        dynamics is therefore mandatory, not cosmetic.
        """
        q = q.clone()
        E = self.potential_energy(q)
        step = torch.full((q.shape[0], 1, 1), step0, device=q.device, dtype=q.dtype)
        for it in range(n_steps):
            F = self.forces(q)
            fmax = F.norm(dim=-1).max(dim=-1).values
            if float(fmax.max()) < f_tol:
                break
            q_try = q + step * F
            E_try = self.potential_energy(q_try)
            ok = (E_try < E)[:, None, None]
            q = torch.where(ok, q_try, q)
            E = torch.where(ok[:, 0, 0], E_try, E)
            step = torch.where(ok, step * 1.15, step * 0.5)
            step = step.clamp(1e-12, 1e-3)
            if verbose and it % 500 == 0:
                print(f"    min it={it} E={float(E.mean()):.3f} "
                      f"fmax={float(fmax.max()):.2f}", flush=True)
        return q, float(self.forces(q).norm(dim=-1).max()), E

    # ----------------------------------------------------------- init pool --
    def load_init_pool(self, path, R, N, seed_offset=0):
        """Deterministic per-seed draw of R*N configurations from the
        equilibrated pool (framework + ethane, unwrapped)."""
        z = np.load(path)
        pool = torch.as_tensor(z["q"], device=self.device, dtype=self.dtype)
        assert pool.shape[1] == self.n_atoms, \
            f"pool has {pool.shape[1]} atoms, system has {self.n_atoms}"
        rng = np.random.default_rng(20260830 + int(seed_offset))
        idx = torch.as_tensor(rng.integers(0, pool.shape[0], size=(R * N,)),
                              device=self.device)
        return pool[idx]



def engine_kwargs(pre):
    """System construction options from the prereg's `engine` block."""
    import torch as _t
    e = pre.get("engine", {})
    dt = {"float64": _t.float64, "float32": _t.float32}
    return dict(dtype=dt[e.get("dtype", "float64")],
                force_dtype=dt[e["force_dtype"]] if e.get("force_dtype") else None,
                chunk=int(e.get("chunk", 1024)))

# --------------------------- estimator helpers (circular) -------------------
def bin_counts_and_sum(phi, values, n_grid):
    """DETERMINISTIC circular binning: returns (counts, weighted sum), both
    (R, n_grid).

    ``scatter_add_`` on CUDA uses floating-point atomics, so its result depends
    on thread completion order.  That is invisible in a diagnostic, but here
    the accumulators feed the ABF bias, which feeds the trajectory -- so two
    arms that should be identical diverge chaotically, and the campaign's
    PAIRED endpoint quietly becomes an unpaired one.  (That is exactly what
    happened to the WCA stage.)  A one-hot ``scatter_`` writes one distinct
    location per sample and the following ``sum`` is a fixed-order reduction,
    so both are deterministic; the cost is R*N*G, negligible beside the force
    kernel.  With this, fr_rate = 0 reproduces ABF bit-for-bit ON THE GPU.
    """
    R, N = phi.shape
    dphi = TWO_PI / n_grid
    idx = (torch.floor((phi + math.pi) / dphi).long() % n_grid).unsqueeze(-1)
    oh = torch.zeros(R, N, n_grid, device=phi.device, dtype=phi.dtype)
    oh.scatter_(2, idx, 1.0)
    return oh.sum(1), (oh * values.unsqueeze(-1)).sum(1)


def mean_force_regularized(fsum, csum, K, min_count):
    """Kernel-smoothed conditional mean force with a low-count damper."""
    return per.smooth(fsum, K) / (per.smooth(csum, K) + min_count + EPS)


def gate_hist(a_gate, phi, sim, k_phi, device, dtype):
    """Gate histogram RESOLVED IN xi inside the band: (R, n_gate_xi, n_gate).

    Resolving in xi is what makes J_gate a conditional comparison rather than a
    marginal one -- see gate_js_series in scripts/run_zif8_screen.py for why
    the unresolved version cannot separate "the gate is in the wrong state"
    from "the population sits at a different place in the band", which is
    precisely what FR changes.

    Out-of-range A_gate is DROPPED, not clamped into the edge bin, so that this
    estimator and the reference's np.histogram agree; the dropped count is
    returned so it can never be silently zero.
    """
    G, X = sim.n_gate_bins, sim.n_gate_xi
    band = sim.gate_band_A * k_phi
    R = a_gate.shape[0]
    ia = torch.floor((a_gate - sim.gate_lo) / (sim.gate_hi - sim.gate_lo) * G).long()
    ix = torch.floor((phi / band * 0.5 + 0.5) * X).long()
    good = (phi.abs() < band) & (ia >= 0) & (ia < G) & (ix >= 0) & (ix < X)
    flat = (ix.clamp(0, X - 1) * G + ia.clamp(0, G - 1))
    out = torch.zeros(R, X * G, device=device, dtype=dtype)
    out.scatter_add_(1, flat, good.to(dtype))
    n_drop = int(((phi.abs() < band) & ~good).sum())
    return out.reshape(R, X, G), n_drop


def js_divergence(p, q):
    """Jensen-Shannon divergence between histograms (normalized here)."""
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-300)
    q = q / np.maximum(q.sum(axis=-1, keepdims=True), 1e-300)
    m = 0.5 * (p + q)

    def kl(x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = x * (np.log(np.maximum(x, 1e-300)) - np.log(np.maximum(y, 1e-300)))
        return np.where(x > 0, t, 0.0).sum(axis=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def uniform_target(R, n_grid, device, dtype):
    return torch.full((R, n_grid), 1.0 / TWO_PI, device=device, dtype=dtype)


# --------------------------------- sampler ---------------------------------
def run_sampler(method, system: ZIF8System, sim: ZIF8SimConfig, seeds, init_pool,
                oracle_free_energy=None, verbose=True, progress_every=0,
                lineage_diagnostics=False, clone_age_edges=(0.5, 5.0)):
    """R = len(seeds) matched-seed populations of ``method`` in one process.

    BAOAB Langevin, one force evaluation per step.  FR clones copy q AND the
    cached force (exact -- F depends only on q) and redraw v from Maxwell using
    the FR stream, so the dynamics stream stays aligned between paired arms.
    """
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method {method!r}")
    assert_no_reference_leakage(method, oracle_free_energy)
    is_fr = method in FR_METHODS
    device, dtype = system.device, system.dtype
    R, N = len(seeds), sim.n_replicas
    A, G = system.n_atoms, sim.n_grid
    grid, dphi = per.periodic_grid(G, device=device, dtype=dtype)
    kf = system.k_phi
    K_abf = per.wrapped_gaussian_kernel_matrix(grid, sim.abf_bandwidth_A * kf)
    K_kde = per.wrapped_gaussian_kernel_matrix(grid, sim.kde_bandwidth_A * kf)
    clip_phi = sim.abf_force_clip_A / kf
    win_phi, cage_phi = sim.window_half_A * kf, sim.cage_half_A * kf
    gate_phi = sim.gate_band_A * kf
    gen_dyn = torch.Generator(device=device).manual_seed(int(sim.rng_seed))
    gen_fr = torch.Generator(device=device).manual_seed(int(sim.rng_seed) + 987654321)

    q = system.load_init_pool(init_pool, R, N, seed_offset=seeds[0])
    v = system.pin_frame_com(system.maxwell_velocities((R * N,), gen_dyn))
    m = system.mass[None, :, None]
    c1 = math.exp(-sim.gamma * sim.dt)
    c2 = math.sqrt(1.0 - c1 * c1)
    vsig = torch.sqrt(system.kT / system.mass)[None, :, None]

    z = lambda: torch.zeros(R, G, device=device, dtype=dtype)
    fsum, csum, fsum_p, csum_p = z(), z(), z(), z()
    usum_p, uhgsum_p, ucnt_p = z(), z(), z()
    # ---- lineage instrumentation (DIAGNOSTIC ONLY; never enters the bias
    # force or the FR score, so the trajectory stays bit-paired between arms).
    # The ordinary ABF estimate weights each lineage by its descendant COUNT;
    # these accumulators also allow a lineage-BALANCED estimate that weights
    # each ancestral discovery once, which is what separates "this bin has 100
    # observations" from "this bin has 100 copies of three discoveries".
    if lineage_diagnostics:
        fsum_ag = torch.zeros(R, N, G, device=device, dtype=dtype)
        csum_ag = torch.zeros(R, N, G, device=device, dtype=dtype)
        n_age = len(clone_age_edges) + 1
        fsum_age = torch.zeros(R, n_age, G, device=device, dtype=dtype)
        csum_age = torch.zeros(R, n_age, G, device=device, dtype=dtype)
        t_clone = torch.zeros(R, N, device=device, dtype=dtype)
        age_edges = torch.tensor(clone_age_edges, device=device, dtype=dtype)
        ess_bin_series = []
    ghist = torch.zeros(R, sim.n_gate_xi, sim.n_gate_bins, device=device,
                        dtype=dtype)
    ghist_c = torch.zeros_like(ghist)
    n_gate_dropped = 0
    ancestors = (torch.arange(N, device=device).expand(R, N).clone() if is_fr else None)
    total_repl = torch.zeros(R, dtype=torch.long)
    crossings = torch.zeros(R, N, dtype=torch.long, device=device)
    cross_gate = []
    score_std_sum, score_absmax, n_score = np.zeros(R), np.zeros(R), 0
    a_gate = torch.zeros(R, N, device=device, dtype=dtype)
    theta_g = torch.zeros(R, N, device=device, dtype=dtype)

    diag = {k: [] for k in ["steps", "times", "mean_force", "pmf", "p_hat",
                            "eff_counts", "ancestor_ess", "n_unique_ancestor",
                            "max_ancestor_frac", "repl_cumulative", "kl_uniform",
                            "tv_uniform", "frac_cage", "frac_window",
                            "n_visited_bins", "gate_hist_block", "gate_mean",
                            "gate_theta_mean", "temp_kin", "n_band",
                            "raw_fsum_t", "raw_csum_t"]}
    t0 = time.perf_counter()

    def full_force(qq):
        F = system.forces(qq)
        f_loc, phi_f = system.cv_local_mean_force(qq, F)
        return F, f_loc.reshape(R, N), phi_f.reshape(R, N)

    F_phys, f_loc, phi = full_force(q)
    phi_unw = phi.clone()
    cage_idx = torch.round((phi_unw - math.pi) / TWO_PI).long()
    last_cage = cage_idx.clone()
    has_cage = (phi.abs() > math.pi - cage_phi)
    mf = torch.zeros(R, G, device=device, dtype=dtype)
    ramp = 0.0

    for step in range(sim.n_steps + 1):
        fl = f_loc.clamp(-clip_phi * 8, clip_phi * 8)
        c_now, f_now = bin_counts_and_sum(phi, fl, G)
        fsum += f_now
        csum += c_now
        if lineage_diagnostics and step >= sim.estimator_burn_in_steps:
            gidx = (torch.floor((phi + math.pi) / dphi).long() % G)
            anc = (ancestors if is_fr else
                   torch.arange(N, device=device).expand(R, N))
            flat = anc * G + gidx
            fsum_ag.view(R, -1).scatter_add_(1, flat, fl)
            csum_ag.view(R, -1).scatter_add_(1, flat, torch.ones_like(fl))
            age = step * sim.dt - t_clone
            ab = torch.bucketize(age, age_edges)
            fa = ab * G + gidx
            fsum_age.view(R, -1).scatter_add_(1, fa, fl)
            csum_age.view(R, -1).scatter_add_(1, fa, torch.ones_like(fl))
        if step >= sim.estimator_burn_in_steps:
            fsum_p += f_now
            csum_p += c_now
            if step % 25 == 0:      # U(xi) conditionals, 1-in-25 stride (declared)
                et, ehg, _ = system.potential_energy(q, split=True)
                cu, su = bin_counts_and_sum(phi, et.reshape(R, N), G)
                _, shg = bin_counts_and_sum(phi, ehg.reshape(R, N), G)
                usum_p += su
                uhgsum_p += shg
                ucnt_p += cu

        if step % sim.gate_every == 0:
            ag, th = system.gate_observables(q)
            a_gate, theta_g = ag.reshape(R, N), th.reshape(R, N)
            gh, nd = gate_hist(a_gate, phi, sim, kf, device, dtype)
            n_gate_dropped += nd
            ghist += gh
            ghist_c += gh

        mf = mean_force_regularized(fsum, csum, K_abf, sim.abf_min_count)
        A_hat = per.free_energy_from_mean_force(mf, grid, dphi)
        ramp = min(1.0, step / max(sim.abf_warmup_steps, 1))
        mf_at = per.circular_interp(mf, grid, phi).clamp(-clip_phi, clip_phi)
        bias_gen = (sim.abf_bias_scale * ramp) * mf_at

        # ---- cage-to-cage transit bookkeeping on the UNWRAPPED phi ----
        in_cage = phi.abs() > math.pi - cage_phi
        moved = in_cage & has_cage & (cage_idx != last_cage)
        if bool(moved.any()):
            crossings += moved.long()
            cross_gate.append(a_gate[moved].detach().cpu().numpy().copy())
        last_cage = torch.where(in_cage, cage_idx, last_cage)
        has_cage = has_cage | in_cage

        if step % sim.save_every == 0 or step == sim.n_steps:
            est_f = fsum_p if float(csum_p.sum()) > 0 else fsum
            est_c = csum_p if float(csum_p.sum()) > 0 else csum
            mf_rep = mean_force_regularized(est_f, est_c, K_abf, sim.abf_min_count)
            diag["steps"].append(step); diag["times"].append(step * sim.dt)
            # the UNSMOOTHED accumulators the estimator used at this save, so
            # the run can be re-scored at any h_read after the fact
            diag["raw_fsum_t"].append(est_f.cpu().numpy())
            diag["raw_csum_t"].append(est_c.cpu().numpy())
            diag["mean_force"].append(mf_rep.cpu().numpy())
            diag["pmf"].append(per.free_energy_from_mean_force(mf_rep, grid, dphi)
                               .cpu().numpy())
            diag["eff_counts"].append(per.smooth(csum, K_abf).cpu().numpy())
            diag["repl_cumulative"].append(total_repl.numpy().copy())
            p_grid = per.kde_marginal(phi, K_kde, G, dphi)
            u_grid = uniform_target(R, G, device, dtype)
            diag["p_hat"].append(p_grid.cpu().numpy())
            diag["kl_uniform"].append(per.marginal_kl(p_grid, u_grid, dphi).cpu().numpy())
            diag["tv_uniform"].append(per.marginal_tv(p_grid, u_grid, dphi).cpu().numpy())
            if is_fr:
                ess, nuq, maxf = _ancestor_stats(ancestors.cpu(), N)
            else:
                ess, nuq, maxf = np.full(R, np.nan), np.full(R, N), np.full(R, np.nan)
            diag["ancestor_ess"].append(ess)
            diag["n_unique_ancestor"].append(nuq)
            diag["max_ancestor_frac"].append(maxf)
            diag["frac_cage"].append(in_cage.to(dtype).mean(-1).cpu().numpy())
            diag["frac_window"].append((phi.abs() < win_phi).to(dtype).mean(-1)
                                       .cpu().numpy())
            diag["n_visited_bins"].append((csum > 0).sum(-1).cpu().numpy())
            diag["gate_hist_block"].append(ghist.cpu().numpy())
            ghist = torch.zeros_like(ghist)
            diag["gate_mean"].append(a_gate.mean(-1).cpu().numpy())
            diag["gate_theta_mean"].append(theta_g.mean(-1).cpu().numpy())
            ke = (0.5 * m * v ** 2).sum(dim=(1, 2))
            diag["temp_kin"].append((2.0 * ke / ((3 * A - 3) * KB)).reshape(R, N)
                                    .mean(-1).cpu().numpy())
            diag["n_band"].append((phi.abs() < gate_phi).to(dtype).sum(-1).cpu().numpy())
            if lineage_diagnostics:
                gidx = (torch.floor((phi + math.pi) / dphi).long() % G)
                anc = (ancestors if is_fr else
                       torch.arange(N, device=device).expand(R, N))
                occ = torch.zeros(R, N * G, device=device, dtype=dtype)
                occ.scatter_add_(1, anc * G + gidx, torch.ones_like(phi))
                occ = occ.view(R, N, G)
                tot = occ.sum(1)
                ess = tot ** 2 / (occ ** 2).sum(1).clamp_min(EPS)
                ess_bin_series.append(torch.where(tot > 0, ess,
                                      torch.zeros_like(ess)).cpu().numpy())
            if progress_every and (step // sim.save_every) % progress_every == 0:
                print(f"    step {step}/{sim.n_steps} t={step*sim.dt:.1f} ps "
                      f"KL={float(np.median(diag['kl_uniform'][-1])):.4f} "
                      f"[{time.perf_counter()-t0:.0f}s]", flush=True)

        if step == sim.n_steps:
            break

        # ---- BAOAB (one force evaluation per step) ----
        F_tot = F_phys + system.bias_cartesian(bias_gen, R, N)
        v = v + (0.5 * sim.dt) * F_tot / m
        q = q + (0.5 * sim.dt) * v
        noise = torch.randn(q.shape, generator=gen_dyn, device=device, dtype=dtype)
        v = system.pin_frame_com(c1 * v + c2 * vsig * noise)
        q = q + (0.5 * sim.dt) * v
        F_phys, f_loc, phi_new = full_force(q)
        phi_unw = phi_unw + per.circular_distance(phi_new, phi)
        phi = phi_new
        cage_idx = torch.round((phi_unw - math.pi) / TWO_PI).long()
        mf_at = per.circular_interp(mf, grid, phi).clamp(-clip_phi, clip_phi)
        F_tot = F_phys + system.bias_cartesian((sim.abf_bias_scale * ramp) * mf_at, R, N)
        v = v + (0.5 * sim.dt) * F_tot / m

        # ---- Fisher--Rao birth-death toward the UNIFORM marginal ----
        if is_fr:
            nxt = step + 1
            if nxt >= sim.fr_start_steps and \
                    (nxt - sim.fr_start_steps) % max(int(sim.fr_every), 1) == 0:
                q_grid = uniform_target(R, G, device, dtype)
                score, _, _ = _fr_score(phi, grid, dphi, K_kde, q_grid,
                                        sim.kde_bandwidth_A * kf, sim.score_clip)
                ss = score.detach().cpu().numpy()
                score_std_sum += ss.std(axis=1)
                score_absmax = np.maximum(score_absmax, np.abs(ss).max(axis=1))
                n_score += 1
                qr, ancestors, n_repl, deaths, births = _birth_death(
                    q.reshape(R, N, A, 3), score, ancestors, sim, gen_fr)
                q = qr.reshape(R * N, A, 3)
                total_repl += n_repl.cpu()
                if int(n_repl.sum()) > 0:
                    Fr = F_phys.reshape(R, N, A, 3)
                    vr = v.reshape(R, N, A, 3)
                    for r in range(R):
                        di = deaths[r]
                        if di is None or di.numel() == 0:
                            continue
                        src = births[r]
                        Fr[r, di] = Fr[r].index_select(0, src)
                        vr[r, di] = (torch.randn(di.numel(), A, 3, generator=gen_fr,
                                                 device=device, dtype=dtype) * vsig[0])
                        for arr in (f_loc, phi, phi_unw, a_gate, theta_g):
                            arr[r, di] = arr[r].index_select(0, src)
                        for arr in (cage_idx, last_cage):
                            arr[r, di] = arr[r].index_select(0, src)
                        if lineage_diagnostics:
                            t_clone[r, di] = (step + 1) * sim.dt
                        # `crossings` is a CUMULATIVE per-walker count, not
                        # state: copying it re-counts the parent's completed
                        # transits once per clone, and the uniform target
                        # clones the window walkers preferentially, so the
                        # population total drifts UPWARD in the FR arm only.
                        # The clone starts from zero; cross_gate_samples (one
                        # entry per real event) stays the ground truth.
                        crossings[r, di] = 0
                        has_cage[r, di] = has_cage[r].index_select(0, src)
                    F_phys = Fr.reshape(R * N, A, 3)
                    v = system.pin_frame_com(vr.reshape(R * N, A, 3))

    u_of_phi = (usum_p / ucnt_p.clamp_min(1.0)).cpu().numpy()
    uhg_of_phi = (uhgsum_p / ucnt_p.clamp_min(1.0)).cpu().numpy()
    out = {"method": method, "grid": grid.cpu().numpy(), "dphi": dphi,
           "period": system.period, "xi_grid": grid.cpu().numpy() / system.k_phi,
           "temperature": system.temperature,
           "runtime_seconds": time.perf_counter() - t0,
           "total_replacement_events": total_repl.numpy(),
           "n_crossings": crossings.sum(-1).cpu().numpy(),
           "n_transit_events": int(sum(x.size for x in cross_gate)),
           "n_gate_dropped": n_gate_dropped,
           "cross_gate_samples": (np.concatenate(cross_gate) if cross_gate
                                  else np.zeros(0)),
           "fr_score_std": score_std_sum / max(n_score, 1),
           "fr_score_absmax": score_absmax,
           "gate_hist_cumulative": ghist_c.cpu().numpy(),
           "gate_edges": np.linspace(sim.gate_lo, sim.gate_hi, sim.n_gate_bins + 1),
           "u_of_phi": u_of_phi, "u_hostguest_of_phi": uhg_of_phi,
           "u_counts": ucnt_p.cpu().numpy(),
           # RAW (unsmoothed) post-burn-in accumulators.  Everything the engine
           # reports has already been convolved with K_abf, and that convolution
           # is not invertible -- so without these the estimator's bandwidth can
           # never be revisited after the fact.  They are what makes an
           # offline h-sweep possible from a single trajectory.
           "raw_fsum": fsum_p.cpu().numpy(), "raw_csum": csum_p.cpu().numpy()}
    if lineage_diagnostics:
        out.update({"lineage_fsum": fsum_ag.cpu().numpy(),
                    "lineage_csum": csum_ag.cpu().numpy(),
                    "cloneage_fsum": fsum_age.cpu().numpy(),
                    "cloneage_csum": csum_age.cpu().numpy(),
                    "clone_age_edges": np.asarray(clone_age_edges),
                    "ess_anc_bin": np.asarray(ess_bin_series)})
    for k in diag:
        out[k] = np.asarray(diag[k])
    if verbose:
        print(f"  {method:12s} T={system.temperature:g} R={R} N={N}: "
              f"{out['runtime_seconds']:.1f}s repl={out['total_replacement_events'].sum()} "
              f"transits={out['n_crossings'].sum()}", flush=True)
    return out


# --------------------------------- umbrella ---------------------------------
def run_umbrella(system: ZIF8System, sim: ZIF8SimConfig, centers, kappa,
                 n_steps, n_replicas, burn_in, sample_every, seed, init_pool,
                 verbose=True):
    """Harmonic umbrella windows on the CIRCULAR phi (BAOAB), saving phi, the
    energy split and the gate observables per retained frame -- the gate
    conditionals at the window are this stage's reference for J_gate."""
    device, dtype = system.device, system.dtype
    W, A = len(centers), system.n_atoms
    c = torch.sort(torch.as_tensor(centers, device=device, dtype=dtype)).values.reshape(W, 1)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    q = system.load_init_pool(init_pool, W, n_replicas, seed_offset=777)
    v = system.pin_frame_com(system.maxwell_velocities((W * n_replicas,), gen))
    m = system.mass[None, :, None]
    c1 = math.exp(-sim.gamma * sim.dt)
    c2 = math.sqrt(1.0 - c1 * c1)
    vsig = torch.sqrt(system.kT / system.mass)[None, :, None]

    def spring(qq):
        phi = system.cv_value(qq).reshape(W, n_replicas)
        return (system.bias_cartesian(-kappa * per.circular_distance(phi, c),
                                      W, n_replicas), phi)

    F = system.forces(q)
    Fu, phi = spring(q)
    phis, us, uhgs, gates, thetas = [], [], [], [], []
    t0 = time.perf_counter()
    for step in range(n_steps):
        v = v + (0.5 * sim.dt) * (F + Fu) / m
        q = q + (0.5 * sim.dt) * v
        noise = torch.randn(q.shape, generator=gen, device=device, dtype=dtype)
        v = system.pin_frame_com(c1 * v + c2 * vsig * noise)
        q = q + (0.5 * sim.dt) * v
        F = system.forces(q)
        Fu, phi = spring(q)
        v = v + (0.5 * sim.dt) * (F + Fu) / m
        if step >= burn_in and step % sample_every == 0:
            phis.append(phi.detach().cpu().numpy().copy())
            et, ehg, _ = system.potential_energy(q, split=True)
            us.append(et.reshape(W, n_replicas).cpu().numpy().copy())
            uhgs.append(ehg.reshape(W, n_replicas).cpu().numpy().copy())
            ag, th = system.gate_observables(q)
            gates.append(ag.reshape(W, n_replicas).cpu().numpy().copy())
            thetas.append(th.reshape(W, n_replicas).cpu().numpy().copy())
    if verbose:
        print(f"  umbrella: {W} windows x {n_replicas}, {n_steps} steps "
              f"-> {len(phis)} frames in {time.perf_counter() - t0:.1f}s", flush=True)
    return (np.array(phis), np.array(us), np.array(uhgs),
            np.array(gates), np.array(thetas))


def wham_periodic(phi_samples, centers, kappa, beta, n_bins=144, n_iter=20000,
                  tol=1e-10):
    """Circular histogram WHAM on [-pi, pi) with harmonic biases in phi."""
    W = phi_samples.shape[1]
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    db = edges[1] - edges[0]
    hist = np.stack([np.histogram(phi_samples[:, w, :].ravel(), bins=edges)[0]
                     for w in range(W)]).astype(float)
    n_w = hist.sum(axis=1)
    d = mids[None, :] - np.asarray(centers)[:, None]
    d = d - 2 * np.pi * np.round(d / (2 * np.pi))
    bias = 0.5 * kappa * d ** 2
    f_w = np.zeros(W)
    converged = False
    for _ in range(n_iter):
        denom = (n_w[:, None] * np.exp(beta * (f_w[:, None] - bias))).sum(axis=0)
        p = hist.sum(axis=0) / np.maximum(denom, 1e-300)
        p = p / max(p.sum() * db, 1e-300)
        f_new = -np.log(np.maximum((np.exp(-beta * bias) * p[None, :] * db),
                                   1e-300).sum(axis=1)) / beta
        if np.abs((f_new - f_new.mean()) - (f_w - f_w.mean())).max() < tol:
            f_w = f_new
            converged = True
            break
        f_w = f_new
    if not converged:
        raise RuntimeError(f"WHAM did not converge in {n_iter} iterations")
    with np.errstate(divide="ignore"):
        F = -np.log(np.maximum(p, 1e-300)) / beta
    return mids, F - np.nanmin(F), p, hist


def conditional_mean_periodic(phi_samples, values, n_bins=144):
    """<value | phi> on the circular grid, plus the per-bin counts."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(phi_samples.ravel(), edges) - 1, 0, n_bins - 1)
    s = np.bincount(idx, weights=values.ravel(), minlength=n_bins)
    cnt = np.bincount(idx, minlength=n_bins)
    return mids, np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan), cnt


__all__ = ["ZIF8SimConfig", "ZIF8System", "run_sampler", "run_umbrella",
           "engine_kwargs", "bin_counts_and_sum",
           "wham_periodic", "conditional_mean_periodic", "gate_hist",
           "js_divergence", "uniform_target", "mean_force_regularized",
           "GUEST", "KB", "COULOMB", "MASS_TO_INTERNAL", "TWO_PI"]
