"""Two-dimensional joint dihedral CV ``xi = (phi1, phi2)`` and the exact 2-D ABF
vector generalized mean force (den Otter), for pentane on the torus ``T^2``.

For CVs ``phi_a`` (a in {1,2}) with per-CV coordinate gradients ``g_a = grad phi_a`` and
Hessians ``H_a = grad^2 phi_a`` (exact autodiff, per-dihedral 12x12 scattered into the
full atomic layout), the Gram matrix is ``G_{ab} = g_a . g_b``.  The dual (Moore--Penrose)
vector fields ``w_a = sum_b (G^{-1})_{ab} g_b`` satisfy ``w_a . g_c = delta_{ac}`` exactly.
The component of the local mean force whose conditional average is ``dF/dphi_a`` is

    f_a = grad V . w_a - beta^{-1} div(w_a).

The divergence is computed **analytically** from the per-CV gradients and Hessians (no
nested autodiff), using ``d(G^{-1})/dq = -G^{-1} (dG/dq) G^{-1}`` and
``d G_{cd}/dq_i = (H_c g_d)_i + (H_d g_c)_i``:

    div(w_a) = sum_b (G^{-1})_{ab} Lap(phi_b)
             - sum_{b,c,d} (G^{-1})_{ac} (G^{-1})_{db} [ T_{bcd} + T_{bdc} ],

    T_{bcd} = g_b . (H_c g_d) = einsum('pbi,pcij,pdj->pbcd', g, H, g),
    Lap(phi_b) = tr(H_b),
    grad V . w_a = - sum_b (G^{-1})_{ab} (F . g_b).

Validated (tests) by: (i) biorthogonality ``w_a . g_c = delta`` (exact); (ii) analytic
``div(w_a)`` vs central finite differences of the analytic ``w_a`` field; (iii) the
decoupled reduction ``f_a -> V4'(phi_a)`` at rigid equilibrium (LJ off); (iv) CPU/GPU
parity.  The Gram condition number/eigenvalues/determinant are exposed for monitoring;
optional ridge regularisation is documented and its activation counted (never silent).

The Cartesian ABF bias force is ``+ sum_a A'_a(z) g_a`` (rows of the CV Jacobian), applied
through the same ``+f grad(phi)`` channel as the scalar dihedral CV.
"""
from __future__ import annotations

import torch

from .cv import _grad_phi4, _hess_phi4, DihedralCV

EPS = 1.0e-12


def sym2x2_eigvals(G):
    """Ascending eigenvalues of a batch of SYMMETRIC 2x2 matrices, in closed form.

    ``torch.linalg.eigvalsh`` dispatches this to ``cusolverDnXsyevBatched``, which is both
    absurd for a 2x2 -- the characteristic polynomial is a quadratic -- and a hard scaling
    wall: it raises ``CUSOLVER_STATUS_INTERNAL_ERROR`` somewhere between batch 32768 and
    65536, which capped how much of an H200 this sampler could use.

    For ``G = [[a, b], [b, c]]`` the roots of ``lam^2 - (a+c) lam + (ac - b^2)`` are

        lam = (a + c)/2 +/- sqrt( ((a - c)/2)^2 + b^2 ),

    and written that way the discriminant is a sum of squares, so it is never negative from
    rounding and the square root is always real -- which the naive ``(a+c)^2 - 4(ac-b^2)``
    form does not guarantee for nearly-degenerate G.  Agreement with ``eigvalsh`` is asserted
    in ``tests/test_alkanes_cv2d.py``.
    """
    a, b, c = G[..., 0, 0], G[..., 0, 1], G[..., 1, 1]
    half_tr = 0.5 * (a + c)
    half_df = 0.5 * (a - c)
    rad = torch.sqrt(half_df * half_df + b * b)
    return half_tr - rad, half_tr + rad


class JointDihedralCV2D:
    """Two dihedrals ``(atoms_a, atoms_b)`` (default pentane phi1=(0123), phi2=(1234))."""

    def __init__(self, atoms_a=(0, 1, 2, 3), atoms_b=(1, 2, 3, 4), n_atoms=5, ridge=0.0,
                 reg_threshold=1e-8):
        self.atoms = (tuple(atoms_a), tuple(atoms_b))
        assert all(len(a) == 4 for a in self.atoms)
        self.n_atoms = int(n_atoms)
        self.ridge = float(ridge)          # constant Tikhonov ridge on G (0 => exact inverse)
        self.reg_threshold = float(reg_threshold)
        self._cv1 = DihedralCV(atoms_a)
        self._cv2 = DihedralCV(atoms_b)
        self._reg_counter = None           # lazy GPU tensor; counts lam_min < reg_threshold
        self.reg_activations = 0           # host copy, refreshed by reg_activation_count()
        # Per-device caches of the constant index tensors used below.  Building them from
        # Python lists on every call is a host->device copy, which (a) costs a launch and a
        # sync-free-but-serialising H2D transfer per call and (b) is forbidden inside CUDA
        # graph capture (alanine.graphed).  Caching changes no arithmetic: the same integer
        # indices select the same elements.
        self._idx_cache = {}

    # -- values --------------------------------------------------------------
    def values(self, q):
        """(phi1, phi2) each ``(B,)`` for ``q`` of shape ``(B, n_atoms, 3)``."""
        return self._cv1.value(q), self._cv2.value(q)

    # -- per-CV grad + Hessian in the full coordinate layout -----------------
    def _grad_hess_full(self, q):
        """Return ``g (B,2,A,3)`` and ``H (B,2,A*3,A*3)`` for the two dihedrals.

        Each dihedral only touches its 4 atoms; grad/Hessian are computed on those 12
        coordinates and scattered into the full ``A*3`` coordinate space (zeros elsewhere),
        so the coupling in ``G`` comes purely from shared atoms.
        """
        B, A, _ = q.shape
        n = A * 3
        g = q.new_zeros(B, 2, A, 3)
        H = q.new_zeros(B, 2, n, n)
        for a, atoms in enumerate(self.atoms):
            atom_idx, lin_idx = self._scatter_indices(a, n, q.device)
            sub = q.index_select(1, atom_idx).reshape(B, 12).detach()
            ga = _grad_phi4(sub)                    # (B,12)
            Ha = _hess_phi4(sub)                    # (B,12,12)
            g[:, a].index_copy_(1, atom_idx, ga.reshape(B, 4, 3))
            # scatter the 12x12 Hessian into the full n x n layout
            H[:, a].reshape(B, n * n).scatter_add_(
                1, lin_idx[None, :].expand(B, -1), Ha.reshape(B, 144))
        return g, H

    def _scatter_indices(self, a, n, device):
        """Cached ``(atom_idx (4,), lin_idx (144,))`` for dihedral ``a`` on ``device``.

        ``lin_idx = rows * n + cols`` over the 12 flat coordinate indices of the dihedral's
        four atoms -- the same values the previous per-call construction produced.
        """
        key = (a, n, str(device))
        hit = self._idx_cache.get(key)
        if hit is None:
            atoms = self.atoms[a]
            atom_idx = torch.tensor(list(atoms), dtype=torch.long, device=device)
            idx = torch.tensor([at * 3 + c for at in atoms for c in range(3)],
                               dtype=torch.long, device=device)     # (12,)
            rows = idx[:, None].expand(12, 12).reshape(-1)
            cols = idx[None, :].expand(12, 12).reshape(-1)
            hit = (atom_idx, (rows * n + cols).contiguous())
            self._idx_cache[key] = hit
        return hit

    def grad_only(self, q):
        """Cheap ``(phi (B,2), g (B,2,A,3))`` using only per-CV gradients (NO Hessian).

        Used every step for the bias-force application and basin tracking; the expensive
        Hessian path (:meth:`geometry`) is needed only when the mean-force estimator is
        accumulated (which may be strided).  Outputs detached.
        """
        B, A, _ = q.shape
        g = q.new_zeros(B, 2, A, 3)
        for a, atoms in enumerate(self.atoms):
            sub = q[:, atoms, :].reshape(B, 12).detach()
            g[:, a][:, atoms, :] = _grad_phi4(sub).reshape(B, 4, 3)
        phi1, phi2 = self.values(q)
        return torch.stack([phi1, phi2], dim=-1).detach(), g.detach()

    def geometry(self, q):
        """Return a dict with phi, per-CV grad ``g (B,2,A,3)``, ``div_v (B,2)`` and Gram
        diagnostics (``G, Ginv, lam_min, lam_max, cond, det``).  Outputs detached."""
        B, A, _ = q.shape
        n = A * 3
        g, H = self._grad_hess_full(q)
        gflat = g.reshape(B, 2, n)                                  # (B,2,n)
        G = torch.einsum("pbi,pci->pbc", gflat, gflat)             # (B,2,2)
        # eigen-diagnostics (symmetric 2x2)
        lam_min, lam_max = sym2x2_eigvals(G)                       # ascending
        det = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
        eye = torch.eye(2, device=q.device, dtype=q.dtype)[None]
        # near-singular guard: branchless (no host sync). A constant tiny ridge stabilises
        # inv; an ADAPTIVE ridge is added only in the rare rows where lam_min < threshold
        # (via torch.where, evaluated on-device). Activations are tallied on-device and read
        # back lazily by reg_activation_count() -- never per step.
        bad = (lam_min < self.reg_threshold)
        adaptive = torch.where(bad[:, None, None], 1e-6 * eye, torch.zeros_like(eye))
        Greg = G + self.ridge * eye + adaptive
        if self._reg_counter is None:
            self._reg_counter = torch.zeros((), device=q.device, dtype=torch.long)
        elif self._reg_counter.device != bad.device:      # one object used on CPU then GPU (tests)
            self._reg_counter = self._reg_counter.to(bad.device)
        self._reg_counter += bad.sum()       # in place: one fixed device address (graph-safe)
        # inv_ex is the same cuSOLVER/LAPACK routine as inv (bitwise-identical result) minus
        # the host-side singularity check, so it is CUDA-graph capturable and sync-free.  A
        # singular G cannot reach it: the adaptive ridge above already excludes lam_min < 1e-8.
        Ginv = torch.linalg.inv_ex(Greg).inverse                   # (B,2,2)
        lap = torch.diagonal(H, dim1=-2, dim2=-1).sum(-1)          # (B,2) tr H_b
        # T[p,b,c,d] = g_b . (H_c g_d)
        T = torch.einsum("pbi,pcij,pdj->pbcd", gflat, H, gflat)    # (B,2,2,2)
        U = T + T.transpose(-1, -2)                                # symmetrise in (c,d)
        term1 = torch.einsum("pab,pb->pa", Ginv, lap)
        term2 = torch.einsum("pac,pdb,pbcd->pa", Ginv, Ginv, U)
        div_v = term1 - term2                                      # (B,2)
        phi1, phi2 = self.values(q)
        return {"phi": torch.stack([phi1, phi2], dim=-1).detach(),
                "g": g.detach(), "gflat": gflat.detach(),
                "div_v": div_v.detach(), "G": G.detach(), "Ginv": Ginv.detach(),
                "lam_min": lam_min.detach(), "lam_max": lam_max.detach(),
                "cond": (lam_max / lam_min.clamp_min(EPS)).detach(), "det": det.detach()}

    def reg_activation_count(self):
        """Host-side count of near-singular Gram regularisations (lazy sync; call at end)."""
        if self._reg_counter is not None:
            self.reg_activations = int(self._reg_counter.item())
        return self.reg_activations

    def dual_fields(self, q):
        """Dual vector fields ``w_a (B,2,A,3)`` with ``w_a . g_c = delta_ac`` (validation)."""
        geo = self.geometry(q)
        g = geo["g"]                                               # (B,2,A,3)
        Ginv = geo["Ginv"]
        w = torch.einsum("pab,pbAc->paAc", Ginv, g)               # (B,2,A,3)
        return w, geo

    def local_mean_force(self, q, physical_forces, beta):
        """Vector local mean force ``f (B,2)`` (den Otter), plus phi ``(B,2)`` and per-CV
        gradients ``g (B,2,A,3)`` for the bias-force application.

        ``physical_forces = -grad V`` of shape ``(B, n_atoms, 3)``.
        """
        geo = self.geometry(q)
        g = geo["g"]; Ginv = geo["Ginv"]; div_v = geo["div_v"]
        Fdotg = (physical_forces[:, None] * g).sum(dim=(-2, -1))  # (B,2)  = F . g_b
        gradV_dot_w = -torch.einsum("pab,pb->pa", Ginv, Fdotg)    # -Ginv (F.g)
        f = gradV_dot_w - (1.0 / beta) * div_v                    # (B,2)
        return f, geo["phi"], g, geo

    # -- divergence autodiff reference (slow; tests only) --------------------
    def divergence_autodiff(self, q, eps=1e-5):
        """Central-FD divergence of the analytic dual field ``w_a`` (validates ``div_v``).

        ``div(w_a) = sum_i d(w_a)_i/dq_i`` by central differences of :meth:`dual_fields`.
        Returns ``(B,2)``.
        """
        B, A, _ = q.shape
        div = q.new_zeros(B, 2)
        for at in range(A):
            for c in range(3):
                qp = q.clone(); qp[:, at, c] += eps
                qm = q.clone(); qm[:, at, c] -= eps
                wp, _ = self.dual_fields(qp)
                wm, _ = self.dual_fields(qm)
                div += (wp[:, :, at, c] - wm[:, :, at, c]) / (2 * eps)
        return div


def abf_bias_force_2d(g, mean_force_at):
    """Cartesian 2-D ABF bias force ``+ sum_a A'_a g_a`` -> ``(B, n_atoms, 3)``.

    ``g`` is ``(B, 2, n_atoms, 3)`` and ``mean_force_at`` is ``(B, 2)`` (the projected
    gradient of the bias potential at each replica along each CV).
    """
    return (mean_force_at[:, :, None, None] * g).sum(dim=1)
