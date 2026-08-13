"""Frozen hydration descriptors for NaCl (SPEC §4): n_NaO, n_ClH, n_ClO, n_bridge.

Rational switch ``s(x; R0) = (1 - (x/R0)^6) / (1 - (x/R0)^12)`` -- the project-standard form
(methane n_gap) -- on minimum-image distances.  ``R0_NaO = 0.315 nm`` is fixed in the SPEC;
``R0_ClH`` and ``R0_ClO`` are frozen from the first minima of the Stage-0 reference RDFs by
``freeze_descriptors`` BEFORE any ABF or screen data exist, and loaded from
``results/nacl/stage0/descriptor_freeze.json`` afterwards.

Everything is batched ``(B, N, 3) -> (B,)`` and cheap relative to a force call; descriptors are
recorded at diagnostic cadence, never used by any sampler.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import system as nsys

R0_NAO_NM = 0.315                      #: SPEC §3.2, fixed in advance
FREEZE_PATH = nsys.STAGE0 / "descriptor_freeze.json"


def rational_switch(x, r0):
    t6 = (x / r0) ** 6
    return (1.0 - t6) / (1.0 - t6 * t6 + 1e-12)


def _min_image(d, L):
    return d - L * torch.round(d / L)


class HydrationDescriptors:
    """Batched n_NaO, n_ClH, n_ClO, n_bridge for positions ``(B, 2465, 3)``."""

    def __init__(self, waters, box_nm, r0_nao=None, r0_clh=None, r0_clo=None,
                 device=None):
        w = torch.as_tensor(np.asarray(waters), device=device, dtype=torch.long)
        self.iO = w[:, 0]
        self.iH = torch.stack([w[:, 1], w[:, 2]], dim=1)       # (n_w, 2)
        self.L = float(box_nm)
        if r0_clh is None or r0_clo is None:
            frozen = json.loads(Path(FREEZE_PATH).read_text())
            r0_clh = frozen["R0_ClH_nm"] if r0_clh is None else r0_clh
            r0_clo = frozen["R0_ClO_nm"] if r0_clo is None else r0_clo
        self.r0_nao = float(r0_nao if r0_nao is not None else R0_NAO_NM)
        self.r0_clh = float(r0_clh)
        self.r0_clo = float(r0_clo)

    def _dist(self, x, i_center, idx):
        d = _min_image(x[:, idx, :] - x[:, i_center, None, :], self.L)
        return d.norm(dim=-1)                                   # (B, len(idx))

    def compute(self, x):
        """Returns dict of ``(B,)`` tensors: n_NaO, n_ClH, n_ClO, n_bridge, n_bridge_hard."""
        iNa, iCl = nsys.ION_INDEX
        r_NaO = self._dist(x, iNa, self.iO)                     # (B, n_w)
        r_ClO = self._dist(x, iCl, self.iO)
        dH = _min_image(x[:, self.iH.reshape(-1), :] - x[:, iCl, None, :], self.L)
        r_ClH = dH.norm(dim=-1).view(x.shape[0], -1, 2)         # (B, n_w, 2)

        s_NaO = rational_switch(r_NaO, self.r0_nao)
        s_ClH = rational_switch(r_ClH, self.r0_clh)
        s_ClO = rational_switch(r_ClO, self.r0_clo)
        r_ClH_min = r_ClH.min(dim=-1).values
        s_ClH_min = rational_switch(r_ClH_min, self.r0_clh)

        hard = ((r_NaO < self.r0_nao) & (r_ClO < self.r0_clo)).to(x.dtype)
        return dict(
            n_NaO=s_NaO.sum(-1), n_ClH=s_ClH.sum(dim=(-2, -1)), n_ClO=s_ClO.sum(-1),
            n_bridge=(s_NaO * s_ClH_min).sum(-1), n_bridge_hard=hard.sum(-1),
        )

    def Y(self, x):
        """The primary descriptor triple ``(B, 3)``: (n_NaO, n_ClH, n_bridge)."""
        d = self.compute(x)
        return torch.stack([d["n_NaO"], d["n_ClH"], d["n_bridge"]], dim=-1)


def freeze_descriptors(r_grid_nm, g_clh, g_clo, out_path=FREEZE_PATH):
    """Freeze R0_ClH / R0_ClO from the first minima of reference RDFs (SPEC §3.2).

    Deterministic rule: the first local minimum of the smoothed RDF after its first maximum,
    rounded to 0.005 nm.  Refuses to overwrite an existing freeze.
    """
    out_path = Path(out_path)
    if out_path.exists():
        raise RuntimeError(f"{out_path} exists; the descriptor freeze is not renegotiable")

    def first_min(r, g):
        g = np.convolve(np.asarray(g), np.ones(5) / 5.0, mode="same")
        imax = int(np.argmax(g))
        rel = g[imax:]
        imin = imax + next(i for i in range(1, len(rel) - 1)
                           if rel[i] < rel[i - 1] and rel[i] <= rel[i + 1])
        return round(float(r[imin]) / 0.005) * 0.005

    frozen = dict(R0_NaO_nm=R0_NAO_NM,
                  R0_ClH_nm=first_min(r_grid_nm, g_clh),
                  R0_ClO_nm=first_min(r_grid_nm, g_clo),
                  rule="first local minimum after first maximum, 5-bin boxcar, rounded 0.005 nm")
    out_path.write_text(json.dumps(frozen, indent=2))
    return frozen
