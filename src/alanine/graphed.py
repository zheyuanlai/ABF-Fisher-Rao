"""CUDA-graph replay of the alanine hot-loop kernels (bitwise-identical to eager).

Why this exists
---------------
At the production batch (16 seeds x 2048 walkers x 22 atoms) one sampler step spends
~20 ms in ``torch.func.vmap(hessian(_phi4))`` (two dihedrals) and ~9 ms in the autograd
physical force, yet the GPU work in both is tiny: every one of the several hundred kernels
runs for a few microseconds and the step is **launch-bound** -- an H200 sits idle most of
the step.  ``torch.compile`` cannot trace ``torch.func`` transforms, and rewriting the
Hessian analytically would change the arithmetic.

CUDA-graph capture records exactly the eager kernel sequence once and replays it with a
single launch, so the arithmetic, the kernels and their order are **unchanged**: outputs
are bitwise identical to the eager call (``tests/test_alanine_graphed.py`` asserts
``torch.equal``).  The physics, the estimator and the RNG streams are untouched -- the
random draws stay OUTSIDE the graphs (eager), so the dynamical noise stream is exactly the
one the eager engine consumes.

Usage
-----
    ff  = GraphedForces(tff, batch=R * N)             # force_fn for run_sampler_ala
    cvg = GraphedCV(cv, batch=R * N, physical_forces_example=..., beta=beta)
    run_sampler_ala(method, tff, cvg, sim, ..., force_fn=ff)

Both wrappers take a fixed batch size (a CUDA graph has static shapes) and refuse any
other; they return CLONES of the graph's static output buffers so callers may hold the
results across replays (the sampler carries ``floc, phi, gfull, geo`` into the next step).
"""
from __future__ import annotations

import torch


class _GraphedCall:
    """Capture ``fn(*static_inputs)`` once; ``__call__`` copies inputs in, replays, clones out."""

    def __init__(self, fn, example_inputs, n_warmup=3):
        self.fn = fn
        self.static_in = [x.detach().clone() for x in example_inputs]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmup):                       # warm up allocator / kernels
                out = fn(*self.static_in)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = fn(*self.static_in)
        self._shapes = [tuple(x.shape) for x in self.static_in]
        self._dtypes = [x.dtype for x in self.static_in]

    @staticmethod
    def _clone(o):
        if isinstance(o, torch.Tensor):
            return o.clone()
        if isinstance(o, dict):
            return {k: _GraphedCall._clone(v) for k, v in o.items()}
        if isinstance(o, (tuple, list)):
            return type(o)(_GraphedCall._clone(v) for v in o)
        return o

    def __call__(self, *inputs):
        if len(inputs) != len(self.static_in):
            raise ValueError("graphed call: wrong number of inputs")
        for buf, x, shp, dt in zip(self.static_in, inputs, self._shapes, self._dtypes):
            if tuple(x.shape) != shp or x.dtype != dt:
                raise ValueError(f"graphed call: static shape {shp}/{dt} but got "
                                 f"{tuple(x.shape)}/{x.dtype}")
            buf.copy_(x)
        self.graph.replay()
        return self._clone(self.static_out)


class GraphedForces:
    """``force_fn(x) -> -grad E`` via a captured graph of ``tff.forces``.  Bitwise = eager."""

    def __init__(self, tff, batch, device="cuda", dtype=torch.float64):
        self.tff = tff
        self.batch = int(batch)
        ex = torch.zeros(self.batch, tff.n_atoms, 3, device=device, dtype=dtype)
        # a non-degenerate example configuration avoids 0/0 in the warm-up passes
        ex += 0.1 * torch.arange(tff.n_atoms, device=device, dtype=dtype)[None, :, None]
        ex += 0.01 * torch.arange(3, device=device, dtype=dtype)[None, None, :]
        self._call = _GraphedCall(lambda x: tff.forces(x), [ex])

    def __call__(self, x):
        return self._call(x)


class GraphedCV:
    """Graphed ``local_mean_force``; ``scatter_bias``/``values`` fall through to ``cv``.

    ``run_sampler_ala`` only consumes ``cv.local_mean_force`` and (optionally)
    ``cv.scatter_bias``; the latter is a single ``index_copy_`` and stays eager.
    """

    def __init__(self, cv, batch, beta, n_atoms, device="cuda", dtype=torch.float64):
        self.cv = cv
        self.batch = int(batch)
        self.beta = float(beta)
        exq = torch.zeros(self.batch, n_atoms, 3, device=device, dtype=dtype)
        exq += 0.1 * torch.arange(n_atoms, device=device, dtype=dtype)[None, :, None]
        exq[:, :, 1] += 0.05 * torch.sin(torch.arange(n_atoms, device=device, dtype=dtype))[None]
        exq[:, :, 2] += 0.03 * torch.cos(torch.arange(n_atoms, device=device, dtype=dtype))[None]
        exf = torch.zeros_like(exq)
        b = self.beta
        self._call = _GraphedCall(lambda q, f: cv.local_mean_force(q, f, b), [exq, exf])
        if hasattr(cv, "scatter_bias"):
            self.scatter_bias = cv.scatter_bias
        # geometry.reg_counter is a device tensor updated inside the graph (in-place add
        # on a captured tensor); the eager attribute is stale after capture, so expose the
        # graph's own view for end-of-run reporting.
        self.reg_activation_count = getattr(cv, "reg_activation_count", None)

    def local_mean_force(self, q, physical_forces, beta):
        if float(beta) != self.beta:
            raise ValueError("GraphedCV was captured for a different beta")
        return self._call(q, physical_forces)

    def values(self, q):
        return self.cv.values(q)
