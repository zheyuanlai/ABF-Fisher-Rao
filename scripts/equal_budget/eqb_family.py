"""Mechanism-campaign extension of the equal-budget metrics library: system 'gateway_family'
(docs/mechanism/SCIENTIFIC_PLAN.md sections 5-6).

WHY A SEPARATE MODULE.  eqb_metrics.py is part of the equal-budget scoring-code digest (eqb_metrics.CODE_FILES ->
code_provenance -> the code_digest of every committed per-run cache key and the provenance of every committed
summary.json).  Editing one byte of it makes every committed equal-budget analysis STALE by the pipeline's own rule
(audit_completeness SUMMARY_STALE, plot_synthesis refuses) although no number changes.  So eqb_metrics.py stays
byte-identical to the committed file and everything the mechanism cells need lives here.  This module only ADDS:
every dispatcher below calls the eqb_metrics function itself (same object, same arguments) for gateway / lta300 /
lta150, so equal-budget results, cache keys, digests and outputs are unchanged.  tests/test_mech_pipeline.py checks
that eqb_metrics.py still hashes to the sha256 recorded in the committed equal-budget provenance.

The callers (analyze_ladder, plot_config, plot_synthesis, audit_completeness, scripts/mechanism/*) use eqb_metrics
(as M) for constants and statistics and this module (as MF) for everything that depends on the system or config:
SYSTEMS, system_kind, default_config_path, load_run, make_scorer, check_plan, run_metrics, threshold_floors,
reference_path, config_digest, code_provenance, run_cache_key.

System 'gateway_family' (kind 'gateway': the gateway result contract, save grid, marginal, establishment and genealogy
metrics).  Its config is a CELL config (configs/mechanism/cells/<experiment>/<variant>.json, scripts/mechanism/
make_cell_configs.py); a cell config is accepted ONLY with --system gateway_family and gateway_family ONLY with a
cell config (check_invocation).
  e_F, e_Fp, e_Fp_stat  the accepted analytic scorer with the SAME primary reference F*, F*' for every variant and
                        lambda (checked at construction against results/equal_budget_v2/references/gateway_reference.npz,
                        bitwise);
  e_F_em, e_Fp_em, e_Fp_em_stat  the per-dynamics frozen-x EM-consistent mean force (family_mean_force; alpha family
                        4Hx(x^2-1) + ((1-alpha)/beta) w'/w + (alpha/beta)(w'/w) / (1 - lam w^(2 alpha) h/2), shifted
                        fibre F*'), verified against the cell's frozen results/mechanism/references/<experiment>_
                        <variant>_reference.npz (scripts/mechanism/build_references.py); at alpha 1, lam 1 it is the
                        equal-budget EM reference bitwise;
  tau                   (plan section 6) on e_F, on the floor-free e_Fp_stat (thresholds.e_Fp_stat; also on the common
                        budget grid, tau_bgrid_e_Fp_stat_*) and on TV_half; raw e_Fp / e_Fp_em are REPORTED, and tau is
                        computed only at their thresholds above the hard zero-noise floor (0.03227): a threshold at or
                        below it gets no tau at all (its tau keys are removed from the per-run scalars);
  check_plan            the run's engine must be the cell's engine_version (gateway_ladder_numba/2 for the reuse cells,
                        gateway_family_numba/1 otherwise: one engine per cell); the model fields variant / alpha / kappa
                        / lam of cfg / meta must equal the cell's; a gateway_ladder_numba/2 file carries none of them and
                        is read as variant alpha, alpha 1, kappa None, lam 1 (FAMILY_MODEL_DEFAULTS); the 'system' label
                        may be 'gateway' or 'gateway_family'.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os

import numpy as np

import eqb_metrics as M
from eqb_metrics import MetricsError, NB, ROOT, THR_NAMES, sha256_file

FAMILY = "gateway_family"
SYSTEMS = dict(M.SYSTEMS)
SYSTEMS[FAMILY] = dict(kind="gateway", config=None, family=True)
# the mechanism-campaign model fields and the values a file WITHOUT them (gateway_ladder_numba/2) represents
FAMILY_MODEL_KEYS = ("variant", "alpha", "kappa", "lam")
FAMILY_MODEL_DEFAULTS = dict(variant="alpha", alpha=1.0, kappa=None, lam=1.0)
FAMILY_ENGINES = ("gateway_ladder_numba/2", "gateway_family_numba/1")
FAMILY_SYSTEM_LABELS = ("gateway", "gateway_family")
CELL_KEYS = ("cell", "results_dir", "analysis_dir", "fig_dir")
# outputs of a mechanism cell (or of any gateway_family invocation) may never land in the equal-budget trees
PROTECTED_OUTPUT_DIRS = (os.path.join(ROOT, "results", "equal_budget_v2"), os.path.join(ROOT, "figures", "equal_budget_v2"))
# scoring code of the family: the gateway files + this module (eqb_metrics.CODE_FILES is not touched)
FAMILY_CODE_FILES = [os.path.abspath(__file__)]


# ------------------------------------------------------------------------------------------------- identity
def is_family(system):
    """True for the mechanism campaign's gateway family (per-cell config, per-dynamics secondary reference)."""
    return bool(SYSTEMS.get(system, {}).get("family"))


def system_kind(system):
    if is_family(system):
        return SYSTEMS[system]["kind"]
    return M.system_kind(system)


def engine_key(system):
    """The run_ladder system key whose engine owns the save-grid rule of ``system`` (gateway family -> gateway)."""
    return "gateway" if is_family(system) else system


def default_config_path(system):
    if is_family(system):
        raise MetricsError(f"system {system!r} has no default config: pass the cell config "
                           f"(configs/mechanism/cells/<experiment>/<variant>.json) with --config")
    return M.default_config_path(system)


def is_cell(P):
    """A mechanism-campaign CELL config (scripts/mechanism/make_cell_configs.py): it names its own result,
    analysis and figure locations."""
    return bool(P) and all(k in P for k in CELL_KEYS)


def check_invocation(system, P, where=""):
    """A cell config is analysed ONLY as system gateway_family, and gateway_family ONLY with a cell config (a cell
    scored as 'gateway' would silently get the equal-budget secondary reference; 'gateway_family' without a cell
    config would have no cell locations and no output guard)."""
    fam, cell = is_family(system), is_cell(P)
    if fam and not cell:
        raise MetricsError(f"{where or 'config'}: --system {system} needs a mechanism CELL config "
                           f"(keys {', '.join(CELL_KEYS)}; configs/mechanism/cells/<experiment>/<variant>.json)")
    if cell and not fam:
        raise MetricsError(f"{where or 'config'}: the mechanism cell config {P.get('experiment')}/{P.get('cell')} "
                           f"must be analysed with --system {FAMILY}, not --system {system}")
    if cell and P.get("system") not in (None, FAMILY):
        raise MetricsError(f"{where or 'config'}: cell config system {P.get('system')!r} != {FAMILY!r}")


def _abs(p):
    return os.path.abspath(p if os.path.isabs(p) else os.path.join(ROOT, p))


def cell_results_root(P):
    """The root under which a cell's runs live as <root>/<out_dir>/N<N>/... (out_dir = basename of results_dir)."""
    rd = _abs(P["results_dir"])
    if os.path.basename(rd.rstrip(os.sep)) != P["out_dir"]:
        raise MetricsError(f"cell config: out_dir {P['out_dir']!r} is not the basename of results_dir {P['results_dir']!r}")
    return os.path.dirname(rd.rstrip(os.sep))


def cell_analysis_dir(P):
    return _abs(P["analysis_dir"])


def cell_fig_dir(P, fig_root=None):
    """A cell's figure directory: fig_dir of the config, or <fig_root>/<cell> when a figure root is given."""
    return os.path.join(os.path.abspath(fig_root), P["cell"]) if fig_root else _abs(P["fig_dir"])


def cell_ledger(P):
    """(ledger path, row filter) of a cell: the config's 'ledger' (production: results/mechanism/<experiment>/
    ledger.csv written by scripts/mechanism/run_cells.py, one per experiment, rows filtered by 'ledger_filter' =
    {experiment, variant}; reuse cells: the equal-budget results/equal_budget_v2/gateway/ledger.csv, no filter), else
    <results_dir>/ledger.csv unfiltered."""
    path = _abs(P["ledger"]) if P.get("ledger") else os.path.join(_abs(P["results_dir"]), "ledger.csv")
    return path, (dict(P["ledger_filter"]) if P.get("ledger_filter") else None)


def guard_output(P, path, system=None):
    """Refuse to write a mechanism cell's output (a cell config, or any gateway_family invocation) into the
    equal-budget trees (results/ or figures/equal_budget_v2)."""
    if not (is_cell(P) or is_family(system)):
        return
    rp = os.path.realpath(os.path.abspath(path))
    for d in PROTECTED_OUTPUT_DIRS:
        dd = os.path.realpath(d)
        if rp == dd or rp.startswith(dd + os.sep):
            raise MetricsError(f"refusing to write mechanism cell {(P or {}).get('experiment')}/{(P or {}).get('cell')} "
                               f"output to {path}: inside the protected equal-budget tree {d}")


# ------------------------------------------------------------------------------------------------- provenance
def config_digest(P):
    """eqb_metrics.config_digest; a cell's digest also covers its engine, locations, reference and model."""
    if not is_cell(P):
        return M.config_digest(P)
    keep = dict(thresholds=P["thresholds"], engine_cfg=P["engine_cfg"], B=P["B"], system=P.get("system"),
                T_K=P.get("T_K"), out_dir=P.get("out_dir"), physical_checkpoints_t=P.get("physical_checkpoints_t"),
                profile_snapshot_u=P.get("profile_snapshot_u"), far=M.FAR, est_reliable_p=M.EST_RELIABLE_P,
                cell={k: P.get(k) for k in ("experiment", "cell", "results_dir", "engine_version", "reference_file",
                                            "model")})
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def code_provenance(system):
    """eqb_metrics.code_provenance; the family: the gateway scoring files + this module."""
    if not is_family(system):
        return M.code_provenance(system)
    files = M.CODE_FILES["common"] + M.CODE_FILES["gateway"] + FAMILY_CODE_FILES
    shas = {os.path.relpath(p, ROOT): sha256_file(p) for p in files}
    digest = hashlib.sha256(json.dumps(shas, sort_keys=True).encode()).hexdigest()[:16]
    return dict(files=shas, digest=digest)


def reference_path(system, P=None):
    """eqb_metrics.reference_path; a gateway-family cell: its own frozen reference (config reference_file)."""
    if not is_family(system):
        return M.reference_path(system, P)
    if not (P or {}).get("reference_file"):
        raise MetricsError("gateway_family needs a cell config with reference_file")
    return _abs(P["reference_file"])


def run_cache_key(path, system, P, code=None, ref_sha=None):
    """eqb_metrics.run_cache_key with the family's config digest, scoring code and reference."""
    if not is_family(system):
        return M.run_cache_key(path, system, P, code=code, ref_sha=ref_sha)
    return dict(src_sha256=sha256_file(path), src_size=int(os.path.getsize(path)), metrics_version=M.METRICS_VERSION,
                config_digest=config_digest(P), code_digest=(code or code_provenance(system))["digest"],
                reference_sha256=ref_sha or sha256_file(reference_path(system, P)))


def load_run(path, system):
    """eqb_metrics.load_run (the family reads the gateway result contract)."""
    return M.load_run(path, "gateway" if is_family(system) else system)


# ------------------------------------------------------------------------------------------------- the family model
FAMILY_SECONDARY_DEFINITION = (
    "per-dynamics frozen-x Euler-Maruyama-consistent mean force at the cell's h and lambda: alpha family "
    "4Hx(x^2-1) + ((1-alpha)/beta) w'/w + (alpha/beta)(w'/w) / (1 - lam w^(2 alpha) h/2) (the EM y-update "
    "y <- (1 - lam w^(2 alpha) h) y + sqrt(2 lam h/beta) z has stationary variance 1/(beta w^(2 alpha) "
    "(1 - lam w^(2 alpha) h/2))); shifted fibre: F*'(x) exactly (the frozen-x EM chain of y has mean m(x) exactly)")


def family_model(engine_cfg):
    """(variant, alpha, kappa, lam) of a cell's engine_cfg, validated; a missing field takes FAMILY_MODEL_DEFAULTS."""
    m = {k: engine_cfg.get(k, FAMILY_MODEL_DEFAULTS[k]) for k in FAMILY_MODEL_KEYS}
    if m["variant"] == "alpha":
        if m["alpha"] is None or m["kappa"] is not None:
            raise MetricsError(f"variant alpha needs alpha (and kappa None): {m}")
        m["alpha"] = float(m["alpha"])
    elif m["variant"] == "shift":
        if m["kappa"] is None or m["alpha"] is not None:
            raise MetricsError(f"variant shift needs kappa (and alpha None): {m}")
        m["kappa"] = float(m["kappa"])
    else:
        raise MetricsError(f"unknown family variant {m['variant']!r}")
    if m["lam"] is None or not float(m["lam"]) > 0:
        raise MetricsError(f"lambda must be > 0: {m}")
    m["lam"] = float(m["lam"])
    return m


def is_original_dynamics(model):
    """alpha 1, lambda 1: the equal-budget gateway itself."""
    return model["variant"] == "alpha" and model["alpha"] == 1.0 and model["lam"] == 1.0


def family_mean_force(x, cell, model):
    """The per-dynamics frozen-x EM-consistent mean force (FAMILY_SECONDARY_DEFINITION).  Written in the operation
    order of analyze_gateway_replica_ladder.em_mean_force, so that at alpha 1, lam 1 it equals it bitwise (tested)."""
    b, H, oo, oi, s, dt = (cell[k] for k in ("beta", "H", "omega_out", "omega_in", "s", "dt"))
    x = np.asarray(x, dtype=float)
    e = np.exp(-x * x / (2 * s * s))
    om = oo + (oi - oo) * e
    dom = -(oi - oo) * (x / (s * s)) * e
    base = 4 * H * x * (x * x - 1)
    if model["variant"] == "shift":
        return base + dom / (om * b)
    a, lam = float(model["alpha"]), float(model["lam"])
    return base + ((1.0 - a) / b) * (dom / om) + a * (dom / (om * b)) / (1.0 - lam * om ** (2.0 * a) * dt / 2.0)


@functools.lru_cache(maxsize=None)
def family_em_scorer_class():
    """analyze_gateway_replica_ladder.Scorer(cell, 'em') with the mean force of one family dynamics (lazy: torch)."""
    import analyze_gateway_replica_ladder as GL
    import torch
    eb = GL.eb

    class FamilyEMScorer(GL.Scorer):
        """The accepted scorer's EM construction (F from the trapezoid integral of the mean force on 180 x 400 + 1
        points, interpolated to the grid and centred on the eval window; e_F' against the mean force on the
        estimator's Gauss-Legendre sub-grid) with family_mean_force in place of em_mean_force."""

        def __init__(self, cell, model):
            super().__init__(cell, "analytic")
            self.model = dict(model)
            self.reference = "family_em"
            xf = np.linspace(eb.XMIN, eb.XMAX, 180 * 400 + 1)
            fp = family_mean_force(xf, cell, self.model)
            F = np.concatenate([[0.0], np.cumsum(0.5 * (fp[1:] + fp[:-1]) * np.diff(xf))])
            Fg = np.interp(self.x_grid.numpy(), xf, F)
            Fg = Fg - Fg[self.eval_mask.numpy()].mean()
            self.F_ref = torch.tensor(Fg[None], dtype=GL.DT)
            self._est = {}

        def est(self, nb, R):
            key = (nb, R)
            if key not in self._est:
                e = eb.HistogramABFEstimator(R, nb, 1.0, self.x_grid, GL.CPU, GL.DT)
                sub = torch.tensor(family_mean_force(e.sub_x.numpy(), self.cell, self.model)[None], dtype=GL.DT)
                mask = (e.sub_x >= eb.EVAL_LO) & (e.sub_x <= eb.EVAL_HI)
                self._est[key] = (e, sub, mask)
            return self._est[key]

    return FamilyEMScorer


def _bin_average(s):
    e, sub, _ = s.est(NB, 1)
    w = e.sub_w.numpy().astype(float)
    s2h = e.sub2h.numpy().astype(np.int64)
    subv = sub.numpy()[0].astype(float)
    return np.bincount(s2h, weights=w * subv, minlength=NB) / np.bincount(s2h, weights=w, minlength=NB)


FAMILY_REF_ARRAYS = ("x_grid", "eval_mask", "F_ref", "Fp_ref", "F_ref_em", "Fp_bin_ref", "Fp_bin_em")
PRIMARY_REF_ARRAYS = ("x_grid", "eval_mask", "F_ref", "Fp_ref")


def primary_digest(ref):
    """sha256 of the primary (analytic) reference arrays: identical for every cell of the family."""
    h = hashlib.sha256()
    for k in PRIMARY_REF_ARRAYS:
        a = np.ascontiguousarray(np.asarray(ref[k]))
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def family_reference_arrays(engine_cfg):
    """(arrays, cell, model) of a cell's frozen reference, computed from its engine_cfg: the primary analytic F*, F*'
    of the accepted scorer, the per-dynamics secondary F (F_ref_em) and the bin averages of both mean forces on the
    scorer's sub-grid (Fp_bin_ref, Fp_bin_em)."""
    import analyze_gateway_replica_ladder as GL
    cell = M.gateway_cell(engine_cfg)
    model = family_model(engine_cfg)
    sa = GL.Scorer(cell, "analytic")
    se = family_em_scorer_class()(cell, model)
    arr = dict(x_grid=sa.x_grid.numpy().copy(), eval_mask=sa.eval_mask.numpy().copy(), F_ref=sa.F_ref.numpy()[0].copy(),
               Fp_ref=sa.Fp_ref.numpy()[0].copy(), F_ref_em=se.F_ref.numpy()[0].copy(), Fp_bin_ref=_bin_average(sa),
               Fp_bin_em=_bin_average(se))
    return arr, cell, model


class GatewayFamilyScorer(M.GatewayScorer):
    """Gateway family (mechanism campaign): the accepted analytic scorer with the SAME primary reference for every
    variant (bitwise the equal-budget results/equal_budget_v2/references/gateway_reference.npz), and the per-dynamics
    EM-consistent secondary, both verified against the cell's frozen reference file at construction.  Inherits the
    scoring methods (bin_reference, stat_error, zero_noise_floors, errors) of eqb_metrics.GatewayScorer unchanged."""
    kind = "gateway"

    def __init__(self, engine_cfg, reference_path):
        import analyze_gateway_replica_ladder as GL
        import torch
        self._torch = torch
        self.cell = M.gateway_cell(engine_cfg)
        self.model = family_model(engine_cfg)
        self.sa = GL.Scorer(self.cell, "analytic")
        self.se = family_em_scorer_class()(self.cell, self.model)
        self.reference_path = reference_path
        if not os.path.exists(reference_path):
            raise MetricsError(f"frozen cell reference {reference_path} does not exist "
                               f"(scripts/mechanism/build_references.py)")
        with np.load(reference_path, allow_pickle=False) as z:
            ref = {k: z[k] for k in z.files}
        miss = [k for k in FAMILY_REF_ARRAYS + ("h", "model_json") if k not in ref]
        if miss:
            raise MetricsError(f"{reference_path}: arrays missing {miss}")
        self._binref = {}
        stored_model = json.loads(str(ref["model_json"]))
        checks = dict(x_grid=np.array_equal(ref["x_grid"], self.sa.x_grid.numpy()),
                      eval_mask=np.array_equal(ref["eval_mask"], self.sa.eval_mask.numpy()),
                      F_ref=np.allclose(ref["F_ref"], self.sa.F_ref.numpy()[0], rtol=0, atol=1e-12),
                      F_ref_em=np.allclose(ref["F_ref_em"], self.se.F_ref.numpy()[0], rtol=0, atol=1e-12),
                      Fp_ref=np.allclose(ref["Fp_ref"], self.sa.Fp_ref.numpy()[0], rtol=0, atol=1e-12),
                      Fp_bin_ref=np.allclose(ref["Fp_bin_ref"], self.bin_reference(self.sa)[0], rtol=0, atol=1e-12),
                      Fp_bin_em=np.allclose(ref["Fp_bin_em"], self.bin_reference(self.se)[0], rtol=0, atol=1e-12),
                      h=float(ref["h"]) == self.cell["dt"],
                      model=stored_model == self.model)
        if not all(checks.values()):
            raise MetricsError(f"family scorer does not reproduce the frozen cell reference {reference_path}: {checks}")
        # the primary reference is identical for every variant and lambda: bitwise the equal-budget gateway reference
        eqb_path = os.path.join(M.REF_DIR, "gateway_reference.npz")
        with np.load(eqb_path, allow_pickle=False) as z:
            eqb = {k: z[k] for k in z.files}
        same = {k: bool(np.array_equal(ref[k], eqb[k]) and np.array_equal(eqb[k], cur))
                for k, cur in (("x_grid", self.sa.x_grid.numpy()), ("eval_mask", self.sa.eval_mask.numpy()),
                               ("F_ref", self.sa.F_ref.numpy()[0]), ("Fp_ref", self.sa.Fp_ref.numpy()[0]))}
        if not all(same.values()):
            raise MetricsError(f"the primary reference of {reference_path} is not the equal-budget gateway reference "
                               f"{eqb_path} bitwise: {same}")
        orig = None
        if is_original_dynamics(self.model):      # alpha 1, lam 1: the secondary IS the equal-budget EM reference
            em = GL.Scorer(self.cell, "em")
            orig = bool(np.array_equal(em.F_ref.numpy(), self.se.F_ref.numpy())
                        and np.array_equal(em.est(NB, 1)[1].numpy(), self.se.est(NB, 1)[1].numpy())
                        and (float(eqb["h"]) != self.cell["dt"] or np.array_equal(ref["F_ref_em"], eqb["F_ref_em"])))
            if not orig:
                raise MetricsError("alpha 1, lam 1: the family secondary reference is not the equal-budget EM reference "
                                   "bitwise")
        info = self._gateway_reference_info(ref)
        info.update(
            em_bias_definition=("em_floor_F_rms: RMS of F_ref_em - F_ref on the eval window (centred); "
                                "em_bias_Fp_bin_rms: RMS over the eval bins of <F'_em>_j - <F'_ref>_j, F'_em = this "
                                "dynamics' frozen-x EM-consistent mean force (NOT the e_F' floor)"),
            scorer=("scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, identical for every variant "
                    "and lambda); secondary = eqb_family.family_mean_force (per-dynamics EM-consistent)"),
            model=dict(self.model), secondary_definition=FAMILY_SECONDARY_DEFINITION,
            primary_reference=dict(path=os.path.relpath(eqb_path, ROOT), sha256=sha256_file(eqb_path),
                                   identical_bitwise=True, arrays=list(PRIMARY_REF_ARRAYS),
                                   digest=primary_digest(ref)),
            secondary_equals_equal_budget_em=orig)
        self.reference_info = info

    def _gateway_reference_info(self, ref):
        """The summary's reference block, computed exactly as eqb_metrics.GatewayScorer.__init__ computes it."""
        m = ref["eval_mask"]
        d = (ref["F_ref_em"] - ref["F_ref"])[m]
        floors = self.zero_noise_floors()
        fbar_a, _, _, _ = self.bin_reference(self.sa)
        fbar_e, w, s2h, mk = self.bin_reference(self.se)
        dd = (fbar_e - fbar_a)[s2h][mk]
        return dict(
            path=os.path.relpath(self.reference_path, ROOT), sha256=sha256_file(self.reference_path),
            n_eval_nodes=int(m.sum()),
            em_floor_F_rms=float(np.sqrt(np.mean((d - d.mean()) ** 2))),
            em_bias_Fp_bin_rms=float(np.sqrt(np.sum(w[mk] * dd * dd) / np.sum(w[mk]))),
            zero_noise_floor=floors,
            floor_is_lower_bound=dict(e_F=False, e_Fp=True, e_F_em=False, e_Fp_em=True, e_Fp_stat=True,
                                      e_Fp_em_stat=True),
            floor_definition=("errors of the scorer fed the exact bin-averaged reference mean force (Gamma_j = "
                              "<F'_ref>_j on the Gauss-Legendre sub-grid, infinite counts); for e_Fp / e_Fp_em it is "
                              "a HARD lower bound: e_Fp^2 = e_Fp_stat^2 + floor^2 for every Gamma (within-bin "
                              "variation of F'_ref, steep at the gate); e_F at Gamma = <F'_ref> is ~0"),
            units="reduced (energy; force per unit x)",
            scorer="scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, em secondary)",
            cell=self.cell)


def make_scorer(system, P):
    """eqb_metrics.make_scorer; the family: GatewayFamilyScorer on the cell's frozen reference."""
    check_invocation(system, P)
    if is_family(system):
        return GatewayFamilyScorer(P["engine_cfg"], reference_path=reference_path(system, P))
    return M.make_scorer(system, P)


# ------------------------------------------------------------------------------------------------- tau thresholds
def _companions_of(kind, thresholds):
    return [k for k in M.error_names(kind)[1] if k in (thresholds or {})]


def tau_companions(kind, P):
    """Companion errors (no frozen thresholds in the equal-budget plan) to which a CELL config gives thresholds of its
    own, keyed by the error name (thresholds.e_Fp_stat); [] for every equal-budget config."""
    if not is_cell(P):
        return []
    return _companions_of(kind, P.get("thresholds"))


def tau_thresholds(kind, P):
    """[(error, its three thresholds)]: the thresholded errors (family thresholds) then tau_companions."""
    th = P["thresholds"]
    return ([(k, th[M.error_family(k)]) for k in M.error_names(kind)[0]] + [(k, th[k]) for k in tau_companions(kind, P)])


def threshold_floors(system, P, scorer):
    """eqb_metrics.threshold_floors; the family: the same records for the thresholded errors AND the companions with
    thresholds of their own (e_Fp_stat), and an unreachable threshold notes that its tau is NOT computed."""
    if not is_family(system):
        return M.threshold_floors(system, P, scorer)
    kind = system_kind(system)
    info = scorer.reference_info
    fl, hard = info.get("zero_noise_floor", {}), info.get("floor_is_lower_bound", {})
    out = {}
    for k, ths in tau_thresholds(kind, P):
        for name, eps in zip(THR_NAMES, ths):
            eps = float(eps)
            f, h = fl.get(k), bool(hard.get(k, False))
            reach = not (h and f is not None and eps <= f)
            budget = (math.sqrt(eps * eps - f * f) if (h and f is not None and eps > f) else
                      (0.0 if h and f is not None else None))
            out[f"tau_{k}_{name}"] = dict(
                metric=k, threshold=name, eps=eps, floor=f, floor_is_lower_bound=h, reachable=reach,
                stat_budget=budget, tau_computed=reach,
                note=(f"UNREACHABLE BY CONSTRUCTION: {k} >= {f:.4g} > eps for every estimate; per plan section 6 no "
                      f"tau is computed at this threshold ({k} itself is reported)" if not reach else
                      (f"reachable only if the statistical part (RMS of Gamma - bin-averaged reference) is <= "
                       f"{budget:.4g}" if h and budget is not None and budget < 0.5 * eps else None)))
    return out


def unreachable_tau_stems(floors):
    """{'tau_<error>_<name>'} of the thresholds at or below a hard floor (no tau computed for a cell)."""
    return {k for k, v in (floors or {}).items() if v.get("reachable") is False}


def tau_groups(system, P, floors=None):
    """(primary tau keys, secondary tau keys) of the paired tau contrasts.  Equal-budget systems: None (the caller
    keeps its own lists, unchanged).  A cell (plan section 6): primary = tau on e_F, on e_Fp_stat and on TV_half (the
    plan's three threshold sets); secondary = tau on e_F_em and on raw e_Fp / e_Fp_em at the thresholds ABOVE their
    hard floor only (``floors`` = threshold_floors; raw e_F' is reported, no tau below its floor).  ``P``: the cell
    config or a summary's plan block (only its thresholds are read)."""
    if not is_family(system):
        return None
    kind = system_kind(system)
    drop = unreachable_tau_stems(floors)
    comp = _companions_of(kind, P.get("thresholds"))
    prim = ([f"tau_e_F_{n}_u" for n in THR_NAMES] + [f"tau_{k}_{n}_u" for k in comp for n in THR_NAMES]
            + ["tau_TV_half_u"])
    sec = [f"tau_{k}_{n}_u" for k in ("e_F_em", "e_Fp", "e_Fp_em") for n in THR_NAMES
           if f"tau_{k}_{n}" not in drop]
    return prim, sec


# ------------------------------------------------------------------------------------------------- reuse gate
REUSE_GATE_SCHEMA = "mechanism_reuse_gate/1"
REUSE_GATE_ENGINE = "gateway_family_numba/1"


def reuse_gate_problems(P):
    """[] iff ``P`` is not a reuse cell, or its reuse record names a gate record (scripts/mechanism/reuse_gate.py)
    that licenses the reuse (plan section 5: the alpha 1 / lam 1 cells may read results/equal_budget_v2/gateway only if
    gateway_family_numba at alpha 1, lam 1 reproduces >= 4 of those files bitwise, both arms, N 128 and 2048):
    schema, verdict PASS, engine gateway_family_numba/1 at the original model, every job bitwise equal, at least
    gate_min_jobs jobs, both arms, every N of gate_required_N, and every compared file IS the file reused now (same
    path under the cell's results_dir, same sha256)."""
    R = (P or {}).get("reuse")
    if not R:
        return []
    rp = R.get("gate_record")
    if not rp:
        return ["the reuse record names no gate_record"]
    path = _abs(rp)
    if not os.path.exists(path):
        return [f"reuse gate record {rp} does not exist (run scripts/mechanism/reuse_gate.py; plan section 5)"]
    try:
        doc = json.load(open(path))
    except Exception as e:  # noqa: BLE001
        return [f"reuse gate record {rp} unreadable: {type(e).__name__}: {e}"]
    probs = []
    if doc.get("schema") != REUSE_GATE_SCHEMA:
        probs.append(f"gate record schema {doc.get('schema')!r} != {REUSE_GATE_SCHEMA!r}")
    if doc.get("verdict") != "PASS":
        probs.append(f"gate verdict {doc.get('verdict')!r}, not PASS")
    if doc.get("engine") != REUSE_GATE_ENGINE:
        probs.append(f"gate engine {doc.get('engine')!r} != {REUSE_GATE_ENGINE!r}")
    model = doc.get("model") or {}
    if any(not _model_equal(model.get(k), FAMILY_MODEL_DEFAULTS[k]) for k in FAMILY_MODEL_KEYS):
        probs.append(f"gate model {model} is not the original dynamics {FAMILY_MODEL_DEFAULTS}")
    jobs = doc.get("jobs") or []
    bad = [j for j in jobs if j.get("bitwise_equal") is not True]
    if bad:
        probs.append(f"{len(bad)} gate job(s) not bitwise equal: "
                     + ", ".join(f"N {j.get('N')} s{j.get('seed')} {j.get('method')}" for j in bad[:4]))
    ok = [j for j in jobs if j.get("bitwise_equal") is True]
    need = int(R.get("gate_min_jobs", 4))
    if len(ok) < need:
        probs.append(f"{len(ok)} bitwise-equal gate jobs < the required {need}")
    if {"abf", "fr"} - {j.get("method") for j in ok}:
        probs.append(f"gate jobs do not cover both arms ({sorted({j.get('method') for j in ok})})")
    reqN = [int(n) for n in R.get("gate_required_N", [2048, 128])]
    missN = sorted(set(reqN) - {int(j.get("N", -1)) for j in ok})
    if missN:
        probs.append(f"gate jobs do not cover N {missN}")
    rd = _abs(P["results_dir"])
    for j in ok:
        f = os.path.join(rd, f"N{j.get('N')}", f"s{j.get('seed')}_{j.get('method')}.npz")
        if not os.path.exists(f):
            probs.append(f"gate job N {j.get('N')} s{j.get('seed')} {j.get('method')}: reused file {f} is absent")
        elif sha256_file(f) != j.get("reused_sha256"):
            probs.append(f"gate job N {j.get('N')} s{j.get('seed')} {j.get('method')}: the reused file changed since "
                         f"the gate compared it (sha256)")
    return probs


def reuse_gate_status(P):
    """The summary / audit record of a cell's reuse gate (reuse False and no problems for a non-reuse cell)."""
    R = (P or {}).get("reuse")
    if not R:
        return dict(reuse=False, problems=[])
    rp = R.get("gate_record")
    path = _abs(rp) if rp else None
    probs = reuse_gate_problems(P)
    return dict(reuse=True, record=rp, record_sha256=(sha256_file(path) if path and os.path.exists(path) else None),
                licensed=not probs, problems=probs)


# ------------------------------------------------------------------------------------------------- plan checks

def _model_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def run_model(meta, cfg, engine):
    """(model fields of a run, problems): each of variant / alpha / kappa / lam from cfg, else meta; cfg and meta must
    agree when both carry it; absent from both -> FAMILY_MODEL_DEFAULTS, allowed ONLY for gateway_ladder_numba/2
    files (the equal-budget engine has no model fields)."""
    out, probs = {}, []
    for k in FAMILY_MODEL_KEYS:
        inc, inm = k in cfg, k in meta
        if inc and inm and not _model_equal(cfg[k], meta[k]):
            probs.append(f"model field {k}: cfg {cfg[k]!r} != meta {meta[k]!r}")
        if inc or inm:
            out[k] = cfg[k] if inc else meta[k]
        elif engine == "gateway_ladder_numba/2":
            out[k] = FAMILY_MODEL_DEFAULTS[k]
        else:
            probs.append(f"model field {k} missing from cfg and meta of a {engine} file")
    return out, probs


def family_plan_problems(meta, cfg, P):
    """Mechanism cell: the run's engine is the cell's engine_version (one engine per cell), its 'system' label is a
    gateway one, and its model fields equal the cell's (gateway_ladder_numba/2 files: defaults)."""
    probs = []
    eng, want = meta.get("engine"), P.get("engine_version")
    if want not in FAMILY_ENGINES:
        probs.append(f"cell engine_version {want!r} is not one of {FAMILY_ENGINES}")
    if eng != want:
        probs.append(f"engine {eng!r} != the cell's engine {want!r} (a cell never mixes engines)")
    for src, d in (("cfg", cfg), ("meta", meta)):
        if "system" in d and d["system"] not in FAMILY_SYSTEM_LABELS:
            probs.append(f"{src} system {d['system']!r} is not a gateway label {FAMILY_SYSTEM_LABELS}")
    got, mp = run_model(meta, cfg, eng)
    probs += mp
    for k in FAMILY_MODEL_KEYS:
        if k in got and not _model_equal(got[k], P["engine_cfg"].get(k, FAMILY_MODEL_DEFAULTS[k])):
            probs.append(f"model field {k}: run {got[k]!r} != cell {P['engine_cfg'].get(k)!r}")
    return probs


def check_plan(res, P, path=None, expect=None):
    """eqb_metrics.check_plan; a cell: the same checks with the model fields and the 'system' label compared by
    family_plan_problems instead of as engine knobs (else MetricsError)."""
    if not is_cell(P):
        return M.check_plan(res, P, path, expect)
    meta, cfg = res["meta"], res["cfg"]
    N, n_steps = int(meta["N"]), int(meta["n_steps"])
    probs = []
    if N * n_steps != int(P["B"]):
        probs.append(f"N n_steps = {N * n_steps} != B {P['B']}")
    if float(meta["h"]) != float(P["engine_cfg"]["h"]):
        probs.append(f"h {meta['h']} != config h {P['engine_cfg']['h']}")
    skip = ("system",) + FAMILY_MODEL_KEYS
    for k, v in P["engine_cfg"].items():
        if k in skip:
            continue
        if k in cfg and cfg[k] != v and not (isinstance(v, (int, float)) and isinstance(cfg[k], (int, float))
                                              and float(cfg[k]) == float(v)):
            probs.append(f"engine knob {k}: run {cfg[k]!r} != config {v!r}")
    probs += family_plan_problems(meta, cfg, P)
    if expect:
        for k in ("N", "seed", "method"):
            if k in expect and meta.get(k) != expect[k]:
                probs.append(f"{k}: file says {meta.get(k)!r}, path says {expect[k]!r}")
    if probs:
        raise MetricsError(f"{path or 'run'} does not match the config: " + "; ".join(probs))


# ------------------------------------------------------------------------------------------------- per-run metrics
def run_metrics(res, system, P, scorer, path=None):
    """eqb_metrics.run_metrics; the family: the gateway metrics of eqb_metrics.run_metrics (system label
    'gateway_family'), plus tau on the companions with thresholds of their own (e_Fp_stat: tau_* over every save and
    tau_bgrid_* on the common budget grid), minus every tau at a threshold at or below a hard zero-noise floor (plan
    section 6: raw e_F' strict, no tau is computed below its floor)."""
    if not is_family(system):
        return M.run_metrics(res, system, P, scorer, path)
    check_invocation(system, P, path or "run")
    kind = system_kind(system)
    curves, sc = M.run_metrics(res, "gateway", P, scorer, path)
    sc["system"] = system
    t = np.asarray(curves["save_t"], float)
    u = np.asarray(curves["save_u"], float)
    bmask = np.asarray(curves["budget_grid_mask"]).astype(bool)
    thr = P["thresholds"]
    for k in tau_companions(kind, P):
        e = np.asarray(curves[k], float)
        for name, eps in zip(THR_NAMES, thr[k]):
            tt = M.tau_persistent(e, t, u, float(eps))
            for f in ("t", "u", "censored", "eps"):
                sc[f"tau_{k}_{name}_{f}"] = tt[f]
            tb = M.tau_persistent(e[bmask], t[bmask], u[bmask], float(eps))
            for f in ("t", "u", "censored"):
                sc[f"tau_bgrid_{k}_{name}_{f}"] = tb[f]
    for stem in unreachable_tau_stems(threshold_floors(system, P, scorer)):
        for pre in (stem, stem.replace("tau_", "tau_bgrid_", 1)):
            for f in ("t", "u", "censored", "eps"):
                sc.pop(f"{pre}_{f}", None)
    return curves, sc
