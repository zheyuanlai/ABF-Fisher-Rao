"""Frozen metrics of the equal-budget replica ladders (docs/equal_budget/SCIENTIFIC_PLAN.md, section 4).

Library used by scripts/equal_budget/analyze_ladder.py and tests/test_eqb_metrics.py.  One result file of the
production layout results/equal_budget_v2/<out_dir>/N<N>/s<seed>_<abf|fr>.npz -> ``run_metrics`` -> (curves,
scalars).  Group statistics (paired contrasts, bootstrap, best allocation, transient improvement) are the
functions at the end.

Errors (references frozen in results/equal_budget_v2/references/, nothing else is read):
  gateway  e_F, e_Fp = the ACCEPTED scorer scripts/analyze_gateway_replica_ladder.Scorer(cell, "analytic")
           (P0 histogram read-out, min_count 1, e_F RMS on the 151 eval-window nodes after centring, e_Fp its
           function-space fp_error); e_F_em, e_Fp_em = Scorer(cell, "em") (secondary); the scorer's profiles are
           checked against gateway_reference.npz at construction.
           ZERO-NOISE FLOOR: e_Fp integrates (Gamma_j - F'_ref(x))^2 over the Gauss-Legendre sub-grid of each bin,
           so it also counts the within-bin variation of F'_ref; for EVERY Gamma
               e_Fp^2 = e_Fp_stat^2 + floor^2   (exactly; the cross term vanishes),
           e_Fp_stat = RMS over the 150 eval-window bins of Gamma_j - <F'_ref>_j (the sub-grid bin average) and
           floor = 0.0323 (analytic and EM reference alike) at h 2.5e-5.  The floor is a hard lower bound of the
           frozen e_Fp: a threshold below it (gateway e_F' strict 0.012) is UNREACHABLE BY CONSTRUCTION and its
           tau is censored for every arm (threshold_floors() reports this; 'both censored' there is not a tie).
           e_Fp_stat / e_Fp_em_stat are reported as labelled descriptive companions (not frozen metrics).
  LTA      Gamma = M / max(C, 1) (0 where C = 0) from the PRODUCTION accumulators (M_prod, C_prod) once C_prod has
           any count, else from the all-steps ones (core_lta's reported estimator); F = histogram_pmf(Gamma);
           e_F = RMS over the 180 bins of F - F_ref after removing the mean difference; e_Fp (= raw) = RMS of
           Gamma - gamma_ref; e_Fp_proj = the same after removing both circular means.  gamma_ref is itself a
           per-bin quantity, so e_Fp has no deterministic floor; Gamma = gamma_ref gives e_F = the reference's own
           mean-force-vs-density consistency (0.0075 kJ/mol at 300 K), reported, far below every threshold.
Marginal: TV_inst (instantaneous walker histogram, 18 coarse bins, vs uniform) with the finite-N floor
  E_N[TV] under multinomial(N, uniform 18), computed EXACTLY ((K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K): each bin is
  marginally binomial); TV_half = TV of the visitation histogram C_all(t) - C_all(s_lo), s_lo the saved step
  nearest t/2 (candidates: the run start, with zero accumulators, and every earlier save; ties -> the earlier
  one), window recorded.
tau(eps): first save after which the error is <= eps at EVERY later save (all saves of the file);
  censored -> inf with a flag; reported as physical t and budget fraction u.  tau_bgrid_*: the same on the
  COMMON budget grid only (200 uniform + 24 log-spaced u, identical in u at every N), used whenever tau is compared
  ACROSS N (best allocation, budget-axis tau vs N); the all-saves tau (which adds the N-dependent physical-time
  checkpoints) is used for the paired comparison at one N, where both arms share the grid.
Establishment: persistent instantaneous far-state fraction >= 1/2 target (frozen), plus its NULL CALIBRATION:
  the chance that an exactly uniform population (far count ~ Bin(N, target) independently at every save) passes
  the same criterion on the run's save grid (est_null_*); est_unreliable = N <= 2 or P(no failure in the second
  half of the saves) < 0.9 -- there the cumulative-visitation version / first arrival are the readable ones.
FR activity: realised deaths (= realised replacements) per opportunity in BOTH systems (gateway: fr_deaths_hist /
  fr_opp_death_cum of engine /2, fallback fr_deaths_cum); the gateway's capped candidate counts kd + kc are kept
  under separate fr_candidate* names.
Integrated errors: mean over the 200 uniform budget fractions u = k/200 (identified exactly as the saves at
  step k n_steps / 200); final errors at u = 1.
Cache provenance: run_cache_key() = content sha256 of the run file + metrics version + config digest (every
  engine knob, thresholds, grid) + sha256 of the scoring code + sha256 of the frozen reference.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

METRICS_VERSION = "eqb_metrics/2"
NB = 180
N_COARSE = 18
N_UNIFORM = 200
TV_FLOOR_NMC = 200_000           # tv_floor_mc only (cross-check of the exact tv_floor)
TV_FLOOR_SEED = 20261010
TV_FLOOR_METHOD = "exact: (K/2) E|X/N - 1/K|, X ~ Bin(N, 1/K), K = 18 (each bin is marginally binomial)"
N_BOOT = 10_000
BOOT_SEED = 20261010
THR_NAMES = ("strict", "mid", "loose")
REF_DIR = os.path.join(ROOT, "results", "equal_budget_v2", "references")
EST_RELIABLE_P = 0.9             # est_unreliable when the null P(no failure in the second half) is below this
# files whose code computes a per-run metric (their sha256 enters the run-metrics cache key)
CODE_FILES = {
    "common": [os.path.join(HERE, "eqb_metrics.py")],
    "gateway": [os.path.join(ROOT, "scripts", "analyze_gateway_replica_ladder.py"),
                os.path.join(ROOT, "src", "eb_abffr_core.py"), os.path.join(ROOT, "src", "gateway_ladder_numba.py")],
    "lta": [os.path.join(ROOT, "src", "lta_ladder_numba.py")],
}

SYSTEMS = {
    "gateway": dict(kind="gateway", config="gateway_production.json"),
    "lta300": dict(kind="lta", config="lta_300K_production.json", T_K=300.0),
    "lta150": dict(kind="lta", config="lta_150K_production.json", T_K=150.0),
}
# population establishment: far state and its uniform-target fraction (frozen, plan section 4)
FAR = {
    "gateway": dict(name="right well x > 0.5", region_col=2, target=0.361),
    "lta": dict(name="window |z| < 1.5 A", region_col=2, target=0.252),
}
GATEWAY_KEYS = ("save_step", "save_t", "save_u", "M_all", "C_all", "hist_inst", "region_frac", "events_cum",
                "lineage_visited", "first_arrival_step", "gen_nuniq_run", "gen_ess_run", "gen_maxfam_run",
                "gen_nuniq_win", "gen_ess_win", "gen_maxfam_win", "fr_deaths_cum", "fr_kd_cum", "fr_kc_cum",
                "fr_opp_cum", "fr_opp_event_cum", "fr_events_hist", "n_force_evals", "wall_s", "peak_rss_mb",
                "meta", "cfg")
LTA_KEYS = ("save_step", "save_t", "save_u", "M_all", "C_all", "M_prod", "C_prod", "hist_inst", "region_frac",
            "events_window_crossings", "events_translocations", "lineage_ever_window", "lineage_ever_opposite",
            "first_window_step", "first_opposite_step", "gen_n_unique", "gen_ess", "gen_max_frac",
            "gen_n_unique_win", "gen_ess_win", "gen_max_frac_win", "cum_deaths", "cum_opps", "cum_opps_with_event",
            "events_per_opp_hist", "n_force_evals", "wall_s", "peak_rss_mb", "meta_json", "cfg_json")


class MetricsError(RuntimeError):
    """A result file violates the engine result contract or the plan (missing key, wrong grid, ...)."""


def system_kind(system):
    if system not in SYSTEMS:
        raise MetricsError(f"unknown system {system!r}; expected one of {sorted(SYSTEMS)}")
    return SYSTEMS[system]["kind"]


def default_config_path(system):
    return os.path.join(ROOT, "configs", "equal_budget_v2", SYSTEMS[system]["config"])


# ------------------------------------------------------------------------------------------------- loading
def engine_module(kind):
    if kind == "gateway":
        import gateway_ladder_numba as E
    else:
        import lta_ladder_numba as E
    return E


def load_run(path, system):
    """Engine load_result of one result file; returns the dict with parsed ``meta`` and ``cfg``.
    Raises MetricsError naming every missing contract key."""
    kind = system_kind(system)
    if kind == "gateway":
        import gateway_ladder_numba as E
        res = E.load_result(path)
        need = GATEWAY_KEYS
    else:
        import lta_ladder_numba as E
        res = E.load_result(path)
        need = LTA_KEYS
    miss = [k for k in need if k not in res]
    if miss:
        raise MetricsError(f"{path}: result contract keys missing for {system}: {miss}")
    if kind == "lta":
        res["meta"] = json.loads(str(res["meta_json"]))
        res["cfg"] = json.loads(str(res["cfg_json"]))
    if res["meta"].get("status") != "complete":
        raise MetricsError(f"{path}: meta status is {res['meta'].get('status')!r}, not 'complete'")
    return res


def file_key(path):
    st = os.stat(path)
    return dict(src_path=os.path.abspath(path), src_mtime_ns=int(st.st_mtime_ns), src_size=int(st.st_size))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def config_digest(P):
    """Digest of everything in the config that a per-run metric depends on or that check_plan compares: every
    engine knob, the thresholds, B, h, the save-grid specification, the system and T, the far-state targets."""
    keep = dict(thresholds=P["thresholds"], engine_cfg=P["engine_cfg"], B=P["B"], system=P.get("system"),
                T_K=P.get("T_K"), out_dir=P.get("out_dir"), physical_checkpoints_t=P.get("physical_checkpoints_t"),
                profile_snapshot_u=P.get("profile_snapshot_u"), far=FAR, est_reliable_p=EST_RELIABLE_P)
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def code_provenance(system):
    """sha256 of every file whose code computes a per-run metric of ``system`` (+ one combined digest)."""
    kind = system_kind(system)
    files = CODE_FILES["common"] + CODE_FILES[kind]
    shas = {os.path.relpath(p, ROOT): sha256_file(p) for p in files}
    digest = hashlib.sha256(json.dumps(shas, sort_keys=True).encode()).hexdigest()[:16]
    return dict(files=shas, digest=digest)


def reference_path(system, P=None):
    """The frozen reference file of ``system`` (REF_DIR is read at call time)."""
    if system_kind(system) == "gateway":
        return os.path.join(REF_DIR, "gateway_reference.npz")
    T = float((P or {}).get("T_K", SYSTEMS[system]["T_K"]))
    return os.path.join(REF_DIR, f"lta_T{int(round(T))}_reference.npz")


def run_cache_key(path, system, P, code=None, ref_sha=None):
    """Key of a cached per-run metric entry: CONTENT of the run file (sha256 + size; mtime is not trusted), the
    metrics version, the config digest, the scoring code and the frozen reference.  A cached entry is valid only
    if its key equals this one."""
    return dict(src_sha256=sha256_file(path), src_size=int(os.path.getsize(path)), metrics_version=METRICS_VERSION,
                config_digest=config_digest(P), code_digest=(code or code_provenance(system))["digest"],
                reference_sha256=ref_sha or sha256_file(reference_path(system, P)))


def cache_key_problems(stored, current):
    """Human-readable differences between a stored cache key and the current one (empty list = valid)."""
    names = dict(src_sha256="run file content changed since it was scored", src_size="run file size changed",
                 metrics_version="metrics version", config_digest="config (engine knobs / thresholds / grid)",
                 code_digest="scoring code changed", reference_sha256="frozen reference changed")
    out = []
    for k, why in names.items():
        if (stored or {}).get(k) != current.get(k):
            out.append(f"{why} ({str((stored or {}).get(k))[:16]} -> {str(current.get(k))[:16]})")
    return out


# ------------------------------------------------------------------------------------------------- scorers
def gateway_cell(engine_cfg):
    return dict(beta=float(engine_cfg["beta"]), H=float(engine_cfg["H"]), omega_out=float(engine_cfg["omega_out"]),
                omega_in=float(engine_cfg["omega_in"]), s=float(engine_cfg["s"]), dt=float(engine_cfg["h"]))


class GatewayScorer:
    """The accepted scorer (analytic, primary) and its EM-consistent variant (secondary), verified against the
    frozen results/equal_budget_v2/references/gateway_reference.npz."""
    kind = "gateway"

    def __init__(self, engine_cfg, reference_path=None):
        import analyze_gateway_replica_ladder as GL
        import torch
        self._torch = torch
        self.cell = gateway_cell(engine_cfg)
        self.sa = GL.Scorer(self.cell, "analytic")
        self.se = GL.Scorer(self.cell, "em")
        self.reference_path = reference_path or os.path.join(REF_DIR, "gateway_reference.npz")
        with np.load(self.reference_path, allow_pickle=False) as z:
            ref = {k: z[k] for k in z.files}
        checks = dict(x_grid=np.array_equal(ref["x_grid"], self.sa.x_grid.numpy()),
                      eval_mask=np.array_equal(ref["eval_mask"], self.sa.eval_mask.numpy()),
                      F_ref=np.allclose(ref["F_ref"], self.sa.F_ref.numpy()[0], rtol=0, atol=1e-12),
                      F_ref_em=np.allclose(ref["F_ref_em"], self.se.F_ref.numpy()[0], rtol=0, atol=1e-12),
                      Fp_ref=np.allclose(ref["Fp_ref"], self.sa.Fp_ref.numpy()[0], rtol=0, atol=1e-12),
                      h=float(ref["h"]) == self.cell["dt"])
        if not all(checks.values()):
            raise MetricsError(f"accepted scorer does not reproduce the frozen gateway reference: {checks}")
        m = ref["eval_mask"]
        d = (ref["F_ref_em"] - ref["F_ref"])[m]
        self._binref = {}
        floors = self.zero_noise_floors()
        fbar_a, _, _, _ = self.bin_reference(self.sa)
        fbar_e, w, s2h, mk = self.bin_reference(self.se)
        dd = (fbar_e - fbar_a)[s2h][mk]
        self.reference_info = dict(
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
            em_bias_definition=("em_floor_F_rms: RMS of F_ref_em - F_ref on the eval window (centred); "
                                "em_bias_Fp_bin_rms: RMS over the eval bins of <F'_em>_j - <F'_ref>_j (the EM bias "
                                "of F', NOT the e_F' floor)"),
            units="reduced (energy; force per unit x)",
            scorer="scripts/analyze_gateway_replica_ladder.py Scorer (analytic primary, em secondary)",
            cell=self.cell)

    # ---- zero-noise floor and the floor-free companion of e_Fp
    def bin_reference(self, s):
        """(<F'_ref>_j over the scorer's own Gauss-Legendre sub-grid (nb,), sub-weights, sub->bin, eval sub-mask)."""
        if id(s) not in self._binref:
            e, sub, mask = s.est(NB, 1)
            w = e.sub_w.numpy().astype(float)
            s2h = e.sub2h.numpy().astype(np.int64)
            subv = sub.numpy()[0].astype(float)
            fbar = np.bincount(s2h, weights=w * subv, minlength=NB) / np.bincount(s2h, weights=w, minlength=NB)
            self._binref[id(s)] = (fbar, w, s2h, mask.numpy().astype(bool))
        return self._binref[id(s)]

    def _gamma(self, s, M, C):
        """The scorer's own bin mean force Gamma = M / (C + min_count) (0 on empty bins), per save."""
        e, _, _ = s.est(NB, M.shape[0])
        e.M = self._torch.as_tensor(M, dtype=self._torch.float64)
        e.C = self._torch.as_tensor(C, dtype=self._torch.float64)
        return e.bin_mean_force().numpy().astype(float)

    def stat_error(self, s, M, C):
        """e_Fp without the deterministic within-bin floor: RMS over the eval-window sub-grid of Gamma_j -
        <F'_ref>_j (same weights as fp_error), so that e_Fp^2 = stat^2 + floor^2 exactly."""
        fbar, w, s2h, mk = self.bin_reference(s)
        G = self._gamma(s, M, C)
        d = (G[:, s2h] - fbar[None, s2h])[:, mk]
        ww = w[mk]
        return np.sqrt(np.sum(d * d * ww[None, :], axis=1) / ww.sum())

    def zero_noise_floors(self):
        out = {}
        for tag, s in (("", self.sa), ("_em", self.se)):
            fbar = self.bin_reference(s)[0]
            C = np.full((1, NB), 1e12)
            Mh = fbar[None, :] * (C + 1.0)
            eF, eFp = s.score(Mh, C)
            out[f"e_F{tag}"] = float(eF[0])
            out[f"e_Fp{tag}"] = float(eFp[0])
            out[f"e_Fp{tag}_stat"] = 0.0
        return out

    def errors(self, res):
        M, C = np.asarray(res["M_all"], float), np.asarray(res["C_all"], float)
        if M.shape[1] != NB:
            raise MetricsError(f"gateway accumulators have {M.shape[1]} bins, expected {NB}")
        eF, eFp = self.sa.score(M, C)
        eFe, eFpe = self.se.score(M, C)
        return dict(e_F=np.asarray(eF, float), e_Fp=np.asarray(eFp, float), e_F_em=np.asarray(eFe, float),
                    e_Fp_em=np.asarray(eFpe, float), e_Fp_stat=self.stat_error(self.sa, M, C),
                    e_Fp_em_stat=self.stat_error(self.se, M, C))


def lta_estimator(M_all, C_all, M_prod, C_prod):
    """core_lta's reported estimator per save: production accumulators once they hold any count, else all-steps;
    Gamma = M / max(C, 1) on populated bins, 0 elsewhere; F = histogram_pmf(Gamma).  Returns (Gamma, F, use_prod)."""
    import lta_ladder_numba as E
    M_all, C_all = np.asarray(M_all, float), np.asarray(C_all, float)
    M_prod, C_prod = np.asarray(M_prod, float), np.asarray(C_prod, float)
    use_prod = C_prod.sum(axis=1) > 0
    M = np.where(use_prod[:, None], M_prod, M_all)
    C = np.where(use_prod[:, None], C_prod, C_all)
    G = E.mean_force(M, C)
    F = E.histogram_pmf(G, 2.0 * math.pi / G.shape[1])
    return G, F, use_prod


def lta_errors_from(G, F, F_ref, gamma_ref):
    dF = F - F_ref[None, :]
    dF = dF - dF.mean(axis=1, keepdims=True)
    e_F = np.sqrt(np.mean(dF ** 2, axis=1))
    e_raw = np.sqrt(np.mean((G - gamma_ref[None, :]) ** 2, axis=1))
    dp = (G - G.mean(axis=1, keepdims=True)) - (gamma_ref - gamma_ref.mean())[None, :]
    e_proj = np.sqrt(np.mean(dp ** 2, axis=1))
    return e_F, e_raw, e_proj


class LTAScorer:
    """Exact-MC reference results/equal_budget_v2/references/lta_T{T}_reference.npz."""
    kind = "lta"

    def __init__(self, T_K, reference_path=None):
        self.T_K = float(T_K)
        self.reference_path = reference_path or os.path.join(REF_DIR, f"lta_T{int(round(self.T_K))}_reference.npz")
        with np.load(self.reference_path, allow_pickle=False) as z:
            ref = {k: z[k] for k in z.files}
        if float(ref["T_K"]) != self.T_K:
            raise MetricsError(f"reference {self.reference_path} is for T {float(ref['T_K'])}, not {self.T_K}")
        for k in ("F_ref", "gamma_ref", "F_ref_se", "gamma_se", "grid_phi"):
            if ref[k].shape != (NB,):
                raise MetricsError(f"reference {k} has shape {ref[k].shape}, expected ({NB},)")
        self.F_ref = ref["F_ref"].astype(float)
        self.gamma_ref = ref["gamma_ref"].astype(float)
        self.grid_phi = ref["grid_phi"].astype(float)
        import lta_ladder_numba as E
        G0 = self.gamma_ref[None, :]
        e0 = lta_errors_from(G0, E.histogram_pmf(G0, 2.0 * math.pi / NB), self.F_ref, self.gamma_ref)
        self.reference_info = dict(
            path=os.path.relpath(self.reference_path, ROOT), sha256=sha256_file(self.reference_path), T_K=self.T_K,
            F_ref_se_rms=float(np.sqrt(np.mean(ref["F_ref_se"] ** 2))),
            gamma_se_rms=float(np.sqrt(np.mean(ref["gamma_se"] ** 2))),
            gamma_ref_circular_mean=float(self.gamma_ref.mean()),
            zero_noise_floor=dict(e_F=float(e0[0][0]), e_Fp=float(e0[1][0]), e_Fp_proj=float(e0[2][0])),
            floor_is_lower_bound=dict(e_F=False, e_Fp=False, e_Fp_proj=False),
            floor_definition=("errors at Gamma = gamma_ref (the reference's own per-bin mean force): e_Fp = 0 by "
                              "construction (no deterministic floor), e_F = the mean-force-route vs density-route "
                              "consistency of the reference; none is a lower bound"),
            estimator="Gamma = M/max(C,1) (0 where C=0) from (M_prod, C_prod) once C_prod has any count else "
                      "(M_all, C_all); F = histogram_pmf(Gamma)",
            units=dict(F="kJ/mol", Fp="kJ/mol/rad"))

    def errors(self, res):
        if float(res["meta"]["T_K"]) != self.T_K:
            raise MetricsError(f"run T {res['meta']['T_K']} vs scorer T {self.T_K}")
        G, F, use_prod = lta_estimator(res["M_all"], res["C_all"], res["M_prod"], res["C_prod"])
        if G.shape[1] != NB:
            raise MetricsError(f"LTA accumulators have {G.shape[1]} bins, expected {NB}")
        e_F, e_raw, e_proj = lta_errors_from(G, F, self.F_ref, self.gamma_ref)
        return dict(e_F=e_F, e_Fp=e_raw, e_Fp_proj=e_proj, est_production=use_prod.astype(np.int8))


def make_scorer(system, P):
    kind = system_kind(system)
    if kind == "gateway":
        return GatewayScorer(P["engine_cfg"], reference_path=reference_path(system, P))
    return LTAScorer(float(P.get("T_K", SYSTEMS[system]["T_K"])), reference_path=reference_path(system, P))


def error_names(kind):
    """Per-save error curves of a run: (thresholded, companions without frozen thresholds)."""
    if kind == "gateway":
        return ["e_F", "e_Fp", "e_F_em", "e_Fp_em"], ["e_Fp_stat", "e_Fp_em_stat"]
    return ["e_F", "e_Fp", "e_Fp_proj"], []


def error_family(k):
    return "e_F" if (k == "e_F" or k.startswith("e_F_")) else "e_Fp"


def threshold_floors(system, P, scorer):
    """For every thresholded error and frozen threshold: the scorer's zero-noise floor, whether it is a hard lower
    bound, whether the threshold is reachable at all and the statistical-error budget sqrt(eps^2 - floor^2) that
    remains (hard floors only).  Keys 'tau_<error>_<strict|mid|loose>' (the tau key stem)."""
    kind = system_kind(system)
    info = scorer.reference_info
    fl, hard = info.get("zero_noise_floor", {}), info.get("floor_is_lower_bound", {})
    out = {}
    for k in error_names(kind)[0]:
        for name, eps in zip(THR_NAMES, P["thresholds"][error_family(k)]):
            eps = float(eps)
            f, h = fl.get(k), bool(hard.get(k, False))
            reach = not (h and f is not None and eps <= f)
            budget = (math.sqrt(eps * eps - f * f) if (h and f is not None and eps > f) else
                      (0.0 if h and f is not None else None))
            out[f"tau_{k}_{name}"] = dict(
                metric=k, threshold=name, eps=eps, floor=f, floor_is_lower_bound=h, reachable=reach,
                stat_budget=budget,
                note=(f"UNREACHABLE BY CONSTRUCTION: {k} >= {f:.4g} > eps for every estimate; tau is censored for "
                      f"every arm and 'both censored' is not a tie" if not reach else
                      (f"reachable only if the statistical part (RMS of Gamma - bin-averaged reference) is <= "
                       f"{budget:.4g}" if h and budget is not None and budget < 0.5 * eps else None)))
    return out


# ------------------------------------------------------------------------------------------------- marginal
def coarse_bins(h, n_coarse=N_COARSE):
    """(..., 180) production-bin histogram -> (..., 18) coarse bins (consecutive groups of 10)."""
    h = np.asarray(h, dtype=float)
    nb = h.shape[-1]
    if nb % n_coarse:
        raise MetricsError(f"{nb} bins are not divisible into {n_coarse} coarse bins")
    return h.reshape(h.shape[:-1] + (n_coarse, nb // n_coarse)).sum(-1)


def tv_to_uniform(counts):
    """Total variation 0.5 sum_k |p_k - 1/K| of each row of nonnegative counts (NaN for an empty row)."""
    c = np.asarray(counts, dtype=float)
    tot = c.sum(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = c / tot
    tv = 0.5 * np.abs(p - 1.0 / c.shape[-1]).sum(-1)
    return np.where(tot[..., 0] > 0, tv, np.nan)


@functools.lru_cache(maxsize=None)
def tv_floor(N, k=N_COARSE):
    """E_N[TV] of the empirical histogram of N iid uniform draws on k bins (multinomial(N, 1/k)), EXACTLY:
    TV = (1/2) sum_b |X_b/N - 1/k| and every X_b ~ Bin(N, 1/k), so E[TV] = (k/2) E|X/N - 1/k|.  Returns
    (mean, 0.0) (the second value, formerly the Monte Carlo standard error, is kept for compatibility)."""
    from scipy.stats import binom
    N = int(N)
    if N < 1:
        raise ValueError("N must be >= 1")
    x = np.arange(N + 1)
    p = binom.pmf(x, N, 1.0 / k)
    return float(0.5 * k * np.sum(p * np.abs(x / N - 1.0 / k))), 0.0


@functools.lru_cache(maxsize=None)
def tv_floor_mc(N, n_mc=TV_FLOOR_NMC, seed=TV_FLOOR_SEED, k=N_COARSE):
    """Monte Carlo cross-check of tv_floor with the fixed stream default_rng([seed, N, k]): (mean, s.e.)."""
    N = int(N)
    if N < 1:
        raise ValueError("N must be >= 1")
    rng = np.random.default_rng([int(seed), N, int(k)])
    p = np.full(k, 1.0 / k)
    tot, tot2, done = 0.0, 0.0, 0
    while done < n_mc:
        m = min(50_000, n_mc - done)
        X = rng.multinomial(N, p, size=m)
        tv = 0.5 * np.abs(X / N - 1.0 / k).sum(1)
        tot += tv.sum(); tot2 += (tv * tv).sum(); done += m
    mean = tot / n_mc
    var = max(0.0, tot2 / n_mc - mean * mean)
    return float(mean), float(math.sqrt(var / n_mc))


def tv_floor_table(Ns):
    return {int(N): dict(mean=tv_floor(int(N))[0]) for N in sorted({int(n) for n in Ns})}


def tv_half(save_step, C_all, n_coarse=N_COARSE):
    """TV of the trailing-half visitation histogram at every save.

    For save i at step s_i the window is C_all(s_i) - C_all(s_lo), where s_lo is the candidate nearest s_i / 2
    among {0 (run start, zero accumulators)} and the saves before i (ties -> the earlier candidate).
    Returns dict(tv (S,), lo_step (S,), lo_index (S,) [-1 = run start], n_counts (S,) = window deposits)."""
    steps = np.asarray(save_step, dtype=np.int64)
    C = coarse_bins(C_all, n_coarse)
    S = steps.size
    cand_steps = np.concatenate([[0], steps])
    cand_C = np.vstack([np.zeros((1, C.shape[1])), C])
    tv = np.full(S, np.nan)
    lo_step = np.zeros(S, dtype=np.int64)
    lo_index = np.zeros(S, dtype=np.int64)
    n_counts = np.zeros(S)
    for i in range(S):
        cs = cand_steps[:i + 1]                      # start + saves 0..i-1
        j = int(np.argmin(np.abs(cs - steps[i] / 2.0)))
        w = C[i] - cand_C[j]
        if np.any(w < -1e-9):
            raise MetricsError("visitation counts decreased between saves (C_all not cumulative)")
        tv[i] = tv_to_uniform(w[None, :])[0]
        lo_step[i] = cs[j]
        lo_index[i] = j - 1
        n_counts[i] = w.sum()
    return dict(tv=tv, lo_step=lo_step, lo_index=lo_index, n_counts=n_counts)


# ------------------------------------------------------------------------------------------------- tau
def persistent_first(ok):
    """Smallest index i with ok[j] True for every j >= i; None if ok[-1] is False (or ok is empty)."""
    ok = np.asarray(ok, dtype=bool)
    if ok.size == 0 or not ok[-1]:
        return None
    bad = np.flatnonzero(~ok)
    return 0 if bad.size == 0 else int(bad[-1] + 1)


def tau_persistent(err, save_t, save_u, eps):
    """Persistent time-to-accuracy: the first save after which err <= eps at ALL later saves (NaN never meets
    eps).  Censored (never / not at the end) -> t = u = inf, index -1, censored True."""
    err = np.asarray(err, dtype=float)
    i = persistent_first(err <= eps)
    if i is None:
        return dict(eps=float(eps), index=-1, t=math.inf, u=math.inf, censored=True)
    return dict(eps=float(eps), index=int(i), t=float(save_t[i]), u=float(save_u[i]), censored=False)


def persistent_at_least(x, save_t, save_u, thr):
    """First save after which x >= thr at ALL later saves (population establishment); censored -> inf."""
    i = persistent_first(np.asarray(x, dtype=float) >= thr)
    if i is None:
        return dict(threshold=float(thr), index=-1, t=math.inf, u=math.inf, censored=True)
    return dict(threshold=float(thr), index=int(i), t=float(save_t[i]), u=float(save_u[i]), censored=False)


def establishment_null(N, save_u, target, frac_threshold=0.5, reliable_p=EST_RELIABLE_P):
    """Null calibration of the frozen persistent establishment criterion (far fraction >= frac_threshold x target
    at every later save) for an EXACTLY uniform population: far count ~ Bin(N, target), independent across saves
    (the saves of a real run are positively correlated, which makes passing EASIER; at small N the save spacing is
    long and they are nearly independent).  Uniform-target FR birth-death can push the instantaneous count BELOW
    binomial scatter, so at N flagged unreliable an FR establishment advantage is not evidence of faster
    convergence.  Returns p_fail per save, P(censored), P(no failure among the saves with u > 1/2), the null
    median establishment u (inf when censored with prob > 1/2) and the 'unreliable' flag (N <= 2 or
    P(no failure in the second half) < reliable_p)."""
    from scipy.stats import binom
    N = int(N)
    u = np.asarray(save_u, dtype=float)
    S = u.size
    thr = float(frac_threshold) * float(target)
    kmin = next(k for k in range(N + 1) if k / N >= thr)          # the same float comparison as the criterion
    p = float(binom.cdf(kmin - 1, N, float(target))) if kmin > 0 else 0.0
    n_late = int(np.sum(u > 0.5))
    p_hold = (1.0 - p) ** n_late
    if p <= 0.0:
        med = float(u[0]) if S else math.nan
    else:
        m = int(math.floor(math.log(0.5) / math.log1p(-p))) if p < 1.0 else 0   # max trailing saves with P >= 1/2
        med = float(u[0]) if m >= S else (math.inf if m <= 0 else float(u[S - m]))
    return dict(N=N, k_min=int(kmin), threshold=thr, target=float(target), p_fail_per_save=p, p_censored=p,
                n_saves=int(S), n_saves_second_half=n_late, p_hold_second_half=float(p_hold), null_median_u=med,
                unreliable=bool(N <= 2 or p_hold < reliable_p))


def uniform_indices(save_step, n_steps, n_lin=N_UNIFORM):
    """Indices of the saves at the n_lin uniform budget fractions u = k/n_lin (k = 1..n_lin).  Exact when
    n_steps % n_lin == 0 (step k n_steps / n_lin, integer arithmetic); otherwise the engine's rounding rule
    round(u n_steps) is used and exact=False.  Raises if any of them is not a save."""
    steps = np.asarray(save_step, dtype=np.int64)
    k = np.arange(1, n_lin + 1, dtype=np.int64)
    if n_steps % n_lin == 0:
        target, exact = k * (n_steps // n_lin), True
    else:
        target = np.clip(np.round(k / n_lin * n_steps).astype(np.int64), 1, n_steps)
        exact = False
    idx = np.searchsorted(steps, target)
    if np.any(idx >= steps.size) or np.any(steps[np.minimum(idx, steps.size - 1)] != target):
        missing = target[(idx >= steps.size) | (steps[np.minimum(idx, steps.size - 1)] != target)]
        raise MetricsError(f"uniform budget fractions missing from the save grid at steps {missing[:5].tolist()}...")
    return idx, exact


# ------------------------------------------------------------------------------------------------- per run
def _nanfinal(a):
    a = np.asarray(a, dtype=float)
    return float(a[-1]) if a.size else math.nan


def _nanmin(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanmin(a)) if np.any(np.isfinite(a)) else math.nan


def _nanmax(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanmax(a)) if np.any(np.isfinite(a)) else math.nan


def _first_event(step, n_steps, h):
    step = int(step)
    if step < 0:
        return dict(step=-1, t=math.inf, u=math.inf, censored=True)
    return dict(step=step, t=step * h, u=step / n_steps, censored=False)


def far_bin_mask(kind, meta):
    """Production bins whose centre lies in the far state (cumulative-visitation establishment)."""
    j = np.arange(NB)
    if kind == "gateway":
        cen = -1.8 + (j + 0.5) * (3.6 / NB)
        return cen > 0.5
    cen = -math.pi + (j + 0.5) * (2.0 * math.pi / NB)
    a = float(meta["a_pseudo"])
    return np.abs(cen) * a / (2.0 * math.pi) < float(meta.get("window_half", 1.5))


def check_plan(res, P, path=None, expect=None):
    """The run's N, n_steps, h, seed, method and engine knobs agree with the config (else MetricsError)."""
    meta, cfg = res["meta"], res["cfg"]
    N, n_steps = int(meta["N"]), int(meta["n_steps"])
    probs = []
    if N * n_steps != int(P["B"]):
        probs.append(f"N n_steps = {N * n_steps} != B {P['B']}")
    if float(meta["h"]) != float(P["engine_cfg"]["h"]):
        probs.append(f"h {meta['h']} != config h {P['engine_cfg']['h']}")
    for k, v in P["engine_cfg"].items():
        if k in cfg and cfg[k] != v and not (isinstance(v, (int, float)) and isinstance(cfg[k], (int, float))
                                              and float(cfg[k]) == float(v)):
            probs.append(f"engine knob {k}: run {cfg[k]!r} != config {v!r}")
    if expect:
        for k in ("N", "seed", "method"):
            if k in expect and meta.get(k) != expect[k]:
                probs.append(f"{k}: file says {meta.get(k)!r}, path says {expect[k]!r}")
    if probs:
        raise MetricsError(f"{path or 'run'} does not match the config: " + "; ".join(probs))


def run_metrics(res, system, P, scorer, path=None):
    """All per-run metrics of one result.  Returns (curves, scalars): curves are arrays over the S saves (plus
    'uniform_index' (200,)), scalars a flat dict of floats / ints / bools / strings (inf = censored)."""
    kind = system_kind(system)
    meta = res["meta"]
    N, n_steps, h = int(meta["N"]), int(meta["n_steps"]), float(meta["h"])
    seed, method = int(meta["seed"]), str(meta["method"])
    T = n_steps * h
    step = np.asarray(res["save_step"], dtype=np.int64)
    t = np.asarray(res["save_t"], dtype=float)
    u = np.asarray(res["save_u"], dtype=float)
    S = step.size
    if S == 0 or np.any(np.diff(step) <= 0):
        raise MetricsError(f"{path}: save_step must be non-empty and strictly increasing")
    if step[-1] != n_steps:
        raise MetricsError(f"{path}: last save at step {step[-1]}, not at u = 1 (n_steps {n_steps})")
    if not (np.allclose(t, step * h, rtol=1e-12, atol=0) and np.allclose(u, step / n_steps, rtol=1e-12, atol=0)):
        raise MetricsError(f"{path}: save_t / save_u inconsistent with save_step")
    iu, exact = uniform_indices(step, n_steps)
    if exact and not np.allclose(u[iu], np.arange(1, N_UNIFORM + 1) / N_UNIFORM, rtol=0, atol=1e-12):
        raise MetricsError(f"{path}: save_u at the uniform saves differs from k/{N_UNIFORM}")
    thr = P["thresholds"]

    # common budget grid (200 uniform + 24 log-spaced u): identical in u at every N
    bsteps = np.asarray(engine_module(kind).budget_save_grid(n_steps)[0], dtype=np.int64)
    bmask = np.isin(step, bsteps)
    if int(bmask.sum()) != bsteps.size:
        raise MetricsError(f"{path}: {bsteps.size - int(bmask.sum())} budget-grid saves are missing")

    # ---- errors
    err = scorer.errors(res)
    curves = dict(save_step=step, save_t=t, save_u=u, uniform_index=iu.astype(np.int64),
                  budget_grid_mask=bmask.astype(np.int8))
    curves.update(err)
    sc = dict(system=system, kind=kind, N=N, seed=seed, method=method, n_steps=n_steps, h=h, T=T, B=N * n_steps,
              n_saves=S, n_budget_grid_saves=int(bmask.sum()), u_grid_exact=bool(exact), metrics_version=METRICS_VERSION)
    err_names, companions = error_names(kind)
    for k in err_names + companions:
        e = np.asarray(err[k], float)
        if e.shape != (S,):
            raise MetricsError(f"{path}: error curve {k} has shape {e.shape}, expected ({S},)")
        sc[f"Ibar_{k[2:]}" if k.startswith("e_") else f"Ibar_{k}"] = float(np.mean(e[iu]))
        sc[f"final_{k}"] = float(e[-1])
        sc[f"n_nan_{k}"] = int(np.sum(~np.isfinite(e)))
    # tau on the frozen thresholds (e_F family on e_F thresholds, e_Fp family on e_Fp thresholds); the primary
    # errors also on the common budget grid only (tau_bgrid_*, for comparisons ACROSS N)
    for k in err_names:
        for name, eps in zip(THR_NAMES, thr[error_family(k)]):
            tt = tau_persistent(err[k], t, u, float(eps))
            for f in ("t", "u", "censored", "eps"):
                sc[f"tau_{k}_{name}_{f}"] = tt[f]
            if k in ("e_F", "e_Fp"):
                tb = tau_persistent(np.asarray(err[k], float)[bmask], t[bmask], u[bmask], float(eps))
                for f in ("t", "u", "censored"):
                    sc[f"tau_bgrid_{k}_{name}_{f}"] = tb[f]
    if kind == "lta":
        sc["est_production_from_u"] = (float(u[np.argmax(err["est_production"] > 0)])
                                       if np.any(err["est_production"] > 0) else math.inf)

    # ---- marginal: TV_inst + floor, TV_half
    hist = np.asarray(res["hist_inst"], dtype=float)
    if hist.shape != (S, NB):
        raise MetricsError(f"{path}: hist_inst shape {hist.shape}, expected ({S}, {NB})")
    if not np.allclose(hist.sum(1), N):
        raise MetricsError(f"{path}: hist_inst rows do not sum to N = {N} (diagnostics off?)")
    tvi = tv_to_uniform(coarse_bins(hist))
    th = tv_half(step, res["C_all"])
    curves.update(TV_inst=tvi, TV_half=th["tv"], TV_half_lo_step=th["lo_step"], TV_half_lo_t=th["lo_step"] * h,
                  TV_half_n_counts=th["n_counts"])
    fmean, fse = tv_floor(N)
    sc.update(TV_inst_floor=fmean, TV_inst_floor_se=fse,
              Ibar_TV_inst=float(np.mean(tvi[iu])), final_TV_inst=float(tvi[-1]),
              Ibar_TV_inst_excess=float(np.mean(tvi[iu]) - fmean), final_TV_inst_excess=float(tvi[-1] - fmean),
              Ibar_TV_half=float(np.mean(th["tv"][iu])), final_TV_half=float(th["tv"][-1]),
              final_TV_half_window_lo_t=float(th["lo_step"][-1] * h), final_TV_half_window_hi_t=float(t[-1]))
    tt = tau_persistent(th["tv"], t, u, float(thr["TV_half"]))
    for f in ("t", "u", "censored", "eps"):
        sc[f"tau_TV_half_{f}"] = tt[f]
    tb = tau_persistent(th["tv"][bmask], t[bmask], u[bmask], float(thr["TV_half"]))
    for f in ("t", "u", "censored"):
        sc[f"tau_bgrid_TV_half_{f}"] = tb[f]

    # ---- region fractions, establishment, first arrivals, true events
    reg = np.asarray(res["region_frac"], dtype=float)
    if reg.shape != (S, 3) or not np.all(np.isfinite(reg)):
        raise MetricsError(f"{path}: region_frac missing or non-finite (diagnostics off?)")
    far = FAR[kind]
    ff = reg[:, far["region_col"]]
    curves.update(region_frac=reg, far_frac=ff)
    names = ["left", "gate", "right"] if kind == "gateway" else ["cage", "neck", "window"]
    for c, nm in enumerate(names):
        sc[f"final_frac_{nm}"] = float(reg[-1, c])
        sc[f"mean_frac_{nm}"] = float(np.mean(reg[iu, c]))
    est = persistent_at_least(ff, t, u, 0.5 * far["target"])
    sc.update(est_t=est["t"], est_u=est["u"], est_censored=est["censored"], est_threshold=est["threshold"],
              est_target=far["target"], est_far_state=far["name"], est_ill_defined=bool(N <= 2))
    nul = establishment_null(N, u, far["target"])
    sc.update(est_null_p_fail_per_save=nul["p_fail_per_save"], est_null_p_censored=nul["p_censored"],
              est_null_p_hold_second_half=nul["p_hold_second_half"], est_null_median_u=nul["null_median_u"],
              est_unreliable=nul["unreliable"])
    mask = far_bin_mask(kind, meta)
    Call = np.asarray(res["C_all"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cum = Call[:, mask].sum(1) / Call.sum(1)
    cum = np.where(np.isfinite(cum), cum, 0.0)
    curves["far_frac_cum"] = cum
    bin_target = float(mask.mean())
    estc = persistent_at_least(cum, t, u, 0.5 * bin_target)
    sc.update(est_cum_t=estc["t"], est_cum_u=estc["u"], est_cum_censored=estc["censored"],
              est_cum_threshold=estc["threshold"], est_cum_bin_target=bin_target, final_far_frac_cum=float(cum[-1]))
    if kind == "gateway":
        fa = _first_event(np.asarray(res["first_arrival_step"]).ravel()[0], n_steps, h)
        sc.update(first_right_t=fa["t"], first_right_u=fa["u"], first_right_censored=fa["censored"])
        ev = np.asarray(res["events_cum"], dtype=np.int64)
        curves.update(trans_LR=ev[:, 0], trans_RL=ev[:, 1], transitions=ev.sum(1))
        sc.update(final_trans_LR=int(ev[-1, 0]), final_trans_RL=int(ev[-1, 1]), final_transitions=int(ev[-1].sum()),
                  final_lineage_right_frac=float(np.asarray(res["lineage_visited"]).reshape(S, -1)[-1, 0] / N))
    else:
        for key, nm in (("first_window_step", "window"), ("first_opposite_step", "opposite")):
            fa = _first_event(np.asarray(res[key]).ravel()[0], n_steps, h)
            sc.update({f"first_{nm}_t": fa["t"], f"first_{nm}_u": fa["u"], f"first_{nm}_censored": fa["censored"]})
        wc = np.asarray(res["events_window_crossings"], dtype=np.int64)
        tl = np.asarray(res["events_translocations"], dtype=np.int64)
        curves.update(window_crossings=wc, transitions=tl)
        sc.update(final_window_crossings=int(wc[-1]), final_transitions=int(tl[-1]),
                  final_lineage_window_frac=float(np.asarray(res["lineage_ever_window"])[-1] / N),
                  final_lineage_opposite_frac=float(np.asarray(res["lineage_ever_opposite"])[-1] / N))
    sc["transitions_per_t"] = sc["final_transitions"] / T

    # ---- genealogy and FR activity.  The reported EVENT = a realised death (= a realised replacement: one death and
    # one copy) in BOTH systems; the gateway's capped candidate counts kd + kc (a kc >= kd opportunity may realise no
    # death, and one replacement can count as 2 candidates) are kept under fr_candidate* names.
    cand = cand_hist = cand_opps = None
    if kind == "gateway":
        g = {k: np.asarray(res[k], dtype=float) for k in ("gen_nuniq_run", "gen_ess_run", "gen_maxfam_run",
                                                          "gen_nuniq_win", "gen_ess_win", "gen_maxfam_win")}
        deaths = np.asarray(res["fr_deaths_cum"], dtype=np.int64)
        opps = np.asarray(res["fr_opp_cum"], dtype=np.int64)
        cand = np.asarray(res["fr_kd_cum"], dtype=np.int64) + np.asarray(res["fr_kc_cum"], dtype=np.int64)
        cand_opps = np.asarray(res["fr_opp_event_cum"], dtype=np.int64)
        cand_hist = np.asarray(res["fr_events_hist"], dtype=np.int64).ravel()
        if "fr_deaths_hist" in res and "fr_opp_death_cum" in res:          # gateway_ladder_numba/2
            ehist = np.asarray(res["fr_deaths_hist"], dtype=np.int64).ravel()
            opps_ev = np.asarray(res["fr_opp_death_cum"], dtype=np.int64)
            ev_src = "fr_deaths_hist / fr_opp_death_cum (realised)"
        else:                                                               # engine /1: mean from fr_deaths_cum only
            ehist, opps_ev = None, np.full(S, np.nan)
            ev_src = "fr_deaths_cum only (engine /1: no realised-death histogram; max and opportunity share unknown)"
        ev_def = ("realised deaths per opportunity (each death is replaced by one copy); candidates kd + kc reported "
                  "separately as fr_candidate*")
        if int(cand_hist.sum()) != int(opps[-1]):
            raise MetricsError(f"{path}: candidate histogram total {int(cand_hist.sum())} != opportunities {int(opps[-1])}")
    else:
        g = dict(gen_nuniq_run=res["gen_n_unique"], gen_ess_run=res["gen_ess"], gen_maxfam_run=res["gen_max_frac"],
                 gen_nuniq_win=res["gen_n_unique_win"], gen_ess_win=res["gen_ess_win"],
                 gen_maxfam_win=res["gen_max_frac_win"])
        g = {k: np.asarray(v, dtype=float) for k, v in g.items()}
        deaths = np.asarray(res["cum_deaths"], dtype=np.int64)
        opps = np.asarray(res["cum_opps"], dtype=np.int64)
        opps_ev = np.asarray(res["cum_opps_with_event"], dtype=np.int64)
        eh = np.asarray(res["events_per_opp_hist"], dtype=np.int64)
        ehist = eh[-1] if eh.ndim == 2 else eh
        ev_src = "events_per_opp_hist / cum_opps_with_event (realised)"
        ev_def = "realised replacements (one death + one copy each) per opportunity"
    n_opp = int(opps[-1])
    if ehist is not None:
        kk = np.arange(ehist.size)
        if int(ehist.sum()) != n_opp:
            raise MetricsError(f"{path}: realised events-per-opportunity histogram total {int(ehist.sum())} != "
                               f"opportunities {n_opp}")
        if int((kk * ehist).sum()) != int(deaths[-1]):
            raise MetricsError(f"{path}: realised events-per-opportunity histogram holds {int((kk * ehist).sum())} "
                               f"events, deaths counter {int(deaths[-1])}")
    curves.update(g)
    curves.update(deaths_cum=deaths, opps_cum=opps, opps_event_cum=opps_ev)
    if cand is not None:
        curves.update(candidates_cum=cand, opps_candidate_cum=cand_opps)
    for k, v in g.items():
        sc[f"final_{k}"] = _nanfinal(v)
    for k in ("gen_ess_run", "gen_ess_win", "gen_nuniq_run", "gen_nuniq_win"):
        sc[f"min_{k}"] = _nanmin(g[k])
    for k in ("gen_maxfam_run", "gen_maxfam_win"):
        sc[f"max_{k}"] = _nanmax(g[k])
    sc["final_gen_nuniq_run_frac"] = sc["final_gen_nuniq_run"] / N
    cap = int((ehist if ehist is not None else cand_hist).size - 1)
    sc.update(fr_cap=cap, fr_deaths=int(deaths[-1]), fr_events=int(deaths[-1]), fr_opps=n_opp,
              fr_opps_with_event=(int(opps_ev[-1]) if np.isfinite(opps_ev[-1]) else -1), fr_event_definition=ev_def,
              fr_event_source=ev_src,
              fr_frac_opps_with_event=(float(opps_ev[-1]) / n_opp if (n_opp and np.isfinite(opps_ev[-1])) else math.nan),
              fr_mean_event_frac_per_opp=(deaths[-1] / n_opp / N if n_opp else math.nan),
              fr_max_event_frac_per_opp=(float(np.arange(ehist.size)[ehist > 0].max()) / N
                                         if (n_opp and ehist is not None) else math.nan),
              fr_deaths_frac_per_opp=(deaths[-1] / n_opp / N if n_opp else math.nan),
              fr_deaths_per_walker=deaths[-1] / N,
              fr_deaths_per_walker_per_t=deaths[-1] / (N * T),
              fr_zero_deaths=bool(method == "fr" and int(deaths[-1]) == 0))
    if cand is not None:
        kc = np.arange(cand_hist.size)
        sc.update(fr_candidates=int(cand[-1]),
                  fr_frac_opps_with_candidate=(cand_opps[-1] / n_opp if n_opp else math.nan),
                  fr_mean_candidate_frac_per_opp=(float((kc * cand_hist).sum()) / n_opp / N if n_opp else math.nan),
                  fr_max_candidate_frac_per_opp=(float(kc[cand_hist > 0].max()) / N if n_opp else math.nan))
    if kind == "gateway":
        # max(1, floor(0.08 N)) IS the historical gateway rule (plan section 1): no finite-N extension here
        sc["fr_cap_extension"] = False
        sc["fr_cap_rule"] = "historical gateway rule max(1, floor(0.08 N)) (contains the floor of 1)"
    else:
        sc["fr_cap_extension"] = bool(method == "fr" and int(meta.get("cap_historical", 0)) < 1 <= int(meta.get("cap", 0)))
        sc["fr_cap_rule"] = ("finite-N extension max(1, floor(0.02 N)) (historical int(0.02 N) = 0)"
                             if sc["fr_cap_extension"] else "historical int(0.02 N) (>= 1 here)")
    sc["N_lt_50_flag"] = bool(kind == "lta" and N < 50)

    # ---- cost
    nfe = int(res["n_force_evals"])
    exp = N * n_steps if kind == "gateway" else N * (n_steps + 1)
    wall = float(res["wall_s"])
    sc.update(n_force_evals=nfe, n_force_evals_expected=exp, n_force_evals_ok=bool(nfe == exp),
              n_force_evals_convention=("N per completed step (N n_steps)" if kind == "gateway"
                                        else "N per evaluated step 0..n_steps (N (n_steps + 1))"),
              wall_s=wall, peak_rss_mb=float(res["peak_rss_mb"]), us_per_walker_step=wall * 1e6 / (N * n_steps))
    return curves, to_python(sc)


# ------------------------------------------------------------------------------------------------- statistics
def quantile_inf(x, q):
    """Linear-interpolation quantile (numpy's default) that is +inf whenever the upper interpolation neighbour
    is +inf (censored values), instead of NaN from inf - inf.  NaNs are dropped."""
    x = np.sort(np.asarray(x, dtype=float))
    x = x[~np.isnan(x)]
    if x.size == 0:
        return math.nan
    p = float(q) / 100.0 * (x.size - 1)
    lo, hi = int(math.floor(p)), int(math.ceil(p))
    if math.isinf(x[hi]) or math.isinf(x[lo]):
        return float(x[hi]) if math.isinf(x[hi]) else float(x[lo])
    return float(x[lo] + (x[hi] - x[lo]) * (p - lo))


def describe(values):
    """values: dict seed -> number (inf allowed).  median / q25 / q75 (quantile_inf), counts."""
    seeds = sorted(values)
    x = np.array([values[s] for s in seeds], dtype=float)
    xf = x[~np.isnan(x)]
    return dict(n=int(xf.size), n_inf=int(np.sum(np.isinf(xf))), median=quantile_inf(x, 50),
                q25=quantile_inf(x, 25), q75=quantile_inf(x, 75), mean=float(np.mean(xf)) if xf.size else math.nan,
                per_seed={int(s): float(values[s]) for s in seeds})


def boot_median_ci(x, n_boot=N_BOOT, seed=BOOT_SEED):
    """(median, 2.5 %, 97.5 %) of a seed bootstrap of the median (resampling with replacement)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    b = np.median(x[rng.integers(0, x.size, size=(int(n_boot), x.size))], axis=1)
    return float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired_contrast(abf, fr, n_boot=N_BOOT, seed=BOOT_SEED):
    """abf, fr: dict seed -> value (same N).  G = (FR - ABF)/ABF per common seed with both finite and ABF != 0;
    median, bootstrap 95 % CI, wins (G < 0), losses, ties, absolute medians."""
    common = sorted(set(abf) & set(fr))
    use = [s for s in common if np.isfinite(abf[s]) and np.isfinite(fr[s]) and abf[s] != 0]
    G = np.array([(fr[s] - abf[s]) / abf[s] for s in use], dtype=float)
    med, lo, hi = boot_median_ci(G, n_boot, seed)
    a = np.array([abf[s] for s in use], float)
    f = np.array([fr[s] for s in use], float)
    return dict(n=len(use), n_common=len(common), G_median=med, G_ci95=[lo, hi],
                wins=int(np.sum(G < 0)), losses=int(np.sum(G > 0)), ties=int(np.sum(G == 0)),
                abf_median=quantile_inf(a, 50), fr_median=quantile_inf(f, 50),
                per_seed={int(s): float(g) for s, g in zip(use, G)})


def sign_test_p(wins, losses):
    """Exact two-sided binomial sign test (ties excluded)."""
    n = int(wins) + int(losses)
    if n == 0:
        return 1.0
    k = min(int(wins), int(losses))
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return float(min(1.0, 2.0 * p))


def paired_tau_contrast(abf, fr, n_boot=N_BOOT, seed=BOOT_SEED):
    """Censoring-aware paired comparison of tau (inf = censored).  Rank part over every common seed: FR wins when
    tau_FR < tau_ABF (any finite beats inf; inf vs inf is a tie), exact sign test.  Ratio part: G = (FR - ABF)/ABF
    on the seeds where both are finite (n_both_finite), median + bootstrap CI.  Censored fractions per arm."""
    common = sorted(set(abf) & set(fr))
    a = np.array([abf[s] for s in common], float)
    f = np.array([fr[s] for s in common], float)
    wins = int(np.sum(f < a))
    losses = int(np.sum(f > a))
    ties = int(len(common) - wins - losses)
    both = [s for s in common if np.isfinite(abf[s]) and np.isfinite(fr[s]) and abf[s] > 0]
    G = np.array([(fr[s] - abf[s]) / abf[s] for s in both], float)
    med, lo, hi = boot_median_ci(G, n_boot, seed)
    return dict(n_common=len(common),
                censored_frac_abf=float(np.mean(np.isinf(a))) if common else math.nan,
                censored_frac_fr=float(np.mean(np.isinf(f))) if common else math.nan,
                abf_median=quantile_inf(a, 50), fr_median=quantile_inf(f, 50),
                rank=dict(wins=wins, losses=losses, ties=ties, sign_test_p=sign_test_p(wins, losses)),
                n_both_finite=len(both), G_median=med, G_ci95=[lo, hi],
                per_seed={int(s): [float(abf[s]), float(fr[s])] for s in common})


def best_allocation(table, Ns_abf, Ns_fr, n_boot=N_BOOT, seed=BOOT_SEED):
    """table: {N: {"abf": {seed: v}, "fr": {seed: v}}} (lower is better, inf = censored allowed, missing seeds
    absent).  Point estimate: min over Ns_abf of the ABF seed median vs min over Ns_fr of the FR seed median.
    Bootstrap: in every resample, for every N the seeds are resampled with replacement (one index draw per N shared
    by both methods, independent across N) and the minimising N is RE-SELECTED; reports how often each N wins
    (TIES share the credit equally: boot_tie_frac = fraction of resamples with a tie at the minimum), CIs of both
    best values and of best_FR - best_ABF (and of the relative difference).  A resample in which an N has no
    usable seed (all resampled seeds missing) drops that N from that resample's selection (NOT treated as
    censored); a resample with no candidate at all is 'no data' (boot_frac_no_data), separate from 'every candidate
    censored' (boot_frac_censored, only possible for censored metrics such as tau)."""
    rng = np.random.default_rng(seed)
    Ns = sorted(set(Ns_abf) | set(Ns_fr))
    med = {m: {} for m in ("abf", "fr")}
    boot = {m: {} for m in ("abf", "fr")}
    for N in Ns:
        d = table.get(N, {})
        seeds = sorted(set(d.get("abf", {})) | set(d.get("fr", {})))
        if not seeds:
            continue
        X = np.array([[d.get(m, {}).get(s, np.nan) for m in ("abf", "fr")] for s in seeds], float)
        idx = rng.integers(0, len(seeds), size=(int(n_boot), len(seeds)))
        Xb = X[idx]                                   # (B, n, 2)
        for c, m in enumerate(("abf", "fr")):
            if np.all(np.isnan(X[:, c])):
                continue
            med[m][N] = quantile_inf(X[:, c], 50)
            with np.errstate(invalid="ignore"):
                cnt = np.sum(~np.isnan(Xb[:, :, c]), axis=1)
                s = np.sort(np.where(np.isnan(Xb[:, :, c]), np.inf, Xb[:, :, c]), axis=1)
            # inf-aware median of the non-NaN part of every resample (NaNs were pushed to the end as +inf,
            # so only the first cnt entries are real; a censored +inf genuinely sorts there too)
            bm = np.full(int(n_boot), np.nan)
            for nn in np.unique(cnt):
                if nn == 0:
                    continue
                rows = cnt == nn
                p = 0.5 * (nn - 1)
                lo, hi = int(math.floor(p)), int(math.ceil(p))
                a, b = s[rows, lo], s[rows, hi]
                with np.errstate(invalid="ignore"):
                    v = a + (b - a) * (p - lo)
                v = np.where(np.isinf(b) | np.isinf(a), np.where(np.isinf(b), b, a), v)
                bm[rows] = v
            boot[m][N] = bm
    out = {}
    for m, Nset in (("abf", Ns_abf), ("fr", Ns_fr)):
        cand = [N for N in sorted(Nset) if N in med[m]]
        if not cand:
            out[m] = None
            continue
        vals = np.array([med[m][N] for N in cand])
        k = int(np.argmin(vals))
        tied_pt = [int(N) for N, v in zip(cand, vals) if math.isfinite(vals[k]) and v == vals[k]]
        B = np.stack([boot[m][N] for N in cand], axis=1)       # (n_boot, n_cand), NaN = no usable seed
        nodata = np.all(np.isnan(B), axis=1)
        Bf = np.where(np.isnan(B), np.inf, B)                  # an N without data cannot be selected
        bmin = Bf.min(axis=1)
        finb = np.isfinite(bmin)
        tied = (Bf == bmin[:, None]) & finb[:, None]           # (n_boot, n_cand): every N at the minimum
        n_tied = tied.sum(axis=1)
        credit = np.where(finb[:, None], tied / np.maximum(n_tied, 1)[:, None], 0.0)
        bestb = np.where(nodata, np.nan, bmin)                 # inf = every candidate censored in that resample
        # frequency of each N being best among ALL resamples (ties share the credit); the remainder is
        # boot_frac_censored (every candidate censored) + boot_frac_no_data
        freq = {int(N): float(np.mean(credit[:, i])) for i, N in enumerate(cand)}
        out[m] = dict(best_N=(int(cand[k]) if math.isfinite(vals[k]) else None), best_value=float(vals[k]),
                      best_N_tied=tied_pt if len(tied_pt) > 1 else [],
                      medians={int(N): float(med[m][N]) for N in cand}, Ns=[int(N) for N in cand],
                      boot_best_N_freq=freq, boot_frac_censored=float(np.mean(np.isinf(bestb))),
                      boot_frac_no_data=float(np.mean(nodata)), boot_tie_frac=float(np.mean(n_tied > 1)),
                      boot_best_value_ci95=[quantile_inf(bestb, 2.5), quantile_inf(bestb, 97.5)],
                      _boot=bestb)
    if out.get("abf") and out.get("fr"):
        da, df = out["abf"]["_boot"], out["fr"]["_boot"]
        with np.errstate(invalid="ignore"):
            diff = df - da                      # inf - inf -> NaN (both censored: undefined)
            rel = np.where(np.isfinite(da) & (da != 0), diff / np.where(da != 0, da, 1.0), np.nan)
        ok = ~np.isnan(diff)
        a0, f0 = out["abf"]["best_value"], out["fr"]["best_value"]
        with np.errstate(invalid="ignore"):
            pt = f0 - a0
        out["fr_minus_abf"] = dict(
            point=float(pt),
            point_rel=float(pt / a0) if (math.isfinite(a0) and a0 != 0 and not math.isnan(pt)) else math.nan,
            ci95=[quantile_inf(diff, 2.5), quantile_inf(diff, 97.5)],
            rel_ci95=[quantile_inf(rel, 2.5), quantile_inf(rel, 97.5)],
            frac_resamples_fr_better=float(np.mean(df[ok] < da[ok])) if ok.any() else math.nan,
            frac_resamples_undefined=float(np.mean(~ok)))
    for m in ("abf", "fr"):
        if out.get(m):
            out[m].pop("_boot")
    out["n_boot"] = int(n_boot)
    out["seed"] = int(seed)
    return out


def max_transient_improvement(u, E_abf, E_fr):
    """Median curves over seeds on the uniform u grid; rel(u) = (ABF - FR)/ABF; max (and min) and where."""
    ma = np.median(np.asarray(E_abf, float), axis=0)
    mf = np.median(np.asarray(E_fr, float), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (ma - mf) / ma
    if not np.any(np.isfinite(rel)):
        return dict(max_rel=math.nan, u_at_max=math.nan, min_rel=math.nan, u_at_min=math.nan, final_rel=math.nan,
                    rel=rel)
    kmax, kmin = int(np.nanargmax(rel)), int(np.nanargmin(rel))
    return dict(max_rel=float(rel[kmax]), u_at_max=float(u[kmax]), min_rel=float(rel[kmin]), u_at_min=float(u[kmin]),
                final_rel=float(rel[-1]), rel=rel)


def to_python(obj):
    """numpy scalars / arrays -> python objects (non-finite floats kept)."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_python(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def json_safe(obj):
    """Strict-JSON copy: numpy scalars -> python; +inf -> the string "inf" (censored: float("inf") reads it
    back), -inf -> "-inf", NaN -> null (undefined)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    return obj
