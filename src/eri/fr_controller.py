"""Online, reference-free stopping controller for the Fisher-Rao (FR) birth-death schedule of histogram ABF
(docs/adaptive_fr_campaign/ONLINE_STOPPING_RULE.md).  Engine-agnostic: the host calls `observe` at fixed check
times with the CURRENT cumulative state of the running FR simulation only.  It never receives a reference free
energy, a mean-force reference, another trajectory or anything from the future, and it does no I/O.

Rule "R1" (frozen spec, all lengths in t.u.):
  checks at t = t_start + n * check_every (n >= 0; the record at t_start anchors the first window);
  window W: for every check, b_r / d_r = births / deaths in coarse region r (R equal arcs of the CV circle) during
    (t - W, t]  (differences of the cumulative birth / death histograms);
  directedness  j(t) = max(0, 1/2 sum_r |b_r - d_r| - 1/2 sum_r sqrt(2 (b_r + d_r) / pi)) / (K W)
    = the systematically relocated fraction of the population per t.u. (the noise expectation of |b - d| under a
    symmetric null is removed);  j_max(t) = running max of j;
  noise floor   s_j(t) = 1/2 sqrt( sum_r (b_r + d_r) (1 - 2/pi) ) / (K W)  (the null s.d. of the same sum);
  support(t)   every fine bin has C_j >= support_count;
  stability(t) || Gamma_t - Gamma_{t-W} ||_2 / ( || Gamma_{t-W} ||_2 + 1e-12 ) <= stability_tol over the bins
    supported at t - W  (False if none is supported);
  low(t) = j(t) <= max(theta * j_max(t), z_floor * s_j(t))  and  support(t)  and  stability(t)
           and  t >= t_start + W;
  t1 = the first check of the first run of `persist` consecutive low checks;
  decision at the check that completes that run; FR stops at t_stop = max(stop_factor * t1, decision time).
  The stop factor multiplies ABSOLUTE time t1 (from t = 0), as the validated offline rule 2 x t_establishment did.
  Safety (evaluated at every check until FR has actually stopped): if the windowed ancestor ESS/K < essw_min or the
  windowed max lineage share > max(maxlin_min, maxlin_k / K), FR stops at that check (reason 'safety'); safety can
  only bring a scheduled stop EARLIER.  A stop is never postponed and FR never restarts.  A run in which neither
  condition fires keeps FR to the horizon and is reported as 'never stopped'.
"""
import math

import numpy as np

R1 = dict(name="R1", check_every=0.25, W=1.0, R=18, theta=0.25, z_floor=2.0, persist=4, stop_factor=2.0,
          support_count=200.0, stability_tol=0.05, essw_min=0.15, maxlin_min=0.10, maxlin_k=6.0)


def directedness(b, d, K, W):
    n = b + d
    moved = 0.5 * np.abs(b - d).sum()
    noise = 0.5 * np.sqrt(2.0 * n / math.pi).sum()
    return max(0.0, moved - noise) / (K * W)


def directedness_noise_sd(b, d, K, W):
    return 0.5 * math.sqrt(float((b + d).sum()) * (1.0 - 2.0 / math.pi)) / (K * W)


def check_grid(spec, dt):
    """The controller's grid must be exact on the engine's step grid (else the trailing window never closes)."""
    ce = spec["check_every"]
    n = round(ce / dt)
    if n < 1 or abs(n * dt - ce) > 1e-9 * max(1.0, ce):
        raise ValueError(f"check_every {ce} is not a multiple of dt {dt}")
    m = round(spec["W"] / ce)
    if m < 1 or abs(m * ce - spec["W"]) > 1e-9:
        raise ValueError(f"W {spec['W']} is not a multiple of check_every {ce}")
    return int(n)


def coarse(x, R):
    x = np.asarray(x, float)
    if x.shape[-1] % R:
        raise ValueError("n_grid must be a multiple of R")
    return x.reshape(R, -1).sum(1)


class OnlineStop:
    """Feed `observe(t, BH, DH, C, M, essw, maxlin)` at every check time (strictly increasing, spaced check_every,
    starting at t_start).  Returns the scheduled stop time (float) once decided, else None."""

    def __init__(self, spec, K, t_start):
        self.s = dict(spec)
        self.K = int(K)
        self.t_start = float(t_start)
        self.hist = []                 # per check: t, BHc, DHc (coarse cumulative), G (fine), Csupp mask
        self.log = []                  # per check: dict of the statistics
        self.j_max = 0.0
        self.run = 0
        self.t1 = None
        self._t_first = None
        self.t_decision = None
        self.t_stop = None
        self.reason = None

    # -- serialisation (checkpoint / result) --
    def state(self):
        return dict(spec=self.s, K=self.K, t_start=self.t_start, j_max=self.j_max, run=self.run, t1=self.t1,
                    t_first=self._t_first,
                    t_decision=self.t_decision, t_stop=self.t_stop, reason=self.reason, log=self.log,
                    hist=[dict(t=h["t"], BHc=h["BHc"].tolist(), DHc=h["DHc"].tolist(), G=h["G"].tolist(),
                               sup=h["sup"].tolist()) for h in self.hist])

    @classmethod
    def from_state(cls, st):
        c = cls(st["spec"], st["K"], st["t_start"])
        for k in ("j_max", "run", "t1", "t_decision", "t_stop", "reason", "log"):
            setattr(c, k, st[k])
        c._t_first = st["t_first"]
        c.hist = [dict(t=h["t"], BHc=np.array(h["BHc"]), DHc=np.array(h["DHc"]), G=np.array(h["G"]),
                       sup=np.array(h["sup"], bool)) for h in st["hist"]]
        return c

    def _lag(self, t):
        """The record at time t - W (exact check time; checks are on a fixed grid)."""
        W = self.s["W"]
        for h in reversed(self.hist):
            if abs(h["t"] - (t - W)) < 1e-9:
                return h
            if h["t"] < t - W - 1e-9:
                break
        return None

    def safety_limit(self):
        return max(self.s["maxlin_min"], self.s["maxlin_k"] / self.K)

    def observe(self, t, BH, DH, C, M, essw, maxlin):
        s = self.s
        if self.t_stop is not None:
            # after the decision only the safety backstop is evaluated, and it can only bring the stop earlier
            if t < self.t_stop - 1e-12 and (essw < s["essw_min"] or maxlin > self.safety_limit()):
                self.log.append(dict(t=float(t), essw=float(essw), maxlin=float(maxlin), safety_override=True))
                self.t_stop = float(t)
                self.reason = self.reason + "+safety"
            return self.t_stop
        C = np.asarray(C, float)
        G = np.where(C > 0, np.asarray(M, float) / np.maximum(C, 1.0), 0.0)
        rec = dict(t=float(t), BHc=coarse(BH, s["R"]), DHc=coarse(DH, s["R"]), G=G, sup=C >= s["support_count"])
        lag = self._lag(t)
        row = dict(t=float(t), essw=float(essw), maxlin=float(maxlin))
        if lag is not None:
            b = rec["BHc"] - lag["BHc"]
            d = rec["DHc"] - lag["DHc"]
            j = directedness(b, d, self.K, s["W"])
            sj = directedness_noise_sd(b, d, self.K, s["W"])
            self.j_max = max(self.j_max, j)
            sup = bool(rec["sup"].all())
            m = lag["sup"]
            stab = (float(np.linalg.norm((G - lag["G"])[m]) / (np.linalg.norm(lag["G"][m]) + 1e-12))
                    if m.any() else float("inf"))
            low = (j <= max(s["theta"] * self.j_max, s["z_floor"] * sj)) and sup and stab <= s["stability_tol"]
            row.update(j=j, j_max=self.j_max, s_j=sj, support=sup, stability=stab, low=bool(low),
                       events=float((b + d).sum() / 2))
            if low:
                self.run += 1
                if self.run == 1:
                    self._t_first = float(t)
                if self.run >= s["persist"]:
                    self.t1 = self._t_first
                    self.t_decision = float(t)
                    self.t_stop = max(s["stop_factor"] * self.t1, float(t))
                    self.reason = "established"
            else:
                self.run = 0
        if self.t_stop is None and (essw < s["essw_min"] or maxlin > self.safety_limit()):
            self.t_decision = float(t)
            self.t_stop = float(t)
            self.reason = "safety"
        row.update(run=self.run, t_stop=self.t_stop)
        self.log.append(row)
        self.hist.append(rec)
        keep = s["W"] + 2 * s["check_every"]          # only the trailing window is ever needed
        self.hist = [h for h in self.hist if h["t"] >= t - keep - 1e-9]
        return self.t_stop
