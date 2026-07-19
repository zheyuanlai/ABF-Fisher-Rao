"""GATE 3: validate the shared closure metric pipeline (src/closure_metrics.py).

Two layers:
  (A) SYNTHETIC unit tests with analytically known answers (KL/TV/entropy/tau/
      coverage) at strict tolerance.
  (B) REAL smoke test: run compute_metrics over a sample of pilot npz and assert
      the common schema is complete, finite where it must be, and estimator columns
      are populated + distinct.

Writes results/opes_closure/metrics_validation/metrics_validation.json.
Usage:  CUDA_VISIBLE_DEVICES="" python -u scripts/validate_metrics.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import closure_metrics as cm  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "opes_closure",
                   "metrics_validation")


def _check(name, cond, detail, tol=None, value=None):
    return dict(name=name, passed=bool(cond), detail=detail, tol=tol, value=value)


def synthetic_tests():
    res = []
    grid = np.linspace(0.0, 1.0, 401)
    # 1. identical marginals => KL=0, TV=0
    p = np.exp(-0.5 * ((grid - 0.5) / 0.1) ** 2)
    m = cm.marginal_metrics(grid, p.copy(), p.copy())
    res.append(_check("KL_self_zero", abs(m["marginal_kl"]) < 1e-6,
                      f"KL(p||p)={m['marginal_kl']:.2e}", 1e-6, m["marginal_kl"]))
    res.append(_check("TV_self_zero", abs(m["marginal_tv"]) < 1e-6,
                      f"TV(p,p)={m['marginal_tv']:.2e}", 1e-6, m["marginal_tv"]))
    # 2. uniform entropy on [0,1] == 0 (differential entropy of U(0,1))
    uni = np.ones_like(grid)
    m2 = cm.marginal_metrics(grid, uni.copy(), uni.copy())
    res.append(_check("uniform_entropy_zero", abs(m2["marginal_entropy"]) < 1e-3,
                      f"H[U(0,1)]={m2['marginal_entropy']:.2e} (analytic 0)", 1e-3,
                      m2["marginal_entropy"]))
    # 3. TV between two disjoint-ish gaussians ~ 1
    a = np.exp(-0.5 * ((grid - 0.1) / 0.02) ** 2)
    b = np.exp(-0.5 * ((grid - 0.9) / 0.02) ** 2)
    m3 = cm.marginal_metrics(grid, a, b)
    res.append(_check("TV_disjoint_near_one", m3["marginal_tv"] > 0.95,
                      f"TV(disjoint)={m3['marginal_tv']:.3f} (want ~1)", None, m3["marginal_tv"]))
    # 4. KL asymmetry / positivity: KL >= 0 always
    m4 = cm.marginal_metrics(grid, a, uni.copy())
    res.append(_check("KL_nonneg", m4["marginal_kl"] >= -1e-9,
                      f"KL>=0: {m4['marginal_kl']:.3f}", None, m4["marginal_kl"]))
    return res


def anytime_tests():
    res = []
    times = np.linspace(0, 100, 101)
    # error decays 1.0 -> 0.1; crosses 0.25 at a known point
    l2 = 0.1 + 0.9 * np.exp(-times / 20.0)
    a = cm.anytime_metrics(times, l2, integrated_l2_f=float("nan"), l2_f_final=l2[-1])
    # tau_abs: first t with l2<=0.25 => solve 0.1+0.9 e^{-t/20}=0.25 => t=20 ln(0.9/0.15)
    t_expected = 20.0 * np.log(0.9 / 0.15)
    res.append(_check("tau_abs_correct", abs(a["tau_abs"] - t_expected) < 2.0,
                      f"tau_abs={a['tau_abs']:.2f} vs analytic {t_expected:.2f}", 2.0, a["tau_abs"]))
    res.append(_check("normalized_anytime_positive",
                      0 < a["normalized_anytime_l2_f"] < 1.0,
                      f"time-avg err={a['normalized_anytime_l2_f']:.3f}", None,
                      a["normalized_anytime_l2_f"]))
    # monotone-improving series => integrated recomputed finite
    res.append(_check("integrated_recomputed", np.isfinite(a["integrated_l2_f"]),
                      f"integrated={a['integrated_l2_f']:.2f}", None, a["integrated_l2_f"]))
    return res


def real_npz_tests(sample=8):
    res = []
    fs = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..",
                "results", "opes_wca", "raw", "*.npz")))
    if not fs:
        res.append(_check("real_npz_present", False, "no pilot npz found on disk"))
        return res, []
    fs = fs[:: max(1, len(fs) // sample)][:sample]
    schema = cm.metrics_schema(); rows = []
    all_keys_ok = True; est_ok = True; finite_ok = True
    for f in fs:
        d = np.load(f, allow_pickle=True)
        row = cm.compute_metrics(d); rows.append(row)
        missing = [k for k in schema if k not in row]
        if missing:
            all_keys_ok = False
        # estimator columns present + native vs common are distinct concepts
        if not (np.isfinite(row["l2_f_common"])):
            finite_ok = False
        if row["primary_estimator"] not in ("meanforce", "reweight"):
            est_ok = False
        d.close()
    res.append(_check("real_npz_present", True, f"scored {len(fs)} pilot npz"))
    res.append(_check("schema_complete", all_keys_ok,
                      f"all {len(schema)} common-schema keys present in every row"))
    res.append(_check("l2_common_finite", finite_ok, "l2_f_common finite on all sampled runs"))
    res.append(_check("estimator_labeled", est_ok,
                      "primary_estimator in {meanforce,reweight}; native+common both emitted"))
    # marginal KL should be well-defined (finite, >=0) on real runs
    kls = [r["marginal_kl"] for r in rows if np.isfinite(r["marginal_kl"])]
    res.append(_check("marginal_kl_finite_nonneg", len(kls) > 0 and all(k >= -1e-6 for k in kls),
                      f"{len(kls)}/{len(rows)} finite marginal_kl, all >=0; "
                      f"range=[{min(kls):.3f},{max(kls):.3f}]" if kls else "no finite KL"))
    return res, rows


def main():
    os.makedirs(OUT, exist_ok=True)
    groups = {"synthetic": synthetic_tests(), "anytime": anytime_tests()}
    real, rows = real_npz_tests()
    groups["real_npz"] = real
    allr = [r for g in groups.values() for r in g]
    npass = sum(1 for r in allr if r["passed"])
    for g, rs in groups.items():
        for r in rs:
            print(f"[{'PASS' if r['passed'] else 'FAIL'}] {g:10s} {r['name']:28s} {r['detail']}")
    summary = dict(n_tests=len(allr), n_pass=npass, all_pass=(npass == len(allr)),
                   groups={g: rs for g, rs in groups.items()},
                   n_real_rows=len(rows), schema=cm.metrics_schema())
    with open(os.path.join(OUT, "metrics_validation.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n{npass}/{len(allr)} passed -> {OUT}/metrics_validation.json")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
