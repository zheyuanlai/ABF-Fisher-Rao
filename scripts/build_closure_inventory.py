#!/usr/bin/env python
"""Build the one authoritative v1 results inventory, from artifacts only.

Every number in the closure documentation and in the report's synthesis comes from here.
Nothing in this file recomputes a scientific estimator: each row is read out of the
aggregated artifact that the study's own analyzer produced, and the only transformation
applied is putting all of them on **one sign convention and one unit**.

That normalisation is the point of the script. The study accumulated three conventions:

* ``median_gain_pct_F`` and friends (WCA production / phase-diagram / representative,
  entropic bottleneck, metastability) are *gains*: ``100*(abf-arm)/abf``, positive = better;
* ``pct`` / ``int_l2_f_pct`` (WCA sham, gateway) are *relative changes*:
  ``100*(arm-abf)/abf``, negative = better;
* ``rel_med`` (alkanes, CV extension) and the alanine decision JSONs are relative changes
  expressed as a **fraction**, not a percentage.

The inventory publishes exactly one numeric convention:

    rel_pct = 100 * (arm_estimate - comparator_estimate) / comparator_estimate
    NEGATIVE = the arm has LOWER error than its comparator = the arm is BETTER.

``endpoint`` names the estimator, and rows carrying different endpoints are never compared
in the prose without saying so. ``favorable_seeds`` always counts matched seeds on which the
arm's error is strictly lower than its comparator's.

Usage:
    python scripts/build_closure_inventory.py            # write + check
    python scripts/build_closure_inventory.py --check    # check only, write nothing

Outputs:
    results/closure/v1_results_inventory.csv     the authoritative table
    results/closure/v1_results_inventory.md      human-readable rendering
    results/closure/v1_regime_map.csv            one row per benchmark
    report/tables/closure_inventory.tex          headline rows, for the report
    report/tables/regime_map.tex                 the regime map, for the report
    report/tables/closure_numbers.tex            \\def macros for in-text numbers
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLUMNS = [
    "system", "family", "cv", "setting", "arm", "comparator",
    "n_seeds", "seeds", "endpoint", "direction",
    "arm_estimate", "comparator_estimate", "abs_diff", "rel_pct", "ci95",
    "favorable_seeds", "test", "status", "prereg", "reference",
    "artifact", "regime", "regime_basis", "section", "notes",
]

DIRECTION = "lower is better; rel_pct<0 = arm better than comparator"


# --------------------------------------------------------------------------- helpers
def rd_json(rel):
    with open(os.path.join(ROOT, rel)) as fh:
        return json.load(fh)


def rd_csv(rel):
    with open(os.path.join(ROOT, rel), newline="") as fh:
        return list(csv.DictReader(fh))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def row(**kw):
    r = {c: "" for c in COLUMNS}
    r["direction"] = DIRECTION
    r.update(kw)
    unknown = set(kw) - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown inventory column(s): {sorted(unknown)}")
    return r


def derive(r, arm_est, comp_est):
    """Fill arm/comparator estimates and the absolute difference consistently.

    abs_diff is the difference of the *published* (rounded) estimates, so the table is
    self-consistent to the digits it shows rather than to digits it does not.
    """
    a, c = float(f"{arm_est:.6g}"), float(f"{comp_est:.6g}")
    r["arm_estimate"] = f"{a:.6g}"
    r["comparator_estimate"] = f"{c:.6g}"
    r["abs_diff"] = f"{a - c:.6g}"
    return r


def seedspan(seeds):
    seeds = sorted(int(s) for s in seeds)
    if not seeds:
        return ""
    contiguous = seeds == list(range(seeds[0], seeds[-1] + 1))
    return f"{seeds[0]}-{seeds[-1]}" if contiguous and len(seeds) > 2 else \
        ",".join(str(s) for s in seeds)


# --------------------------------------------------------------------------- loaders
def gateway_rows():
    """Gateway: constructed establishment-limited regime. v1 and v2 reported separately."""
    out = []
    frozen = rd_json("results/gateway_phase/production/phase_classification.frozen.json")
    counts = frozen["regime_counts"]
    anchor = frozen["anchor"]
    setting = (f"beta={anchor['beta']:g}, s={anchor['s']:g}, r={anchor['r']:g}, "
               f"beta*H=8 kT")
    for tag, label, status in [
        ("confirmatory", "gateway confirmatory v1", "confirmatory"),
        ("confirmatory_v2", "gateway confirmatory v2 (replicate; quoted)", "confirmatory"),
    ]:
        d = rd_json(f"results/gateway_anchor/{tag}/confirmatory_summary.json")
        pre, prim = d["preregistration"], d["primary"]
        seeds = seedspan(range(pre["seeds"]["first"],
                               pre["seeds"]["first"] + pre["seeds"]["count"]))
        base = dict(system=label, family="constructed model (entropic gateway)",
                    cv="xi = x (longitudinal channel coordinate)", setting=setting,
                    n_seeds=prim["n_seeds"], seeds=seeds,
                    endpoint="int_l2_f (time-integrated L2 free-energy error)",
                    test="paired per seed; bootstrap median, 10000 resamples, seed 20260803",
                    status=status,
                    prereg="results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.{md,json}; "
                           "frozen at cdb8209 (+Amendments 1-2 at 706d4c1, 7c7cde4)",
                    reference="analytic free energy (no reference error)",
                    artifact=f"results/gateway_anchor/{tag}/confirmatory_summary.json",
                    regime="establishment-limited",
                    regime_basis=(
                        "frozen ABF-only classifier (phase_classification.frozen.json, "
                        "commit 61a8c1d), run before any FR arm existed. At the anchor: "
                        f"T_hit/T_run={anchor['T_hit_frac']:.3f} (found early), "
                        f"T_est/T_run={anchor['T_est_frac']:.3f} (populated late), "
                        f"gap={anchor['est_gap_frac']:.3f}, "
                        f"below_half_frac={anchor['below_half_frac']:.3f}, "
                        f"0/{anchor['n_seeds']} seeds with late discovery"),
                    section="sec:gateway")
        out.append(derive(row(**base, arm="practical mFR (fr_estimated)", comparator="ABF",
                              rel_pct=f"{prim['int_l2_f_pct']:.3f}",
                              ci95=f"[{prim['int_l2_f_ci95'][0]:.2f}, "
                                   f"{prim['int_l2_f_ci95'][1]:.2f}]",
                              favorable_seeds=f"{prim['int_l2_f_wins']}/{prim['n_seeds']}",
                              notes=("primary criterion "
                                     f"{'PASS' if d['primary_pass'] else 'FAIL'}; "
                                     f"init=left; baseline noise-matched="
                                     f"{d.get('baseline_noise_matched')}")),
                          prim["int_l2_f_arm"], prim["int_l2_f_abf"]))
        ds = next(x for x in d["direct_vs_sham"]
                  if x["arm"] == "fr_estimated" and x["init"] == "left")
        tost = d["sham_tost"]["sham_practical"]
        out.append(row(**base, arm="practical mFR (fr_estimated)",
                       comparator="its own matched-turnover sham (sham_practical)",
                       rel_pct=f"{ds['pct']:.3f}",
                       ci95=f"[{ds['ci95'][0]:.2f}, {ds['ci95'][1]:.2f}]",
                       favorable_seeds=f"{ds['wins']}/{ds['n_seeds']}",
                       notes=("attribution statistic: same event schedule and count, only "
                              "direction differs. POST HOC for v1 (added during v1 analysis); "
                              "preregistered for the WCA sham")))
        shamrow = next(x for x in d["rows"]
                       if x["arm"] == "sham_practical" and x["init"] == "left")
        out.append(derive(row(**{**base, "test": "TOST, alpha=0.05, 90% CI inside +/-5%"},
                              arm="matched-turnover sham (sham_practical)",
                              comparator="ABF",
                              rel_pct=f"{shamrow['int_l2_f_pct']:.3f}",
                              ci95=f"[{shamrow['int_l2_f_ci95'][0]:.2f}, "
                                   f"{shamrow['int_l2_f_ci95'][1]:.2f}]",
                              favorable_seeds=f"{shamrow['int_l2_f_wins']}/"
                                              f"{shamrow['n_seeds']}",
                              notes=(f"TOST 90% CI [{tost['ci90'][0]:.2f}, "
                                     f"{tost['ci90'][1]:.2f}] -> "
                                     f"{'EQUIVALENT' if tost['equivalent'] else 'NOT shown equivalent'}")),
                          shamrow["int_l2_f_arm"], shamrow["int_l2_f_abf"]))
        out.append(row(**{**base,
                          "endpoint": "frozen_l2_f_kT (F = B - kT log p_B, fresh population, "
                                      "no adaptation, no birth-death)"},
                       arm="practical mFR (fr_estimated), frozen-bias endpoint",
                       comparator="ABF, frozen-bias endpoint",
                       rel_pct=f"{prim['frozen_l2_f_kT_pct']:.3f}",
                       ci95=f"[{prim['frozen_l2_f_kT_ci95'][0]:.2f}, "
                            f"{prim['frozen_l2_f_kT_ci95'][1]:.2f}]",
                       favorable_seeds=f"{prim['frozen_l2_f_kT_wins']}/{prim['n_seeds']}",
                       notes="estimator-independent endpoint; discards the online accumulators"))
    out.append(row(system="gateway establishment map (ABF only)",
                   family="constructed model (entropic gateway)",
                   cv="xi = x", setting="128 cells: beta x s x r = 4 x 4 x 4, 2 inits",
                   arm="ABF only", comparator="n/a (classification screen)",
                   n_seeds=16, seeds="0-15",
                   endpoint="regime label from T_hit_frac / est_gap_frac / below_half_frac",
                   test="frozen threshold rule", status="preregistered screen",
                   prereg="results/gateway_phase/production/PREREGISTRATION.md; frozen at 61a8c1d",
                   reference="analytic free energy",
                   artifact="results/gateway_phase/production/phase_classification.frozen.json",
                   regime="map: %d ABF-sufficient / %d intermediate / %d establishment-limited "
                          "/ %d discovery-limited"
                          % (counts.get("ABF-sufficient", 0), counts.get("intermediate", 0),
                             counts.get("establishment-limited", 0),
                             counts.get("discovery-limited", 0)),
                   regime_basis="the frozen classifier itself",
                   section="sec:gateway",
                   notes="classification and anchor frozen before any FR arm ran; the beta "
                         "axis is a TIME-BUDGET axis (beta*H fixed), not a landscape axis"))
    return out, counts, anchor


def wca_sham_rows():
    d = rd_json("results/wca_sham/sham/sham_summary.json")
    seeds = seedspan(d["seeds"])
    base = dict(system="WCA dimer, matched sham (Case IX)",
                family="many-body condensed phase",
                cv="xi = dimer bond coordinate",
                setting="cell b1_h2: beta=1, h=2, w=2, M=100, a=1.5; N=1024; 120000 steps; "
                        "FR rate 0.10",
                n_seeds=len(d["seeds"]), seeds=seeds,
                endpoint="integrated_l2_f (time-integrated L2 free-energy error)",
                test="paired per seed; bootstrap median, 10000 resamples, seed 20260803",
                status="confirmatory",
                prereg="results/wca_sham/PREREGISTRATION.md; frozen at 9e7496d, run at fc29d5e",
                reference="cached thermodynamic-integration reference (cache/phase)",
                artifact="results/wca_sham/sham/sham_summary.json",
                regime="establishment-limited",
                regime_basis="state is reached by every arm (round trips equal within ~1.3%); "
                             "the deficit is post-discovery representation",
                section="sec:wca_sham")
    out = []
    for arm, label in [("fr_estimated", "practical mFR (fr_estimated)"),
                       ("fr_oracle", "oracle mFR (fr_oracle)"),
                       ("sham_practical", "matched-turnover sham (sham_practical)"),
                       ("sham_oracle", "matched-turnover sham (sham_oracle)")]:
        r = next(x for x in d["vs_abf"] if x["arm"] == arm)
        note = ""
        if arm in d["tost"]:
            t = d["tost"][arm]
            note = (f"TOST 90% CI [{t['ci90'][0]:.2f}, {t['ci90'][1]:.2f}] -> "
                    f"{'EQUIVALENT' if t['equivalent'] else 'NOT shown equivalent'} "
                    f"(fails on the HARMFUL side)")
        out.append(derive(row(**base, arm=label, comparator="ABF",
                              rel_pct=f"{r['pct']:.3f}",
                              ci95=f"[{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]",
                              favorable_seeds=f"{r['wins']}/{r['n_seeds']}",
                              notes=note),
                          r["median_metric"], r["abf_metric"]))
    for arm, label in [("fr_estimated", "practical mFR (fr_estimated)"),
                       ("fr_oracle", "oracle mFR (fr_oracle)")]:
        dr = d["direct"][arm]
        out.append(row(**base, arm=label,
                       comparator=f"its own matched-turnover sham ({dr['vs']})",
                       rel_pct=f"{dr['pct']:.3f}",
                       ci95=f"[{dr['ci95'][0]:.2f}, {dr['ci95'][1]:.2f}]",
                       favorable_seeds=f"{dr['wins']}/{dr['n_seeds']}",
                       notes="PREREGISTERED attribution statistic: identical event schedule "
                             "and count, only the selection direction differs"))
    rt = {x["arm"]: x["round_trips"] for x in d["vs_abf"]}
    rt["abf"] = d["abf_round_trips"]
    est = next(x for x in d["vs_abf"] if x["arm"] == "fr_estimated")
    out.append(row(**{**base,
                      "endpoint": "n_round_trips (barrier round trips per run)",
                      "test": "paired per seed; bootstrap median, 10000 resamples, "
                              "independent stream (seed 20260804)"},
                   arm="practical mFR (fr_estimated)", comparator="ABF",
                   arm_estimate=f"{rt['fr_estimated']:.0f}",
                   comparator_estimate=f"{rt['abf']:.0f}",
                   abs_diff=f"{rt['fr_estimated'] - rt['abf']:.0f}",
                   rel_pct=f"{est['round_trips_paired_pct']:.3f}",
                   ci95=f"[{est['round_trips_paired_ci95'][0]:.2f}, "
                        f"{est['round_trips_paired_ci95'][1]:.2f}]",
                   notes=f"transport diagnostic, NOT an accuracy endpoint, so "
                         f"favorable_seeds is left empty (more crossings is not 'better'): "
                         f"mFR has MORE crossings than ABF on "
                         f"{est['round_trips_wins']}/{est['n_seeds']} seeds. A ~1% change in "
                         f"crossings alongside a >20% accuracy change is what the "
                         f"establishment reading rests on"))
    return out, d


def wca_earlier_rows():
    n = rd_json("report/tables/report_numbers.json")
    w = n["wca"]
    out = [derive(row(system="WCA dimer, tuned production (Case III)",
                      family="many-body condensed phase",
                      cv="xi = dimer bond coordinate",
                      setting="tuned production cell; N and budget per configs/wca_production.yaml",
                      arm="practical mFR (fr_estimated, tuned)", comparator="ABF",
                      n_seeds=w["tuned_nseeds"], seeds="0-7,42,123",
                      endpoint="final l2_f (L2 free-energy error at the final budget)",
                      rel_pct=f"{-w['tuned_gain_pct']:.3f}",
                      favorable_seeds=f"{w['tuned_win']}/{w['tuned_nseeds']}",
                      test="matched-seed win rate; median gain over seeds",
                      status="exploratory (hyperparameters selected on this study)",
                      prereg="none (pre-dates the preregistration practice)",
                      reference="thermodynamic-integration reference",
                      artifact="results/wca_production/summaries/winrates.csv",
                      regime="establishment-limited",
                      regime_basis="same system and mechanism as the preregistered sham test "
                                   "(Case IX), which supplies the controlled evidence",
                      section="sec:case_wca",
                      notes="hyperparameters were tuned here, so this is the exploratory "
                            "positive that Case IX later re-tested on fresh seeds with "
                            "nothing retuned"),
                  w["tuned_l2f"], w["abf_l2f"])]
    p = [r for r in rd_csv("results/wca_phase_diagram/production/summaries/phase_main_table.csv")]
    for r in p:
        out.append(derive(row(
            system=f"WCA phase diagram cell {r['physics_tag']}",
            family="many-body condensed phase",
            cv="xi = dimer bond coordinate",
            setting=f"beta={r['beta']}, h={r['h']}, M={r['M']}, beta*h={r['beta_h']}",
            arm="practical mFR (fr_estimated)", comparator="ABF",
            n_seeds=int(f(r["n_seeds"])), seeds="0-3",
            endpoint="final l2_f",
            rel_pct=f"{-f(r['median_gain_pct']):.3f}",
            favorable_seeds=f"{int(f(r['n_wins']))}/{int(f(r['n_seeds']))}",
            test="matched-seed median gain",
            status="exploratory (difficulty sweep)",
            prereg="none; starvation thresholds are CLI defaults of analyze_wca_starvation.py",
            reference="cached thermodynamic-integration reference (cache/phase)",
            artifact="results/wca_phase_diagram/production/summaries/phase_main_table.csv",
            regime="establishment-limited where starved",
            regime_basis="matched-seed gain tracks MEASURED ABF baseline error "
                         "(Spearman rho ~ +0.80), not nominal beta*h (rho ~ -0.57)",
            section="sec:wca_phase",
            notes="one of 14 cells; reported in full, favourable and unfavourable alike"),
            f(r["fr_est_l2_f"]), f(r["abf_l2_f"])))
    return out


def toy_rows():
    n = rd_json("report/tables/report_numbers.json")
    imp, eb, meta = n["improvements"], n["entropic_bottleneck"], n["meta"]
    out = [row(system="2-D metastability model (Case I)", family="model potential",
               cv="xi = x", setting=f"beta={meta['beta']:g}, N={meta['n_particles']}, "
                                    f"{meta['n_steps']} steps",
               arm="practical mFR (estimated target)", comparator="ABF",
               n_seeds=meta["n_seeds"], seeds=seedspan(meta["seeds"]),
               endpoint="integrated L2(F) (time-integrated free-energy error)",
               rel_pct=f"{-imp['est_int_integratedF_pct']:.3f}",
               test="median over seeds of the per-config selection metric",
               status="exploratory (hyperparameters selected on this study)",
               prereg="none", reference="quadrature-exact reference",
               artifact="report/tables/report_numbers.json (from "
                        "results/two_dim_xi_x/production_gpu/)",
               regime="ABF-sufficient",
               regime_basis="ABF converges on the biased coordinate within the budget; the "
                            "residual headroom is target DIRECTION, not support",
               section="sec:case_meta",
               notes="modest accelerator; the oracle target is best here, which is the "
                     "signature of a target-quality rather than a support bottleneck"),
           row(system="2-D metastability model (Case I)", family="model potential",
               cv="xi = x", setting=f"beta={meta['beta']:g}",
               arm="oracle mFR (diagnostic control)", comparator="ABF",
               n_seeds=meta["n_seeds"], seeds=seedspan(meta["seeds"]),
               endpoint="integrated L2(F)",
               rel_pct=f"{-imp['oracle_integratedF_pct']:.3f}",
               test="median over seeds", status="diagnostic control",
               prereg="none", reference="quadrature-exact reference",
               artifact="report/tables/report_numbers.json",
               regime="ABF-sufficient", regime_basis="see the practical row",
               section="sec:case_meta",
               notes="NOT a deployable method: the oracle target is built from the reference "
                     "free energy, i.e. the unknown being estimated")]
    for beta, key, win, note in [
        (4, "warm_gain_beta4_pct", "", "mFR is mildly HARMFUL here: birth-death injects "
                                       "resampling noise into an estimator that is not starved"),
        (8, "gain_strong_pct", f"{eb['cold_win']}/{eb['cold_nseeds']}",
         "large gain; the uniform target nearly matches it and the ORACLE target is worse "
         "than ABF, so this is density correction, not free-energy-shape steering"),
    ]:
        r = row(system=f"entropic bottleneck, beta={beta} (Case II)", family="model potential",
                cv="xi = x", setting=f"beta={beta}, narrow transverse channel",
                arm="practical mFR (estimated target)", comparator="ABF",
                n_seeds=eb["cold_nseeds"] if beta == 8 else "",
                seeds="" , endpoint="final l2_f",
                rel_pct=f"{-eb[key]:.3f}", favorable_seeds=win,
                test="matched-seed median gain / win rate",
                status="exploratory (temperature sweep)", prereg="none",
                reference="analytic reference free energy and analytic conditional law",
                artifact="results/entropic_bottleneck/summaries/config_summary.csv",
                regime="ABF-sufficient" if beta == 4 else "establishment-limited",
                regime_basis="the transverse channel is smooth, so every walker reaches every "
                             "x and there is no discovery barrier; what changes with beta is "
                             "how fast the channel region is POPULATED. Inferred from the "
                             "study's evidence, not produced by the frozen gateway classifier",
                section="sec:case_eb", notes=note)
        if beta == 8:
            derive(r, eb["fr_l2f"], eb["abf_l2f"])
        out.append(r)
    out.append(row(system="entropic bottleneck, oracle target, beta=8", family="model potential",
                   cv="xi = x", setting="beta=8", arm="oracle mFR (diagnostic control)",
                   comparator="ABF", endpoint="final l2_f",
                   rel_pct=f"{-eb['oracle_gain_pct']:.3f}",
                   test="matched-seed median gain", status="diagnostic control", prereg="none",
                   reference="analytic reference", n_seeds=eb["cold_nseeds"], seeds="",
                   artifact="results/entropic_bottleneck/summaries/config_summary.csv",
                   regime="establishment-limited", regime_basis="see the practical row",
                   section="sec:case_eb",
                   notes="the oracle is WORSE than ABF here: an exact free-energy-shaped "
                         "target fights the transient ABF bias. The oracle is a diagnostic "
                         "of regime, not a performance upper bound"))
    return out


def alkane_rows():
    out = []
    eq = rd_csv("results/alkanes/production/summaries/alkanes_equivalence.csv")
    pr = rd_csv("results/alkanes/production/summaries/alkanes_paired.csv")
    paired = {(r["cell"], r["method"], r["metric"]): r for r in pr}
    for r in eq:
        if r["metric"] != "final_l2_F" or r["method"] not in ("fr_estimated", "fr_active"):
            continue
        mol = "butane" if r["cell"].startswith("butane") else "pentane"
        p = paired.get((r["cell"], r["method"], r["metric"]), {})
        n_pairs = int(f(p.get("n_pairs", "nan"))) if p else ""
        wins = round(f(r["win_rate"]) * n_pairs) if n_pairs else ""
        arm = ("practical mFR (fr_estimated, rate 0.02)" if r["method"] == "fr_estimated"
               else "aggressive mFR (fr_active, rate 0.20; mechanism probe)")
        rw = row(system=f"{mol} torsion, cell {r['cell']}",
                 family="united-atom alkane",
                 cv="xi = phi1 (signed dihedral)" if mol == "butane"
                    else "xi = phi1 (signed dihedral; phi2 hidden)",
                 setting=r["cell"], arm=arm, comparator="ABF",
                 n_seeds=n_pairs, seeds="1-16" if "b1" in r["cell"] else "1-12",
                 endpoint="final_l2_F",
                 rel_pct=f"{100.0 * f(r['rel_med']):.3f}",
                 ci95=f"[{100.0 * f(r['rel_lo']):.2f}, {100.0 * f(r['rel_hi']):.2f}]",
                 favorable_seeds=f"{wins}/{n_pairs}" if n_pairs else "",
                 test=f"matched-seed bootstrap; TOST margin +/-{100 * f(r['margin']):.0f}%",
                 status="confirmatory within a pre-declared equivalence margin",
                 prereg="equivalence margin fixed in code before the production stage "
                        "(analyze_alkanes.py --margin, default 0.10)",
                 reference=("exact analytic F(phi1)=V4(phi1)+C" if mol == "butane"
                            else "cached quadrature reference (cache/alkanes)"),
                 artifact="results/alkanes/production/summaries/alkanes_equivalence.csv",
                 regime="ABF-sufficient",
                 regime_basis=("the single dihedral mixes freely under the bias; no support "
                               "deficit forms on the biased coordinate" if mol == "butane"
                               else "phi1 is easy to flatten; the residual difficulty is the "
                                    "HIDDEN conditional p(phi2|phi1), a coordinate mFR does "
                                    "not bias"),
                 section="sec:case_butane" if mol == "butane" else "sec:case_pentane",
                 notes=f"verdict: {r['verdict']}")
        if p:
            derive(rw, f(p.get("method_med", "nan")), f(p.get("abf_med", "nan")))
        out.append(rw)
    for stage, label, cv, sect, regime, basis, note in [
        ("r15_methods", "pentane end-to-end distance R15",
         "xi = R15 = |q5 - q1| (curved distance CV)", "sec:cv_harder_1d",
         "discovery-limited",
         "ABF-only screen labels this the study's FIRST genuinely starved biased coordinate "
         "(norm L2 0.144, ~22% of thermal bins under-supported); the deficit is a rare "
         "torsional-barrier CROSSING, not a support imbalance reallocation can fill",
         "mFR is EQUIVALENT at the deployable rate and sharply HARMFUL when pushed: "
         "fr_active repairs nominal support (0.223 -> 0.082 under-supported) yet makes the "
         "free energy ~33% worse, with ancestor ESS collapsing and round trips FALLING. "
         "Amplification is not discovery"),
        ("2d_methods", "pentane torsion torus",
         "xi = (phi1, phi2) in T^2 (both torsions biased)", "sec:cv_does_it_help",
         "ABF-sufficient",
         "promoting the hidden torsion into the CV removes the starvation: the ABF-only "
         "screen labels every production cell 'easy'",
         "biasing both dihedrals resolves both barriers directly, leaving no marginal "
         "imbalance for reallocation to correct"),
    ]:
        pr2 = rd_csv(f"results/alkanes_cv_extension/{stage}/summaries/cv_paired.csv")
        for r in pr2:
            if r["metric"] != "final_l2_F" or not r["cell"].startswith("production"):
                continue
            if r["method"] not in ("fr_estimated", "fr_oracle", "fr_active"):
                continue
            n_pairs = int(f(r["n_pairs"]))
            arm = {"fr_estimated": "practical mFR (fr_estimated)",
                   "fr_oracle": "oracle mFR (fr_oracle; diagnostic control)",
                   "fr_active": "aggressive mFR (fr_active; mechanism probe)"}[r["method"]]
            out.append(derive(row(
                system=f"{label}, cell {r['cell']}", family="united-atom alkane",
                cv=cv, setting=r["cell"], arm=arm, comparator="ABF",
                n_seeds=n_pairs, seeds=seedspan(range(1, n_pairs + 1)),
                endpoint="final_l2_F",
                rel_pct=f"{100.0 * f(r['rel_med']):.3f}",
                ci95=f"[{100.0 * f(r['rel_lo']):.2f}, {100.0 * f(r['rel_hi']):.2f}]",
                favorable_seeds=f"{round(f(r['win_rate']) * n_pairs)}/{n_pairs}",
                test="matched-seed bootstrap; success rule in cv_success.csv",
                status="confirmatory within a pre-declared success rule",
                prereg="success rule fixed in analyze_alkanes_cv_extension.py before the "
                       "methods stage",
                reference="cached importance-sampling reference with an independent "
                          "uniform-proposal cross-check (cache/alkanes_cv)",
                artifact=f"results/alkanes_cv_extension/{stage}/summaries/cv_paired.csv",
                regime=regime, regime_basis=basis, section=sect, notes=note),
                f(r["method_med"]), f(r["abf_med"])))
    return out


def dipeptide_rows():
    out = []
    ala_seeds = {"N2048": "0-3", "N4096": "10-13", "N2048_refeq": "20-23"}
    ala_note = {"N2048": "pilot, N=2048 walkers, C7eq initialisation",
                "N4096": "pilot, N=4096 walkers, C7eq initialisation (headline pilot stage)",
                "N2048_refeq": "crossed control: reference-equilibrium initialisation"}
    for stage in ("N2048", "N4096", "N2048_refeq"):
        p = f"results/alanine_oracle/pilot/analysis/pilot_decision_{stage}.json"
        if not os.path.exists(os.path.join(ROOT, p)):
            continue
        d = rd_json(p)
        pr = d["primary_kernel_matched_integrated_FES"]["equilibrium"]
        out.append(row(
            system=f"alanine dipeptide, pilot {stage} (Case VI)",
            family="all-atom peptide (vacuum)",
            cv="xi = (phi, psi) (Ramachandran)",
            setting="Ace-Ala-Nme, ff14SB, vacuum, 300 K, 100 ps run, 20-100 ps window",
            arm="ORACLE mFR (fr_oracle)", comparator="ABF",
            n_seeds=int(pr["n"]), seeds=ala_seeds[stage],
            endpoint="int_eF_km_equilibrium (kernel-matched integrated FES error, "
                     "equilibrium weighting)",
            rel_pct=f"{100.0 * pr['median']:.4f}",
            ci95=f"[{100.0 * pr['lo']:.4f}, {100.0 * pr['hi']:.4f}]",
            favorable_seeds=f"{round(pr['win_rate'] * pr['n'])}/{int(pr['n'])}",
            test="paired BCa bootstrap; pre-declared threshold -10% to call an improvement",
            status="confirmatory against a pre-declared threshold",
            prereg="ALANINE_ORACLE_PILOT_HANDOFF.md sec.0; thresholds fixed before the run",
            reference="results/alanine/reference/reference.npz (24x24 periodic umbrella + "
                      "MBAR, 16 copies/window; dG = 3.419 +/- 0.079 kT, systematic floor "
                      "0.25 kT)",
            artifact=p,
            regime="ABF-sufficient",
            regime_basis="ABF reaches C7ax in 3.1-4.1 ps and holds ~5.6% occupancy, so the "
                         "state is both found and populated; the reference shows psi carries "
                         "at most a 0.75 kT internal barrier, so there is no hidden slow "
                         "coordinate for reallocation to repair",
            section="sec:neutrality",
            notes=f"{ala_note[stage]}; classification {d['classification']}. Only the ORACLE "
                  f"target was ever run for alanine -- the ideal-information control also "
                  f"changes nothing, which is stronger than a practical-arm null"))
    for e in rd_json("results/alanine_oracle/rate_ladder/analysis/ladder_summary.json"):
        out.append(row(
            system=f"alanine dipeptide, FR-rate ladder rate={e['fr_rate']:g} (Case VI)",
            family="all-atom peptide (vacuum)",
            cv="xi = (phi, psi)",
            setting=f"Ace-Ala-Nme, ff14SB, vacuum, 300 K, N=4096, FR rate {e['fr_rate']:g}",
            arm="ORACLE mFR (fr_oracle)", comparator="ABF",
            n_seeds=4, seeds="10-13",
            endpoint="int_eF_km_equilibrium",
            rel_pct=f"{100.0 * e['median']:.4f}",
            ci95=f"[{100.0 * e['lo']:.4f}, {100.0 * e['hi']:.4f}]",
            favorable_seeds=f"{e['wins']}/4",
            test="paired BCa bootstrap",
            status="post-pilot rate sensitivity (declared after the pilot concluded; "
                   "removes the 'rate too gentle' objection, NOT an independent replication)",
            prereg="results/alanine_oracle/rate_ladder/PREREGISTRATION.json "
                   "(declared_at_commit c065c2d)",
            reference="results/alanine/reference/reference.npz",
            artifact="results/alanine_oracle/rate_ladder/analysis/ladder_summary.json",
            regime="ABF-sufficient", regime_basis="see the pilot rows",
            section="sec:neutrality",
            notes=f"{e['events']} birth-death events ({100 * e['ev_per_opp']:.3f}% per "
                  f"opportunity), age-aware ancestor ESS/N {e['ess_age_min']:.3f}; "
                  f"classification {e['classification']}. Intensity spans 22x in events and "
                  f"ESS falls 0.966 -> 0.602, so the mechanism verifiably worked harder and "
                  f"accuracy did not move"))
    v = rd_json("results/valine/v3_screen/v3_metrics_concentrated.json")
    prov = rd_json("results/valine/closure/provenance.json")
    out.append(row(
        system="valine dipeptide, gate V3 (Case VII)",
        family="all-atom peptide (vacuum)",
        cv="xi = (phi, chi1)",
        setting="Ace-Val-Nme, vacuum, N=2048, 300 ps run, 8 labelled regions",
        arm="ABF only (no mFR arm was run)", comparator="n/a (establishment screen)",
        n_seeds=16, seeds="0-15",
        endpoint="frac_below_half_target (fraction of the run a region sits below half its "
                 "bias-aware target occupancy)",
        rel_pct="", ci95="",
        test=f"frozen gate: fires only if frac_below_half_target >= "
             f"{v.get('thresholds', {}).get('below_half_frac', 0.20)}",
        status="preregistered screen; the study STOPPED here",
        prereg="VALINE_SCREEN_SPEC.md; gate frozen at 7a20e9c, outcome at 00797cb",
        reference="results/valine/pilot_reference/pilot_reference.npz (18x18 windows over "
                  "(phi,chi1); meta.json explicitly flags IS_NOT_PUBLICATION_QUALITY -- it "
                  "supplies target populations for the establishment metric only)",
        artifact="results/valine/closure/valine_results_table.csv",
        regime="ABF-sufficient",
        regime_basis="ABF first-touches all 8 regions by 5.4 ps, holds them from 18 ps and "
                     "ESTABLISHES all 8 by 52 ps of a 300 ps run; the rarest region "
                     "(population 0.0014) ends at 1.46x its bias-aware target. No "
                     "discovered-but-under-established state exists",
        section="sec:neutrality",
        notes=f"verdict {v.get('verdict', 'FAIL-B')}. NO mFR ARM WAS RUN -- the gate that "
              f"would have justified one never fired, so valine's classification rests on "
              f"the ABF-only establishment screen, not on a measured mFR outcome. The 11-18 "
              f"kT chi1 barrier that motivated the system is a backbone-CLAMPED conditional "
              f"barrier worth 1.1-7.4 kT with the backbone free; the slow coordinate is phi, "
              f"which is already in the CV"))
    return out, prov


# --------------------------------------------------------------------------- checks
def check(rows):
    """Consistency gates. Any failure is fatal: the inventory must not ship half-true."""
    errs, warns = [], []

    seen = {}
    for r in rows:
        k = (r["system"], r["arm"], r["comparator"], r["endpoint"])
        if k in seen:
            errs.append(f"duplicate row: {k}")
        seen[k] = r

    for r in rows:
        rp = r["rel_pct"]
        if rp == "":
            continue
        v = f(rp)
        if v != v:
            errs.append(f"nonfinite rel_pct: {r['system']} / {r['arm']}")
            continue
        if abs(v) > 200.0:
            # Not a defect. Where the ABF baseline error is already tiny a modest absolute
            # change is a huge relative one, so these rows must be read with the absolute
            # estimates in hand. Flagged so nobody quotes the percentage on its own.
            warns.append(f"|rel_pct| > 200 (small-baseline amplification; quote the absolute "
                         f"estimates alongside): {r['system']} / {r['arm']} = {v:+.1f}%  "
                         f"[arm {r['arm_estimate']} vs comparator {r['comparator_estimate']}]")
        a, c = f(r["arm_estimate"]), f(r["comparator_estimate"])
        if a == a and c == c and c != 0:
            recomputed = 100.0 * (a - c) / c
            # Estimates are medians-over-seeds while rel_pct is the median of the paired
            # per-seed ratio; those differ by construction, so this only catches a SIGN
            # disagreement, which would mean a convention was mixed up.
            if recomputed * v < 0 and min(abs(recomputed), abs(v)) > 0.5:
                errs.append(f"sign disagreement: {r['system']} / {r['arm']}: "
                            f"rel_pct={v:+.3f} but (arm-comp)/comp={recomputed:+.3f}")
        d = f(r["abs_diff"])
        if a == a and c == c and d == d and abs(d - (a - c)) > 1e-6 * max(1.0, abs(a)):
            errs.append(f"abs_diff inconsistent: {r['system']} / {r['arm']}")

    for r in rows:
        fs = r["favorable_seeds"]
        if not fs:
            continue
        try:
            w, n = (int(x) for x in fs.split("/"))
        except ValueError:
            errs.append(f"unparseable favorable_seeds {fs!r}: {r['system']}")
            continue
        if not 0 <= w <= n:
            errs.append(f"favorable_seeds out of range {fs}: {r['system']}")
        if r["n_seeds"] not in ("", None) and int(f(r["n_seeds"])) != n:
            errs.append(f"n_seeds {r['n_seeds']} != denominator {n}: {r['system']}")

    for r in rows:
        for col in ("endpoint", "artifact", "regime", "status"):
            if not r[col]:
                errs.append(f"missing {col}: {r['system']} / {r['arm']}")
        p = r["artifact"].split(" ")[0]
        if p and not os.path.exists(os.path.join(ROOT, p)):
            errs.append(f"artifact path does not exist: {p}")

    allowed = {"ABF-sufficient", "discovery-limited", "establishment-limited",
               "establishment-limited where starved"}
    for r in rows:
        if not (r["regime"] in allowed or r["regime"].startswith("map:")):
            errs.append(f"unrecognised regime {r['regime']!r}: {r['system']}")
        if not r["regime_basis"]:
            errs.append(f"regime with no stated basis: {r['system']}")

    for r in rows:
        if "pilot" in r["artifact"] and "alanine" not in r["artifact"]:
            warns.append(f"row sourced from a pilot artifact: {r['artifact']}")
        if "/smoke/" in r["artifact"] or "/tuning/" in r["artifact"]:
            errs.append(f"row sourced from a smoke/tuning artifact: {r['artifact']}")

    return errs, warns


# --------------------------------------------------------------------------- rendering
def esc(s):
    for a, b in [("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde ")]:
        s = s.replace(a, b)
    return s


# (system, arm, comparator, endpoint-prefix). The endpoint has to be part of the key:
# the WCA sham rows carry both an accuracy endpoint and a round-trip diagnostic for the same
# arm-vs-comparator pair, and picking the wrong one puts a transport number in an accuracy
# column.
HEADLINE = [
    ("gateway confirmatory v2 (replicate; quoted)", "practical mFR (fr_estimated)", "ABF",
     "int_l2_f"),
    ("gateway confirmatory v2 (replicate; quoted)", "practical mFR (fr_estimated)",
     "its own matched-turnover sham (sham_practical)", "int_l2_f"),
    ("gateway confirmatory v2 (replicate; quoted)",
     "matched-turnover sham (sham_practical)", "ABF", "int_l2_f"),
    ("gateway confirmatory v1", "practical mFR (fr_estimated)", "ABF", "int_l2_f"),
    ("gateway confirmatory v1", "practical mFR (fr_estimated)",
     "its own matched-turnover sham (sham_practical)", "int_l2_f"),
    ("gateway confirmatory v1", "matched-turnover sham (sham_practical)", "ABF", "int_l2_f"),
    ("WCA dimer, matched sham (Case IX)", "practical mFR (fr_estimated)", "ABF",
     "integrated_l2_f"),
    ("WCA dimer, matched sham (Case IX)", "practical mFR (fr_estimated)",
     "its own matched-turnover sham (sham_practical)", "integrated_l2_f"),
    ("WCA dimer, matched sham (Case IX)", "matched-turnover sham (sham_practical)", "ABF",
     "integrated_l2_f"),
]

SHORT = {
    "gateway confirmatory v2 (replicate; quoted)": "Gateway, confirmatory v2",
    "gateway confirmatory v1": "Gateway, confirmatory v1",
    "WCA dimer, matched sham (Case IX)": "WCA dimer, Case IX",
    "practical mFR (fr_estimated)": "practical mFR",
    "matched-turnover sham (sham_practical)": "matched sham",
    "its own matched-turnover sham (sham_practical)": "its own sham",
    "ABF": "ABF",
}


def key_of(r):
    return (r["system"], r["arm"], r["comparator"], r["endpoint"].split(" (")[0])


def write_inventory_tex(rows, path):
    idx = {key_of(r): r for r in rows}
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Authoritative headline comparisons for the two "
             r"establishment-limited systems, regenerated from artifacts by "
             r"\texttt{scripts/build\_closure\_inventory.py}. The endpoint is the "
             r"time-integrated $L^2$ free-energy error in every row; the statistic is the "
             r"median over matched seeds of the per-seed relative change, so "
             r"\textbf{negative is better}. Intervals are 95\,\% bootstrap intervals of that "
             r"median. The full inventory, including every neutral and adverse row, is "
             r"\texttt{results/closure/v1\_results\_inventory.csv}.}",
             r"\label{tab:closure_inventory}",
             r"\begin{tabular}{llrrl}", r"\toprule",
             r"System & Comparison & $\Delta I_F$ (\%) & 95\,\% CI & seeds \\", r"\midrule"]
    missing = [k for k in HEADLINE if k not in idx]
    if missing:
        raise KeyError(f"headline rows absent from the inventory: {missing}")
    for key in HEADLINE:
        r = idx[key]
        lines.append(f"{esc(SHORT.get(r['system'], r['system']))} & "
                     f"{esc(SHORT.get(r['arm'], r['arm']))} vs.\\ "
                     f"{esc(SHORT.get(r['comparator'], r['comparator']))} & "
                     f"${f(r['rel_pct']):+.2f}$ & {esc(r['ci95'])} & "
                     f"{esc(r['favorable_seeds'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(path, "w").write("\n".join(lines))


# The regime map is authored in PLAIN TEXT, and the LaTeX renderer adds markup on the way
# out. It used to be authored in LaTeX and stripped back for the CSV, which silently turned
# "$R_{15}$" into "R_{15" -- so the machine-readable copy disagreed with the typeset one.
# Plain text is the source of truth; TEX below is a display-only substitution table.
REGIME_MAP = [
    ("Butane", "phi1", "neither: ABF resolves the barrier",
     "found immediately; single torsion mixes under the bias",
     "no deficit forms", "equivalent within +/-10%", "ABF-sufficient"),
    ("Pentane", "phi1", "conditional sampling of the hidden phi2",
     "found immediately on the biased coordinate",
     "no marginal deficit on phi1", "equivalent within +/-10%", "ABF-sufficient"),
    ("Pentane", "R15", "rare torsional-barrier crossing (dynamical)",
     "FAILS: about 22% of thermal bins under-supported",
     "not reached, so nothing to establish",
     "equivalent when gentle; HARMFUL when pushed", "discovery-limited"),
    ("Pentane", "(phi1,phi2)", "neither: both barriers biased directly",
     "screen labels every production cell easy", "no deficit forms",
     "equivalent", "ABF-sufficient"),
    ("Alanine dipeptide", "(phi,psi)", "neither",
     "C7ax reached in 3.1-4.1 ps", "held at about 5.6% occupancy",
     "equivalent across a 22x rate ladder", "ABF-sufficient"),
    ("Valine dipeptide", "(phi,chi1)", "neither",
     "all 8 regions touched by 5.4 ps", "all 8 established by 52 ps of 300 ps",
     "not run: the gate never fired", "ABF-sufficient"),
    ("Metastability model", "x", "target direction, not support",
     "both basins visited", "marginal converges within budget",
     "modest accelerator", "ABF-sufficient"),
    ("Entropic bottleneck", "x, beta<=4", "neither",
     "smooth channel: no discovery barrier", "marginal converges within budget",
     "mildly adverse", "ABF-sufficient"),
    ("Entropic bottleneck", "x, beta=8", "population of the channel",
     "smooth channel: no discovery barrier", "channel under-populated within budget",
     "large gain", "establishment-limited"),
    ("Entropic gateway", "x", "population of the far side",
     "gateway crossed early in every cell", "far side populated slowly",
     "-12.5% vs ABF, -14.9% vs its sham", "establishment-limited"),
    ("WCA dimer", "bond coordinate", "population among crossing replicas",
     "round trips equal across arms to about 1%",
     "stretched state under-represented", "-22.8% vs ABF, -26.4% vs its sham",
     "establishment-limited"),
]

# Display-only: plain text -> LaTeX. Longest first so substrings do not clobber.
TEX = [
    ("(phi1,phi2)", "$(\\varphi_1,\\varphi_2)$"),
    ("(phi,chi1)", "$(\\varphi,\\chi_1)$"),
    ("(phi,psi)", "$(\\varphi,\\psi)$"),
    ("x, beta<=4", "$x$, $\\beta\\!\\leq\\!4$"),
    ("x, beta=8", "$x$, $\\beta\\!=\\!8$"),
    ("bond coordinate", "bond coordinate"),
    ("phi1", "$\\varphi_1$"),
    ("phi2", "$\\varphi_2$"),
    ("R15", "$R_{15}$"),
    ("FAILS", "\\textbf{fails}"),
    ("HARMFUL", "\\textbf{harmful}"),
    ("not run:", "\\emph{not run}:"),
    ("+/-10%", "$\\pm10\\%$"),
    ("22x", "22$\\times$"),
    ("-12.5%", "$-12.5\\%$"),
    ("-14.9%", "$-14.9\\%$"),
    ("-22.8%", "$-22.8\\%$"),
    ("-26.4%", "$-26.4\\%$"),
    ("22%", "22\\%"),
    ("5.6%", "5.6\\%"),
    ("1%", "1\\%"),
    (" vs ", " vs.\\ "),
    ("^x$", "$x$"),
]


def totex(s):
    if s == "x":
        return "$x$"
    for a, b in TEX:
        if a.startswith("^"):
            continue
        s = s.replace(a, b)
    return s


def write_regime_tex(path):
    lines = [r"\begin{table}[p]", r"\centering", r"\footnotesize",
             r"\caption{Regime map over every completed benchmark. The regime is decided by "
             r"two timescales relative to the run --- how long a relevant state takes to be "
             r"\emph{found}, and how long it then takes to be \emph{populated} --- and not by "
             r"whether mFR happened to help, which would be circular. Discovery and "
             r"establishment evidence are therefore stated independently of the mFR column. "
             r"Generated by \texttt{scripts/build\_closure\_inventory.py}.}",
             r"\label{tab:regime_map}",
             # Narrow p-columns must be ragged-right: justifying a 2 cm column stretches interword
             # space to the point that TeX emits an underfull box for nearly every line, and
             # the result reads worse than a ragged edge does.
             r"\newcolumntype{R}[1]{>{\raggedright\arraybackslash}p{#1}}",
             r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{R{1.9cm}R{1.3cm}R{2.2cm}R{2.5cm}R{2.45cm}R{2.2cm}R{2.15cm}}",
             r"\toprule",
             r"System & CV & Dominant finite-time limitation & Discovery evidence & "
             r"Establishment evidence & mFR result & Regime \\", r"\midrule"]
    prev = None
    for s, cv, lim, disc, est, res, reg in REGIME_MAP:
        if prev is not None and reg != prev:
            lines.append(r"\midrule")
        prev = reg
        lines.append(" & ".join(totex(x) for x in (s, cv, lim, disc, est, res, reg))
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(path, "w").write("\n".join(lines))


SYNTH = [
    ("Metastability", "2-D metastability model (Case I)",
     "integrated L2(F)", "oracle target is best: the residual bottleneck is target direction"),
    ("Entropic bottleneck, $\\beta{=}4$", "entropic bottleneck, beta=4 (Case II)",
     "final l2_f", "n/a --- ABF is not starved, and reallocation adds resampling noise"),
    ("Entropic bottleneck, $\\beta{=}8$", "entropic bottleneck, beta=8 (Case II)",
     "final l2_f", "oracle is \\emph{worse} than ABF; a uniform target nearly matches"),
    ("WCA dimer (Case IX)", "WCA dimer, matched sham (Case IX)",
     "integrated_l2_f", "oracle $\\approx$ estimated; both beat their own matched sham"),
]


def write_synthesis_tex(rows, path):
    """Regenerate the cross-case table on the inventory's sign convention.

    The hand-maintained version of this table published *gains* (positive = better) while
    every other closure table published relative changes (negative = better), and put an
    integrated-error row and three final-error rows in one unlabelled column. Both are fixed
    here by generating it, with the endpoint named per row.
    """
    idx = {key_of(r): r for r in rows}
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Cross-case summary for the model potentials and the condensed-phase "
             r"dimer, on the closure sign convention: entries are the paired relative change "
             r"in error, so \textbf{negative is better}. The endpoint differs between rows and "
             r"is named in its own column, because an integrated error and a final error are "
             r"not interchangeable. Regenerated by "
             r"\texttt{scripts/build\_closure\_inventory.py}.}",
             r"\label{tab:synthesis}",
             r"\begin{tabular}{llrl>{\raggedright\arraybackslash}p{5.2cm}}", r"\toprule",
             r"Case & Endpoint & $\Delta$err (\%) & seeds & What the oracle target says \\",
             r"\midrule"]
    for label, system, endpoint, note in SYNTH:
        r = idx.get((system, "practical mFR (fr_estimated)", "ABF", endpoint))
        if r is None:
            r = idx.get((system, "practical mFR (estimated target)", "ABF", endpoint))
        if r is None:
            raise KeyError(f"synthesis row missing from the inventory: {system} / {endpoint}")
        lines.append(f"{label} & \\texttt{{{esc(endpoint)}}} & "
                     f"${f(r['rel_pct']):+.1f}$ & "
                     f"{esc(r['favorable_seeds']) or '--'} & {note} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(path, "w").write("\n".join(lines))


def _rows_table(rows, path, picks, caption, label, colspec, header, ncols, prec=2):
    """Shared renderer for the compact closure tables.

    ``prec`` matters: the dipeptide effects are of order a hundredth of a per cent, and
    printing them at two decimals renders them all as "+0.01"/"-0.00", which reads as
    rounding noise rather than as the measured near-exact equality it is.
    """
    idx = {key_of(r): r for r in rows}
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{" + caption + "}", r"\label{" + label + "}"]
    lines += [r"\begin{tabular}{" + colspec + "}", r"\toprule", header, r"\midrule"]
    for group, entries in picks:
        if group:
            lines.append(r"\multicolumn{" + str(ncols) + r"}{l}{\emph{" + group + r"}}\\")
        for label_txt, key, extra in entries:
            r = idx.get(key)
            if r is None:
                raise KeyError(f"closure table row missing from the inventory: {key}")
            pct = f"${f(r['rel_pct']):+.{prec}f}$" if r["rel_pct"] else "--"
            ci = esc(r["ci95"]) if r["ci95"] else "--"
            seeds = esc(r["favorable_seeds"]) if r["favorable_seeds"] else "--"
            lines.append(f"{label_txt} & {pct} & {ci} & {seeds} & {extra} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(path, "w").write("\n".join(lines))


def write_alkane_tex(rows, path):
    P = "practical mFR (fr_estimated, rate 0.02)"
    A = "aggressive mFR (fr_active, rate 0.20; mechanism probe)"
    picks = [
        ("Torsion CV: ABF already resolves the barrier", [
            (r"Butane $\varphi_1$, $\beta{=}1$",
             ("butane torsion, cell butane_b1_s2.3_full_trans", P, "ABF", "final_l2_F"),
             "ABF-sufficient"),
            (r"Pentane $\varphi_1$, $\beta{=}1$",
             ("pentane torsion, cell pentane_b1_s2.3_full_trans", P, "ABF", "final_l2_F"),
             "ABF-sufficient"),
            (r"Pentane $\varphi_1$, $\beta{=}2$",
             ("pentane torsion, cell pentane_b2_s2.3_full_trans", P, "ABF", "final_l2_F"),
             "ABF-sufficient"),
        ]),
        (r"Distance CV $R_{15}$, $\beta{=}2$: the one starved coordinate", [
            (r"practical rate",
             ("pentane end-to-end distance R15, cell production_dist_pentane_b2_trans_g256",
              "practical mFR (fr_estimated)", "ABF", "final_l2_F"), "discovery-limited"),
            (r"oracle target",
             ("pentane end-to-end distance R15, cell production_dist_pentane_b2_trans_g256",
              "oracle mFR (fr_oracle; diagnostic control)", "ABF", "final_l2_F"),
             "ideal target does not help either"),
            (r"pushed (rate 0.20)",
             ("pentane end-to-end distance R15, cell production_dist_pentane_b2_trans_g256",
              "aggressive mFR (fr_active; mechanism probe)", "ABF", "final_l2_F"),
             r"\textbf{adverse}: support repaired, lineages collapse"),
        ]),
        (r"Torsion torus $(\varphi_1,\varphi_2)$: promoting the hidden torsion", [
            (r"practical rate",
             ("pentane torsion torus, cell production_joint2d_pentane_b2_trans_g48",
              "practical mFR (fr_estimated)", "ABF", "final_l2_F"), "ABF-sufficient"),
        ]),
    ]
    _rows_table(rows, path, picks,
                r"Alkane closure. All entries are the paired relative change in "
                r"\texttt{final\_l2\_F} against ABF, so \textbf{negative is better}; the "
                r"seeds column counts matched seeds on which mFR is strictly more accurate. "
                r"The three blocks are three different limitations, not three instances of "
                r"one. Regenerated by \texttt{scripts/build\_closure\_inventory.py}.",
                "tab:alkane_closure", r"lrl l>{\raggedright\arraybackslash}p{4.6cm}",
                r"Arm & $\Delta$err (\%) & 95\,\% CI & seeds & Reading \\", ncols=5)


def write_dipeptide_tex(rows, path):
    O = "ORACLE mFR (fr_oracle)"
    picks = [
        ("Alanine dipeptide, oracle target, pilot stages", [
            (r"$N{=}2048$, C7eq",
             ("alanine dipeptide, pilot N2048 (Case VI)", O, "ABF",
              "int_eF_km_equilibrium"), "equivalent"),
            (r"$N{=}4096$, C7eq",
             ("alanine dipeptide, pilot N4096 (Case VI)", O, "ABF",
              "int_eF_km_equilibrium"), "equivalent"),
            (r"$N{=}2048$, reference-equilibrium start",
             ("alanine dipeptide, pilot N2048\\_refeq (Case VI)".replace("\\_", "_"), O,
              "ABF", "int_eF_km_equilibrium"), "equivalent (crossed control)"),
        ]),
        ("Alanine dipeptide, FR-rate ladder (post-pilot sensitivity)", [
            (r"rate $0.02$ (3\,021 events, ESS/$N$ 0.966)",
             ("alanine dipeptide, FR-rate ladder rate=0.02 (Case VI)", O, "ABF",
              "int_eF_km_equilibrium"), "equivalent"),
            (r"rate $0.15$ (22\,830 events, ESS/$N$ 0.813)",
             ("alanine dipeptide, FR-rate ladder rate=0.15 (Case VI)", O, "ABF",
              "int_eF_km_equilibrium"), "equivalent"),
            (r"rate $0.45$ (66\,818 events, ESS/$N$ 0.602)",
             ("alanine dipeptide, FR-rate ladder rate=0.45 (Case VI)", O, "ABF",
              "int_eF_km_equilibrium"), "equivalent"),
        ]),
    ]
    _rows_table(rows, path, picks,
                r"Dipeptide closure. Entries are the paired relative change in the "
                r"kernel-matched integrated FES error against ABF, so \textbf{negative is "
                r"better}; the pre-declared threshold for calling an improvement was "
                r"$-10\%$. Note the arm: only the \emph{oracle} target was ever run on "
                r"alanine, so this is the ideal-information control returning nothing, which "
                r"is a stronger null than a practical arm would have been. Valine has no mFR "
                r"row because its establishment gate never fired --- ABF reaches all 8 "
                r"regions by 5.4\,ps and establishes all 8 by 52\,ps of a 300\,ps run --- so "
                r"no arm was justified and none was run. Regenerated by "
                r"\texttt{scripts/build\_closure\_inventory.py}.",
                "tab:dipeptide_closure",
                r"lrl l>{\raggedright\arraybackslash}p{3.4cm}",
                r"Arm & $\Delta$err (\%) & 95\,\% CI & seeds & Classification \\",
                ncols=5, prec=4)


def write_macros(rows, path, counts, sham, prov):
    idx = {key_of(r): r for r in rows}

    def g(key, field):
        r = idx.get(key)
        return r[field] if r else "--"

    gw2 = ("gateway confirmatory v2 (replicate; quoted)", "practical mFR (fr_estimated)",
           "ABF", "int_l2_f")
    gw1 = ("gateway confirmatory v1", "practical mFR (fr_estimated)", "ABF", "int_l2_f")
    wca = ("WCA dimer, matched sham (Case IX)", "practical mFR (fr_estimated)", "ABF",
           "integrated_l2_f")
    wcad = ("WCA dimer, matched sham (Case IX)", "practical mFR (fr_estimated)",
            "its own matched-turnover sham (sham_practical)", "integrated_l2_f")
    wcas = ("WCA dimer, matched sham (Case IX)", "matched-turnover sham (sham_practical)",
            "ABF", "integrated_l2_f")
    rt = next(r for r in rows if r["endpoint"].startswith("n_round_trips"))
    m = {
        "ClosureRows": str(len(rows)),
        "ClosureGwOneGain": f"{f(g(gw1, 'rel_pct')):+.2f}",
        "ClosureGwTwoGain": f"{f(g(gw2, 'rel_pct')):+.2f}",
        "ClosureGwOneWins": g(gw1, "favorable_seeds"),
        "ClosureGwTwoWins": g(gw2, "favorable_seeds"),
        "ClosureGwSuffCells": str(counts.get("ABF-sufficient", 0)),
        "ClosureGwInterCells": str(counts.get("intermediate", 0)),
        "ClosureGwEstCells": str(counts.get("establishment-limited", 0)),
        "ClosureGwDiscCells": str(counts.get("discovery-limited", 0)),
        "ClosureWcaGain": f"{f(g(wca, 'rel_pct')):+.2f}",
        "ClosureWcaWins": g(wca, "favorable_seeds"),
        "ClosureWcaDirect": f"{f(g(wcad, 'rel_pct')):+.2f}",
        "ClosureWcaDirectWins": g(wcad, "favorable_seeds"),
        "ClosureWcaSham": f"{f(g(wcas, 'rel_pct')):+.2f}",
        "ClosureWcaRoundTripPct": f"{f(rt['rel_pct']):+.2f}",
        "ClosureWcaRoundTripCIlo": f"{f(rt['ci95'].strip('[]').split(',')[0]):+.2f}",
        "ClosureWcaRoundTripCIhi": f"{f(rt['ci95'].strip('[]').split(',')[1]):+.2f}",
        "ClosureWcaAbfRoundTrips": rt["comparator_estimate"],
        "ClosureWcaMfrRoundTrips": rt["arm_estimate"],
        "ClosureValRunDate": str(prov.get("generated", ""))[:10],
    }
    lines = ["% Auto-generated by scripts/build_closure_inventory.py -- do not edit.",
             "% Source: results/closure/v1_results_inventory.csv"]
    lines += [f"\\def\\{k}{{{v}}}" for k, v in m.items()]
    open(path, "w").write("\n".join(lines) + "\n")
    return m


def write_md(rows, path, errs, warns):
    L = ["# v1 authoritative results inventory", "",
         "Generated by `scripts/build_closure_inventory.py` from the aggregated artifacts "
         "listed in the `artifact` column. Do not edit by hand.", "",
         "**Sign convention (uniform across every row).** "
         "`rel_pct = 100 x (arm - comparator) / comparator` on the stated `endpoint`. "
         "**Negative means the arm has lower error, i.e. the arm is better.** Several source "
         "artifacts publish the opposite ('gain') convention or express the change as a "
         "fraction; this table normalises all of them. Rows carrying different `endpoint` "
         "values are not comparable and are never mixed in a single claim.", "",
         f"Rows: **{len(rows)}**. Consistency checks: "
         f"**{'PASS' if not errs else 'FAIL'}** ({len(errs)} errors, {len(warns)} warnings).",
         ""]
    for fam in sorted({r["family"] for r in rows}):
        L += [f"## {fam}", ""]
        L += ["| System | Arm | Comparator | Endpoint | rel % | 95% CI | seeds | Regime | Status |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in [x for x in rows if x["family"] == fam]:
            L.append(f"| {r['system']} | {r['arm']} | {r['comparator']} | "
                     f"{r['endpoint'].split(' (')[0]} | {r['rel_pct']} | {r['ci95']} | "
                     f"{r['favorable_seeds']} | {r['regime']} | {r['status'].split(' (')[0]} |")
        L.append("")
    L += ["## Notes and caveats, row by row", ""]
    for r in rows:
        if r["notes"]:
            L.append(f"- **{r['system']} / {r['arm']} vs {r['comparator']}** — {r['notes']}")
    L.append("")
    open(path, "w").write("\n".join(L))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run checks, write nothing")
    a = ap.parse_args()

    rows = []
    gw, counts, anchor = gateway_rows()
    rows += gw
    sham_rows, sham = wca_sham_rows()
    rows += sham_rows
    rows += wca_earlier_rows()
    rows += toy_rows()
    rows += alkane_rows()
    dip, prov = dipeptide_rows()
    rows += dip

    errs, warns = check(rows)
    for w in warns:
        print(f"  WARN  {w}")
    for e in errs:
        print(f"  ERROR {e}")
    print(f"\n{len(rows)} rows; {len(errs)} errors, {len(warns)} warnings")
    if errs:
        return 1
    if a.check:
        print("check-only: nothing written")
        return 0

    outdir = os.path.join(ROOT, "results/closure")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "v1_results_inventory.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    write_md(rows, os.path.join(outdir, "v1_results_inventory.md"), errs, warns)

    with open(os.path.join(outdir, "v1_regime_map.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "cv", "dominant_limitation", "discovery_evidence",
                    "establishment_evidence", "mfr_result", "regime"])
        w.writerows(REGIME_MAP)

    tdir = os.path.join(ROOT, "report/tables")
    write_inventory_tex(rows, os.path.join(tdir, "closure_inventory.tex"))
    write_synthesis_tex(rows, os.path.join(tdir, "synthesis.tex"))
    write_alkane_tex(rows, os.path.join(tdir, "alkane_closure.tex"))
    write_dipeptide_tex(rows, os.path.join(tdir, "dipeptide_closure.tex"))
    write_regime_tex(os.path.join(tdir, "regime_map.tex"))
    macros = write_macros(rows, os.path.join(tdir, "closure_numbers.tex"),
                          counts, sham, prov)

    print(f"wrote {csv_path}")
    print(f"wrote {outdir}/v1_results_inventory.md, v1_regime_map.csv")
    print(f"wrote {tdir}/closure_inventory.tex, synthesis.tex, alkane_closure.tex, "
          f"dipeptide_closure.tex, regime_map.tex, closure_numbers.tex")
    print(f"macros: {', '.join(sorted(macros))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
