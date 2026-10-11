#!/usr/bin/env python
"""Frozen outcome classification of Experiment A (docs/final_campaign/PREREGISTRATION.md section 1.7).

    python scripts/final_campaign/expA_outcome.py [--cells-dir configs/final_campaign/cells/expA]
        [--synthesis-dir results/final_campaign/expA/synthesis] [--out-dir <synthesis-dir>] [--no-verify-runs]
        [--provisional] [--force] [--recompute] [--dry-run]

Pure functions (unit-tested branch by branch, tests/test_final_expA_analysis.py):

  classify(R, p_holm, R_ci95, G_hi, guards, n=None)  -> dict(label, wording, failed_guards, ...)
      per N and per contrast family (E1 primary: lambda 16 vs 1; E2 extension: lambda 128 vs 1).
      R, p_holm, R_ci95: the contrast's ratio, Holm-adjusted p and 95 % CI (expA_cross_cell.py); G_hi: the upper end of
      the 95 % CI of G(Ibar_F) at the higher lambda; guards: {g1..g4: True (holds) / False (fails) / None (not
      evaluable)} at the higher lambda and the same N (expA_mech.py).
        A               p_Holm < 0.05, R < 1, G_hi < 0, and g1-g4 hold      "faster mobility rescues FR"
        A-qualified     p_Holm < 0.05, R < 1, G_hi < 0, a guard fails       "FR better than ABF but with a new problem: gX"
        B               p_Holm < 0.05, R < 1, G_hi >= 0                     "reduces FR's disadvantage; FR not better than ABF"
        I               not (p_Holm < 0.05 and R < 1), but G_hi < 0         "FR better than ABF at the higher lambda;
                                                                             improvement over lambda = 1 not supported"
        C-equivalent    p_Holm >= 0.05, 95 % CI of R in [0.90, 1.11], G_hi >= 0   "no rescue"
        C-reverse       p_Holm < 0.05, R > 1                                "faster mobility makes FR relatively worse"
        C-inconclusive  every other case                                    "no detectable rescue; R bounded by its CI"
                                                                            (never "no rescue")
      Strict inequalities exactly as frozen: p_Holm = 0.05 is NOT significant; G_hi = 0 is 'G_hi >= 0'; the
      equivalence band is inclusive (a CI end exactly at 0.90 or 1.11 is inside); R = 1 exactly is neither R < 1 nor
      R > 1.
      Resolutions of cases the frozen table leaves open (RESOLUTIONS; stated in each result's 'resolution' field).
      They are analysis decisions taken AFTER the freeze and BEFORE any production data were read; they must be
      recorded as a dated amendment appended to PREREGISTRATION.md carrying the marker AMENDMENT_MARKER.  Until that
      amendment exists, outcome.json is never complete (prereg_amendment(); the amendment text block enters the input
      signature, and outcome.json / outcome.md cite it).
        (r1) p_Holm < 0.05, R > 1 and G_hi < 0 satisfy BOTH the I rule and the C-reverse rule.  The contrast is
             significant, and the significant-contrast rows (A, A-qualified, B, C-reverse) decide on R; I is the
             'no significant rescue' row; and the combination rule's "opposite significant signs -> N-dependent"
             sentence treats a significant R > 1 as C-reverse.  So the label is C-reverse, with also_satisfies = ['I']
             and the I wording appended ("FR is nonetheless better than ABF at the higher lambda").  (A top-down
             reading of the table would give I.)
        (r2) A guard that cannot be evaluated (None) does not 'hold': a case that would be A becomes A-qualified
             naming 'gX (not evaluable)'.
        (r3) n < 10 paired seeds (PREREGISTRATION 1.6) -> label 'no-claim'; a non-finite R, p_Holm, CI or G_hi ->
             'not-available'.  Neither is one of the seven frozen labels; an N with either makes the combination
             "no global claim (incomplete)".
  combine_across_N({N: label}, failed={N: [guards]})  -> dict(global_label, governing_label, wording, ...)
        both N in {A, A-qualified, B}                                 -> "rescue (<weaker> governs)"; the WEAKER label
                                                                         (B < A-qualified < A) governs, and the wording is
                                                                         ONLY that label's ("faster mobility rescues FR"
                                                                         only when both N are A)
        one N in {A, A-qualified, B}, the other C-equivalent, C-inconclusive or I
                                                                      -> "rescue at N = n only"
        one N in {A, A-qualified, B}, the other C-reverse              -> "N-dependent" (no global claim)
        both N in {C-equivalent, C-reverse}                            -> "no rescue"
        any other combination                                          -> "no global claim" ("no detectable rescue" when
                                                                         neither N is a rescue; never "no rescue")
        a no-claim / not-available N                                   -> "no global claim (incomplete)"
  interpretation(...)  the frozen interpretation map (PREREGISTRATION 1.7) as typed statements ([observed] /
        [hypothesis]); every outcome is to be read with MC1, MC2 (validation tool), the ABF-only descriptors and the
        E decomposition.

CLI: reads <synthesis-dir>/cross_cell.json (R, p_Holm, CI, n, G_hi; E rows of Ibar_F; decomposition) and mech.json
(guards g1-g4 at the higher lambda; MC3).  Refused unless both are signed (not an interrupted write), were made from
the same cell configs and summaries (sha256) AND those are the CURRENT cell configs and summaries on disk, and the
summaries are current and fresh (cross_cell.check_summary / check_freshness; --no-verify-runs skips freshness and
forbids complete); these checks also run before an up-to-date output is skipped.  Classifies E1 and E2 at every N,
combines across N and writes <out-dir>/outcome.json and outcome.md (placeholder JSON first, then the markdown, then the
final JSON).  input_signature = the two inputs' sha256, the code, the completeness basis (audits, --provisional,
freshness) and the amendment block.  complete iff both inputs are complete AND the CURRENT
expA_analysis_common.completeness() holds (every cell's completeness audit PASS on the current summary / config,
frozen engine only, n_boot 10 000 in every summary, no fixture, no --provisional) AND the resolutions amendment is in
PREREGISTRATION.md.  Output policy as expA_analysis_common.decide_write.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import expA_analysis_common as C  # noqa: E402

SCHEMA = "final_expA_outcome/1"
LABELS = ("A", "A-qualified", "B", "I", "C-equivalent", "C-reverse", "C-inconclusive")
RESCUE = ("A", "A-qualified", "B")
NOT_RESCUE_OTHER = ("C-equivalent", "C-inconclusive", "I")
NO_RESCUE = ("C-equivalent", "C-reverse")
STRENGTH = {"A": 3, "A-qualified": 2, "B": 1}          # weaker label governs the combined wording
GUARDS = ("g1", "g2", "g3", "g4")
GUARD_TEXT = dict(g1="g1: the 95 % CI of G(final e_F) lies entirely above 0",
                  g2="g2: the 95 % CI of G(Ibar_F'_stat) lies entirely above 0",
                  g3="g3: genealogy collapse (ESS/N ratio upper CI < 0.8 or max-lineage ratio lower CI > 1.25)",
                  g4="g4: a non-finite state or a cap violation in a run")
WORDING = {
    "A": "faster mobility rescues FR",
    "A-qualified": "FR better than ABF but with a new problem: {guards}",
    "B": "reduces FR's disadvantage; FR not better than ABF",
    "I": "FR better than ABF at the higher lambda; improvement over lambda = 1 not supported",
    "C-equivalent": "no rescue",
    "C-reverse": "faster mobility makes FR relatively worse",
    "C-inconclusive": "no detectable rescue; R bounded by its CI",
    "no-claim": "no claim (fewer than 10 complete seed pairs)",
    "not-available": "not available (contrast or G(Ibar_F) not computable)",
}
RESOLUTIONS = dict(
    r1=("p_Holm < 0.05, R > 1 and G_hi < 0 satisfy both the I and the C-reverse rule: labelled C-reverse (the contrast "
        "is significant; the significant-contrast rows decide on R, consistent with the combination rule's 'opposite "
        "significant signs -> N-dependent'), also_satisfies I, I wording appended; a top-down reading would give I"),
    r2="a guard that cannot be evaluated does not hold: an A case becomes A-qualified naming 'gX (not evaluable)'",
    r3=("n < 10 paired seeds -> 'no-claim'; non-finite inputs -> 'not-available' (outside the seven frozen labels); "
        "either at one N makes the combination 'no global claim (incomplete)'"))
AMENDMENT_MARKER = "EXPA-OUTCOME-RESOLUTIONS-r1-r3"
AMENDMENTS_LINE = "*Amendments (dated, appended below"
CODE_FILES = [os.path.abspath(__file__), os.path.join(HERE, "expA_analysis_common.py")]


def prereg_amendment(text=None, path=None):
    """The dated amendment of PREREGISTRATION.md that records RESOLUTIONS r1-r3: the first line BELOW the frozen
    amendments line that carries AMENDMENT_MARKER, and its block (that line up to the next markdown heading or
    horizontal rule).  dict(found, line, block_sha256, path); found False when there is none."""
    path = path or C.PREREG
    if text is None:
        try:
            text = open(path).read()
        except OSError:
            text = ""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip().startswith(AMENDMENTS_LINE)), None)
    out = dict(found=False, marker=AMENDMENT_MARKER, line=None, block_sha256=None, path=C.rel(path))
    if start is None:
        out["why"] = "the frozen amendments line was not found in the preregistration"
        return out
    for i in range(start + 1, len(lines)):
        if AMENDMENT_MARKER in lines[i]:
            j = i + 1
            while j < len(lines) and not (lines[j].startswith("#") or lines[j].strip() == "---"):
                j += 1
            block = "\n".join(lines[i:j]).strip()
            out.update(found=True, line=lines[i].strip(), line_no=i + 1,
                       block_sha256=hashlib.sha256(block.encode()).hexdigest())
            return out
    out["why"] = f"no amendment carrying {AMENDMENT_MARKER!r} below the frozen amendments line"
    return out


def _finite(*v):
    try:
        return all(v_ is not None and math.isfinite(float(v_)) for v_ in v)
    except (TypeError, ValueError):
        return False


def classify(R, p_holm, R_ci95, G_hi, guards=None, n=None, alpha=C.ALPHA, equiv=C.EQUIV, small_n=C.SMALL_N):
    """The PREREGISTRATION 1.7 label of one (family, N) (module docstring, incl. resolutions r1-r3)."""
    guards = dict(guards or {})
    base = dict(inputs=dict(R=R, p_holm=p_holm, R_ci95=list(R_ci95) if R_ci95 is not None else None, G_hi=G_hi,
                            guards={g: guards.get(g) for g in GUARDS}, n=n),
                failed_guards=[], also_satisfies=[], resolution=None)
    if n is not None and int(n) == 0:
        return dict(base, label="not-available", wording=WORDING["not-available"], resolution="r3",
                    rule="no paired seed (contrast not computable)")
    if n is not None and int(n) < small_n:
        return dict(base, label="no-claim", wording=WORDING["no-claim"], resolution="r3",
                    rule=f"n = {n} < {small_n} paired seeds")
    lo, hi = (list(R_ci95) + [None, None])[:2] if R_ci95 is not None else (None, None)
    if not _finite(R, p_holm, lo, hi, G_hi):
        return dict(base, label="not-available", wording=WORDING["not-available"], resolution="r3",
                    rule="a non-finite input")
    R, p_holm, lo, hi, G_hi = float(R), float(p_holm), float(lo), float(hi), float(G_hi)
    sig = p_holm < alpha
    fr_better = G_hi < 0
    in_band = equiv[0] <= lo and hi <= equiv[1]
    if sig and R < 1:
        if not fr_better:
            return dict(base, label="B", wording=WORDING["B"], rule="p_Holm < 0.05, R < 1, G_hi >= 0")
        failed = []
        for g in GUARDS:
            v = guards.get(g)
            if v is None:
                failed.append(f"{g} (not evaluable)")
            elif not v:
                failed.append(g)
        if not failed:
            return dict(base, label="A", wording=WORDING["A"], rule="p_Holm < 0.05, R < 1, G_hi < 0, g1-g4 hold")
        res = "r2" if any("not evaluable" in f for f in failed) else None
        return dict(base, label="A-qualified", failed_guards=failed, resolution=res,
                    wording=WORDING["A-qualified"].format(guards=", ".join(failed)),
                    rule="p_Holm < 0.05, R < 1, G_hi < 0, a guard fails")
    if sig and R > 1:
        out = dict(base, label="C-reverse", wording=WORDING["C-reverse"], rule="p_Holm < 0.05, R > 1")
        if fr_better:
            out.update(also_satisfies=["I"], resolution="r1",
                       wording=WORDING["C-reverse"] + "; FR is nonetheless better than ABF at the higher lambda (G_hi < 0)")
        return out
    if fr_better:
        return dict(base, label="I", wording=WORDING["I"], rule="not (p_Holm < 0.05 and R < 1), G_hi < 0")
    if not sig and in_band:
        return dict(base, label="C-equivalent", wording=WORDING["C-equivalent"],
                    rule="p_Holm >= 0.05, 95 % CI of R inside [0.90, 1.11], G_hi >= 0")
    return dict(base, label="C-inconclusive", wording=WORDING["C-inconclusive"], rule="every other case")


def combine_across_N(labels, family_text="faster mobility", failed=None):
    """The PREREGISTRATION 1.7 combination rule over the two N; labels = {N: label}; failed = {N: [failed guards]}
    (named in an A-qualified wording).  When both N are rescues the WEAKER label governs and the combined wording is
    ONLY that label's wording (so "rescues FR" appears only when both N are A)."""
    Ns = sorted(labels, key=lambda x: -int(x))
    labs = [labels[N] for N in Ns]
    failed = {int(k): v for k, v in (failed or {}).items()}
    out = dict(per_N={str(N): labels[N] for N in Ns}, family=family_text)

    def guards_of(lab, Nsel):
        g = [f"{x} (N = {N})" for N in Nsel if labels[N] == lab for x in failed.get(int(N), [])]
        return ", ".join(g) or "see the per-N guards"
    if len(Ns) != 2:
        return dict(out, global_label="no global claim (incomplete)", wording=f"no global claim: {len(Ns)} N classified")
    bad = [N for N in Ns if labels[N] not in LABELS]
    if bad:
        good = [N for N in Ns if labels[N] in LABELS]
        txt = "; ".join(f"N = {N}: {labels[N]}" for N in Ns)
        return dict(out, global_label="no global claim (incomplete)",
                    wording=f"no global claim: N = {', '.join(str(b) for b in bad)} makes no claim ({txt})",
                    claimable_N=[str(g) for g in good])
    a, b = labs
    if a in RESCUE and b in RESCUE:
        gov = min(labs, key=lambda x: STRENGTH[x])
        return dict(out, global_label=f"rescue ({gov} governs)", governing_label=gov,
                    wording=WORDING[gov].format(guards=guards_of(gov, Ns)),
                    wording_context=(f"{family_text}: A, A-qualified or B at both N (N = {Ns[0]}: {a}; N = {Ns[1]}: "
                                     f"{b}); the weaker label {gov} governs the wording"))
    for (Nr, lr), (No, lo) in (((Ns[0], a), (Ns[1], b)), ((Ns[1], b), (Ns[0], a))):
        if lr in RESCUE and lo in NOT_RESCUE_OTHER:
            return dict(out, global_label=f"rescue at N = {Nr} only", governing_label=lr,
                        wording=f"rescue at N = {Nr} only ({lr}: {WORDING[lr].format(guards=guards_of(lr, [Nr]))}); "
                                f"N = {No}: {lo} ({WORDING[lo]})")
        if lr in RESCUE and lo == "C-reverse":
            return dict(out, global_label="N-dependent",
                        wording=f"N-dependent, no global claim: N = {Nr} {lr}, N = {No} C-reverse (opposite "
                                "significant signs)")
    if a in NO_RESCUE and b in NO_RESCUE:
        return dict(out, global_label="no rescue", wording="no rescue (both N C-equivalent or C-reverse)")
    return dict(out, global_label="no global claim",
                wording="no detectable rescue at either N; no global claim (per-N labels: "
                        + ", ".join(f"N = {N} {labels[N]}" for N in Ns) + ")")


# ============================================================================================ interpretation map
def interpretation(fam, lam_hi, per_N, mc3, abf_decomp, other_family_global=None, tv_rows=None):
    """Frozen interpretation statements (PREREGISTRATION 1.7) for one family; per_N = {N: classify() result}."""
    S = ["[identity] lambda changes gate transport (MC2) as well as conditional relaxation (MC1): any change in R is "
         "attributed to transverse mobility; an attribution to conditional relaxation alone is a [hypothesis].  Read "
         "with MC1 / MC2 (results/final_campaign/expA/validation), the ABF-only descriptors and the decomposition."]
    for N, res in per_N.items():
        lab = res["label"]
        m = ((mc3 or {}).get("per_N") or {}).get(str(N)) or {}
        mc3_pass = m.get("passes")
        d = (abf_decomp or {}).get(str(N)) or {}
        abf_ci = d.get("ratio_ci95") or [None, None]
        abf_improves = _finite(abf_ci[1]) and float(abf_ci[1]) < 1
        p = res["inputs"].get("p_holm")
        if abf_improves and _finite(p) and float(p) < C.ALPHA:
            S.append(f"[observed] N = {N}: ABF improves strongly with lambda (ABF Ibar_F ratio {lam_hi:g}/1 CI upper "
                     f"{float(abf_ci[1]):.3g} < 1) while R changes: the change may come from ABF's own improved "
                     "transport / establishment rather than from FR's clones (confound).")
        if lab in RESCUE:
            if mc3_pass:
                S.append(f"[hypothesis] N = {N}: rescue ({lab}) with MC3 passing: consistent with the clone fibre-lag "
                         "hypothesis.")
            else:
                S.append(f"[observed] N = {N}: rescue ({lab}) without MC3 passing (MC3 {mc3_pass}): the rescue is not "
                         "explained by reduced young-clone bias (transport or other).")
        elif lab.startswith("C-"):
            if mc3_pass:
                S.append(f"[observed] N = {N}: {lab} with MC3 passing: young-clone bias falls but FR's relative "
                         "performance does not improve; young-clone lag is not what limits FR here.  [hypothesis] "
                         "candidates: establishment share of ABF's error, entry-side over-population, ABF's own lag.")
            else:
                S.append(f"[observed] N = {N}: {lab} without MC3 passing (MC3 {mc3_pass}): the intervention did not "
                         "reach the mechanism at the tested lambda; read against MC1 (predicted flank speed-up 1.09x / "
                         "1.44x / 4.74x at lambda 4 / 16 / 128).")
        if tv_rows is not None:
            tv = tv_rows.get(str(N)) or {}
            ci = tv.get("ratio_ci95") or [None, None]
            if _finite(p) and float(p) < C.ALPHA and _finite(*ci) and float(ci[0]) <= 1 <= float(ci[1]):
                S.append(f"[observed] N = {N}: an F effect without a marginal (Ibar_TV_half) effect points to the force "
                         "estimate, not to establishment.")
    if other_family_global is not None:
        S.append(f"(E1 vs E2) other family: {other_family_global}")
    return S


# ============================================================================================ CLI
def _rows(cc, metric, fid):
    return {str(r["N"]): r for r in cc["contrasts"][metric]["rows"] if r["id"] == fid and r["kind"] == "family"}


def build(cc, mech):
    fams = {}
    lam_hi_of = dict(E1=16.0, E2=128.0)
    for fid in ("E1", "E2"):
        lam_hi = lam_hi_of[fid]
        lk = C.lam_key(lam_hi)
        rows = _rows(cc, "Ibar_F", fid)
        tv = _rows(cc, "Ibar_TV_half", fid)
        per_N = {}
        for N in [str(n) for n in cc["N"]]:
            r = rows.get(N)
            if r is None:
                per_N[N] = classify(None, None, None, None, n=0)
                continue
            g = (mech.get("guards") or {}).get(lk, {}).get(N) or {}
            guards = {k: (g.get(k) or {}).get("holds") for k in GUARDS}
            G_hi = (cc.get("G_hi") or {}).get(lk, {}).get(N)
            per_N[N] = classify(C.fnum(r["ratio"]), C.fnum(r.get("p_holm")), [C.fnum(x) for x in r["ratio_ci95"]],
                                C.fnum(G_hi), guards, n=r["n"])
            per_N[N]["contrast"] = dict(ratio_ci975=r.get("ratio_ci975"), p=r.get("p"), wins_a=r.get("wins_a"),
                                        equivalent=r.get("equivalent"))
        comb = combine_across_N({int(N): v["label"] for N, v in per_N.items()},
                                family_text=f"faster mobility (lambda {lam_hi:g} vs 1)",
                                failed={int(N): v.get("failed_guards") or [] for N, v in per_N.items()})
        decomp = {N: ((rows.get(N) or {}).get("decomposition") or {}).get("abf") for N in per_N}
        fams[fid] = dict(lam_hi=lk, lam_ref="1", per_N=per_N, combined=comb,
                         mc3=(mech.get("mc3") or {}).get(lk), abf_decomposition=decomp, tv_rows=tv)
    for fid, other in (("E1", "E2"), ("E2", "E1")):
        f = fams[fid]
        f["interpretation"] = interpretation(fid, float(f["lam_hi"]), f["per_N"], f["mc3"], f["abf_decomposition"],
                                             other_family_global=fams[other]["combined"]["global_label"],
                                             tv_rows=f.pop("tv_rows"))
    e1, e2 = fams["E1"]["combined"]["global_label"], fams["E2"]["combined"]["global_label"]
    disagree = (e1.startswith("rescue")) != (e2.startswith("rescue"))
    notes = []
    if disagree:
        notes.append("[hypothesis] E1 and E2 disagree (e.g. no rescue at lambda 16, rescue at lambda 128): consistent "
                     "with flank relaxation (which changes materially only at lambda >> m'^2) mattering, with transport "
                     "also changed.")
    return dict(families=fams, e1_e2_disagree=disagree, notes=notes)


def consistency_problems(cc, mech):
    """Both inputs must have been made from the same cell configs and summaries."""
    probs = []
    for lk in sorted(set(cc.get("cells", {})) | set(mech.get("cells", {}))):
        a, b = (cc.get("cells") or {}).get(lk), (mech.get("cells") or {}).get(lk)
        if a is None or b is None:
            probs.append(f"lambda {lk} present in only one input")
            continue
        for k in ("config_sha256", "summary_sha256"):
            if a.get(k) != b.get(k):
                probs.append(f"lambda {lk}: {k} {str(a.get(k))[:12]} != {str(b.get(k))[:12]}")
    return probs


def md(doc):
    am = doc.get("prereg_amendment") or {}
    L = ["# Experiment A: frozen outcome classification (PREREGISTRATION.md 1.7)", "",
         f"Generated {doc['provenance']['generated_utc']} by scripts/final_campaign/expA_outcome.py ({SCHEMA}) from "
         f"cross_cell.json (sha256 {doc['inputs']['cross_cell_sha256'][:12]}) and mech.json (sha256 "
         f"{doc['inputs']['mech_sha256'][:12]}).  Complete: {doc['complete']}"
         + (" (PROVISIONAL)" if doc.get("provisional") else "") + ".", ""]
    if doc.get("fixture"):
        L += ["**FIXTURE INPUT: synthetic tiny-budget data, never production evidence.**", ""]
    comp = doc.get("completeness") or {}
    if comp.get("reasons"):
        L += ["Not complete because:", ""] + [f"* {r}" for r in comp["reasons"][:20]] + [""]
    L += ["Resolutions r1-r3 of cases the frozen table leaves open (analysis decisions after the freeze): "
          + (f"recorded in the dated amendment {am.get('path')} line {am.get('line_no')}: \"{am.get('line')}\" "
             f"(block sha256 {str(am.get('block_sha256'))[:12]})." if am.get("found") else
             f"**NOT YET recorded as a dated amendment of {am.get('path')}** (marker {am.get('marker')}): this "
             "classification is not final."), ""]
    for fid, f in doc["families"].items():
        L += [f"## {fid}: lambda {f['lam_hi']} vs lambda 1", "",
              "| N | label | wording | R [95 % CI] | p Holm | G_hi | guards g1 g2 g3 g4 | failed | rule |",
              "|---|---|---|---|---|---|---|---|---|"]
        for N, r in f["per_N"].items():
            i = r["inputs"]
            gs = " ".join(str(i["guards"].get(g)) for g in GUARDS)
            L.append(f"| {N} | **{r['label']}** | {r['wording']} | {C.fmt(i['R'], 4)} {C.fmt_ci(i['R_ci95'], 4)} | "
                     f"{C.fmt(i['p_holm'])} | {C.fmt(i['G_hi'])} | {gs} | {', '.join(r['failed_guards']) or '--'} | "
                     f"{r['rule']}{' (resolution ' + r['resolution'] + ')' if r.get('resolution') else ''} |")
        c = f["combined"]
        L += ["", f"**Combined across N: {c['global_label']}** -- {c['wording']}"]
        if c.get("wording_context"):
            L.append(f"({c['wording_context']})")
        L += ["", "Interpretation (frozen map):", ""]
        L += [f"* {s}" for s in f["interpretation"]]
        L.append("")
    if doc["notes"]:
        L += ["## Notes", ""] + [f"* {s}" for s in doc["notes"]] + [""]
    L += ["## Resolutions of cases the frozen table leaves open", ""] + [f"* {k}: {v}" for k, v in RESOLUTIONS.items()]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", default=C.CELLS_DIR)
    ap.add_argument("--synthesis-dir", default=C.SYN_DIR)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-verify-runs", action="store_true",
                    help="skip the summaries' run-tree freshness check (forbids complete)")
    ap.add_argument("--provisional", action="store_true",
                    help="force complete = false (the synthesis runs over a failed completeness audit)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    sd = os.path.abspath(a.synthesis_dir)
    out = os.path.abspath(a.out_dir or sd)
    cp, mp = os.path.join(sd, "cross_cell.json"), os.path.join(sd, "mech.json")
    cc, mech = C.load_signed(cp, "cross_cell.json"), C.load_signed(mp, "mech.json")
    cells = C.load_cells(a.cells_dir)
    if not cells:
        raise C.ExpAError(f"no cell config under {a.cells_dir}")
    verify = not a.no_verify_runs
    diff = []
    if [int(n) for n in cc["N"]] != [int(n) for n in mech["N"]]:
        diff.append(f"cross_cell N {cc['N']} != mech N {mech['N']}")
    diff += consistency_problems(cc, mech)
    diff += C.synthesis_input_problems(cells, cross_cell=cc, mech=mech)
    am = prereg_amendment()
    extra = [f"{nm} is not complete" for nm, d in (("cross_cell.json", cc), ("mech.json", mech)) if not d.get("complete")]
    if not am["found"]:
        extra.append(f"resolutions r1-r3 are not yet recorded as a dated amendment of {am['path']} (marker "
                     f"{AMENDMENT_MARKER}): {am.get('why')}")
    comp = C.completeness(cells, None, verify, a.provisional, extra_reasons=extra)
    sig = C.digest(dict(schema=SCHEMA, cross_cell=C.sha256_file(cp), mech=C.sha256_file(mp),
                        code=C.code_digests(CODE_FILES), audits=C.audit_signature(comp["audits"]),
                        provisional=bool(a.provisional), verify_runs=verify,
                        amendment=dict(found=am["found"], block_sha256=am["block_sha256"])))
    jp = os.path.join(out, "outcome.json")
    if a.dry_run:
        try:
            decision = C.decide_write(jp, sig, a.force, a.recompute)
        except C.ExpAError as e:
            decision = f"REFUSE ({e})"
        sp = C.summary_problems(cells, verify)
        print(f"complete-eligible {comp['complete']}; amendment {'found' if am['found'] else 'MISSING'}; would "
              f"{decision} {jp} (signature {sig[:16]})" + (f"; inputs inconsistent / not current: would stop "
                                                           f"({diff[:2]})" if diff else "")
              + (f"; summaries not current for lambda {[f'{l:g}' for l in sp]}: would stop" if sp else ""))
        return 0
    if diff:
        raise C.ExpAError("cross_cell.json and mech.json were not made from the same, current cell configs and "
                          "summaries: " + "; ".join(diff[:6]) + " (re-run expA_cross_cell.py and expA_mech.py)")
    C.require_current_summaries(cells, verify)          # also before skipping
    decision = C.decide_write(jp, sig, a.force, a.recompute)
    if decision == "skip":
        print(f"{jp} is up to date (signature {sig[:16]}; inputs verified current)")
        return 0
    prev = C.previous_signature(jp)
    doc = dict(schema=SCHEMA, input_signature=sig, provenance=C.provenance(CODE_FILES),
               inputs=dict(cross_cell=C.rel(cp), cross_cell_sha256=C.sha256_file(cp), mech=C.rel(mp),
                           mech_sha256=C.sha256_file(mp), cross_cell_signature=cc.get("input_signature"),
                           mech_signature=mech.get("input_signature")),
               complete=comp["complete"], completeness=comp, provisional=bool(a.provisional),
               freshness_verified=verify, cells_dir=C.rel(a.cells_dir),
               fixture=bool(cc.get("fixture") or mech.get("fixture")), labels=list(LABELS), wording=WORDING,
               resolutions=RESOLUTIONS, prereg_amendment=am, **build(cc, mech))
    if prev and prev != sig:
        doc["replaced_signature"] = prev
    C.begin_output(jp, SCHEMA, prev)
    C.write_text(os.path.join(out, "outcome.md"), md(C.reload_json(C.M.json_safe(doc))))
    C.write_json(jp, doc)
    for fid, f in doc["families"].items():
        print(f"{fid}: " + "; ".join(f"N {N} {r['label']}" for N, r in f["per_N"].items())
              + f" -> {f['combined']['global_label']}", flush=True)
    print(f"wrote {jp} and outcome.md (complete {doc['complete']}; amendment {'found' if am['found'] else 'MISSING'})",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except C.ExpAError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
