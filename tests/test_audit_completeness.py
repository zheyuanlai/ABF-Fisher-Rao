"""Tests of scripts/equal_budget/audit_completeness.py on the REAL fixture tree (scripts/equal_budget/make_fixtures.py).

A good copy of the fixture (analysed with analyze_ladder, figures + MANIFEST.json written here as tiny valid
PNG / PDF files) must PASS; deliberately broken copies (missing / empty / corrupt figure, note instead of a figure,
missing array, incomplete meta status, missing run, wrong save grid, wrong engine knob, STATUS.json contradiction,
stale or tampered summary, too few bootstrap resamples, missing synthesis figure, undocumented reference) must each
FAIL with the expected failure code; a partial N documented in STATUS.json and re-analysed passes with flagged CIs.
All scratch lives under this file's directory (audit_test/).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=~/.cache/numba_eqb_dev \
      python -m pytest -p no:cacheprovider <this file> -q
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import zlib

import numpy as np
import pytest

REPO = "/home/zheyuanlai/ABF-Fisher-Rao"
sys.path.insert(0, os.path.join(REPO, "scripts", "equal_budget"))
sys.path.insert(0, os.path.join(REPO, "src"))
import audit_completeness as AC  # noqa: E402
import analyze_ladder as A  # noqa: E402
import eqb_metrics as M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import tempfile  # noqa: E402
FIXTURE_ROOT = os.environ.get("EQB_FIXTURE_ROOT", os.path.join(tempfile.gettempdir(), "eqb_fixtures"))
SCRATCH = os.path.join(tempfile.gettempdir(), "eqb_audit_test")
FIX = {"gateway": "gateway_fixture.json", "lta300": "lta_300K_fixture.json"}
OUT_DIR = {"gateway": "gateway", "lta300": "lta_T300"}


# ------------------------------------------------------------------------------------------------ helpers
def png_bytes(w=4, h=3):
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x20\x60\xa0" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"


def cfg_path(root, system):
    return os.path.join(root, "configs", FIX[system])


def res_root(root):
    return os.path.join(root, "results", "equal_budget_v2")


def fig_root(root):
    return os.path.join(root, "figures", "equal_budget_v2")


PER_N_TAGS = ["A1_eF_vs_t", "A2_eF_vs_u", "A3_F_profiles", "B1_eFp_vs_t", "B2_eFp_vs_u", "B3_Fp_profiles",
              "C1_marginal_inst_snapshots", "C2_TV_vs_t", "C2_TV_vs_u", "C3_density_heatmap_u", "C3_density_heatmap_t",
              "C4_visitation_snapshots", "C5_traces", "D_establishment_vs_t", "D_establishment_vs_u",
              "E_genealogy_vs_t", "E_genealogy_vs_u", "F_summary_t", "F_summary_u"]
SYNTH_IDS = ["S1_free_energy_abs_vs_N", "S2_mean_force_abs_vs_N", "S2s_secondary_reference_abs_vs_N", "S3_fr_gain_vs_N",
             "S4a_tau_free_energy_time", "S4b_tau_free_energy_budget", "S5a_tau_mean_force_time",
             "S5b_tau_mean_force_budget", "S6a_establishment_budget", "S6b_establishment_time",
             "S7_convergence_selected_N", "S7s_convergence_all_N", "S8_best_allocation", "S8s_best_allocation_secondary"]


def make_figures(root, system):
    """Tiny valid PNG + PDF files and manifests in the formats of scripts/equal_budget/plot_config.py (one
    MANIFEST.json per (system, N), 'figures' with tags; E at N = 1 a placeholder) and plot_synthesis.py
    (<out_dir>/synthesis/MANIFEST.json, ids S1..S8s, the summary sha256, freshness).  Seeds = files on disk."""
    P = json.load(open(cfg_path(root, system)))
    od = P["out_dir"]
    top = os.path.join(fig_root(root), od)
    shutil.rmtree(top, ignore_errors=True)

    def put(d, fn):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, fn), "wb") as fh:
            fh.write(PDF if fn.endswith(".pdf") else png_bytes())
        return fn

    for N in P["N_ladder"]:
        d = os.path.join(top, f"N{N}")
        tags = PER_N_TAGS + (["B4_Fp_profiles_projected"] if system != "gateway" else [])
        figs = []
        for tag in tags:
            desc = "placeholder: no FR arm" if (N == 1 and tag.startswith("E_")) else f"figure {tag}"
            figs.append(dict(tag=tag, title=tag, description=desc,
                             files=[put(d, f"{od}_N{N}_{tag}.{x}") for x in ("png", "pdf")]))
        seeds = {m: [s for s in P["seeds"] if os.path.exists(run_file(root, system, N, s, m))]
                 for m in (["abf"] if N == 1 else ["abf", "fr"])}
        man = dict(schema="eqb_figures_manifest/1", plot_version="test", metrics_version=M.METRICS_VERSION,
                   system=system, system_dir=od, N=N, fixture=True, seeds=seeds, figures=figs)
        json.dump(man, open(os.path.join(d, "MANIFEST.json"), "w"), indent=1)
    d = os.path.join(top, "synthesis")
    figs = [dict(id=i, title=i, files=[put(d, f"{i}.{x}") for x in ("png", "pdf")]) for i in SYNTH_IDS]
    summ = os.path.join(res_root(root), od, "analysis", "summary.json")
    S = json.load(open(summ))
    man = dict(schema="eqb_synthesis_manifest/1", system=system, fixture=True,
               summary=dict(path=summ, sha256=AC.sha256_file(summ), generated_utc=S["generated_utc"]),
               freshness=dict(verified=True, stale=False, stale_reasons=[]),
               plan=dict(B=P["B"], h=P["engine_cfg"]["h"], N_ladder=P["N_ladder"], seeds=P["seeds"],
                         thresholds=P["thresholds"]), figures=figs)
    json.dump(man, open(os.path.join(d, "MANIFEST.json"), "w"), indent=1)
    return top


def edit_json(path, fn):
    d = json.load(open(path))
    fn(d)
    json.dump(d, open(path, "w"), indent=1)


def audit(root, system, extra_cfg=None, quiet=True):
    out = os.path.join(root, f"audit_{system}.json")
    argv = ["--system", system, "--results-root", res_root(root), "--fig-root", fig_root(root), "--out", out]
    if system == "all":
        for s in FIX:
            argv += ["--config", f"{s}={cfg_path(root, s)}"]
    else:
        argv += ["--config", cfg_path(root, system)]
    if quiet:
        argv.append("--quiet")
    rc = AC.main(argv)

    def bad(x):
        raise ValueError(f"non-strict JSON constant {x}")
    doc = json.load(open(out), parse_constant=bad)                  # strict JSON
    return rc, doc


def codes(doc, system):
    return {f["code"] for f in doc["systems"][system]["failures"]}


def fails(doc, system, code):
    return [f for f in doc["systems"][system]["failures"] if f["code"] == code]


def copy_case(base, name):
    d = os.path.join(SCRATCH, f"case_{name}")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(base, d)                                         # copy2: mtimes preserved
    return d


def rewrite_npz(path, fn):
    with np.load(path, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    arr = fn(arr)
    np.savez_compressed(path, **arr)


def run_file(root, system, N, seed, method):
    return os.path.join(res_root(root), OUT_DIR[system], f"N{N}", f"s{seed}_{method}.npz")


# ------------------------------------------------------------------------------------------------ the good tree
@pytest.fixture(scope="module")
def base():
    need = [s for s, f in FIX.items() if not os.path.exists(os.path.join(FIXTURE_ROOT, "configs", f))]
    if need:
        import make_fixtures
        make_fixtures.main(["--root", FIXTURE_ROOT, "--systems", *need])
    b = os.path.join(SCRATCH, "base")
    shutil.rmtree(b, ignore_errors=True)
    os.makedirs(os.path.join(b, "configs"))
    for s, f in FIX.items():
        shutil.copy2(os.path.join(FIXTURE_ROOT, "configs", f), os.path.join(b, "configs", f))
        shutil.copytree(os.path.join(FIXTURE_ROOT, "results", "equal_budget_v2", OUT_DIR[s]),
                        os.path.join(res_root(b), OUT_DIR[s]), ignore=shutil.ignore_patterns("analysis"))
        A.analyze(s, res_root(b), cfg_path(b, s), None, verbose=False)
        make_figures(b, s)
    return b


@pytest.mark.parametrize("system", ["gateway", "lta300"])
def test_good_tree_passes(base, system):
    rc, doc = audit(base, system)
    a = doc["systems"][system]
    assert rc == 0 and doc["verdict"] == "PASS" and a["verdict"] == "PASS", a["failures"]
    P = json.load(open(cfg_path(base, system)))
    assert [r["N"] for r in a["N"]] == P["N_ladder"] and all(r["status"] == "complete" for r in a["N"])
    n_runs = sum(len(P["seeds"]) * (1 if N == 1 else 2) for N in P["N_ladder"])
    rf = a["run_files"]
    assert rf["n_complete"] == n_runs and rf["n_invalid"] == 0
    req = AC.GATEWAY_ARRAYS if system == "gateway" else AC.LTA_ARRAYS
    assert rf["required_arrays"] == [k for k, _, _ in req]
    assert all(v["n_uniform_saves"] == 200 for v in rf["files"].values())
    s = a["summary"]
    assert s["provenance"]["runs_checked"] == n_runs and s["provenance"]["values_checked"] == s["n_per_seed_values"] > 0
    for N, blk in s["per_N_backing"].items():
        for m, x in blk.items():
            assert x["n_files"] == x["n_seeds_summary"] == len(P["seeds"])
    ci = a["confidence_intervals"]
    assert ci["n_total"] == ci["n_full"] + ci["n_flagged"] > 0
    # in this short fixture most tau ratios are censored: those CIs are flagged with the reason, the rest are full
    assert ci["n_flagged"] > 0 and all(x["reasons"] for x in ci["flagged"])
    assert any("contrasts/" in p and "/primary/Ibar_F/" in p for p in ci["full"])
    for N, f in a["figures"]["per_N"].items():
        assert f["required"] and f["present"] and f["n_ok"] == f["n_items"] == 15
        assert f["items"]["E"] == ("note" if N == "1" else "ok")
        assert f["supplementary"] == []                                     # LTA's B4 counts for B_profiles
    assert set(a["figures"]["synthesis"]["items"].values()) == {"ok"}
    n_tags = len(PER_N_TAGS) + (1 if system == "lta300" else 0)
    assert a["figures"]["n_files_checked"] == 2 * (n_tags * len(P["N_ladder"]) + len(SYNTH_IDS))
    ref = a["references"]
    if system == "lta300":
        assert ref["units_F"] == "kJ/mol" and ref["units_gamma"] == "kJ/mol/rad" and ref["gauge"]
        assert abs(ref["gauge_check"]["mean_F_ref"]) < 1e-9
    else:
        assert ref["units"] and ref["gauge"] and abs(ref["gauge_check"]["mean_F_ref_eval_window"]) < 1e-9


def test_cli_exit_codes(base):
    """The CLI returns 0 on the good tree and 1 on a broken copy (subprocess, as the campaign will call it)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    cmd = [sys.executable, os.path.join(REPO, "scripts", "equal_budget", "audit_completeness.py"), "--system", "lta300",
           "--results-root", res_root(base), "--config", cfg_path(base, "lta300"), "--fig-root", fig_root(base),
           "--out", os.path.join(SCRATCH, "cli_good.json")]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "COMPLETENESS AUDIT PASS" in p.stdout and "lta300 (lta_T300): PASS" in p.stdout
    d = copy_case(base, "cli_broken")
    os.remove(os.path.join(fig_root(d), "lta_T300", "N4", "lta_T300_N4_C2_TV_vs_t.png"))
    cmd[cmd.index("--results-root") + 1] = res_root(d)
    cmd[cmd.index("--config") + 1] = cfg_path(d, "lta300")
    cmd[cmd.index("--fig-root") + 1] = fig_root(d)
    cmd[cmd.index("--out") + 1] = os.path.join(d, "cli.json")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert p.returncode == 1 and "[FIG_MISSING] N4 C2:" in p.stdout and "COMPLETENESS AUDIT FAIL" in p.stdout


def test_all_systems_and_documented_not_run(base):
    d = copy_case(base, "all")
    rc, doc = audit(d, "all")
    assert rc == 1 and doc["systems_audited"] == ["gateway", "lta300", "lta150"]
    assert doc["systems"]["gateway"]["verdict"] == "PASS" and doc["systems"]["lta300"]["verdict"] == "PASS"
    l150 = doc["systems"]["lta150"]
    assert l150["verdict"] == "FAIL" and {r["status"] for r in l150["N"]} == {"NOT RUN"}
    assert len(fails(doc, "lta150", "N_UNEXPLAINED")) == 11 and codes(doc, "lta150") == {"N_UNEXPLAINED"}
    # every N documented as NOT RUN -> the whole ladder is accounted for (no summary / figures can exist)
    st = os.path.join(res_root(d), "lta_T150", "STATUS.json")
    os.makedirs(os.path.dirname(st))
    P150 = json.load(open(M.default_config_path("lta150")))
    json.dump({"N": {str(N): {"status": "NOT RUN", "reason": "test: resource ceiling"} for N in P150["N_ladder"]}},
              open(st, "w"))
    rc, doc = audit(d, "all")
    assert rc == 0 and doc["verdict"] == "PASS", doc["systems"]["lta150"]["failures"]
    assert all(r["documented"] and r["reason"] == "test: resource ceiling" for r in doc["systems"]["lta150"]["N"])
    # 'running' is not a final status
    json.dump({"N": {"1024": {"status": "running", "reason": "in progress"}}}, open(st, "w"))
    rc, doc = audit(d, "all")
    assert rc == 1 and "STATUS_NOT_FINAL" in codes(doc, "lta150")


# ------------------------------------------------------------------------------------------------ broken copies
def break_missing_figure(d):
    os.remove(os.path.join(fig_root(d), "gateway", "N8", "gateway_N8_C3_density_heatmap_u.png"))


def break_empty_figure(d):
    open(os.path.join(fig_root(d), "gateway", "synthesis", "S5a_tau_mean_force_time.png"), "wb").close()


def break_corrupt_figure(d):
    open(os.path.join(fig_root(d), "gateway", "N2", "gateway_N2_D_establishment_vs_u.pdf"), "wb").write(b"not a pdf")


def break_note_instead_of_figure(d):
    def f(m):
        for e in m["figures"]:
            if e["tag"].startswith("E_"):
                e["description"] = "placeholder: no FR arm"
    edit_json(os.path.join(fig_root(d), "gateway", "N2", "MANIFEST.json"), f)


def break_N1_note_missing(d):
    edit_json(os.path.join(fig_root(d), "gateway", "N1", "MANIFEST.json"),
              lambda m: m.update(figures=[e for e in m["figures"] if not e["tag"].startswith("E_")]))


def break_synthesis_missing(d):
    edit_json(os.path.join(fig_root(d), "gateway", "synthesis", "MANIFEST.json"),
              lambda m: m.update(figures=[e for e in m["figures"] if not e["id"].startswith("S7")]))


def break_per_N_manifest_missing(d):
    os.remove(os.path.join(fig_root(d), "gateway", "N2", "MANIFEST.json"))


def break_figures_stale_seeds(d):
    edit_json(os.path.join(fig_root(d), "gateway", "N8", "MANIFEST.json"),
              lambda m: m["seeds"].update(fr=[8100, 8101]))


def break_synthesis_stale(d):
    edit_json(os.path.join(fig_root(d), "gateway", "synthesis", "MANIFEST.json"),
              lambda m: m.update(freshness=dict(verified=True, stale=True, stale_reasons=["N 2 fr s8101 rewritten"])))


def break_missing_array(d):
    rewrite_npz(run_file(d, "gateway", 2, 8101, "fr"), lambda a: {k: v for k, v in a.items() if k != "hist_inst"})


def break_incomplete_status(d):
    def f(a):
        meta = json.loads(str(a["meta_json"]))
        meta["status"] = "partial"
        a["meta_json"] = np.array(json.dumps(meta))
        return a
    rewrite_npz(run_file(d, "gateway", 2, 8100, "abf"), f)


def break_missing_run(d):
    os.remove(run_file(d, "gateway", 8, 8102, "fr"))


def break_wrong_grid(d):
    """Drop the save at u = 0.5 from one run (every per-save array, meta n_saves kept consistent)."""
    def f(a):
        S = a["save_step"].shape[0]
        n_steps = json.loads(str(a["meta_json"]))["n_steps"]
        i = int(np.flatnonzero(a["save_step"] == n_steps // 2)[0])
        for k, v in list(a.items()):
            if v.ndim >= 1 and v.shape[0] == S and k not in ("traces_step", "traces_t", "traces", "traces_anc"):
                a[k] = np.delete(v, i, axis=0)
        meta = json.loads(str(a["meta_json"]))
        meta["n_saves"] = S - 1
        a["meta_json"] = np.array(json.dumps(meta))
        return a
    rewrite_npz(run_file(d, "gateway", 2, 8101, "fr"), f)


def break_engine_knob(d):
    def f(a):
        c = json.loads(str(a["cfg_json"]))
        c["gamma"] = 3.0
        a["cfg_json"] = np.array(json.dumps(c))
        return a
    rewrite_npz(run_file(d, "gateway", 8, 8100, "fr"), f)


def break_status_contradiction(d):
    json.dump({"N": {"8": {"status": "NOT RUN", "reason": "claimed but false"}}},
              open(os.path.join(res_root(d), "gateway", "STATUS.json"), "w"))


def break_summary_value(d):
    p = os.path.join(res_root(d), "gateway", "analysis", "summary.json")
    S = json.load(open(p))
    S["per_N"]["8"]["abf"]["Ibar_F"]["per_seed"]["8100"] *= 0.9
    json.dump(S, open(p, "w"), indent=1)


def break_summary_nboot(d):
    p = os.path.join(res_root(d), "gateway", "analysis", "summary.json")
    S = json.load(open(p))
    S["n_boot"] = 500
    json.dump(S, open(p, "w"), indent=1)


def break_summary_stale(d):
    """A run file rewritten after the analysis with DIFFERENT data but the same byte size and mtime (uncompressed
    rewrite first, then re-analysis, then the data change): the content-keyed cache no longer matches."""
    f = run_file(d, "gateway", 2, 8100, "fr")
    with np.load(f, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    np.savez(f[:-4], **arr)
    A.analyze("gateway", res_root(d), cfg_path(d, "gateway"), None, verbose=False)
    make_figures(d, "gateway")
    st = os.stat(f)
    arr["M_all"] = arr["M_all"] * 1.37
    np.savez(f[:-4], **arr)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert os.stat(f).st_size == st.st_size


def break_summary_missing(d):
    os.remove(os.path.join(res_root(d), "gateway", "analysis", "summary.json"))


def break_checkpoint_only(d):
    f = run_file(d, "gateway", 8, 8101, "abf")
    os.rename(f, f + ".ckpt.npz")


BROKEN = [
    ("missing_figure", break_missing_figure, {"FIG_MISSING"}, "N8 C3"),
    ("empty_figure", break_empty_figure, {"FIG_EMPTY"}, "S5a"),
    ("corrupt_figure", break_corrupt_figure, {"FIG_CORRUPT"}, "N2 D"),
    ("note_instead_of_figure", break_note_instead_of_figure, {"FIG_NOTE"}, "N2 E"),
    ("N1_note_missing", break_N1_note_missing, {"FIG_MISSING"}, "N1 E"),
    ("synthesis_missing", break_synthesis_missing, {"SYNTH_MISSING"}, "S7"),
    ("per_N_manifest_missing", break_per_N_manifest_missing, {"MANIFEST_MISSING"}, "N2 figures"),
    ("figures_stale_seeds", break_figures_stale_seeds, {"FIG_STALE"}, "N8 figures"),
    ("synthesis_stale", break_synthesis_stale, {"MANIFEST_STALE"}, "synthesis"),
    ("missing_array", break_missing_array, {"ARRAY_MISSING", "SUMMARY_STALE"}, "N2/s8101_fr.npz"),
    ("incomplete_status", break_incomplete_status, {"FILE_STATUS", "N_UNEXPLAINED", "LEDGER_CLAIM", "SUMMARY_STALE"},
     "N2/s8100_abf.npz"),
    ("missing_run", break_missing_run, {"N_UNEXPLAINED", "LEDGER_CLAIM", "SUMMARY_STALE"}, "N8"),
    ("checkpoint_only", break_checkpoint_only, {"N_UNEXPLAINED", "LEDGER_CLAIM", "SUMMARY_STALE"}, "N8"),
    ("wrong_grid", break_wrong_grid, {"GRID"}, "N2/s8101_fr.npz"),
    ("engine_knob", break_engine_knob, {"PLAN_MISMATCH"}, "N8/s8100_fr.npz"),
    ("status_contradiction", break_status_contradiction, {"STATUS_CONTRADICTION"}, "N8"),
    ("summary_value", break_summary_value, {"SUMMARY_UNBACKED", "MANIFEST_STALE"}, "N8 s8100 abf"),
    ("summary_nboot", break_summary_nboot, {"SUMMARY_NBOOT", "MANIFEST_STALE"}, "summary"),
    ("summary_stale", break_summary_stale, {"SUMMARY_STALE"}, "N2 s8100 fr"),
    ("summary_missing", break_summary_missing, {"SUMMARY_MISSING"}, "summary"),
]


@pytest.mark.parametrize("name,brk,want,where", BROKEN, ids=[b[0] for b in BROKEN])
def test_broken_copy_fails(base, name, brk, want, where):
    d = copy_case(base, name)
    brk(d)
    rc, doc = audit(d, "gateway")
    got = codes(doc, "gateway")
    assert rc == 1 and doc["verdict"] == "FAIL" and doc["systems"]["gateway"]["verdict"] == "FAIL"
    assert want <= got, (name, got)
    assert any(f["where"] == where for f in doc["systems"]["gateway"]["failures"] if f["code"] in want), \
        [f for f in doc["systems"]["gateway"]["failures"] if f["code"] in want]
    if name == "checkpoint_only":
        r8 = [r for r in doc["systems"]["gateway"]["N"] if r["N"] == 8][0]
        assert r8["status"] == "partial" and r8["methods"]["abf"]["running"] == 1
    if name == "missing_array":
        f = fails(doc, "gateway", "ARRAY_MISSING")[0]
        assert "hist_inst" in f["message"]
    if name == "summary_value":
        f = [f for f in fails(doc, "gateway", "SUMMARY_UNBACKED") if f["where"] == "N8 s8100 abf"][0]
        assert f["message"].startswith("1 summary value(s) differ") and "Ibar_F" in f["message"]
    if name == "wrong_grid":
        msg = " | ".join(f["message"] for f in fails(doc, "gateway", "GRID"))
        assert "k = [100]" in msg and "profile snapshot fractions u [0.5]" in msg


def test_combined_broken_copy_fails_on_every_defect(base):
    """The requested broken copy: missing figure + missing array + incomplete status in one tree."""
    d = copy_case(base, "combined")
    break_missing_figure(d)
    rewrite_npz(run_file(d, "lta300", 4, 30001, "fr"), lambda a: {k: v for k, v in a.items() if k != "C_prod"})
    break_incomplete_status(d)
    rc, doc = audit(d, "all")
    assert rc == 1 and doc["verdict"] == "FAIL"
    assert {"FIG_MISSING", "FILE_STATUS"} <= codes(doc, "gateway")
    assert "ARRAY_MISSING" in codes(doc, "lta300")
    m = fails(doc, "lta300", "ARRAY_MISSING")[0]
    assert m["where"] == "N4/s30001_fr.npz" and "C_prod" in m["message"]


def test_documented_partial_N_passes_after_reanalysis(base):
    d = copy_case(base, "documented_partial")
    os.remove(run_file(d, "lta300", 2, 30002, "fr"))
    st = os.path.join(res_root(d), "lta_T300", "STATUS.json")
    # a wrong 'missing' list contradicts the disk
    json.dump({"N": {"2": {"status": "partial", "reason": "test: run quarantined", "missing": {"fr": [30001]}}}},
              open(st, "w"))
    A.analyze("lta300", res_root(d), cfg_path(d, "lta300"), None, verbose=False)
    make_figures(d, "lta300")
    rc, doc = audit(d, "lta300")
    assert rc == 1 and "STATUS_CONTRADICTION" in codes(doc, "lta300")
    # a completed run (ledger 'complete') removed: documenting the N is NOT enough, the deletion must be named
    json.dump({"N": {"N2": {"status": "partial", "reason": "test: run quarantined", "missing": {"fr": [30002]}}}},
              open(st, "w"))
    rc, doc = audit(d, "lta300")
    assert rc == 1 and "LEDGER_CLAIM" in codes(doc, "lta300")
    json.dump({"N": {"N2": {"status": "partial", "reason": "test: run quarantined (file corrupted, deleted)",
                            "missing": {"fr": [30002]}, "deleted": {"fr": [30002]}}}}, open(st, "w"))
    rc, doc = audit(d, "lta300")
    a = doc["systems"]["lta300"]
    assert rc == 0 and a["verdict"] == "PASS", a["failures"]
    r2 = [r for r in a["N"] if r["N"] == 2][0]
    assert r2["status"] == "partial" and r2["documented"] and r2["methods"]["fr"]["complete"] == 2
    assert any(w["code"] == "LEDGER_CLAIM" for w in a["warnings"])          # documented -> warning, not failure
    fl = [c for c in a["confidence_intervals"]["flagged"] if c["path"].startswith("contrasts/2/primary/")]
    assert fl and all("1 planned seed(s) without both arms on disk" in c["reasons"][0] for c in fl)
    assert all(c["n_seeds"] == 2 for c in fl)
    ba = [c for c in a["confidence_intervals"]["flagged"] if c["path"] == "best_allocation/Ibar_F/fr/boot_best_value_ci95"]
    assert ba and ba[0]["seeds_per_N"] == {"2": 2, "4": 3}                   # N 1 (ABF only) is not an FR candidate


def test_reference_must_document_units_and_gauge(base, monkeypatch):
    d = copy_case(base, "bad_reference")
    rdir = os.path.join(d, "refs_bad")
    os.makedirs(rdir)
    for f in os.listdir(M.REF_DIR):
        shutil.copy2(os.path.join(M.REF_DIR, f), os.path.join(rdir, f))
    p = os.path.join(rdir, "lta_T300_reference.npz")
    with np.load(p, allow_pickle=False) as z:
        ref = {k: z[k] for k in z.files if k != "gauge"}
    ref["units_gamma"] = np.array("kcal/mol/rad")
    ref["F_ref"] = ref["F_ref"] + 0.5                        # violates the stated gauge (mean 0)
    np.savez(p, **ref)
    monkeypatch.setattr(M, "REF_DIR", rdir)
    rc, doc = audit(d, "lta300")
    got = codes(doc, "lta300")
    assert rc == 1 and {"REF_UNDOCUMENTED", "REF_UNITS", "REF_GAUGE", "REF_SUMMARY"} <= got, got
    assert any("'gauge'" in f["message"] for f in fails(doc, "lta300", "REF_UNDOCUMENTED"))


def test_unknown_ci_location_is_unauditable(base):
    d = copy_case(base, "unknown_ci")
    p = os.path.join(res_root(d), "lta_T300", "analysis", "summary.json")
    S = json.load(open(p))
    S["extra_block"] = {"something": {"G_ci95": [0.1, 0.2]}}
    json.dump(S, open(p, "w"), indent=1)
    make_figures(d, "lta300")                                # re-pin the manifest to the edited summary
    rc, doc = audit(d, "lta300")
    assert rc == 1 and codes(doc, "lta300") == {"CI_UNAUDITABLE"}


def test_synthesis_under_results_root_passes_with_warning(base):
    """plot_synthesis.py's default --fig-root is the results root: accepted, with a SYNTH_LOCATION warning."""
    d = copy_case(base, "synth_location")
    shutil.move(os.path.join(fig_root(d), "lta_T300", "synthesis"), os.path.join(res_root(d), "lta_T300", "synthesis"))
    rc, doc = audit(d, "lta300")
    a = doc["systems"]["lta300"]
    assert rc == 0 and a["verdict"] == "PASS", a["failures"]
    assert any(w["code"] == "SYNTH_LOCATION" for w in a["warnings"])
    assert set(a["figures"]["synthesis"]["items"].values()) == {"ok"}


@pytest.mark.skipif(os.environ.get("EQB_AUDIT_REAL_PLOTS") != "1", reason="slow (~3 min): set EQB_AUDIT_REAL_PLOTS=1")
def test_real_plot_scripts_pass_the_audit(base):
    """End to end: the REAL scripts/equal_budget/plot_config.py and plot_synthesis.py on the fixture copy."""
    d = copy_case(base, "real_plots_test")
    shutil.rmtree(fig_root(d))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    for s in FIX:
        for script in ("plot_config.py", "plot_synthesis.py"):
            subprocess.run([sys.executable, os.path.join(REPO, "scripts", "equal_budget", script), "--system", s,
                            "--results-root", res_root(d), "--config", cfg_path(d, s), "--fig-root", fig_root(d)],
                           check=True, env=env, capture_output=True)
    rc, doc = audit(d, "all")
    for s in FIX:
        assert doc["systems"][s]["verdict"] == "PASS", doc["systems"][s]["failures"]
        assert doc["systems"][s]["figures"]["per_N"]["1"]["items"]["E"] == "note"


# ------------------------------------------------------------------------------------------------ review fixes
def _rerun(d, system="gateway"):
    A.analyze(system, res_root(d), cfg_path(d, system), None, verbose=False)
    make_figures(d, system)


def test_unpaired_arm_fails(base):
    """An FR arm regenerated with another Langevin noise seed is not paired with its ABF arm (plan section 6)."""
    import gateway_ladder_numba as E
    import run_ladder as RL
    d = copy_case(base, "unpaired")
    P = json.load(open(cfg_path(d, "gateway")))
    N, s = 8, 8100
    p = run_file(d, "gateway", N, s, "fr")
    os.remove(p)
    n_steps = int(P["B"]) // N
    E.run_job(P["engine_cfg"], "fr", N, s, n_steps, p, save_steps=RL.save_grid(dict(P, system_key="gateway"), n_steps),
              noise_seed=E.default_noise_seed(s, N) + 12345)
    _rerun(d)
    rc, doc = audit(d, "gateway")
    f = fails(doc, "gateway", "PAIRING")
    assert rc == 1 and f and all(x["where"] == "N8 s8100" for x in f)
    assert any("noise_seed" in x["message"] for x in f) and any("differ between the arms" in x["message"] for x in f)


def test_cfg_and_engine_build_consistency(base):
    d = copy_case(base, "cfg_engine")
    p = run_file(d, "lta300", 4, 30000, "fr")
    def phys(arr):
        c = json.loads(str(arr["cfg_json"]))
        c["rc"] = float(c["rc"]) * 0.8
        c["eps_go_K"] = float(c["eps_go_K"]) * 1.1
        arr["cfg_json"] = np.array(json.dumps(c, sort_keys=True))
        return arr
    rewrite_npz(p, phys)
    _rerun(d, "lta300")
    rc, doc = audit(d, "lta300")
    f = fails(doc, "lta300", "CFG_INCONSISTENT")
    assert rc == 1 and f and "rc" in f[0]["message"] and "eps_go_K" in f[0]["message"]
    # two engine builds in one ladder: FAIL unless STATUS.json documents it
    d2 = copy_case(base, "engine_mixed")
    def build(arr):
        m = json.loads(str(arr["meta_json"]))
        m["engine_sha256"] = "0" * 64
        arr["meta_json"] = np.array(json.dumps(m, sort_keys=True))
        return arr
    rewrite_npz(run_file(d2, "gateway", 2, 8101, "abf"), build)
    _rerun(d2)
    rc, doc = audit(d2, "gateway")
    assert rc == 1 and "ENGINE_MIXED" in codes(doc, "gateway")
    json.dump({"engine_builds": {"reason": "test: comment-only engine edit between builds"}, "N": {}},
              open(os.path.join(res_root(d2), "gateway", "STATUS.json"), "w"))
    rc, doc = audit(d2, "gateway")
    assert rc == 0, doc["systems"]["gateway"]["failures"]
    assert any(w["code"] == "ENGINE_MIXED" for w in doc["systems"]["gateway"]["warnings"])


def test_ledger_rules(base):
    d = copy_case(base, "no_ledger")
    os.remove(os.path.join(res_root(d), "gateway", "ledger.csv"))
    rc, doc = audit(d, "gateway")
    assert rc == 1 and "LEDGER_MISSING" in codes(doc, "gateway")
    # a completed N deleted afterwards and declared NOT RUN: the ledger still says complete -> FAIL
    d2 = copy_case(base, "dropped_N")
    shutil.rmtree(os.path.join(res_root(d2), "gateway", "N2"))
    json.dump({"N": {"2": {"status": "NOT RUN", "reason": "x"}}}, open(os.path.join(res_root(d2), "gateway", "STATUS.json"), "w"))
    _rerun(d2)
    rc, doc = audit(d2, "gateway")
    assert rc == 1 and {f["where"] for f in fails(doc, "gateway", "LEDGER_CLAIM")} == {
        f"N2 s{s} {m}" for s in (8100, 8101, 8102) for m in ("abf", "fr")}


def test_reduced_ladder_flags_best_allocation(base):
    d = copy_case(base, "reduced")
    shutil.rmtree(os.path.join(res_root(d), "gateway", "N2"))
    lp = os.path.join(res_root(d), "gateway", "ledger.csv")
    lines = open(lp).read().splitlines()
    open(lp, "w").write("\n".join([lines[0]] + [l for l in lines[1:] if l.split(",")[1] != "2"]) + "\n")
    json.dump({"N": {"2": {"status": "NOT RUN", "reason": "test: resource ceiling"}}},
              open(os.path.join(res_root(d), "gateway", "STATUS.json"), "w"))
    _rerun(d)
    rc, doc = audit(d, "gateway")
    a = doc["systems"]["gateway"]
    assert rc == 0, a["failures"]
    ba = [c for c in a["confidence_intervals"]["flagged"] if c["path"].startswith("best_allocation/")]
    assert ba and all(any("REDUCED ladder" in r for r in c["reasons"]) for c in ba)
    assert not [p for p in a["confidence_intervals"]["full"] if p.startswith("best_allocation/")]
    S = json.load(open(os.path.join(res_root(d), "gateway", "analysis", "summary.json")))
    assert S["best_allocation"]["Ibar_F"]["fr"]["Ns_missing"] == [2]
    assert "REDUCED ladder" in open(os.path.join(res_root(d), "gateway", "analysis", "tables.md")).read()


def test_tampered_aggregate_and_reference_change(base, monkeypatch):
    d = copy_case(base, "tamper_aggregate")
    sp = os.path.join(res_root(d), "gateway", "analysis", "summary.json")
    S = json.load(open(sp))
    c = S["contrasts"]["8"]["primary"]["Ibar_F"]
    c["G_median"], c["G_ci95"] = -0.9, [-0.95, -0.85]
    S["max_transient"]["8"]["e_F"]["max_rel"] = 0.99
    json.dump(S, open(sp, "w"), indent=1, allow_nan=False)
    make_figures(d, "gateway")
    rc, doc = audit(d, "gateway")
    f = fails(doc, "gateway", "SUMMARY_AGGREGATE")
    assert rc == 1 and f and "contrasts/8/primary/Ibar_F/G_median" in f[0]["message"]
    # the frozen reference changed after the analysis (gauge kept): cache keys and the summary's sha disagree
    d2 = copy_case(base, "ref_change")
    refs = os.path.join(d2, "refs")
    shutil.copytree(M.REF_DIR, refs)
    z = dict(np.load(os.path.join(refs, "lta_T300_reference.npz")))
    z["F_ref"] = z["F_ref"] + 0.5 * np.cos(z["grid_phi"]) - np.mean(0.5 * np.cos(z["grid_phi"]))
    np.savez(os.path.join(refs, "lta_T300_reference"), **z)
    monkeypatch.setattr(M, "REF_DIR", refs)
    rc, doc = audit(d2, "lta300")
    cs = codes(doc, "lta300")
    assert rc == 1 and "REF_SUMMARY" in cs and "SUMMARY_STALE" in cs


def test_synthesis_needs_every_id(base):
    d = copy_case(base, "synth_subset")
    mp = os.path.join(fig_root(d), "gateway", "synthesis", "MANIFEST.json")
    edit_json(mp, lambda m: m.update(figures=[e for e in m["figures"]
                                              if not e["id"].startswith(("S4b", "S5b", "S6b", "S2s", "S7s", "S8s"))]))
    rc, doc = audit(d, "gateway")
    assert rc == 1 and {f["where"] for f in fails(doc, "gateway", "SYNTH_MISSING")} == {"S4b", "S5b", "S6b", "S2s", "S7s", "S8s"}


def test_disjoint_seeds_documented_passes(base):
    """Both arms at an N complete but no common seed: the analysis does not crash and the audit accepts n = 0
    transients once the partial N is documented."""
    d = copy_case(base, "disjoint")
    g = os.path.join(res_root(d), "gateway", "N8")
    for f in ("s8102_abf.npz", "s8100_fr.npz", "s8101_fr.npz"):
        os.remove(os.path.join(g, f))
    lp = os.path.join(res_root(d), "gateway", "ledger.csv")
    gone = {("8102", "abf"), ("8100", "fr"), ("8101", "fr")}
    lines = open(lp).read().splitlines()
    open(lp, "w").write("\n".join([lines[0]] + [l for l in lines[1:] if not (l.split(",")[1] == "8" and
                                                (l.split(",")[2], l.split(",")[3]) in gone)]) + "\n")
    json.dump({"N": {"8": {"status": "partial", "reason": "test: mid-campaign",
                           "missing": {"abf": [8102], "fr": [8100, 8101]}}}},
              open(os.path.join(res_root(d), "gateway", "STATUS.json"), "w"))
    _rerun(d)
    rc, doc = audit(d, "gateway")
    assert rc == 0, doc["systems"]["gateway"]["failures"]
