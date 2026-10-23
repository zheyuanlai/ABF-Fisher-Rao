#!/usr/bin/env python
"""Rebuild the whole UROPS 2026 figure-data package from the raw simulation data.

    ABFFR_DATA_ROOT=/path/to/main/checkout python export_all.py [--skip-export]

1. the seven exporters (export_gateway, export_lta [150 K + 300 K], export_ite, export_metastability, export_alanine,
   export_wca) -> figs/data/*.csv, figs/data_supplementary/*.csv, per_seed/*.csv, meta/*.json;
2. validate_exports.py -> VALIDATION_REPORT.md, validation_results.json;
3. plot_previews.py -> figs/previews/*.pdf;
4. manifest.json (provenance summary + sha256 of every package file);
5. ../UROPS_2026_figure_data.zip (self-contained: CSVs, per-seed CSVs, meta, previews, documentation, scripts).
The metastability confirmation itself (meta_validate_dt.py, meta_pilot.py, meta_confirm.py) is NOT rerun here.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

EXPORTERS = ["export_gateway.py", "export_lta.py", "export_ite.py", "export_metastability.py", "export_alanine.py",
             "export_wca.py"]
DOCS = ["README.md", "FIGURE_DATA.md", "FIGURE_PROVENANCE.md", "RESULTS_SUMMARY.md", "VALIDATION_REPORT.md",
        "INSTRUCTIONS_FOR_CLAUDE.md", "manifest.json", "validation_results.json"]
ZIP = os.path.join(os.path.dirname(C.PKG), "UROPS_2026_figure_data.zip")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def run(script, *args):
    print(f"== {script} {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, script), *args], cwd=HERE)
    return r.returncode


def package_files():
    out = []
    for sub in ("figs/data", "figs/data_supplementary", "figs/previews", "per_seed", "meta", "metastability", "scripts"):
        d = os.path.join(C.PKG, sub)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                if os.path.isfile(p) and not f.endswith((".pyc",)) and "__pycache__" not in p:
                    out.append(os.path.relpath(p, C.PKG))
    out += [d for d in DOCS if os.path.exists(os.path.join(C.PKG, d)) and d != "manifest.json"]
    return out


def write_manifest():
    figs = {}
    for exp in C.EXPERIMENTS:
        mp = os.path.join(C.META_DIR, f"{exp}.json")
        if not os.path.exists(mp):
            figs[exp] = dict(status="MISSING")
            continue
        m = json.load(open(mp))
        figs[exp] = dict(
            title=m["title"], status=m["status"], seeds=m["seeds"], n_seeds=m["n_seeds"], n_times=m["n_times"],
            t_range=[m["t_first"], m["t_last"]], source=m["source"], config=m["config"], reference=m["reference"],
            definitions=m["definitions"], units=m["units"], domain=m["domain"], weighting=m["weighting"],
            marginal_type=m["marginal_type"], integrated_definition=m["integrated_definition"],
            csv={k: v["csv"] for k, v in m["files"].items()}, preview=f"figs/previews/{exp}.pdf",
            supplementary=m.get("supplementary"), floors=m.get("floors"), provenance=m["provenance"])
    files = {f: dict(sha256=sha(os.path.join(C.PKG, f)), bytes=os.path.getsize(os.path.join(C.PKG, f))) for f in package_files()}
    man = dict(package="UROPS 2026 convergence-figure data (ABF vs ABF + marginal Fisher-Rao birth-death)",
               generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               repository="https://github.com/zheyuanlai/ABF-Fisher-Rao", branch="urop-report-figures",
               csv_columns=list(C.COLUMNS), statistic="median and 25th / 75th percentiles across seeds at every saved checkpoint",
               figures=figs, files=files)
    json.dump(C.json_safe(man), open(os.path.join(C.PKG, "manifest.json"), "w"), indent=1)


def write_zip():
    tmp = ZIP + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in package_files() + ["manifest.json"]:
            z.write(os.path.join(C.PKG, f), os.path.join("UROPS_2026", f))
    os.replace(tmp, ZIP)
    with zipfile.ZipFile(ZIP) as z:
        bad = z.testzip()
        names = z.namelist()
    assert bad is None, bad
    need = [f"UROPS_2026/figs/data/{e}_{m}.csv" for e in C.EXPERIMENTS for m in C.METRICS]
    missing = [n for n in need if n not in names]
    print(f"zip: {len(names)} entries, {os.path.getsize(ZIP) / 1e6:.2f} MB; missing CSVs: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-export", action="store_true")
    a = ap.parse_args()
    if not a.skip_export:
        for s in EXPORTERS:
            if run(s) != 0:
                print(f"!! {s} failed; continuing with the others (see its traceback)", flush=True)
    rc = run("validate_exports.py")
    run("plot_previews.py")
    write_manifest()
    write_zip()
    sys.exit(rc)


if __name__ == "__main__":
    main()
