#!/usr/bin/env python
"""Write manifest.json (sha256 of every package file + provenance summary) and the self-contained
deliverables/UROPS_2026_WCA_timing.zip (no raw trajectories).  Run last."""
import datetime
import hashlib
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_wca as CW  # noqa: E402

ZIP = os.path.join(os.path.dirname(CW.PKG), "UROPS_2026_WCA_timing.zip")
DOCS = ["README.md", "PREREGISTRATION.md", "frozen_config.json", "RESULTS_SUMMARY.md", "VALIDATION_REPORT.md",
        "FIGURE_PROVENANCE.md", "INSTRUCTIONS_FOR_CLAUDE.md"]


def files():
    out = [d for d in DOCS if os.path.exists(os.path.join(CW.PKG, d))]
    for sub in ("data", "figures", "scripts"):
        for root, dirs, fs in os.walk(os.path.join(CW.PKG, sub)):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for f in sorted(fs):
                if not f.endswith((".pyc", ".npz")):
                    out.append(os.path.relpath(os.path.join(root, f), CW.PKG))
    return out


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    A = json.load(open(os.path.join(CW.PKG, "data", "summary", "analysis.json")))
    fz = CW.frozen()
    man = dict(package="UROPS 2026 WCA Fisher-Rao activation / early-stopping study (validated dt, exact MC reference)",
               generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               repository="https://github.com/zheyuanlai/ABF-Fisher-Rao", branch="urop-wca-timing-validated",
               confirmation_runs_commit=sorted({r["commit"] for r in A["run_checks"]}),
               analysis_commit=A["analysis_commit"], arms=fz["arms"], seeds=fz["seeds"], reference=fz["reference"],
               raw_data="results/urops_2026_wca_timing/raw/ in the main research checkout (git-ignored, not packaged)",
               threepanel_csv_columns="t,abf_median,abf_q25,abf_q75,fr_median,fr_q25,fr_q75",
               files={f: dict(sha256=sha(os.path.join(CW.PKG, f)), bytes=os.path.getsize(os.path.join(CW.PKG, f))) for f in files()})
    json.dump(man, open(os.path.join(CW.PKG, "manifest.json"), "w"), indent=1)
    tmp = ZIP + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files() + ["manifest.json"]:
            z.write(os.path.join(CW.PKG, f), os.path.join("UROPS_2026_WCA_timing", f))
    os.replace(tmp, ZIP)
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        n = len(z.namelist())
    print(f"manifest: {len(man['files'])} files; zip: {n} entries, {os.path.getsize(ZIP) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
