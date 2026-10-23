#!/usr/bin/env python
"""Write manifest.json (sha256 of every package file) and deliverables/UROPS_2026_Alanine_CV.zip (no raw data)."""
import datetime
import hashlib
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_ala as CA  # noqa: E402

ZIP = os.path.join(os.path.dirname(CA.PKG), "UROPS_2026_Alanine_CV.zip")
DOCS = ["README.md", "PROVENANCE.md", "RESULTS_SUMMARY.md", "VALIDATION_REPORT.md", "INSTRUCTIONS_FOR_CLAUDE.md"]


def files():
    out = [d for d in DOCS if os.path.exists(os.path.join(CA.PKG, d))]
    for sub in ("figs", "per_seed", "data", "meta", "scripts"):
        for root, dirs, fs in os.walk(os.path.join(CA.PKG, sub)):
            dirs[:] = sorted(x for x in dirs if x != "__pycache__")
            for f in sorted(fs):
                if not f.endswith((".pyc", ".npz")):
                    out.append(os.path.relpath(os.path.join(root, f), CA.PKG))
    return out


def main():
    E = json.load(open(os.path.join(CA.PKG, "meta", "export.json")))
    man = dict(package="UROPS 2026 alanine incomplete-CV figure data (phi alone / psi alone; ABF vs ABF + uniform FR)",
               generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               repository="https://github.com/zheyuanlai/ABF-Fisher-Rao", branch="urop-alanine-incomplete-cv",
               raw_data={st: dict(file=os.path.relpath(CA.raw_file(st), CA.DATA_ROOT), sha256=E["arms"][st]["raw_sha256"])
                         for st in CA.STAGES}, provenance=E["provenance"],
               files={f: dict(sha256=hashlib.sha256(open(os.path.join(CA.PKG, f), "rb").read()).hexdigest(),
                              bytes=os.path.getsize(os.path.join(CA.PKG, f))) for f in files()})
    json.dump(man, open(os.path.join(CA.PKG, "manifest.json"), "w"), indent=1)
    tmp = ZIP + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files() + ["manifest.json"]:
            z.write(os.path.join(CA.PKG, f), os.path.join("UROPS_2026_Alanine_CV", f))
    os.replace(tmp, ZIP)
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        n = len(z.namelist())
    print(f"manifest {len(man['files'])} files; zip {n} entries, {os.path.getsize(ZIP) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
