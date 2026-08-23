#!/usr/bin/env bash
# Regenerate docs/MOLECULAR_TABLES.md from the stored result archives.
set -e
cd "$(dirname "$0")/.."
OUT=docs/MOLECULAR_TABLES.md
{
echo "# Molecular campaign: every table, machine-generated"
echo
echo "Regenerate with \`bash scripts/make_mol_tables.sh\`.  Numbers here come from"
echo "the stored \`.npz\` archives, never retyped."
echo
echo "## Pentane: frozen hyper-parameters (chosen on 8 screening seeds)"
echo
cat results/mol/frozen_table.md 2>/dev/null || echo "(not yet run)"
echo
echo "## Pentane: confirmation, 32 fresh seeds"
echo
python scripts/mol_report.py --system PEN --tag confirm 2>/dev/null || true
echo
echo "## Pentane: convergence rate"
echo
python scripts/mol_slopes.py --system PEN --tag confirm 2>/dev/null || true
echo
echo "## Alanine dipeptide: frozen hyper-parameters"
echo
cat results/mol/frozen_ALA_table.md 2>/dev/null || echo "(not yet run)"
echo
echo "## Alanine dipeptide: confirmation, 16 fresh seeds (kJ/mol)"
echo
python scripts/mol_report.py --system ALA --tag confirm --floor 0.156 --unit "kJ/mol" 2>/dev/null || true
echo
echo "## Alanine dipeptide: convergence rate"
echo
python scripts/mol_slopes.py --system ALA --tag confirm 2>/dev/null || true
echo
echo "## Hexane: which fiber mode has to be promoted?"
echo
python scripts/mol_report.py --system HEX --hexane 2>/dev/null || true
echo
echo "## Switch campaign: does turning transport off restore the rate?"
echo
python scripts/mol_switch_report.py --system PEN 2>/dev/null || true
echo
echo "## Alanine, z = (phi, psi): the complete-coordinate control (kJ/mol)"
echo
python - <<'PY'
import numpy as np, os, sys
sys.path.insert(0, "src")
from rcwfr.campaign import paired_bootstrap, rel_change
b = "results/mol/campaign2d/ALA2D_ti_cold.npz"
if os.path.exists(b):
    base = np.load(b)["e_F_final"]
    print("| arm | e_F | I_F | curl | coverage | vs stratified TI |")
    print("|---|---|---|---|---|---|")
    for a, lab in [("ti_cold", "stratified constrained TI, cold"),
                   ("wfr", "RC-WFR (W + Fisher-Rao)"),
                   ("w_only", "RC-WFR, W only"),
                   ("wfr_sw50000", "RC-WFR -> TI, switch @5e4"),
                   ("abf", "ABF, multiple walkers")]:
        p = f"results/mol/campaign2d/ALA2D_{a}.npz"
        if not os.path.exists(p):
            continue
        d = np.load(p); e = d["e_F_final"]
        if a == "ti_cold":
            ch = "-"
        else:
            m, lo, hi = paired_bootstrap(rel_change(e, base))
            st = "**" if lo * hi > 0 else ""
            ch = f"{st}{100*m:+.1f}%{st} [{100*lo:+.1f}, {100*hi:+.1f}]"
        print(f"| {lab} | {np.median(e):.3f} | {np.median(d['I_F']):.3f} | "
              f"{np.median(d['curl'][-1]):.3f} | {np.median(d['cov'][-1]):.3f} | {ch} |")
PY
echo
echo "## Slow-mode ranking predicted from the run's own statistics"
echo
for s in PEN HEX ALA; do
  [ -f "results/mol/${s}_mode_diagnostic.json" ] && \
    { echo "\`\`\`json"; echo "// $s"; cat "results/mol/${s}_mode_diagnostic.json"; echo "\`\`\`"; echo; }
done
} > $OUT
echo "wrote $OUT"
