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
echo "## Slow-mode ranking predicted from the run's own statistics"
echo
for s in PEN HEX ALA; do
  [ -f "results/mol/${s}_mode_diagnostic.json" ] && \
    { echo "\`\`\`json"; echo "// $s"; cat "results/mol/${s}_mode_diagnostic.json"; echo "\`\`\`"; echo; }
done
} > $OUT
echo "wrote $OUT"
