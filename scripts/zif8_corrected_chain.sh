#!/usr/bin/env bash
# Corrected-baseline chain: 5 safety rates CONCURRENTLY, then the two arms.
# The 5 calibration runs are B=384 each, far below the H200's saturation point
# (measured at B>=1024), so running them together costs far less than 5x the
# wall clock of one.  The two production arms are B=6144 each; running them
# concurrently was measured to give +60% aggregate throughput.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
O=results/information_campaign/corrected
mkdir -p "$O"
export PYTHONUNBUFFERED=1

echo "[$(date +%H:%M:%S)] safety calibration, 5 rates concurrently"
for r in 0.005 0.01 0.02 0.05 0.1; do
  python -u scripts/run_zif8_corrected.py --stage calibrate --rate "$r" \
    > "$O/cal_${r}.log" 2>&1 &
done
wait
echo "[$(date +%H:%M:%S)] calibration done"
python - <<'PY'
import json, glob, os
O="results/information_campaign/corrected"
rows=[json.load(open(f)) for f in sorted(glob.glob(f"{O}/cal_rate_*.json"))]
rows.sort(key=lambda r: r["rate"])
for r in rows:
    print(f"  rate {r['rate']:>6.3f}: ESS/N {r['ess_min']:.3f} wmax {r['wmax_max']:.4f} "
          f"evfrac {r['event_fraction']:.4f} ok={r['ok']}")
safe=[r for r in rows if r["ok"]]
sel=max(safe, key=lambda r: r["rate"])["rate"] if safe else None
json.dump(dict(ladder=rows, selected=sel, h_bias=rows[0]["h_bias"],
               note="safety only; no error metric read"),
          open(f"{O}/fr_rate_selection.json","w"), indent=2)
print(f"  SELECTED fr_rate = {sel}")
PY
SEL=$(python -c "import json;print(json.load(open('$O/fr_rate_selection.json'))['selected'])")
[ "$SEL" = "None" ] && { echo "NO SAFE RATE -- two-arm run refused"; exit 1; }

echo "[$(date +%H:%M:%S)] production, both arms concurrently (fr_rate=$SEL)"
python -u scripts/run_zif8_corrected.py --stage produce --only abf        > "$O/prod_abf.log" 2>&1 &
python -u scripts/run_zif8_corrected.py --stage produce --only fr_uniform > "$O/prod_fr.log"  2>&1 &
wait
echo "[$(date +%H:%M:%S)] production done"
python -u scripts/analyze_zif8_corrected.py > "$O/analysis.log" 2>&1
tail -40 "$O/analysis.log"
echo "[$(date +%H:%M:%S)] corrected-baseline experiment complete"
