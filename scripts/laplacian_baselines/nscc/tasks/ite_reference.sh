#!/bin/bash
# Rebuild the exact-MC ITE references (300 K: configs/overnight/ite/reference.json; 150 K: ite150) into
# results/laplacian_baselines/references/<cell>/ite_itq3/{MC, reference_profile.npz, reference_fine.npz, gate1.json}.
# The archived reference_profile.npz files are gitignored and absent on NSCC; the frozen protocol and seeds are re-run
# unchanged (seed_base 8100000 / 8200000), BEFORE any Laplacian production result exists.  Gate 1 numbers are printed
# next to the tracked gate1.json of the archive for a reproducibility check.
set -uo pipefail
cd "$LB_ROOT"
REF=results/laplacian_baselines/references
mkdir -p "$REF"
for cell in ite300:ite:300:0.03:0.045 ite150:ite150:150:0.03:0.045; do
  IFS=: read -r name src T tol tolup <<< "$cell"
  cfg=$REF/${name}_reference_config.json
  if [ -f "$REF/$name/ite_itq3/reference_fine.npz" ] && [ -f "$REF/$name/ite_itq3/gate1.json" ]; then
    echo "== $name: reference_fine.npz + gate1.json exist, skipping (idempotent; the MC windows are skipped by run_reference.py anyway) =="
    continue
  fi
  python - "$src" "$REF/$name" "$cfg" <<'PY'
import json, sys
src, out, dst = sys.argv[1:]
c = json.load(open(f"configs/overnight/{src}/reference.json"))
c["results_dir"] = out
c["_nscc_rebuild"] = f"copy of configs/overnight/{src}/reference.json with results_dir -> {out} (2026-11-01, LRS baseline prerequisites)"
json.dump(c, open(dst, "w"), indent=1)
print("config", dst, "windows", c["windows"]["n"], "groups", c["groups"], "MC steps", c["MC"]["n_steps"])
PY
  echo "== $name: MC umbrella windows =="
  ERI_REFERENCE_CFG=$cfg python scripts/eri_campaign/run_reference.py --variant ite_itq3 --scheme MC --groups 0-15 --procs "$NPROCS"
  echo "== $name: Gate 1 analysis (MC only; the Gate 2 Langevin ladders are not rebuilt) =="
  python scripts/eri_campaign/analyze_reference.py --variant ite_itq3 --base "$REF/$name" --T "$T" --tol "$tol" --tolup "$tolup" --tags none --nboot 2000
  python scripts/eri_campaign/reference_fine.py --base "$REF/$name/ite_itq3" --T "$T"
  echo "== $name: rebuilt vs archived gate1.json =="
  python - "$REF/$name/ite_itq3/gate1.json" "results/overnight/$src/reference/ite_itq3/gate1.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for k in ("gate1", "check1_overlap", "overlap_min_fine_count", "check6_reference_noise_rms"):
    print(f"  {k}: rebuilt {a.get(k)} | archived {b.get(k)}")
for k in ("check2_density_vs_MF", "check3_split_half", "check4_equilibration", "check5_closure"):
    print(f"  {k}: rebuilt {a[k].get('verdict')} D={a[k].get('D', a[k].get('mean_gamma'))} | archived {b[k].get('verdict')} D={b[k].get('D', b[k].get('mean_gamma'))}")
print("  barrier rebuilt:", json.dumps(a.get("barrier"))[:300])
print("  barrier archived:", json.dumps(b.get("barrier"))[:300])
PY
  ls -la "$REF/$name/ite_itq3/" | grep -v "^d"
done
echo ITE_REFERENCES_DONE
