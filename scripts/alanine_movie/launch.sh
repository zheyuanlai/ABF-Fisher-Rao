#!/usr/bin/env bash
# Movie runs, one at a time on GPU 3: 2-D stages (abf, fr) then 1-D stages (phi_abf, phi_fr, psi_abf, psi_fr).
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=3
LOG=results/alanine_movie/logs; mkdir -p "$LOG"
run () {  # runner config stage init
  echo "[$(date -Is)] start $3" >> "$LOG/driver.log"
  python -u "$1" --config "$2" --stage "$3" --init-cache "$4" --cuda-graph > "$LOG/$3.log" 2>&1
  echo "[$(date -Is)] end   $3 rc=$?" >> "$LOG/driver.log"
}
for s in abf fr; do run scripts/run_alanine_study.py configs/alanine_movie/ala2d.yaml $s results/alanine_movie/ala2d/init_c7eq_seed0_N2048.npz; done
for s in phi_abf phi_fr psi_abf psi_fr; do run scripts/run_alanine_1d_study.py configs/alanine_movie/ala1d.yaml $s results/alanine_movie/ala1d/init_c7eq_seed0_N2048.npz; done
echo "[$(date -Is)] DONE" >> "$LOG/driver.log"
