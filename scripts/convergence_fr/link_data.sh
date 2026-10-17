#!/usr/bin/env bash
# Link the git-ignored inputs a worktree of this branch needs from the main checkout (read-only use):
# framework voxel files and the exact-MC reference profiles. Historical RUN files are read through
# ABFFR_DATA_ROOT instead (scripts/convergence_fr/systems.py), so nothing else is linked.
#   bash scripts/convergence_fr/link_data.sh /home/zheyuanlai/ABF-Fisher-Rao
set -euo pipefail
SRC=${1:-/home/zheyuanlai/ABF-Fisher-Rao}
DST=$(cd "$(dirname "$0")/../.." && pwd)
[ "$SRC" = "$DST" ] && { echo "running in the main checkout: nothing to link"; exit 0; }
for f in cache/eri/framework_gard_tait.npz cache/eri/framework_schlenker.npz cache/eri/framework_ite_itq3.npz \
         results/eri_campaign/reference/gard_tait/reference_profile.npz results/eri_campaign/reference/gard_tait/reference_fine.npz \
         results/overnight/ite/reference/ite_itq3/reference_profile.npz results/overnight/ite/reference/ite_itq3/reference_fine.npz \
         results/overnight/ite150/reference/ite_itq3/reference_profile.npz results/overnight/ite150/reference/ite_itq3/reference_fine.npz; do
  mkdir -p "$DST/$(dirname "$f")"
  [ -e "$DST/$f" ] || ln -s "$SRC/$f" "$DST/$f"
done
echo linked
