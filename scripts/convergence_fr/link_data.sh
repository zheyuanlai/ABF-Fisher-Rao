#!/usr/bin/env bash
# Link the git-ignored inputs a worktree of this branch needs from the main checkout (read-only use):
# framework voxel files and the exact-MC reference profiles. Historical RUN files are read through
# ABFFR_DATA_ROOT instead (scripts/convergence_fr/systems.py), so nothing else is linked.
#   bash scripts/convergence_fr/link_data.sh /home/zheyuanlai/ABF-Fisher-Rao
# Optional second argument (adaptive-FR campaign, 2026-10-17): the data root that holds the campaign's run
# directories (results/adaptive_fr_campaign/<block>/*.npz, git-ignored, kept ONLY there). Each block directory that
# does not exist locally is symlinked; inside a block directory that does exist locally (Ecal_K* hold tracked
# calibration JSONs) the npz files are linked one by one. The symlinks are added to this worktree's
# .git/info/exclude so they never show up as untracked.
#   bash scripts/convergence_fr/link_data.sh /home/zheyuanlai/ABF-Fisher-Rao /home/zheyuanlai/ABF-Fisher-Rao-adaptive
set -euo pipefail
SRC=${1:-/home/zheyuanlai/ABF-Fisher-Rao}
ADA=${2:-}
DST=$(cd "$(dirname "$0")/../.." && pwd)
[ "$SRC" = "$DST" ] && { echo "running in the main checkout: nothing to link"; exit 0; }
for f in cache/lta/framework.npz cache/eri/framework_gard_tait.npz cache/eri/framework_schlenker.npz cache/eri/framework_ite_itq3.npz \
         results/eri_campaign/reference/gard_tait/reference_profile.npz results/eri_campaign/reference/gard_tait/reference_fine.npz \
         results/overnight/ite/reference/ite_itq3/reference_profile.npz results/overnight/ite/reference/ite_itq3/reference_fine.npz \
         results/overnight/ite150/reference/ite_itq3/reference_profile.npz results/overnight/ite150/reference/ite_itq3/reference_fine.npz; do
  mkdir -p "$DST/$(dirname "$f")"
  [ -e "$DST/$f" ] || ln -s "$SRC/$f" "$DST/$f"
done
if [ -n "$ADA" ] && [ "$ADA" != "$DST" ]; then
  REL=results/adaptive_fr_campaign
  mkdir -p "$DST/$REL"
  for d in "$ADA/$REL"/*/; do
    b=$(basename "$d")
    case "$b" in analysis|mechanism|logs) continue;; esac
    if [ -e "$DST/$REL/$b" ] && [ ! -L "$DST/$REL/$b" ]; then
      for f in "$d"*.npz; do [ -e "$f" ] && { [ -e "$DST/$REL/$b/$(basename "$f")" ] || ln -s "$f" "$DST/$REL/$b/$(basename "$f")"; }; done
    else
      [ -e "$DST/$REL/$b" ] || ln -s "${d%/}" "$DST/$REL/$b"
    fi
  done
  EX=$(git -C "$DST" rev-parse --git-path info/exclude)
  for pat in "$REL/*_*" "$REL/dev" "$REL/Ecal_K*" "!$REL/Ecal_K*/frcal_K*.json"; do
    grep -qxF "$pat" "$EX" 2>/dev/null || echo "$pat" >> "$EX"
  done
fi
echo linked
