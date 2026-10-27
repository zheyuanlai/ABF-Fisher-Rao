# Whole study in one job: stage 1 (gates -> reference -> ABF pilot) -> c_ss bench -> ladder -> confirmation.
# Idempotent: run_batch skips complete batches with the same signature, so a resubmission resumes.
set -u
bash scripts/lta_loading/tasks/stage1.sh || exit 1
[ -f results/lta_loading/bench/bench.json ] || bash scripts/lta_loading/tasks/bench.sh || exit 2
bash scripts/lta_loading/tasks/ladder.sh || exit 3
bash scripts/lta_loading/tasks/confirm.sh || exit 4
echo PIPELINE_DONE
