# Environment of every NSCC A100 study process (sourced by the PBS wrappers and by interactive use).
# Home is over quota on this account: every cache and build directory lives on scratch.
S=/scratch/users/nus/e1297740
export ABFFR_ROOT=$S/abffr_gpu
export ABFFR_LOGS=$S/logs
export PATH=/app/apps/cuda/12.8.1/bin:$ABFFR_ROOT/.venv/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=$S/cache UV_CACHE_DIR=$S/cache/uv
export NUMBA_CACHE_DIR=$S/cache/numba
export GW_PARALLEL_BUILD_DIR=$S/cache/gw_build
export GPU_BUILD_DIR=$ABFFR_ROOT/build/gpu
export MPLCONFIGDIR=$S/cache/mpl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -f "$ABFFR_ROOT/.git_commit" ] && export ABFFR_GIT_COMMIT=$(cat "$ABFFR_ROOT/.git_commit")
mkdir -p "$ABFFR_LOGS" "$NUMBA_CACHE_DIR" "$GW_PARALLEL_BUILD_DIR" "$GPU_BUILD_DIR" "$MPLCONFIGDIR"
cd "$ABFFR_ROOT"
