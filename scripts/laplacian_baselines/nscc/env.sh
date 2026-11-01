# Environment of every LRS-Laplacian-baseline process on NSCC (sourced by the PBS wrappers and for interactive use).
# CPU-only numba engines (gateway_ladder_numba, lta_ladder_numba, eri.engine) in the uv env `ltafl`
# (numpy / scipy / numba / matplotlib / pytest; rebuilt by scripts/lta_loading/setup_env.sh after a purge).
S=/scratch/users/nus/e1297740
export LB_ROOT=$S/ABF-Fisher-Rao
export LB_LOGS=$S/logs
export PATH=$S/envs/ltafl/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$LB_ROOT/src
export XDG_CACHE_HOME=$S/cache NUMBA_CACHE_DIR=$S/cache/numba_lapbase MPLCONFIGDIR=$S/cache/mpl
export CUDA_VISIBLE_DEVICES=""
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ABFFR_DATA_ROOT=$LB_ROOT
mkdir -p "$LB_LOGS" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
cd "$LB_ROOT"
