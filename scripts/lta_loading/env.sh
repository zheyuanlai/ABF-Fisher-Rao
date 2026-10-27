# Environment of every finite-loading LTA process (sourced by the PBS wrapper and interactive use).
S=/scratch/users/nus/e1297740
export LF_ROOT=$S/ABF-Fisher-Rao
export LF_LOGS=$S/logs
export PATH=$S/envs/ltafl/bin:/app/apps/cuda/12.8.1/bin:$PATH
export CUDA_PATH=/app/apps/cuda/12.8.1 LD_LIBRARY_PATH=/app/apps/cuda/12.8.1/lib64:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export XDG_CACHE_HOME=$S/cache NUMBA_CACHE_DIR=$S/cache/numba_ltafl CUPY_CACHE_DIR=$S/cache/cupy
export MPLCONFIGDIR=$S/cache/mpl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$LF_LOGS" "$NUMBA_CACHE_DIR" "$CUPY_CACHE_DIR" "$MPLCONFIGDIR"
cd "$LF_ROOT"
