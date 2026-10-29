# Environment of every deca_fr_2026 process (sourced by the PBS wrappers and for interactive use).
S=/scratch/users/nus/e1297740
export DF_ROOT=$S/ABF-Fisher-Rao
export DF_LOGS=$S/logs
export PATH=$S/envs/deca26/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$DF_ROOT/src
export XDG_CACHE_HOME=$S/cache TORCHINDUCTOR_CACHE_DIR=$S/cache/inductor_deca26 TRITON_CACHE_DIR=$S/cache/triton_deca26
export MPLCONFIGDIR=$S/cache/mpl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$DF_LOGS" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
cd "$DF_ROOT"
