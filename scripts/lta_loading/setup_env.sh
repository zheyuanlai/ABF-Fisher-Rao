#!/usr/bin/env bash
# Environment of the finite-loading ethane/LTA study (CuPy + custom CUDA kernels on one A100).
# Rebuild after a scratch purge, on the login node (download + install only):
#   bash scripts/lta_loading/setup_env.sh
set -euo pipefail
S=/scratch/users/nus/e1297740
ENV=$S/envs/ltafl
export CONDA_PKGS_DIRS=$S/conda_pkgs
module load miniforge3
rm -rf "$ENV"
conda create -y -p "$ENV" -c conda-forge python=3.12 numpy scipy matplotlib pytest numba \
    cupy "cuda-version=12.8" cuda-nvrtc
"$ENV/bin/python" -c "import numpy, cupy, numba; print('numpy', numpy.__version__, 'cupy', cupy.__version__, 'numba', numba.__version__)"
echo SETUP_DONE
