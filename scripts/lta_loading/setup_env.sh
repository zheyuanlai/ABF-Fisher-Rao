#!/usr/bin/env bash
# Environment of the finite-loading ethane/LTA study: uv venv (Python 3.12) + CuPy wheels, CUDA from the system
# toolkit (/app/apps/cuda/12.8.1, set by env.sh).  Rebuild after a scratch purge, on the login node:
#   bash scripts/lta_loading/setup_env.sh
# (conda-forge was tried first: extraction on GPFS took > 35 min and did not finish.)
set -euo pipefail
S=/scratch/users/nus/e1297740
ENV=$S/envs/ltafl
export UV_CACHE_DIR=$S/cache/uv UV_PYTHON_INSTALL_DIR=$S/cache/uv-python
rm -rf "$ENV"
$S/bin/uv venv --python-preference only-managed --python 3.12 "$ENV"   # /usr/bin/python3.12 is login-node only
$S/bin/uv pip install --python "$ENV/bin/python" numpy scipy numba matplotlib pytest "cupy-cuda12x>=13.3"
# CPU torch for the gateway scorer (scripts/analyze_gateway_replica_ladder.py), used by the LRS-baseline analysis
$S/bin/uv pip install --python "$ENV/bin/python" --extra-index-url https://download.pytorch.org/whl/cpu torch
"$ENV/bin/python" -c "import numpy, numba, cupy; print('numpy', numpy.__version__, 'numba', numba.__version__, 'cupy', cupy.__version__)"
echo SETUP_DONE
