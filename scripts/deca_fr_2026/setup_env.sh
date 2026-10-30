#!/usr/bin/env bash
# Environment of the 2026-10 deca-alanine study on NSCC ASPIRE 2A: uv venv on scratch (home is over
# quota; conda-forge extraction on GPFS stalls).  Rebuild after a scratch purge, on the login node:
#   bash scripts/deca_fr_2026/setup_env.sh
# torch 2.6.0 (PyPI wheel, CUDA 12.4 runtime bundled; driver 570 on the A100 nodes), OpenMM from PyPI
# (CPU only is enough: it parametrises the ff14SB system, the torch engine does the dynamics), pymbar
# for the umbrella reference.
set -euo pipefail
S=/scratch/users/nus/e1297740
ENV=$S/envs/deca26
export UV_CACHE_DIR=$S/cache/uv UV_PYTHON_INSTALL_DIR=$S/cache/uv-python
rm -rf "$ENV"
$S/bin/uv venv --python-preference only-managed --python 3.11 "$ENV"
$S/bin/uv pip install --python "$ENV/bin/python" "torch==2.6.0" "numpy<2.3" scipy pyyaml pandas \
    matplotlib pytest openmm pymbar setuptools wheel   # setuptools: triton.runtime.build imports it
"$ENV/bin/python" -c "import torch, openmm, pymbar, numpy; print('torch', torch.__version__, torch.version.cuda, 'openmm', openmm.__version__, 'pymbar', pymbar.__version__, 'numpy', numpy.__version__)"
echo SETUP_DONE
