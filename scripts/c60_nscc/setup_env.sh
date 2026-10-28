#!/usr/bin/env bash
# C60/NSCC environment: OpenMM (CUDA) + numpy stack, no torch (torch in an OpenMM-CUDA process deadlocked, NaCl).
# Rebuild after a scratch purge: bash scripts/c60_nscc/setup_env.sh   (login node; download + install only)
set -euo pipefail
S=/scratch/users/nus/e1297740
ENV=$S/envs/c60omm
export CONDA_PKGS_DIRS=$S/conda_pkgs
module load miniforge3
rm -rf "$ENV"
conda create -y -p "$ENV" -c conda-forge python=3.12 openmm "cuda-version>=12,<12.9" \
    numpy scipy pyyaml pytest matplotlib
"$ENV/bin/python" -c "import openmm, numpy; print('openmm', openmm.__version__, 'numpy', numpy.__version__)"
echo SETUP_DONE
