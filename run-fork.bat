#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_EXE="${PROJECT_DIR}/miniconda3/bin/conda"
ENV_DIR="${PROJECT_DIR}/env"
ENV_PYTHON="${ENV_DIR}/bin/python"
export CONDA_PKGS_DIRS="${PROJECT_DIR}/.conda-pkgs"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This launcher is intended for Linux." >&2
    exit 1
fi
if [[ ! -x "${CONDA_EXE}" ]]; then
    echo "Local Miniconda not found. Run ./run-install.bat." >&2
    exit 1
fi
if [[ ! -x "${ENV_PYTHON}" || ! -f "${ENV_DIR}/conda-meta/history" ]]; then
    echo "Conda environment not found. Run ./run-install.bat." >&2
    exit 1
fi
if ! "${ENV_PYTHON}" -c \
    'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))'
then
    echo "The environment does not use Python 3.10. Run ./run-install.bat to repair it." >&2
    exit 1
fi

cd "${PROJECT_DIR}"
exec "${CONDA_EXE}" run \
    --no-capture-output \
    --prefix "${ENV_DIR}" \
    python app.py --open "$@"
