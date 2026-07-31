#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINICONDA_DIR="${PROJECT_DIR}/miniconda3"
CONDA_EXE="${MINICONDA_DIR}/bin/conda"
ENV_DIR="${PROJECT_DIR}/env"
PYTHON_VERSION="${PYTHON_VERSION:-3.10.18}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
MINICONDA_VERSION="${MINICONDA_VERSION:-latest}"
CONDA_PKGS_DIRS="${PROJECT_DIR}/.conda-pkgs"
export CONDA_PKGS_DIRS
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This installer is intended for Linux." >&2
    exit 1
fi

case "$(uname -m)" in
    x86_64|amd64)
        MINICONDA_ARCH="x86_64"
        ;;
    aarch64|arm64)
        MINICONDA_ARCH="aarch64"
        ;;
    *)
        echo "Unsupported Linux architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

cd "${PROJECT_DIR}"

backup_directory() {
    local source_dir="$1"
    local label="$2"
    local backup_dir="${source_dir}.${label}-$(date +%Y%m%d-%H%M%S)-$$"
    mv -- "${source_dir}" "${backup_dir}"
    echo "Incompatible environment preserved at: ${backup_dir}"
}

install_miniconda() {
    local installer
    local url
    installer="$(mktemp --tmpdir codename-miniconda-XXXXXX.sh)"
    if [[ "${MINICONDA_VERSION}" == "latest" ]]; then
        url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh"
    else
        url="https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-${MINICONDA_ARCH}.sh"
    fi

    echo "Downloading Miniconda for ${MINICONDA_ARCH}..."
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --output "${installer}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --output-document="${installer}" "${url}"
    else
        echo "Install curl or wget to download Miniconda." >&2
        exit 1
    fi

    bash "${installer}" -b -p "${MINICONDA_DIR}"
    rm -f -- "${installer}"
}

if [[ -d "${MINICONDA_DIR}" && ! -x "${CONDA_EXE}" ]]; then
    backup_directory "${MINICONDA_DIR}" "incomplete"
fi
if [[ ! -x "${CONDA_EXE}" ]]; then
    install_miniconda
fi

environment_is_compatible=false
if [[ -x "${ENV_DIR}/bin/python" && -f "${ENV_DIR}/conda-meta/history" ]]; then
    if "${ENV_DIR}/bin/python" -c \
        'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))'
    then
        environment_is_compatible=true
    fi
fi

if [[ "${environment_is_compatible}" != true && -e "${ENV_DIR}" ]]; then
    backup_directory "${ENV_DIR}" "python-incompatible"
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "Creating a local Conda environment with Python ${PYTHON_VERSION}..."
    "${CONDA_EXE}" create \
        --yes \
        --prefix "${ENV_DIR}" \
        --override-channels \
        --channel conda-forge \
        "python=${PYTHON_VERSION}" \
        pip
fi

ENV_PYTHON="${ENV_DIR}/bin/python"
ENV_UV="${ENV_DIR}/bin/uv"

"${CONDA_EXE}" run --no-capture-output --prefix "${ENV_DIR}" \
    python -m pip install --upgrade pip uv

"${ENV_UV}" pip install \
    --python "${ENV_PYTHON}" \
    --upgrade \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url "${TORCH_INDEX_URL}"

"${ENV_UV}" pip install \
    --python "${ENV_PYTHON}" \
    -r "${PROJECT_DIR}/requirements.txt"

"${ENV_PYTHON}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Unexpected Python version in the environment: {sys.version}")
import torch
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY

mkdir -p "${PROJECT_DIR}/rvc/models/vocoders"
VOCODER="${PROJECT_DIR}/rvc/models/vocoders/pc_nsf_hifigan_44.1k_hop512_128bin.pth"
if [[ ! -f "${VOCODER}" ]]; then
    echo
    echo "Dependencies installed. The pc-NSF-HiFiGAN checkpoint is not present yet:"
    echo "  ${VOCODER}"
    echo "It will be downloaded automatically when the interface starts."
fi

echo
echo "Installation complete using the local Miniconda:"
echo "  ${MINICONDA_DIR}"
echo "Run: ./run-fork.bat"
echo "With torch.compile: RVC_TORCH_COMPILE=1 ./run-fork.bat"
