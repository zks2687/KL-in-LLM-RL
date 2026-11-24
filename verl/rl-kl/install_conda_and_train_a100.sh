#!/usr/bin/env bash
# One command on Linux x86_64: conda + pip -> assets -> checks -> 4-GPU training.
# Requires an NVIDIA driver and CUDA 12.8 development tools (nvcc, gcc/g++).
# Usage: bash verl/rl-kl/install_conda_and_train_a100.sh [Hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$VERL_ROOT/.." && pwd)
cd -- "$VERL_ROOT"

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    printf '%s\n' 'This installer targets Linux x86_64 with NVIDIA GPUs. Run it on your GPU server.' >&2
    exit 1
fi
if [[ -n ${CUDA_HOME:-} ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
fi
for executable in nvidia-smi nvcc gcc g++; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        printf 'Missing prerequisite: %s. Configure the NVIDIA driver and CUDA 12.8 development toolchain first.\n' "$executable" >&2
        exit 1
    fi
done
nvidia-smi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MAX_JOBS=${MAX_JOBS:-4}
export CUDA_HOME=${CUDA_HOME:-"$(dirname -- "$(dirname -- "$(readlink -f -- "$(command -v nvcc)")")")"}
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    printf 'CUDA_HOME does not contain bin/nvcc: %s\n' "$CUDA_HOME" >&2
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
cuda_compiler_version=$(nvcc --version)
printf '%s\n' "$cuda_compiler_version"
if [[ "$cuda_compiler_version" != *"release 12.8,"* ]]; then
    printf '%s\n' 'This pinned stack expects CUDA toolkit 12.8. Load it or set CUDA_HOME to its installation directory.' >&2
    exit 1
fi

# Prefer an existing conda installation. If absent, install Miniforge under HOME.
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
elif [[ -n ${CONDA_EXE:-} && -x "$CONDA_EXE" ]]; then
    CONDA_BASE=$("$CONDA_EXE" info --base)
else
    CONDA_BASE=${CONDA_INSTALL_DIR:-"$HOME/miniforge3"}
    if [[ ! -x "$CONDA_BASE/bin/conda" ]]; then
        if [[ -e "$CONDA_BASE" ]]; then
            printf 'Conda destination already exists but has no bin/conda: %s\nSet CONDA_INSTALL_DIR to a new directory.\n' "$CONDA_BASE" >&2
            exit 1
        fi
        command -v curl >/dev/null 2>&1 || { printf '%s\n' 'Install curl first.' >&2; exit 1; }
        installer=$(mktemp "${TMPDIR:-/tmp}/kl-miniforge.XXXXXX.sh")
        trap 'rm -f -- "$installer"' EXIT
        curl --fail --location --retry 3 \
            "${MINIFORGE_INSTALLER_URL:-https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh}" \
            --output "$installer"
        bash "$installer" -b -p "$CONDA_BASE"
        rm -f -- "$installer"
        trap - EXIT
    fi
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

CONDA_ENV_NAME=${CONDA_ENV_NAME:-kl-rl}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-"$CONDA_BASE/envs/$CONDA_ENV_NAME"}
if [[ ! -d "$CONDA_ENV_PREFIX" ]]; then
    conda create --yes --prefix "$CONDA_ENV_PREFIX" --override-channels \
        --channel conda-forge python=3.12 pip
fi
if [[ ! -x "$CONDA_ENV_PREFIX/bin/python" ]]; then
    printf 'Existing environment has no Python: %s\nChoose a new CONDA_ENV_NAME or CONDA_ENV_PREFIX.\n' "$CONDA_ENV_PREFIX" >&2
    exit 1
fi
conda activate "$CONDA_ENV_PREFIX"
export PYTHON_BIN="$CONDA_ENV_PREFIX/bin/python"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use a fresh Python 3.12 environment"; print("Python:", sys.executable)'

CONSTRAINTS_FILE="$SCRIPT_DIR/constraints-a100-cu128.txt"
if [[ ${SKIP_INSTALL:-0} != 1 ]]; then
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
    "$PYTHON_BIN" -m pip install -c "$CONSTRAINTS_FILE" \
        'vllm==0.11.0' 'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0' \
        --extra-index-url https://download.pytorch.org/whl/cu128
    "$PYTHON_BIN" -m pip install -c "$CONSTRAINTS_FILE" -e '.[vllm,math]' \
        'flashinfer-python==0.3.1' latex2sympy2_extended liger-kernel \
        ninja packaging psutil orjson swanlab
    "$PYTHON_BIN" -m pip install -c "$CONSTRAINTS_FILE" \
        'flash-attn==2.8.1' --no-build-isolation
fi
"$PYTHON_BIN" -m pip check

export KL_REPO_ROOT="$REPO_ROOT"
export DATA_ROOT=${DATA_ROOT:-$VERL_ROOT}
export MODEL_ID=${MODEL_ID:-Qwen/Qwen3-1.7B-Base}
export MODEL_PATH=${MODEL_PATH:-"$VERL_ROOT/models/Qwen3-1.7B-Base"}
export TRAIN_FILE=${TRAIN_FILE:-"$DATA_ROOT/data/dapo_filter.parquet"}
export VAL_FILE=${VAL_FILE:-"$DATA_ROOT/data/math500/test.parquet"}
RECORD_DIR=${RECORD_DIR:-"$REPO_ROOT/environment-records"}
mkdir -p -- "$RECORD_DIR"
"$PYTHON_BIN" -m pip freeze > "$RECORD_DIR/requirements-a100-resolved.txt"
conda env export --prefix "$CONDA_ENV_PREFIX" --no-builds > "$RECORD_DIR/conda-a100.yml"
cp -- "$CONSTRAINTS_FILE" "$RECORD_DIR/constraints-a100-cu128.txt"
nvidia-smi > "$RECORD_DIR/nvidia-smi.txt"
nvcc --version > "$RECORD_DIR/nvcc.txt"

if [[ ${INSTALL_ONLY:-0} == 1 ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/prepare_a100_training.py" --check-only
    printf 'Environment ready: %s\nActivate with: conda activate "%s"\n' "$CONDA_ENV_PREFIX" "$CONDA_ENV_PREFIX"
    exit 0
fi
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_a100_training.py"

export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","swanlab"]'}
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
for argument in "$@"; do
    if [[ "$argument" == trainer.logger=* ]]; then
        export TRAINER_LOGGER=${argument#trainer.logger=}
    fi
done
if [[ "$TRAINER_LOGGER" == *swanlab* && "$SWANLAB_MODE" == cloud && -z ${SWANLAB_API_KEY:-} ]]; then
    # Reuses saved credentials, or asks for a key before Ray starts.
    swanlab login
fi
# Authentication has already been handled here; the child launcher must not prompt twice.
export SWANLAB_LOGIN=0
printf '%s\n' 'Environment and assets ready. Starting four-GPU training...'
exec bash "$SCRIPT_DIR/train_a100_40g_4gpu.sh" "$@"
