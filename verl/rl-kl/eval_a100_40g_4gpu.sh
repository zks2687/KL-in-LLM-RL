#!/usr/bin/env bash
# Evaluate a merged HF model on all six CorrectKLinRL math benchmarks.
# Usage: EVAL_MODEL_PATH=/absolute/merged_hf bash verl/rl-kl/eval_a100_40g_4gpu.sh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$VERL_ROOT/.." && pwd)
cd -- "$REPO_ROOT"
export PYTHON_BIN=${PYTHON_BIN:-python3}
export PYTHONUNBUFFERED=1
export DATA_ROOT=${DATA_ROOT:-$VERL_ROOT}
if [[ -z ${EVAL_MODEL_PATH:-} ]]; then
    printf '%s\n' 'Set EVAL_MODEL_PATH to your merged Hugging Face model directory (not an actor shard checkpoint).' >&2
    exit 1
fi
if [[ ! -f "$EVAL_MODEL_PATH/config.json" ]]; then
    printf 'No config.json in EVAL_MODEL_PATH: %s\nMerge the FSDP checkpoint first.\n' "$EVAL_MODEL_PATH" >&2
    exit 1
fi
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.device_count() >= 4, "Need 4 visible CUDA GPUs"'

# Datasets are reused when present; no model/training environment setup here.
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_a100_training.py" --benchmarks-only
data_args=()
for name in olympiadbench minerva_math amc23 aime25 aime24 math500; do
    path="$DATA_ROOT/data/$name/test.parquet"
    if [[ "$name" == math500 ]]; then
        path=${VAL_FILE:-$path}
    fi
    [[ -f "$path" ]] || { printf 'Missing benchmark: %s\n' "$path" >&2; exit 1; }
    data_args+=(--data "$name=$path")
done

EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"$REPO_ROOT/eval/results"}
EVAL_OUTPUT_NAME=${EVAL_OUTPUT_NAME:-"qwen3-1.7b-six-benchmarks-$(date +%Y%m%d-%H%M%S)"}
mkdir -p -- "$EVAL_OUTPUT_DIR"
"$PYTHON_BIN" "$REPO_ROOT/eval/eval.py" \
    --model "$EVAL_MODEL_PATH" \
    "${data_args[@]}" \
    --output-dir "$EVAL_OUTPUT_DIR" --output-name "$EVAL_OUTPUT_NAME" \
    --n "${EVAL_N:-16}" --temperature "${EVAL_TEMPERATURE:-1.0}" \
    --max-response-length "${EVAL_MAX_RESPONSE_LENGTH:-8192}" \
    --max-model-len "${EVAL_MAX_MODEL_LEN:-10240}" \
    --tp 1 --dp 4 --dtype bfloat16 \
    --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.8}" \
    --max-num-seqs "${EVAL_MAX_NUM_SEQS:-16}" \
    --generate-batch-size "${EVAL_GENERATE_BATCH_SIZE:-1}" \
    --enforce-eager "$@"
