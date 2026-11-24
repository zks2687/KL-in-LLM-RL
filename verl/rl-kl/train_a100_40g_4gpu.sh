#!/usr/bin/env bash
# Qwen3-1.7B-Base, one node with 4 x A100 40GB, FSDP + vLLM.
# Activate the prepared conda environment first. Defaults need GPU validation.
# Usage: bash verl/rl-kl/train_a100_40g_4gpu.sh [Hydra overrides...]
# Modes: TRAIN_MODE=grpo (default), prm, prl_current_kl.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd -- "$VERL_ROOT"

TRAIN_MODE=${TRAIN_MODE:-grpo}
case "$TRAIN_MODE" in
    grpo) TRAIN_SCRIPT=train.sh ;;
    prm) TRAIN_SCRIPT=train_prl.sh ;;
    prl_current_kl) TRAIN_SCRIPT=train_prl_current_policy.sh ;;
    *) printf 'Unknown TRAIN_MODE: %s (use grpo, prm, prl_current_kl)\n' "$TRAIN_MODE" >&2; exit 1 ;;
esac

export PYTHON_BIN=${PYTHON_BIN:-python3}
export PYTHONUNBUFFERED=1
export DATA_ROOT=${DATA_ROOT:-$VERL_ROOT}
export TRAIN_FILE=${TRAIN_FILE:-"$DATA_ROOT/data/dapo_filter.parquet"}
VAL_FILE=${VAL_FILE:-"$DATA_ROOT/data/math500/test.parquet"}
if [[ -z ${MODEL_PATH:-} ]]; then
    if [[ -d "$VERL_ROOT/models/Qwen3-1.7B-Base" ]]; then
        export MODEL_PATH="$VERL_ROOT/models/Qwen3-1.7B-Base"
    else
        export MODEL_PATH=Qwen/Qwen3-1.7B-Base
    fi
fi

for data_file in "$TRAIN_FILE" "$VAL_FILE"; do
    if [[ ! -f "$data_file" ]]; then
        printf 'Missing parquet: %s\nSet TRAIN_FILE and VAL_FILE to existing absolute paths.\n' "$data_file" >&2
        exit 1
    fi
done

# All four GPUs participate in FSDP; TP=1 gives independent rollout replicas.
"$PYTHON_BIN" -c 'import torch; n = torch.cuda.device_count(); assert n >= 4, f"Need 4 visible CUDA GPUs, found {n}. Check CUDA_VISIBLE_DEVICES and your job allocation."'

export PROJECT_NAME=${PROJECT_NAME:-kl-in-llm-rl}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3-1.7b-${TRAIN_MODE}-a100x4-$(date +%Y%m%d-%H%M%S)"}
export CKPTS_DIR=${CKPTS_DIR:-"$DATA_ROOT/ckpts/$PROJECT_NAME/$EXPERIMENT_NAME"}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","swanlab"]'}
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-"$VERL_ROOT/swanlog"}
LOG_DIR=${LOG_DIR:-"$VERL_ROOT/logs"}
mkdir -p -- "$LOG_DIR"
LOG_FILE=${LOG_FILE:-"$LOG_DIR/${EXPERIMENT_NAME}.log"}
mkdir -p -- "$(dirname -- "$LOG_FILE")"

# Start with 2048 generated tokens; increase to 4096/8192 after profiling.
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_N=${ROLLOUT_N:-5}
export VAL_N=${VAL_N:-1}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16}
export OFFLOAD=${OFFLOAD:-True}

printf 'Mode: %s\nExperiment: %s\nCheckpoint directory: %s\nConsole log: %s\n' \
    "$TRAIN_MODE" "$EXPERIMENT_NAME" "$CKPTS_DIR" "$LOG_FILE"

# Extra Hydra overrides are last, so callers can change steps, batches, etc.
# Keep the new worker path for current-policy PRL reference-logprob support.
bash "$SCRIPT_DIR/$TRAIN_SCRIPT" \
    "data.val_files=[\"$VAL_FILE\"]" \
    "data.train_batch_size=${TRAIN_BATCH_SIZE:-32}" \
    "data.val_max_samples=${VAL_MAX_SAMPLES:-32}" \
    data.filter_overlong_prompts_workers=4 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.use_legacy_worker_impl=disable \
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-3000}" \
    trainer.val_before_train=False \
    "trainer.test_freq=${TEST_FREQ:-20}" \
    "trainer.save_freq=${SAVE_FREQ:-100}" \
    "trainer.log_val_generations=${LOG_VAL_GENERATIONS:-2}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    "$@" 2>&1 | tee -a -- "$LOG_FILE"
