#!/usr/bin/env bash
# GRPO with an explicit KL loss. See docs for the matching environment.

set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/setup_logging.sh"

project_name=${PROJECT_NAME:-rl-kl}
MODEL_NAME=Qwen3-1.7B-Base
MODEL_PATH=${MODEL_PATH:-Qwen/${MODEL_NAME}}
DATA_ROOT=${DATA_ROOT:-"~/rl-kl/verl"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/data/dapo_filter.parquet"}
TEST_FILE='["~/rl-kl/verl/data/math500/test.parquet"]'

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}

# KL type options include "kl"/"k1", "mse"/"k2", "low_var_kl"/"k3", "k3+", and "forward_kl"/"forwardKL".
use_kl_in_reward=False
kl_in_reward_type="forward_kl"
reward_kl_coef=1e-6

use_kl_loss=True
kl_in_loss_type="k3" # k3+, forward_kl
kl_loss_coef=1e-2
correct_reverseKL_is=False

ADV_ESTIMATOR=grpo
GAE_GAMMA=1.0
GAE_LAM=1.0
CRITIC_LR=1e-6

USE_CRITIC=False
if [ "$ADV_ESTIMATOR" == "gae" ]; then
    USE_CRITIC=True
fi

# Conservative defaults for Qwen3-30B-A3B-Base. Override from CLI/env if you have more headroom.
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}
ROLLOUT_N=${ROLLOUT_N:-5}
VAL_N=${VAL_N:-16}
OFFLOAD=${OFFLOAD:-True}

experiment_name=${MODEL_NAME}-dapo_filter-$ADV_ESTIMATOR-KLinLoss_${use_kl_loss}-KLlossCoef${kl_loss_coef}-KLtype_${kl_in_loss_type}-AddIS${correct_reverseKL_is}-useKLInReward_${use_kl_in_reward}-rewardKLCoef${reward_kl_coef}-KLtype_${kl_in_reward_type}
experiment_name=${EXPERIMENT_NAME:-$experiment_name}
CKPTS_DIR=${CKPTS_DIR:-"${DATA_ROOT}/ckpts/${project_name}/${experiment_name}"}

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.gamma=$GAE_GAMMA \
    algorithm.lam=$GAE_LAM \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=256 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.filter_overlong_prompts_workers=64 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_in_loss_type \
    actor_rollout_ref.actor.correct_reverseKL_is=$correct_reverseKL_is \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$reward_kl_coef \
    algorithm.kl_penalty=$kl_in_reward_type \
    critic.enable=$USE_CRITIC \
    critic.model.path="${MODEL_PATH}" \
    critic.optim.lr=$CRITIC_LR \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.forward_micro_batch_size_per_gpu=2 \
    critic.fsdp.param_offload=False \
    critic.fsdp.optimizer_offload=False \
    trainer.critic_warmup=0 \
    "trainer.logger=${TRAINER_LOGGER}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    trainer.val_before_train=True \
    trainer.total_training_steps=3000 \
    "$@"
