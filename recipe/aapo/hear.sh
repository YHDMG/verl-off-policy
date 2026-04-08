#!/usr/bin/env bash

set -xeuo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

export RAY_NAMESPACE=verl
export RAY_worker_register_timeout_seconds=120
export RAY_health_check_timeout_ms=60000
export RAY_health_check_period_ms=10000
export RAY_ignore_unhandled_errors=1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29517

export TORCH_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_SHM_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_TREE_THRESHOLD=0
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
export CUDA_LAUNCH_BLOCKING=0

export HYDRA_FULL_ERROR=1


# ================================ Project configuration ================================
project_name='off-policy-hear'
exp_name='hear-ds1.5b-ratio-0.2'

# ================================ Paths ================================
model_path='/home/cxy/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B'
train_file='/home/cxy/verl_async/dapodataset/train-00000-of-00001_converted_final.parquet'
test_file='/home/cxy/verl_async/DeepscalerDataset/converted/aime2025_converted_verl_fixed.parquet'
ckpts_dir="/mnt/data1/ckpts/${project_name}/${exp_name}"

# ================================ Resume configuration ================================
resume_mode='disable'
resume_from_path=''

# ================================ GPU / parallel configuration ================================
nnodes=1
n_gpus_per_node=4
gen_tp=1
sp_size=1
fsdp_size=4

# ================================ Algorithm parameters ================================
adv_estimator='grpo'
loss_mode='hear'
loss_agg_mode='token-mean'

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.001
kl_loss_type='low_var_kl'

clip_ratio_low=0.2
clip_ratio_high=0.28

# ================================ HEAR parameters ================================
high_entropy_ratio=0.2
high_entropy_last_epoch_enabled=True
enable_correction=True
correction_history_size=10
correction_beta=2.0
correction_lambda=2.0

enable_high_entropy_guard=True
# Guard coverage is measured on the selected HEAR token subset, not on all response tokens.
high_entropy_guard_min_ratio=0.95
high_entropy_guard_select_ratio=0.2
high_entropy_guard_max_iters=25
high_entropy_guard_low_step=0.01
high_entropy_guard_high_step=0.05
high_entropy_guard_lower_min=0.7
high_entropy_guard_upper_max=2.0

entropy_history_size=50
entropy_ema_beta=0.1
entropy_coeff=1e-4

# ================================ Off-policy replay parameters ================================
off_policy_enable=True
# `seq_mean_entropy` prioritizes lower-entropy sequences and can couple with HEAR's
# high-entropy protection. For attribution runs, compare both `seq_mean_entropy` and `seq_reward`.
off_policy_quality_metric='seq_mean_entropy'
off_policy_replay_mini_batch_multiplier=1
off_policy_replay_schedule_type='cosine_decay'
off_policy_replay_start_ratio=1.0
off_policy_replay_end_ratio=0.25
off_policy_replay_anneal_steps=0
off_policy_warmup_steps=4
off_policy_quality_alpha=0.5
off_policy_late_quality_alpha_multiplier=1.0
off_policy_staleness_horizon=4
off_policy_uniform_mix=0.3
off_policy_late_uniform_mix=0.3
off_policy_max_age_steps=4
off_policy_zero_adv_epsilon=1e-6
off_policy_cpu_offload=True
off_policy_capacity_steps=4
off_policy_enable_difficulty_sampling=True
off_policy_difficulty_metric='pass_rate'
off_policy_difficulty_pass_threshold=0.9
off_policy_difficulty_alpha=1.0
off_policy_difficulty_min_priority_scale=0.5
off_policy_difficulty_medium_lower=0.25
off_policy_difficulty_medium_upper=0.75

# ================================ Ablation presets ================================
# ppo_baseline: replay / correction / guard / last_epoch all disabled
# replay_only: only replay enabled
# hear_only: HEAR correction / guard / last_epoch enabled, replay disabled
# full_hear: replay + HEAR all enabled
ablation_mode='full_hear'

case "${ablation_mode}" in
  ppo_baseline)
    off_policy_enable=False
    enable_correction=False
    enable_high_entropy_guard=False
    high_entropy_last_epoch_enabled=False
    ;;
  replay_only)
    off_policy_enable=True
    enable_correction=False
    enable_high_entropy_guard=False
    high_entropy_last_epoch_enabled=False
    ;;
  hear_only)
    off_policy_enable=False
    enable_correction=True
    enable_high_entropy_guard=True
    high_entropy_last_epoch_enabled=True
    ;;
  full_hear)
    off_policy_enable=True
    enable_correction=True
    enable_high_entropy_guard=True
    high_entropy_last_epoch_enabled=True
    ;;
  *)
    echo "Unknown ablation_mode: ${ablation_mode}" >&2
    exit 1
    ;;
esac

# ================================ Response length parameters ================================
max_prompt_length=1024
max_response_length=$((1024 * 8))

# ================================ Sampling parameters ================================
temperature=1.0
top_p=1.0
top_k=-1
val_temperature=0.6
val_top_p=0.7
val_top_k=-1

# ================================ Performance related parameters ================================
use_dynamic_bsz=False
ref_offload=True
actor_offload=False
rollout_gpu_memory_utilization=0.8

# ================================ Batch parameters ================================
train_prompt_bsz=64
n_resp_per_prompt=8
ppo_mini_batch_size=16
ppo_micro_batch_size_per_gpu=4
ppo_epochs=2
rollout_log_prob_micro_batch_size_per_gpu=4
ref_log_prob_micro_batch_size_per_gpu=4

off_policy_capacity=$((train_prompt_bsz * n_resp_per_prompt * off_policy_capacity_steps))

# ================================ Training schedule ================================
test_freq=10
save_freq=50
save_best_checkpoint=True
total_epochs=1
total_training_steps=500
val_before_train=False

# ================================ Misc ================================
trainer_logger='["console","swanlab"]'
reward_manager='dapo'
validation_data_dir="${ckpts_dir}/validation_generations"

rollout_model_len=$((max_prompt_length + max_response_length))
actor_ppo_max_token_len=$((rollout_model_len * 2))
infer_ppo_max_token_len=$((rollout_model_len * 3))

python3 -m recipe.aapo.main_aapo \
    algorithm.adv_estimator="${adv_estimator}" \
    actor_rollout_ref.actor.policy_loss.loss_mode="${loss_mode}" \
    actor_rollout_ref.actor.loss_agg_mode="${loss_agg_mode}" \
    data.train_files="${train_file}" \
    data.val_files="${test_file}" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.filter_overlong_prompts=True \
    data.shuffle=False \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.train_batch_size="${train_prompt_bsz}" \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    algorithm.use_kl_in_reward="${use_kl_in_reward}" \
    algorithm.kl_ctrl.kl_coef="${kl_coef}" \
    algorithm.off_policy.enable="${off_policy_enable}" \
    algorithm.off_policy.capacity="${off_policy_capacity}" \
    algorithm.off_policy.warmup_steps="${off_policy_warmup_steps}" \
    algorithm.off_policy.replay_mini_batch_multiplier="${off_policy_replay_mini_batch_multiplier}" \
    algorithm.off_policy.replay_schedule_type="${off_policy_replay_schedule_type}" \
    algorithm.off_policy.replay_start_ratio="${off_policy_replay_start_ratio}" \
    algorithm.off_policy.replay_end_ratio="${off_policy_replay_end_ratio}" \
    algorithm.off_policy.replay_anneal_steps="${off_policy_replay_anneal_steps}" \
    algorithm.off_policy.quality_metric="${off_policy_quality_metric}" \
    algorithm.off_policy.quality_alpha="${off_policy_quality_alpha}" \
    algorithm.off_policy.late_quality_alpha_multiplier="${off_policy_late_quality_alpha_multiplier}" \
    algorithm.off_policy.staleness_horizon="${off_policy_staleness_horizon}" \
    algorithm.off_policy.uniform_mix="${off_policy_uniform_mix}" \
    algorithm.off_policy.late_uniform_mix="${off_policy_late_uniform_mix}" \
    algorithm.off_policy.max_age_steps="${off_policy_max_age_steps}" \
    algorithm.off_policy.zero_adv_epsilon="${off_policy_zero_adv_epsilon}" \
    algorithm.off_policy.cpu_offload="${off_policy_cpu_offload}" \
    algorithm.off_policy.enable_difficulty_sampling="${off_policy_enable_difficulty_sampling}" \
    algorithm.off_policy.difficulty_metric="${off_policy_difficulty_metric}" \
    algorithm.off_policy.difficulty_pass_threshold="${off_policy_difficulty_pass_threshold}" \
    algorithm.off_policy.difficulty_alpha="${off_policy_difficulty_alpha}" \
    algorithm.off_policy.difficulty_min_priority_scale="${off_policy_difficulty_min_priority_scale}" \
    algorithm.off_policy.difficulty_medium_lower="${off_policy_difficulty_medium_lower}" \
    algorithm.off_policy.difficulty_medium_upper="${off_policy_difficulty_medium_upper}" \
    actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
    actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
    actor_rollout_ref.actor.kl_loss_type="${kl_loss_type}" \
    actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
    actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
    actor_rollout_ref.actor.high_entropy_ratio="${high_entropy_ratio}" \
    actor_rollout_ref.actor.high_entropy_last_epoch_enabled="${high_entropy_last_epoch_enabled}" \
    actor_rollout_ref.actor.policy_loss.enable_correction="${enable_correction}" \
    actor_rollout_ref.actor.policy_loss.correction_history_size="${correction_history_size}" \
    actor_rollout_ref.actor.policy_loss.correction_beta="${correction_beta}" \
    actor_rollout_ref.actor.policy_loss.correction_lambda="${correction_lambda}" \
    actor_rollout_ref.actor.policy_loss.enable_high_entropy_guard="${enable_high_entropy_guard}" \
    actor_rollout_ref.actor.policy_loss.high_entropy_guard_min_ratio="${high_entropy_guard_min_ratio}" \
    actor_rollout_ref.actor.policy_loss.high_entropy_guard_select_ratio="${high_entropy_guard_select_ratio}" \
    actor_rollout_ref.actor.policy_loss.high_entropy_guard_max_iters="${high_entropy_guard_max_iters}" \
    actor_rollout_ref.actor.policy_loss.high_entropy_guard_low_step="${high_entropy_guard_low_step}" \
    actor_rollout_ref.actor.policy_loss.high_entropy_guard_high_step="${high_entropy_guard_high_step}" \
    actor_rollout_ref.actor.policy_loss.high_entropy_guard_lower_min="${high_entropy_guard_lower_min}" \
    actor_rollout_ref.actor.policy_loss.high_entropy_guard_upper_max="${high_entropy_guard_upper_max}" \
    actor_rollout_ref.actor.policy_loss.entropy_history_size="${entropy_history_size}" \
    actor_rollout_ref.actor.policy_loss.entropy_ema_beta="${entropy_ema_beta}" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_epochs="${ppo_epochs}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size_per_gpu}" \
    actor_rollout_ref.actor.entropy_coeff="${entropy_coeff}" \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload="${actor_offload}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${actor_offload}" \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${fsdp_size}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${sp_size}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_gpu_memory_utilization}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="${rollout_model_len}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.temperature="${temperature}" \
    actor_rollout_ref.rollout.top_p="${top_p}" \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${val_temperature}" \
    actor_rollout_ref.rollout.val_kwargs.n="${n_resp_per_prompt}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p="${val_top_p}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${val_top_k}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${rollout_log_prob_micro_batch_size_per_gpu}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ref_log_prob_micro_batch_size_per_gpu}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${ref_offload}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${sp_size}" \
    reward.reward_manager.name="${reward_manager}" \
    +reward.reward_kwargs.max_resp_len="${max_response_length}" \
    trainer.critic_warmup=0 \
    trainer.logger="${trainer_logger}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train="${val_before_train}" \
    trainer.validation_data_dir="${validation_data_dir}" \
    trainer.save_freq="${save_freq}" \
    trainer.save_best_checkpoint="${save_best_checkpoint}" \
    trainer.test_freq="${test_freq}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.total_training_steps="${total_training_steps}" \
    trainer.default_local_dir="${ckpts_dir}" \
    trainer.resume_mode="${resume_mode}" \
    ${resume_from_path:+trainer.resume_from_path="${resume_from_path}"} \
    trainer.nnodes="${nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    actor_rollout_ref.rollout.prompt_length="${max_prompt_length}" \
    actor_rollout_ref.rollout.response_length="${max_response_length}" \
    actor_rollout_ref.rollout.max_model_len="${rollout_model_len}" \
    "$@"
