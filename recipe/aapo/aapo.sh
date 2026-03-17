#!/usr/bin/env bash

set -xeuo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# This script is intended for a quick server-side smoke test of GRPO with
# recipe-local off-policy replay enabled.


# ================================ Project configuration ================================
project_name='off-policy-grpo'
exp_name='grpo-qwen3-1.7b-buffer-test'

# ================================ Paths ================================
model_path='/home/cxy/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B'
train_file='/home/cxy/verl_async/dapodataset/train-00000-of-00001_converted_final.parquet'
test_file='/home/cxy/verl_async/DeepscalerDataset/converted/aime2025_converted_verl_fixed.parquet'
ckpts_dir="/home/cxy/ckpts/${project_name}/${exp_name}"

# ================================ Resume configuration ================================
resume_mode='disable'
resume_from_path=''

# ================================ GPU / parallel configuration ================================
# For dual-socket / dual-NUMA 8-GPU machines, prefer one NUMA island first:
#   CUDA_VISIBLE_DEVICES=0,1,2,3  or  CUDA_VISIBLE_DEVICES=4,5,6,7
nnodes=1
n_gpus_per_node=8
gen_tp=1
sp_size=1
fsdp_size=8

# ================================ Algorithm parameters ================================
adv_estimator='grpo'
loss_mode='vanilla'
loss_agg_mode='token-mean'

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.001
kl_loss_type='low_var_kl'

clip_ratio_low=0.2
clip_ratio_high=0.28

# ================================ Off-policy replay parameters ================================
# Start with a conservative replay setting so we can first validate that replay
# metrics move in the expected direction without destabilizing training.
off_policy_enable=True
off_policy_quality_metric='seq_reward'
off_policy_replay_mini_batch_multiplier=1
off_policy_warmup_steps=20
off_policy_quality_alpha=0.5
off_policy_staleness_horizon=64
off_policy_uniform_mix=0.3
off_policy_max_age_steps=128
off_policy_cpu_offload=True
off_policy_capacity_steps=128

# ================================ Response length parameters ================================
max_prompt_length=1024
max_response_length=$((1024 * 7))

# ================================ Sampling parameters ================================
temperature=1.0
top_p=1.0
top_k=-1
val_temperature=1.0
val_top_p=0.7
val_top_k=-1

# ================================ Performance related parameters ================================
use_dynamic_bsz=True
ref_offload=True
actor_offload=False
rollout_gpu_memory_utilization=0.8

# ================================ Batch parameters ================================
train_prompt_bsz=64
n_resp_per_prompt=8
ppo_mini_batch_size=16
ppo_micro_batch_size_per_gpu=4
ref_log_prob_micro_batch_size_per_gpu=4

# Keep roughly `off_policy_capacity_steps` historical training steps in replay.
off_policy_capacity=$((train_prompt_bsz * n_resp_per_prompt * off_policy_capacity_steps))

# ================================ Training schedule ================================
test_freq=10
save_freq=-1
total_epochs=1
total_training_steps=500
val_before_train=True

# ================================ Misc ================================
trainer_logger='["console","swanlab"]'
reward_manager='dapo'

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
    algorithm.off_policy.quality_metric="${off_policy_quality_metric}" \
    algorithm.off_policy.quality_alpha="${off_policy_quality_alpha}" \
    algorithm.off_policy.staleness_horizon="${off_policy_staleness_horizon}" \
    algorithm.off_policy.uniform_mix="${off_policy_uniform_mix}" \
    algorithm.off_policy.max_age_steps="${off_policy_max_age_steps}" \
    algorithm.off_policy.cpu_offload="${off_policy_cpu_offload}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
    actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
    actor_rollout_ref.actor.kl_loss_type="${kl_loss_type}" \
    actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
    actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
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
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size_per_gpu}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
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
    trainer.save_freq="${save_freq}" \
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
