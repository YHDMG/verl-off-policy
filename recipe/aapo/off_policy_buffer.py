# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.config.algorithm import OffPolicyConfig
import verl.utils.torch_functional as verl_F

OFF_POLICY_BATCH_KEYS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "prompts",
    "responses",
    "response_mask",
    "old_log_probs",
    "advantages",
    "returns",
    "values",
    "token_level_scores",
    "token_level_rewards",
)
OFF_POLICY_OPTIONAL_BATCH_KEYS = (
    "rollout_log_probs",
    "rollout_is_weights",
    "ref_log_prob",
    "rm_scores",
    "dummy_tensor",
    "seq_mean_entropy",
)
OFF_POLICY_NON_TENSOR_KEYS = ("uid", "multi_modal_inputs")


def compute_off_policy_warmup_size(config: OffPolicyConfig, train_batch_size: int, rollout_n: int) -> int:
    raw_warmup_size = max(int(config.warmup_steps), 0) * int(train_batch_size) * int(rollout_n)
    return min(raw_warmup_size, int(config.capacity))


def compute_replay_batch_size(config: OffPolicyConfig, actor_mini_batch_size: int, rollout_n: int) -> int:
    return max(int(config.replay_mini_batch_multiplier), 0) * int(actor_mini_batch_size) * int(rollout_n)


@dataclass
class OffPolicySampleMetrics:
    metrics: dict[str, float]
    batch: Optional[DataProto]


@dataclass
class OffPolicyAddMetrics:
    metrics: dict[str, float]
    batch: Optional[DataProto]


class OffPolicyReplayBuffer:
    """Ring buffer for replaying past online rollout samples in synchronous PPO training."""

    def __init__(self, config: OffPolicyConfig):
        self.config = config
        self.capacity = int(config.capacity)
        if self.capacity <= 0:
            raise ValueError(f"OffPolicyReplayBuffer capacity must be positive, got {self.capacity}")
        supported_quality_metrics = {"seq_reward", "seq_mean_entropy"}
        if config.quality_metric not in supported_quality_metrics:
            raise ValueError(
                "OffPolicyReplayBuffer v1 only supports quality_metric in "
                f"{supported_quality_metrics}. Got {config.quality_metric}."
            )
        supported_difficulty_metrics = {"group_relative_dispersion", "pass_rate"}
        if config.difficulty_metric not in supported_difficulty_metrics:
            raise ValueError(
                "OffPolicyReplayBuffer difficulty sampling only supports "
                f"{supported_difficulty_metrics}. Got {config.difficulty_metric}."
            )

        self.pool: Optional[TensorDict] = None
        self.non_tensor_pool: dict[str, np.ndarray] = {}
        self.size = 0
        self.position = 0
        self.storage_device = torch.device("cpu") if config.cpu_offload else None

        self.source_steps = np.full(self.capacity, -1, dtype=np.int64)
        self.seq_rewards = np.zeros(self.capacity, dtype=np.float32)
        self.seq_mean_entropy = np.zeros(self.capacity, dtype=np.float32)
        self.quality_z = np.zeros(self.capacity, dtype=np.float32)
        self.seq_lens = np.zeros(self.capacity, dtype=np.int32)
        self.prompt_uids = np.empty(self.capacity, dtype=object)
        self.prompt_success_rates = np.full(self.capacity, 0.5, dtype=np.float32)
        self.prompt_informativeness = np.ones(self.capacity, dtype=np.float32)
        self.prompt_reward_spans = np.zeros(self.capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def empty_metrics(self, requested_batch_size: int) -> dict[str, float]:
        return self._empty_metrics(requested_batch_size)

    def add_batch(self, batch: DataProto, global_step: int) -> dict[str, float]:
        selected = self._select_replay_fields(batch)
        filtered_output = self._filter_zero_adv_groups(selected)
        metrics = filtered_output.metrics
        selected = filtered_output.batch
        if selected is None or len(selected) == 0:
            return metrics

        if self.pool is None:
            self._lazy_init(selected)

        total_selected = len(selected)
        block_size = min(total_selected, self.capacity)
        source_start = total_selected - block_size
        insert_indices = (self.position + np.arange(block_size)) % self.capacity
        insert_index_tensor = torch.as_tensor(insert_indices, dtype=torch.long, device=self.pool.device)

        seq_reward = selected.batch["token_level_scores"].sum(dim=-1).detach().to(torch.float32).cpu()
        seq_mean_entropy = self._extract_seq_mean_entropy(selected)
        response_mask = selected.batch["response_mask"]
        seq_len = response_mask.sum(dim=-1).detach().to(torch.int32).cpu().numpy()
        uid_array = selected.non_tensor_batch.get("uid")
        prompt_success_rate, prompt_informativeness, prompt_reward_span = self._compute_prompt_difficulty_features(
            seq_reward, uid_array
        )
        quality_values, lower_is_better = self._compute_quality_values(
            seq_reward=seq_reward, seq_mean_entropy=seq_mean_entropy
        )
        quality_z = self._compute_quality_z(quality_values, lower_is_better=lower_is_better)

        for key, value in selected.batch.items():
            source_value = value[source_start : source_start + block_size].detach()
            target_value = source_value.to(self.pool.device)
            self.pool[key].index_copy_(0, insert_index_tensor, target_value)

        for key, value in selected.non_tensor_batch.items():
            self.non_tensor_pool[key][insert_indices] = value[source_start : source_start + block_size]

        if uid_array is not None:
            self.prompt_uids[insert_indices] = uid_array[source_start : source_start + block_size]
        else:
            self.prompt_uids[insert_indices] = ""

        self.source_steps[insert_indices] = int(global_step)
        self.seq_rewards[insert_indices] = seq_reward[source_start : source_start + block_size].numpy()
        self.seq_mean_entropy[insert_indices] = seq_mean_entropy[source_start : source_start + block_size].numpy()
        self.quality_z[insert_indices] = quality_z[source_start : source_start + block_size]
        self.seq_lens[insert_indices] = seq_len[source_start : source_start + block_size]
        self.prompt_success_rates[insert_indices] = prompt_success_rate[source_start : source_start + block_size]
        self.prompt_informativeness[insert_indices] = prompt_informativeness[source_start : source_start + block_size]
        self.prompt_reward_spans[insert_indices] = prompt_reward_span[source_start : source_start + block_size]

        self.position = (self.position + block_size) % self.capacity
        self.size = min(self.size + block_size, self.capacity)
        return metrics

    def sample_batch(
        self,
        batch_size: int,
        current_step: int,
        quality_alpha: Optional[float] = None,
        uniform_mix: Optional[float] = None,
        bias_beta: Optional[float] = None,
        bias_weight_clip: Optional[float] = None,
    ) -> OffPolicySampleMetrics:
        metrics = self._empty_metrics(batch_size)
        if batch_size <= 0 or self.size == 0:
            return OffPolicySampleMetrics(metrics=metrics, batch=None)

        valid_indices = self._get_valid_indices(current_step)
        if valid_indices.size == 0 or valid_indices.size < batch_size:
            metrics["off_policy/replay_hit_rate"] = valid_indices.size / float(max(batch_size, 1))
            return OffPolicySampleMetrics(metrics=metrics, batch=None)

        ages = current_step - self.source_steps[valid_indices]
        priorities = self._compute_priorities(valid_indices, ages, quality_alpha=quality_alpha)
        probabilities = self._mix_uniform(priorities, uniform_mix=uniform_mix)

        sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=False, p=probabilities)
        sample_positions = np.searchsorted(valid_indices, sampled_indices)
        sampled_ages = ages[sample_positions]
        sampled_probabilities = probabilities[sample_positions]

        batch = self._gather_batch(sampled_indices)
        replay_sampling_weight = self._compute_replay_sampling_weight(
            sampled_probabilities=sampled_probabilities,
            valid_count=valid_indices.size,
            bias_beta=bias_beta,
            bias_weight_clip=bias_weight_clip,
            device=batch.batch.device,
        )
        batch = batch.union(DataProto.from_dict(tensors={"replay_sampling_weight": replay_sampling_weight}))
        metrics.update(self._compute_sample_health_metrics(batch))
        metrics.update(self._compute_sample_difficulty_metrics(sampled_indices))
        metrics.update(
            {
                "off_policy/buffer_size": float(self.size),
                "off_policy/replay_batch_size": float(batch_size),
                "off_policy/replay_fraction": 0.0,
                "off_policy/sample_age_mean": float(sampled_ages.mean()),
                "off_policy/sample_age_max": float(sampled_ages.max()),
                "off_policy/seq_reward_mean": float(self.seq_rewards[sampled_indices].mean()),
                "off_policy/sample_seq_mean_entropy_mean": float(self.seq_mean_entropy[sampled_indices].mean()),
                "off_policy/priority_entropy": self._entropy(probabilities),
                "off_policy/replay_hit_rate": 1.0,
                "off_policy/sample_probability_mean": float(sampled_probabilities.mean()),
                "off_policy/sample_replay_weight_mean": float(replay_sampling_weight.mean().item()),
                "off_policy/sample_replay_weight_std": float(replay_sampling_weight.std(unbiased=False).item())
                if replay_sampling_weight.numel() > 1
                else 0.0,
                "off_policy/sample_replay_weight_max": float(replay_sampling_weight.max().item()),
            }
        )
        return OffPolicySampleMetrics(metrics=metrics, batch=batch)

    def _select_replay_fields(self, batch: DataProto) -> DataProto:
        batch_keys = [key for key in OFF_POLICY_BATCH_KEYS if key in batch.batch.keys()]
        batch_keys.extend(key for key in OFF_POLICY_OPTIONAL_BATCH_KEYS if key in batch.batch.keys())
        non_tensor_keys = [key for key in OFF_POLICY_NON_TENSOR_KEYS if key in batch.non_tensor_batch]
        return batch.select(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_keys, deepcopy=True)

    def _filter_zero_adv_groups(self, batch: DataProto) -> OffPolicyAddMetrics:
        total_sequences = len(batch)
        metrics = {
            "off_policy/skipped_zero_adv_groups": 0.0,
            "off_policy/skipped_zero_adv_sequences": 0.0,
            "off_policy/accepted_sequences": float(total_sequences),
            "off_policy/accepted_ratio": 1.0 if total_sequences > 0 else 0.0,
        }
        if total_sequences == 0:
            return OffPolicyAddMetrics(metrics=metrics, batch=None)
        if "advantages" not in batch.batch.keys() or "response_mask" not in batch.batch.keys():
            return OffPolicyAddMetrics(metrics=metrics, batch=batch)

        advantages = batch.batch["advantages"].detach().to(torch.float32)
        response_mask = batch.batch["response_mask"].detach().to(torch.float32)
        seq_adv_abs_max = (advantages.abs() * response_mask).amax(dim=-1).cpu().numpy()

        uid_values = batch.non_tensor_batch.get("uid")
        if uid_values is None:
            uid_values = np.arange(total_sequences, dtype=np.int64)

        group_to_indices: dict[object, list[int]] = {}
        for idx, uid in enumerate(uid_values):
            group_key = self._normalize_group_key(uid)
            group_to_indices.setdefault(group_key, []).append(idx)

        accepted_mask = np.ones(total_sequences, dtype=bool)
        skipped_groups = 0
        skipped_sequences = 0
        zero_adv_epsilon = float(self.config.zero_adv_epsilon)

        for indices in group_to_indices.values():
            group_adv_abs_max = seq_adv_abs_max[indices]
            if np.all(group_adv_abs_max <= zero_adv_epsilon):
                accepted_mask[np.asarray(indices, dtype=np.int64)] = False
                skipped_groups += 1
                skipped_sequences += len(indices)

        accepted_indices = np.flatnonzero(accepted_mask)
        metrics.update(
            {
                "off_policy/skipped_zero_adv_groups": float(skipped_groups),
                "off_policy/skipped_zero_adv_sequences": float(skipped_sequences),
                "off_policy/accepted_sequences": float(accepted_indices.size),
                "off_policy/accepted_ratio": accepted_indices.size / float(max(total_sequences, 1)),
            }
        )
        if accepted_indices.size == 0:
            return OffPolicyAddMetrics(metrics=metrics, batch=None)

        filtered_batch = batch.select_idxs(torch.as_tensor(accepted_indices, dtype=torch.long))
        return OffPolicyAddMetrics(metrics=metrics, batch=filtered_batch)

    def _lazy_init(self, sample: DataProto) -> None:
        device = self.storage_device if self.storage_device is not None else next(iter(sample.batch.values())).device
        self.pool = TensorDict(
            {
                key: torch.zeros((self.capacity, *value.shape[1:]), dtype=value.dtype, device=device)
                for key, value in sample.batch.items()
            },
            batch_size=[self.capacity],
            device=device,
        )
        for key, value in sample.non_tensor_batch.items():
            self.non_tensor_pool[key] = np.empty((self.capacity, *value.shape[1:]), dtype=object)

    def _get_valid_indices(self, current_step: int) -> np.ndarray:
        if self.size == self.capacity:
            candidates = np.arange(self.capacity)
        else:
            candidates = np.arange(self.size)

        age_mask = np.ones_like(candidates, dtype=bool)
        if int(self.config.max_age_steps) > 0:
            ages = current_step - self.source_steps[candidates]
            age_mask = ages <= int(self.config.max_age_steps)

        filled_mask = self.source_steps[candidates] >= 0
        return candidates[filled_mask & age_mask]

    def _compute_priorities(
        self,
        valid_indices: np.ndarray,
        ages: np.ndarray,
        quality_alpha: Optional[float] = None,
    ) -> np.ndarray:
        if quality_alpha is None:
            quality_alpha = float(self.config.quality_alpha)
        quality_term = np.exp(float(quality_alpha) * self.quality_z[valid_indices])
        difficulty_term = self._compute_difficulty_priority_term(valid_indices)
        if int(self.config.staleness_horizon) > 0:
            staleness_term = np.exp(-ages / float(self.config.staleness_horizon))
        else:
            staleness_term = np.ones_like(quality_term)
        priorities = quality_term * difficulty_term * staleness_term
        priorities = np.clip(priorities, a_min=1e-8, a_max=None)
        return priorities.astype(np.float64, copy=False)

    def _mix_uniform(self, priorities: np.ndarray, uniform_mix: Optional[float] = None) -> np.ndarray:
        normalized = priorities / priorities.sum()
        if uniform_mix is None:
            uniform_mix = float(self.config.uniform_mix)
        if uniform_mix <= 0:
            return normalized
        uniform = np.full_like(normalized, 1.0 / normalized.size, dtype=np.float64)
        return (1.0 - uniform_mix) * normalized + uniform_mix * uniform

    def _gather_batch(self, indices: np.ndarray) -> DataProto:
        assert self.pool is not None
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.pool.device)
        tensor_batch = {
            key: value.index_select(0, index_tensor).clone()
            for key, value in self.pool.items()
        }
        non_tensor_batch = {
            key: value[indices].copy()
            for key, value in self.non_tensor_pool.items()
        }
        return DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch)

    def _compute_replay_sampling_weight(
        self,
        sampled_probabilities: np.ndarray,
        valid_count: int,
        bias_beta: Optional[float],
        bias_weight_clip: Optional[float],
        device: torch.device,
    ) -> torch.Tensor:
        if bias_beta is None:
            bias_beta = float(self.config.replay_bias_beta_start)
        if bias_weight_clip is None:
            bias_weight_clip = float(self.config.replay_bias_weight_clip)

        sampled_probabilities = np.clip(sampled_probabilities.astype(np.float64, copy=False), a_min=1e-12, a_max=None)
        valid_count = max(int(valid_count), 1)
        uniform_probability = 1.0 / float(valid_count)
        weights = np.power(uniform_probability / sampled_probabilities, float(bias_beta)).astype(np.float64, copy=False)
        weights = np.clip(weights, a_min=0.0, a_max=float(bias_weight_clip))
        weights = weights / max(weights.mean(), 1e-12)
        return torch.as_tensor(weights, dtype=torch.float32, device=device)

    def _compute_sample_health_metrics(self, batch: DataProto) -> dict[str, float]:
        metrics = {
            "off_policy/sample_rollout_is_eff_sample_size": 1.0,
            "off_policy/sample_rollout_is_seq_abs_mean_deviation": 0.0,
            "off_policy/sample_rollout_is_seq_max_deviation": 0.0,
        }
        if "rollout_is_weights" not in batch.batch.keys() or "response_mask" not in batch.batch.keys():
            return metrics

        response_mask = batch.batch["response_mask"]
        seq_mask = response_mask.sum(dim=-1) > 0
        if not torch.any(seq_mask):
            return metrics

        seq_mean_weight = verl_F.masked_mean(batch.batch["rollout_is_weights"], response_mask, axis=-1)[seq_mask]
        if seq_mean_weight.numel() == 0:
            return metrics

        mean_weight = seq_mean_weight.mean()
        if mean_weight.abs().item() <= 1e-8:
            ess = 0.0
        else:
            normalized_weight = seq_mean_weight / mean_weight
            ess = float((1.0 / normalized_weight.square().mean()).item())

        seq_deviation = (seq_mean_weight - 1.0).abs()
        metrics["off_policy/sample_rollout_is_eff_sample_size"] = ess
        metrics["off_policy/sample_rollout_is_seq_abs_mean_deviation"] = float(seq_deviation.mean().item())
        metrics["off_policy/sample_rollout_is_seq_max_deviation"] = float(seq_deviation.max().item())
        return metrics

    def _compute_quality_values(
        self,
        seq_reward: torch.Tensor,
        seq_mean_entropy: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        if self.config.quality_metric == "seq_mean_entropy":
            return seq_mean_entropy, True
        return seq_reward, False

    def _extract_seq_mean_entropy(self, selected: DataProto) -> torch.Tensor:
        if "seq_mean_entropy" in selected.batch.keys():
            return selected.batch["seq_mean_entropy"].detach().to(torch.float32).cpu().reshape(-1)

        if self.config.quality_metric == "seq_mean_entropy":
            raise ValueError(
                "quality_metric='seq_mean_entropy' requires seq_mean_entropy in the replay batch, "
                "but the field is missing."
            )
        return torch.zeros(len(selected), dtype=torch.float32)

    def _compute_quality_z(self, quality_values: torch.Tensor, lower_is_better: bool = False) -> np.ndarray:
        quality_np = quality_values.numpy().astype(np.float32, copy=False)
        if lower_is_better:
            quality_np = -quality_np
        if quality_np.size <= 1:
            return np.zeros_like(quality_np, dtype=np.float32)

        quality_mean = quality_np.mean()
        quality_std = quality_np.std()
        if quality_std < 1e-6:
            z = np.zeros_like(quality_np, dtype=np.float32)
        else:
            z = (quality_np - quality_mean) / quality_std
        return np.clip(z, -2.0, 2.0).astype(np.float32, copy=False)

    def _compute_prompt_difficulty_features(
        self,
        seq_reward: torch.Tensor,
        uid_values: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.config.difficulty_metric == "pass_rate":
            return self._compute_prompt_pass_rate_features(seq_reward, uid_values)
        return self._compute_prompt_dispersion_features(seq_reward, uid_values)

    def _compute_prompt_pass_rate_features(
        self,
        seq_reward: torch.Tensor,
        uid_values: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reward_np = seq_reward.numpy().astype(np.float32, copy=False)
        total_sequences = reward_np.size
        if total_sequences == 0:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )

        prompt_success_rate = np.full(total_sequences, 0.5, dtype=np.float32)
        prompt_informativeness = np.ones(total_sequences, dtype=np.float32)
        prompt_reward_span = np.zeros(total_sequences, dtype=np.float32)
        if uid_values is None:
            return prompt_success_rate, prompt_informativeness, prompt_reward_span

        success_mask = reward_np > float(self.config.difficulty_pass_threshold)
        group_to_indices: dict[object, list[int]] = {}
        for idx, uid in enumerate(uid_values):
            group_key = self._normalize_group_key(uid)
            group_to_indices.setdefault(group_key, []).append(idx)

        for indices in group_to_indices.values():
            if len(indices) <= 1:
                continue
            group_indices = np.asarray(indices, dtype=np.int64)
            group_success_rate = float(success_mask[group_indices].mean())
            group_informativeness = float(np.clip(4.0 * group_success_rate * (1.0 - group_success_rate), 0.0, 1.0))
            prompt_success_rate[group_indices] = group_success_rate
            prompt_informativeness[group_indices] = group_informativeness
            prompt_reward_span[group_indices] = float(reward_np[group_indices].max() - reward_np[group_indices].min())

        return prompt_success_rate, prompt_informativeness, prompt_reward_span

    def _compute_prompt_dispersion_features(
        self,
        seq_reward: torch.Tensor,
        uid_values: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reward_np = seq_reward.numpy().astype(np.float32, copy=False)
        total_sequences = reward_np.size
        if total_sequences == 0:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )

        prompt_success_rate = np.full(total_sequences, 0.5, dtype=np.float32)
        prompt_informativeness = np.ones(total_sequences, dtype=np.float32)
        prompt_reward_span = np.zeros(total_sequences, dtype=np.float32)
        if uid_values is None:
            return prompt_success_rate, prompt_informativeness, prompt_reward_span

        group_to_indices: dict[object, list[int]] = {}
        for idx, uid in enumerate(uid_values):
            group_key = self._normalize_group_key(uid)
            group_to_indices.setdefault(group_key, []).append(idx)

        for indices in group_to_indices.values():
            if len(indices) <= 1:
                continue

            group_indices = np.asarray(indices, dtype=np.int64)
            group_rewards = reward_np[group_indices]
            reward_min = float(group_rewards.min())
            reward_max = float(group_rewards.max())
            reward_span = reward_max - reward_min
            prompt_reward_span[group_indices] = reward_span
            if reward_span <= 1e-6:
                prompt_informativeness[group_indices] = 0.0
                continue

            normalized_rewards = (group_rewards - reward_min) / reward_span
            pairwise_distance = np.abs(normalized_rewards[:, None] - normalized_rewards[None, :]).mean()
            group_informativeness = float(np.clip(2.0 * pairwise_distance, 0.0, 1.0))
            prompt_informativeness[group_indices] = group_informativeness

        return prompt_success_rate, prompt_informativeness, prompt_reward_span

    def _compute_difficulty_priority_term(self, valid_indices: np.ndarray) -> np.ndarray:
        if not bool(self.config.enable_difficulty_sampling):
            return np.ones(valid_indices.shape[0], dtype=np.float64)

        difficulty_alpha = max(float(self.config.difficulty_alpha), 0.0)
        min_priority_scale = float(np.clip(self.config.difficulty_min_priority_scale, 0.0, 1.0))
        informativeness = np.nan_to_num(
            self.prompt_informativeness[valid_indices].astype(np.float64, copy=False),
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        )
        informativeness = np.clip(informativeness, 0.0, 1.0)
        if difficulty_alpha != 1.0:
            informativeness = np.power(informativeness, difficulty_alpha).astype(np.float64, copy=False)
        return min_priority_scale + (1.0 - min_priority_scale) * informativeness

    def _compute_sample_difficulty_metrics(self, sampled_indices: np.ndarray) -> dict[str, float]:
        metrics = {
            "off_policy/sample_prompt_success_rate_mean": 0.0,
            "off_policy/sample_prompt_informativeness_mean": 0.0,
            "off_policy/sample_prompt_reward_span_mean": 0.0,
            "off_policy/sample_prompt_easy_fraction": 0.0,
            "off_policy/sample_prompt_medium_fraction": 0.0,
            "off_policy/sample_prompt_hard_fraction": 0.0,
        }
        if sampled_indices.size == 0:
            return metrics

        lower_bound, upper_bound = self._get_prompt_difficulty_bucket_bounds()
        sampled_success_rates = self.prompt_success_rates[sampled_indices].astype(np.float64, copy=False)
        sampled_informativeness = self.prompt_informativeness[sampled_indices].astype(np.float64, copy=False)
        sampled_reward_spans = self.prompt_reward_spans[sampled_indices].astype(np.float64, copy=False)
        if self.config.difficulty_metric == "pass_rate":
            hard_mask = sampled_success_rates <= lower_bound
            easy_mask = sampled_success_rates >= upper_bound
            medium_mask = ~(hard_mask | easy_mask)
        else:
            hard_mask = sampled_informativeness <= lower_bound
            easy_mask = sampled_informativeness >= upper_bound
            medium_mask = ~(hard_mask | easy_mask)

        metrics["off_policy/sample_prompt_success_rate_mean"] = float(sampled_success_rates.mean())
        metrics["off_policy/sample_prompt_informativeness_mean"] = float(sampled_informativeness.mean())
        metrics["off_policy/sample_prompt_reward_span_mean"] = float(sampled_reward_spans.mean())
        metrics["off_policy/sample_prompt_easy_fraction"] = float(easy_mask.mean())
        metrics["off_policy/sample_prompt_medium_fraction"] = float(medium_mask.mean())
        metrics["off_policy/sample_prompt_hard_fraction"] = float(hard_mask.mean())
        return metrics

    def _get_prompt_difficulty_bucket_bounds(self) -> tuple[float, float]:
        lower_bound = float(np.clip(self.config.difficulty_medium_lower, 0.0, 1.0))
        upper_bound = float(np.clip(self.config.difficulty_medium_upper, 0.0, 1.0))
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound
        return lower_bound, upper_bound

    def _empty_metrics(self, requested_batch_size: int) -> dict[str, float]:
        return {
            "off_policy/buffer_size": float(self.size),
            "off_policy/replay_batch_size": 0.0,
            "off_policy/replay_fraction": 0.0,
            "off_policy/sample_age_mean": 0.0,
            "off_policy/sample_age_max": 0.0,
            "off_policy/seq_reward_mean": 0.0,
            "off_policy/sample_seq_mean_entropy_mean": 0.0,
            "off_policy/priority_entropy": 0.0,
            "off_policy/replay_hit_rate": 0.0 if requested_batch_size > 0 else 1.0,
            "off_policy/sample_probability_mean": 0.0,
            "off_policy/sample_replay_weight_mean": 0.0,
            "off_policy/sample_replay_weight_std": 0.0,
            "off_policy/sample_replay_weight_max": 0.0,
            "off_policy/sample_rollout_is_eff_sample_size": 0.0,
            "off_policy/sample_rollout_is_seq_abs_mean_deviation": 0.0,
            "off_policy/sample_rollout_is_seq_max_deviation": 0.0,
            "off_policy/sample_prompt_success_rate_mean": 0.0,
            "off_policy/sample_prompt_informativeness_mean": 0.0,
            "off_policy/sample_prompt_reward_span_mean": 0.0,
            "off_policy/sample_prompt_easy_fraction": 0.0,
            "off_policy/sample_prompt_medium_fraction": 0.0,
            "off_policy/sample_prompt_hard_fraction": 0.0,
        }

    @staticmethod
    def _entropy(probabilities: np.ndarray) -> float:
        safe_prob = np.clip(probabilities, a_min=1e-12, a_max=None)
        return float(-(safe_prob * np.log(safe_prob)).sum())

    @staticmethod
    def _normalize_group_key(uid: object) -> object:
        if isinstance(uid, np.ndarray):
            if uid.ndim == 0:
                return uid.item()
            return tuple(uid.tolist())
        if isinstance(uid, np.generic):
            return uid.item()
        if isinstance(uid, list):
            return tuple(uid)
        return uid
