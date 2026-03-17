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

OFF_POLICY_BATCH_KEYS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "responses",
    "response_mask",
    "old_log_probs",
    "advantages",
    "token_level_scores",
    "token_level_rewards",
)
OFF_POLICY_OPTIONAL_BATCH_KEYS = ("rollout_log_probs", "ref_log_prob")
OFF_POLICY_NON_TENSOR_KEYS = ("uid", "multi_modal_inputs")


def compute_off_policy_warmup_size(config: OffPolicyConfig, train_batch_size: int, rollout_n: int) -> int:
    return max(int(config.warmup_steps), 0) * int(train_batch_size) * int(rollout_n)


def compute_replay_batch_size(config: OffPolicyConfig, actor_mini_batch_size: int, rollout_n: int) -> int:
    return max(int(config.replay_mini_batch_multiplier), 0) * int(actor_mini_batch_size) * int(rollout_n)


@dataclass
class OffPolicySampleMetrics:
    metrics: dict[str, float]
    batch: Optional[DataProto]


class OffPolicyReplayBuffer:
    """Ring buffer for replaying past online rollout samples in synchronous PPO training."""

    def __init__(self, config: OffPolicyConfig):
        self.config = config
        self.capacity = int(config.capacity)
        if self.capacity <= 0:
            raise ValueError(f"OffPolicyReplayBuffer capacity must be positive, got {self.capacity}")
        if config.quality_metric != "seq_reward":
            raise ValueError(
                "OffPolicyReplayBuffer v1 only supports quality_metric='seq_reward'. "
                f"Got {config.quality_metric}."
            )

        self.pool: Optional[TensorDict] = None
        self.non_tensor_pool: dict[str, np.ndarray] = {}
        self.size = 0
        self.position = 0
        self.storage_device = torch.device("cpu") if config.cpu_offload else None

        self.source_steps = np.full(self.capacity, -1, dtype=np.int64)
        self.seq_rewards = np.zeros(self.capacity, dtype=np.float32)
        self.quality_z = np.zeros(self.capacity, dtype=np.float32)
        self.seq_lens = np.zeros(self.capacity, dtype=np.int32)
        self.prompt_uids = np.empty(self.capacity, dtype=object)

    def __len__(self) -> int:
        return self.size

    def empty_metrics(self, requested_batch_size: int) -> dict[str, float]:
        return self._empty_metrics(requested_batch_size)

    def add_batch(self, batch: DataProto, global_step: int) -> None:
        selected = self._select_replay_fields(batch)
        if len(selected) == 0:
            return

        if self.pool is None:
            self._lazy_init(selected)

        block_size = min(len(selected), self.capacity)
        insert_indices = (self.position + np.arange(block_size)) % self.capacity
        insert_index_tensor = torch.as_tensor(insert_indices, dtype=torch.long, device=self.pool.device)

        seq_reward = selected.batch["token_level_scores"].sum(dim=-1).detach().to(torch.float32).cpu()
        response_mask = selected.batch["response_mask"]
        seq_len = response_mask.sum(dim=-1).detach().to(torch.int32).cpu().numpy()
        quality_z = self._compute_quality_z(seq_reward)

        for key, value in selected.batch.items():
            source_value = value[:block_size].detach()
            target_value = source_value.to(self.pool.device)
            self.pool[key].index_copy_(0, insert_index_tensor, target_value)

        for key, value in selected.non_tensor_batch.items():
            self.non_tensor_pool[key][insert_indices] = value[:block_size]

        uid_array = selected.non_tensor_batch.get("uid")
        if uid_array is not None:
            self.prompt_uids[insert_indices] = uid_array[:block_size]
        else:
            self.prompt_uids[insert_indices] = ""

        self.source_steps[insert_indices] = int(global_step)
        self.seq_rewards[insert_indices] = seq_reward[:block_size].numpy()
        self.quality_z[insert_indices] = quality_z[:block_size]
        self.seq_lens[insert_indices] = seq_len[:block_size]

        self.position = (self.position + block_size) % self.capacity
        self.size = min(self.size + block_size, self.capacity)

    def sample_batch(self, batch_size: int, current_step: int) -> OffPolicySampleMetrics:
        metrics = self._empty_metrics(batch_size)
        if batch_size <= 0 or self.size == 0:
            return OffPolicySampleMetrics(metrics=metrics, batch=None)

        valid_indices = self._get_valid_indices(current_step)
        if valid_indices.size == 0 or valid_indices.size < batch_size:
            metrics["off_policy/replay_hit_rate"] = valid_indices.size / float(max(batch_size, 1))
            return OffPolicySampleMetrics(metrics=metrics, batch=None)

        ages = current_step - self.source_steps[valid_indices]
        priorities = self._compute_priorities(valid_indices, ages)
        probabilities = self._mix_uniform(priorities)

        sampled_indices = np.random.choice(valid_indices, size=batch_size, replace=False, p=probabilities)
        sample_positions = np.searchsorted(valid_indices, sampled_indices)
        sampled_ages = ages[sample_positions]
        sampled_probabilities = probabilities[sample_positions]

        batch = self._gather_batch(sampled_indices)
        metrics.update(
            {
                "off_policy/buffer_size": float(self.size),
                "off_policy/replay_batch_size": float(batch_size),
                "off_policy/replay_fraction": 0.0,
                "off_policy/sample_age_mean": float(sampled_ages.mean()),
                "off_policy/sample_age_max": float(sampled_ages.max()),
                "off_policy/seq_reward_mean": float(self.seq_rewards[sampled_indices].mean()),
                "off_policy/priority_entropy": self._entropy(probabilities),
                "off_policy/replay_hit_rate": 1.0,
                "off_policy/sample_probability_mean": float(sampled_probabilities.mean()),
            }
        )
        return OffPolicySampleMetrics(metrics=metrics, batch=batch)

    def _select_replay_fields(self, batch: DataProto) -> DataProto:
        batch_keys = [key for key in OFF_POLICY_BATCH_KEYS if key in batch.batch.keys()]
        batch_keys.extend(key for key in OFF_POLICY_OPTIONAL_BATCH_KEYS if key in batch.batch.keys())
        non_tensor_keys = [key for key in OFF_POLICY_NON_TENSOR_KEYS if key in batch.non_tensor_batch]
        return batch.select(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_keys, deepcopy=True)

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

    def _compute_priorities(self, valid_indices: np.ndarray, ages: np.ndarray) -> np.ndarray:
        quality_term = np.exp(float(self.config.quality_alpha) * self.quality_z[valid_indices])
        if int(self.config.staleness_horizon) > 0:
            staleness_term = np.exp(-ages / float(self.config.staleness_horizon))
        else:
            staleness_term = np.ones_like(quality_term)
        priorities = quality_term * staleness_term
        priorities = np.clip(priorities, a_min=1e-8, a_max=None)
        return priorities.astype(np.float64, copy=False)

    def _mix_uniform(self, priorities: np.ndarray) -> np.ndarray:
        normalized = priorities / priorities.sum()
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

    def _compute_quality_z(self, seq_reward: torch.Tensor) -> np.ndarray:
        reward_np = seq_reward.numpy()
        if reward_np.size <= 1:
            return np.zeros_like(reward_np, dtype=np.float32)

        reward_mean = reward_np.mean()
        reward_std = reward_np.std()
        if reward_std < 1e-6:
            z = np.zeros_like(reward_np, dtype=np.float32)
        else:
            z = (reward_np - reward_mean) / reward_std
        return np.clip(z, -2.0, 2.0).astype(np.float32, copy=False)

    def _empty_metrics(self, requested_batch_size: int) -> dict[str, float]:
        return {
            "off_policy/buffer_size": float(self.size),
            "off_policy/replay_batch_size": 0.0,
            "off_policy/replay_fraction": 0.0,
            "off_policy/sample_age_mean": 0.0,
            "off_policy/sample_age_max": 0.0,
            "off_policy/seq_reward_mean": 0.0,
            "off_policy/priority_entropy": 0.0,
            "off_policy/replay_hit_rate": 0.0 if requested_batch_size > 0 else 1.0,
            "off_policy/sample_probability_mean": 0.0,
        }

    @staticmethod
    def _entropy(probabilities: np.ndarray) -> float:
        safe_prob = np.clip(probabilities, a_min=1e-12, a_max=None)
        return float(-(safe_prob * np.log(safe_prob)).sum())
