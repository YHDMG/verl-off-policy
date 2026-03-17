# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
AAPO trainer that adds synchronous off-policy replay on top of RayPPOTrainer
without modifying the core PPO trainer implementation.
"""

from copy import deepcopy

from verl import DataProto
from verl.trainer.config.algorithm import OffPolicyConfig
from verl.trainer.ppo.core_algos import AdvantageEstimator
from .off_policy_buffer import (
    OffPolicyReplayBuffer,
    compute_off_policy_warmup_size,
    compute_replay_batch_size,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.config import omega_conf_to_dataclass


class RayAAPOTrainer(RayPPOTrainer):
    """Recipe-local PPO trainer with synchronous GRPO off-policy replay."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.off_policy_config: OffPolicyConfig | None = None
        self.off_policy_buffer: OffPolicyReplayBuffer | None = None
        self._init_off_policy_replay()

    def _init_off_policy_replay(self) -> None:
        off_policy_cfg = self.config.algorithm.get("off_policy", None)
        if not off_policy_cfg or not off_policy_cfg.get("enable", False):
            return

        off_policy_config = omega_conf_to_dataclass(off_policy_cfg, dataclass_type=OffPolicyConfig)
        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError(
                "Synchronous off-policy replay v1 only supports algorithm.adv_estimator=grpo. "
                f"Got {self.config.algorithm.adv_estimator}."
            )
        if self.use_critic:
            raise ValueError("Synchronous off-policy replay v1 is actor-only and does not support critic replay.")

        self.off_policy_config = off_policy_config
        self.off_policy_buffer = OffPolicyReplayBuffer(off_policy_config)

    def _build_actor_training_batch(self, batch: DataProto) -> tuple[DataProto, dict[str, float]]:
        on_policy_tokens = float(batch.batch["response_mask"].sum().item()) if "response_mask" in batch.batch else 0.0
        metrics = {
            "actor/on_policy_tokens": on_policy_tokens,
            "actor/replay_tokens": 0.0,
        }
        if self.off_policy_buffer is None or self.off_policy_config is None:
            return batch, metrics

        replay_batch_size = compute_replay_batch_size(
            self.off_policy_config,
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            self.config.actor_rollout_ref.rollout.n,
        )
        metrics.update(self.off_policy_buffer.empty_metrics(replay_batch_size))
        if replay_batch_size <= 0:
            return batch, metrics

        warmup_size = compute_off_policy_warmup_size(
            self.off_policy_config,
            self.config.data.train_batch_size,
            self.config.actor_rollout_ref.rollout.n,
        )
        if len(self.off_policy_buffer) < warmup_size:
            return batch, metrics

        sample_output = self.off_policy_buffer.sample_batch(replay_batch_size, current_step=self.global_steps)
        metrics.update(sample_output.metrics)
        replay_batch = sample_output.batch
        if replay_batch is None:
            return batch, metrics

        replay_batch.meta_info = deepcopy(batch.meta_info)
        actor_train_batch = DataProto.concat([batch, replay_batch])
        replay_tokens = float(replay_batch.batch["response_mask"].sum().item())
        metrics["actor/replay_tokens"] = replay_tokens
        metrics["off_policy/replay_fraction"] = len(replay_batch) / float(len(actor_train_batch))
        return actor_train_batch, metrics

    def _record_off_policy_batch(self, batch: DataProto) -> None:
        if self.off_policy_buffer is None:
            return
        self.off_policy_buffer.add_batch(batch, global_step=self.global_steps)

    def _update_actor(self, batch: DataProto) -> DataProto:
        actor_train_batch, off_policy_metrics = self._build_actor_training_batch(batch)
        actor_output = super()._update_actor(actor_train_batch)

        metrics = actor_output.meta_info.setdefault("metrics", {})
        metrics.update(off_policy_metrics)

        self._record_off_policy_batch(batch)
        return actor_output
