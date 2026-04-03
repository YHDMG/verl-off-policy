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

Replay is only used for actor updates. Critic/value updates, when enabled by
PPO/GAE, remain strictly on-policy.
"""

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import torch

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


@dataclass
class OffPolicyScheduleState:
    base_replay_batch_size: int
    effective_replay_batch_size: int
    effective_replay_ratio: float
    effective_quality_alpha: float
    effective_uniform_mix: float
    replay_bias_beta: float
    health_factor: float
    schedule_progress: float


class RayAAPOTrainer(RayPPOTrainer):
    """Recipe-local PPO trainer with synchronous actor replay for GRPO/PPO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.off_policy_config: OffPolicyConfig | None = None
        self.off_policy_buffer: OffPolicyReplayBuffer | None = None
        self._replay_health_state = {
            "ess_ema": 1.0,
            "seq_dev_ema": 0.0,
        }
        self._init_off_policy_replay()

    def _init_off_policy_replay(self) -> None:
        off_policy_cfg = self.config.algorithm.get("off_policy", None)
        if not off_policy_cfg or not off_policy_cfg.get("enable", False):
            return

        off_policy_config = omega_conf_to_dataclass(off_policy_cfg, dataclass_type=OffPolicyConfig)
        adv_estimator = self._normalize_adv_estimator(self.config.algorithm.adv_estimator)
        supported_adv_estimators = {AdvantageEstimator.GRPO, AdvantageEstimator.GAE}
        if adv_estimator not in supported_adv_estimators:
            raise ValueError(
                "Synchronous off-policy replay v1 only supports algorithm.adv_estimator in {grpo, gae}. "
                f"Got {self.config.algorithm.adv_estimator}."
            )

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

        schedule_state = self._compute_off_policy_schedule()
        metrics.update(
            {
                "off_policy/schedule_progress": schedule_state.schedule_progress,
                "off_policy/effective_replay_batch_size": float(schedule_state.effective_replay_batch_size),
                "off_policy/effective_replay_ratio": schedule_state.effective_replay_ratio,
                "off_policy/effective_quality_alpha": schedule_state.effective_quality_alpha,
                "off_policy/effective_uniform_mix": schedule_state.effective_uniform_mix,
                "off_policy/replay_bias_beta": schedule_state.replay_bias_beta,
                "off_policy/health_factor": schedule_state.health_factor,
            }
        )
        metrics.update(self.off_policy_buffer.empty_metrics(schedule_state.effective_replay_batch_size))
        if schedule_state.effective_replay_batch_size <= 0:
            return batch, metrics

        warmup_size = compute_off_policy_warmup_size(
            self.off_policy_config,
            self.config.data.train_batch_size,
            self.config.actor_rollout_ref.rollout.n,
        )
        if len(self.off_policy_buffer) < warmup_size:
            return batch, metrics

        sample_output = self.off_policy_buffer.sample_batch(
            schedule_state.effective_replay_batch_size,
            current_step=self.global_steps,
            quality_alpha=schedule_state.effective_quality_alpha,
            uniform_mix=schedule_state.effective_uniform_mix,
            bias_beta=schedule_state.replay_bias_beta,
            bias_weight_clip=float(self.off_policy_config.replay_bias_weight_clip),
        )
        metrics.update(sample_output.metrics)
        replay_batch = sample_output.batch
        if replay_batch is None:
            return batch, metrics

        self._update_replay_health_state(sample_output.metrics)
        batch = self._attach_on_policy_replay_sampling_weight(batch)

        replay_batch = self._align_replay_batch_to_current(batch, replay_batch)
        replay_batch.meta_info = deepcopy(batch.meta_info)

        if self.config.actor_rollout_ref.actor.use_dynamic_bsz:
            actor_train_batch, mix_metrics = self._build_dynamic_mixed_batch(batch, replay_batch)
            metrics.update(mix_metrics)
        else:
            actor_train_batch = DataProto.concat([batch, replay_batch])
            replay_tokens = float(replay_batch.batch["response_mask"].sum().item())
            metrics["actor/replay_tokens"] = replay_tokens
            metrics["off_policy/replay_fraction"] = len(replay_batch) / float(len(actor_train_batch))

        return actor_train_batch, metrics

    def _record_off_policy_batch(self, batch: DataProto) -> dict[str, float]:
        if self.off_policy_buffer is None:
            return {}
        add_metrics = self.off_policy_buffer.add_batch(batch, global_step=self.global_steps)
        add_metric_map = {
            "off_policy/skipped_zero_adv_groups": "off_policy/add_skipped_zero_adv_groups",
            "off_policy/skipped_zero_adv_sequences": "off_policy/add_skipped_zero_adv_sequences",
            "off_policy/accepted_sequences": "off_policy/add_accepted_sequences",
            "off_policy/accepted_ratio": "off_policy/add_accepted_ratio",
        }
        remapped_metrics = {}
        for key, value in add_metrics.items():
            remapped_metrics[add_metric_map.get(key, key)] = value
        return remapped_metrics

    def _get_actor_dp_size(self) -> int:
        if getattr(self, "actor_rollout_wg", None) is not None:
            return self._get_dp_size(self.actor_rollout_wg, "actor")
        trainer_cfg = getattr(self.config, "trainer", None)
        if trainer_cfg is not None:
            fallback_dp_size = trainer_cfg.get("n_gpus_per_node", 1)
            return max(int(fallback_dp_size), 1)
        return 1

    def _compute_off_policy_schedule(self) -> OffPolicyScheduleState:
        assert self.off_policy_config is not None

        base_replay_batch_size = compute_replay_batch_size(
            self.off_policy_config,
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            self.config.actor_rollout_ref.rollout.n,
        )
        if base_replay_batch_size <= 0:
            return OffPolicyScheduleState(
                base_replay_batch_size=0,
                effective_replay_batch_size=0,
                effective_replay_ratio=0.0,
                effective_quality_alpha=float(self.off_policy_config.quality_alpha),
                effective_uniform_mix=float(self.off_policy_config.uniform_mix),
                replay_bias_beta=float(self.off_policy_config.replay_bias_beta_start),
                health_factor=1.0,
                schedule_progress=0.0,
            )

        if self.off_policy_config.replay_schedule_type != "cosine_decay":
            raise ValueError(
                "AAPO replay only supports replay_schedule_type='cosine_decay'. "
                f"Got {self.off_policy_config.replay_schedule_type}."
            )

        anneal_steps = int(self.off_policy_config.replay_anneal_steps)
        if anneal_steps <= 0:
            anneal_steps = int(getattr(self, "total_training_steps", 0) or self.config.trainer.get("total_training_steps", 0) or 0)
        anneal_steps = max(anneal_steps, 1)

        progress = float(np.clip(self.global_steps / float(anneal_steps), 0.0, 1.0))
        replay_start_ratio = float(self.off_policy_config.replay_start_ratio)
        replay_end_ratio = float(self.off_policy_config.replay_end_ratio)
        scheduled_replay_ratio = replay_end_ratio + (replay_start_ratio - replay_end_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        base_scheduled_replay_batch_size = math.floor(base_replay_batch_size * scheduled_replay_ratio)
        base_scheduled_quality_alpha = float(self.off_policy_config.quality_alpha) * (
            1.0 + (float(self.off_policy_config.late_quality_alpha_multiplier) - 1.0) * progress
        )
        base_scheduled_uniform_mix = float(self.off_policy_config.uniform_mix) + (
            float(self.off_policy_config.late_uniform_mix) - float(self.off_policy_config.uniform_mix)
        ) * progress
        base_scheduled_uniform_mix = float(np.clip(base_scheduled_uniform_mix, 0.0, 1.0))
        replay_bias_beta = float(
            self.off_policy_config.replay_bias_beta_start
            + (self.off_policy_config.replay_bias_beta_end - self.off_policy_config.replay_bias_beta_start) * progress
        )
        health_factor = self._compute_replay_health_factor()

        raw_effective_replay_batch_size = math.floor(base_scheduled_replay_batch_size * health_factor)
        rollout_n = max(int(self.config.actor_rollout_ref.rollout.n), 1)
        is_grpo = self._normalize_adv_estimator(self.config.algorithm.adv_estimator) == AdvantageEstimator.GRPO
        if self.config.actor_rollout_ref.actor.use_dynamic_bsz:
            align_unit = rollout_n if is_grpo else 1
        else:
            actor_dp_size = self._get_actor_dp_size()
            align_unit = actor_dp_size
            if is_grpo:
                align_unit = math.lcm(align_unit, rollout_n)
                if getattr(self, "use_prefix_grouper", False):
                    align_unit = actor_dp_size * rollout_n
        if raw_effective_replay_batch_size <= 0:
            effective_replay_batch_size = 0
        elif align_unit > 1:
            aligned_replay_batch_size = (raw_effective_replay_batch_size // align_unit) * align_unit
            if aligned_replay_batch_size > 0:
                effective_replay_batch_size = aligned_replay_batch_size
            elif align_unit <= base_replay_batch_size:
                effective_replay_batch_size = align_unit
            else:
                effective_replay_batch_size = 0
        else:
            effective_replay_batch_size = raw_effective_replay_batch_size
        effective_replay_batch_size = min(base_replay_batch_size, effective_replay_batch_size)
        effective_quality_alpha = base_scheduled_quality_alpha * max(
            float(health_factor),
            float(self.off_policy_config.replay_health_alpha_min_scale),
        )
        effective_uniform_mix = float(
            np.clip(
                base_scheduled_uniform_mix
                + (1.0 - float(health_factor)) * float(self.off_policy_config.replay_health_uniform_mix_boost),
                0.0,
                1.0,
            )
        )
        effective_replay_ratio = effective_replay_batch_size / float(base_replay_batch_size)

        return OffPolicyScheduleState(
            base_replay_batch_size=base_replay_batch_size,
            effective_replay_batch_size=effective_replay_batch_size,
            effective_replay_ratio=effective_replay_ratio,
            effective_quality_alpha=effective_quality_alpha,
            effective_uniform_mix=effective_uniform_mix,
            replay_bias_beta=replay_bias_beta,
            health_factor=health_factor,
            schedule_progress=progress,
        )

    def _compute_replay_health_factor(self) -> float:
        assert self.off_policy_config is not None

        ess_target = max(float(self.off_policy_config.replay_health_target_ess), 1e-8)
        seq_dev_target = max(float(self.off_policy_config.replay_health_target_seq_dev), 1e-8)
        ess_ema = float(self._replay_health_state["ess_ema"])
        seq_dev_ema = float(self._replay_health_state["seq_dev_ema"])

        ess_factor = float(np.clip(ess_ema / ess_target, 0.0, 1.0))
        dev_factor = float(np.clip(seq_dev_target / max(seq_dev_ema, 1e-8), 0.0, 1.0))
        return min(ess_factor, dev_factor)

    def _update_replay_health_state(self, sample_metrics: dict[str, float]) -> None:
        assert self.off_policy_config is not None

        ema_beta = float(np.clip(self.off_policy_config.replay_health_ema_beta, 0.0, 1.0))
        sample_ess = float(sample_metrics.get("off_policy/sample_rollout_is_eff_sample_size", 1.0))
        sample_seq_dev = float(sample_metrics.get("off_policy/sample_rollout_is_seq_abs_mean_deviation", 0.0))

        self._replay_health_state["ess_ema"] = (
            (1.0 - ema_beta) * float(self._replay_health_state["ess_ema"]) + ema_beta * sample_ess
        )
        self._replay_health_state["seq_dev_ema"] = (
            (1.0 - ema_beta) * float(self._replay_health_state["seq_dev_ema"]) + ema_beta * sample_seq_dev
        )

    @staticmethod
    def _attach_on_policy_replay_sampling_weight(batch: DataProto) -> DataProto:
        if "replay_sampling_weight" in batch.batch.keys():
            return batch

        response_mask = batch.batch["response_mask"]
        replay_sampling_weight = torch.ones(len(batch), dtype=torch.float32, device=response_mask.device)
        return batch.union(DataProto.from_dict(tensors={"replay_sampling_weight": replay_sampling_weight}))

    def _align_replay_batch_to_current(self, batch: DataProto, replay_batch: DataProto) -> DataProto:
        replay_size = len(replay_batch)

        aligned_tensors = {}
        for key, current_value in batch.batch.items():
            if key in replay_batch.batch.keys():
                replay_value = replay_batch.batch[key]
                if replay_value.device != current_value.device:
                    replay_value = replay_value.to(current_value.device)
                aligned_tensors[key] = replay_value
            elif key in {"rollout_is_weights", "replay_sampling_weight"}:
                aligned_tensors[key] = torch.ones(
                    (replay_size, *current_value.shape[1:]),
                    dtype=current_value.dtype,
                    device=current_value.device,
                )
            else:
                aligned_tensors[key] = torch.zeros(
                    (replay_size, *current_value.shape[1:]),
                    dtype=current_value.dtype,
                    device=current_value.device,
                )

        aligned_non_tensors = {}
        for key, current_value in batch.non_tensor_batch.items():
            if key in replay_batch.non_tensor_batch:
                aligned_non_tensors[key] = replay_batch.non_tensor_batch[key]
                continue

            shape = (replay_size, *current_value.shape[1:])
            if current_value.dtype == object:
                fill_value = None
            else:
                fill_value = 0
            aligned_non_tensors[key] = np.full(shape, fill_value, dtype=current_value.dtype)

        return DataProto.from_dict(
            tensors=aligned_tensors,
            non_tensors=aligned_non_tensors,
            meta_info=deepcopy(replay_batch.meta_info),
        )

    def _build_dynamic_mixed_batch(self, batch: DataProto, replay_batch: DataProto) -> tuple[DataProto, dict[str, float]]:
        """Build a fixed-size mixed batch for synchronous dynamic-bsz training.

        Unlike the async trainer, the sync PPO path already has a fully assembled
        on-policy batch of fixed size. To keep the actor token budget stable, we
        replace part of the current on-policy batch with replay samples instead of
        appending replay samples on top of it.
        """

        batch_size = len(batch)
        requested_replay_size = len(replay_batch)
        replay_size, replace_indices = self._compute_sync_replay_mix_plan(batch_size, requested_replay_size)
        if replay_size <= 0:
            return batch, {
                "actor/replay_tokens": 0.0,
                "off_policy/replay_batch_size": 0.0,
                "off_policy/replay_fraction": 0.0,
            }

        replay_batch = replay_batch[:replay_size]
        mixed_batch = batch.select_idxs(torch.arange(batch_size))

        for key, replay_value in replay_batch.batch.items():
            mixed_batch.batch[key][replace_indices] = replay_value.to(mixed_batch.batch[key].device)

        replace_indices_np = replace_indices.detach().cpu().numpy()
        for key, replay_value in replay_batch.non_tensor_batch.items():
            mixed_batch.non_tensor_batch[key][replace_indices_np] = replay_value

        replaced_on_policy_tokens = (
            float(batch.batch["response_mask"][replace_indices].sum().item()) if "response_mask" in batch.batch else 0.0
        )
        total_on_policy_tokens = float(batch.batch["response_mask"].sum().item()) if "response_mask" in batch.batch else 0.0
        on_policy_tokens = total_on_policy_tokens - replaced_on_policy_tokens
        replay_tokens = float(replay_batch.batch["response_mask"].sum().item())
        metrics = {
            "actor/on_policy_tokens": on_policy_tokens,
            "actor/replay_tokens": replay_tokens,
            "off_policy/replay_batch_size": float(replay_size),
            "off_policy/replay_fraction": replay_size / float(len(mixed_batch)),
        }
        return mixed_batch, metrics

    @staticmethod
    def _normalize_adv_estimator(adv_estimator) -> AdvantageEstimator | str:
        if isinstance(adv_estimator, AdvantageEstimator):
            return adv_estimator
        try:
            return AdvantageEstimator(adv_estimator)
        except ValueError:
            return adv_estimator

    def _compute_sync_replay_mix_plan(self, batch_size: int, requested_replay_size: int) -> tuple[int, torch.Tensor]:
        """Choose which on-policy rows to replace with replay rows.

        For synchronous GRPO, batches naturally come in prompt groups of
        ``rollout.n`` sequences. When possible we replace a whole number of prompt
        groups and always keep at least one fresh on-policy group in the batch.
        PPO/GAE does not require prompt-group structure, so it falls back to
        per-sequence replacement.
        """

        rollout_n = max(int(self.config.actor_rollout_ref.rollout.n), 1)
        generator = torch.Generator()
        generator.manual_seed(int(self.global_steps))

        is_grpo = self._normalize_adv_estimator(self.config.algorithm.adv_estimator) == AdvantageEstimator.GRPO
        if is_grpo and batch_size % rollout_n == 0 and requested_replay_size % rollout_n == 0:
            total_groups = batch_size // rollout_n
            replay_groups = min(requested_replay_size // rollout_n, max(total_groups - 1, 0))
            replay_size = replay_groups * rollout_n
            if replay_size <= 0:
                return 0, torch.empty(0, dtype=torch.long)

            selected_groups = torch.randperm(total_groups, generator=generator)[:replay_groups]
            replace_indices = []
            for group_idx in selected_groups.tolist():
                start = group_idx * rollout_n
                replace_indices.extend(range(start, start + rollout_n))
            replace_indices = torch.tensor(replace_indices, dtype=torch.long)
            return replay_size, replace_indices

        min_on_policy = rollout_n if is_grpo else 1
        replay_size = min(requested_replay_size, max(batch_size - min_on_policy, 0))
        if replay_size <= 0:
            return 0, torch.empty(0, dtype=torch.long)
        replace_indices = torch.randperm(batch_size, generator=generator)[:replay_size]
        return replay_size, replace_indices

    def _update_actor(self, batch: DataProto) -> DataProto:
        actor_train_batch, off_policy_metrics = self._build_actor_training_batch(batch)
        balance_metrics = {}
        has_replay = off_policy_metrics.get("actor/replay_tokens", 0.0) > 0

        if has_replay and self.config.trainer.balance_batch:
            actor_dp_size = self._get_actor_dp_size()
            if len(actor_train_batch) % actor_dp_size == 0:
                self._balance_batch(actor_train_batch, metrics=balance_metrics, logging_prefix="off_policy_seqlen")
            else:
                balance_metrics["off_policy_seqlen/skipped_nondivisible_batch"] = 1.0

        actor_train_batch.meta_info["global_token_num"] = torch.sum(
            actor_train_batch.batch["attention_mask"], dim=-1
        ).tolist()

        actor_output = super()._update_actor(actor_train_batch)

        metrics = actor_output.meta_info.setdefault("metrics", {})
        metrics.update(off_policy_metrics)
        metrics.update(balance_metrics)
        metrics.update(self._record_off_policy_batch(batch))
        return actor_output
