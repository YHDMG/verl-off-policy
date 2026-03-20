import numpy as np
import pytest
import torch

pytest.importorskip("ray")

from recipe.aapo.off_policy_buffer import OffPolicyReplayBuffer
from verl import DataProto
from verl.trainer.config.algorithm import OffPolicyConfig


def _make_batch(uid_values, advantages):
    batch_size, response_length = advantages.shape
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.ones(batch_size, response_length + 2, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, response_length + 2, dtype=torch.long),
            "position_ids": torch.arange(response_length + 2, dtype=torch.long).repeat(batch_size, 1),
            "responses": torch.ones(batch_size, response_length, dtype=torch.long),
            "response_mask": torch.ones(batch_size, response_length, dtype=torch.float32),
            "old_log_probs": torch.zeros(batch_size, response_length, dtype=torch.float32),
            "advantages": advantages.to(torch.float32),
            "token_level_scores": torch.ones(batch_size, response_length, dtype=torch.float32),
            "token_level_rewards": torch.ones(batch_size, response_length, dtype=torch.float32),
        },
        non_tensors={"uid": np.asarray(uid_values, dtype=object)},
    )


def test_zero_adv_group_is_not_added_to_buffer():
    buffer = OffPolicyReplayBuffer(OffPolicyConfig(enable=True, capacity=8, zero_adv_epsilon=1e-6))
    batch = _make_batch(["prompt-a", "prompt-a"], torch.zeros(2, 3))

    metrics = buffer.add_batch(batch, global_step=1)

    assert len(buffer) == 0
    assert metrics["off_policy/skipped_zero_adv_groups"] == 1.0
    assert metrics["off_policy/skipped_zero_adv_sequences"] == 2.0
    assert metrics["off_policy/accepted_sequences"] == 0.0
    assert metrics["off_policy/accepted_ratio"] == 0.0


def test_only_non_zero_adv_groups_are_kept():
    buffer = OffPolicyReplayBuffer(OffPolicyConfig(enable=True, capacity=8, zero_adv_epsilon=1e-6))
    batch = _make_batch(
        ["prompt-a", "prompt-a", "prompt-b", "prompt-b"],
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ]
        ),
    )

    metrics = buffer.add_batch(batch, global_step=1)

    assert len(buffer) == 2
    assert metrics["off_policy/skipped_zero_adv_groups"] == 1.0
    assert metrics["off_policy/skipped_zero_adv_sequences"] == 2.0
    assert metrics["off_policy/accepted_sequences"] == 2.0
    assert metrics["off_policy/accepted_ratio"] == 0.5


def test_single_sequence_gae_sample_is_kept_when_advantage_non_zero():
    buffer = OffPolicyReplayBuffer(OffPolicyConfig(enable=True, capacity=8, zero_adv_epsilon=1e-6))
    batch = _make_batch(["prompt-a"], torch.tensor([[0.0, 0.2, 0.0]]))

    metrics = buffer.add_batch(batch, global_step=1)

    assert len(buffer) == 1
    assert metrics["off_policy/skipped_zero_adv_groups"] == 0.0
    assert metrics["off_policy/skipped_zero_adv_sequences"] == 0.0
    assert metrics["off_policy/accepted_sequences"] == 1.0
    assert metrics["off_policy/accepted_ratio"] == 1.0
