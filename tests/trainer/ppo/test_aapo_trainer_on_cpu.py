import numpy as np
import pytest
import torch

pytest.importorskip("ray")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from recipe.aapo.aapo_ray_trainer import RayAAPOTrainer
from recipe.aapo.off_policy_buffer import OffPolicyReplayBuffer
from verl import DataProto
from verl.trainer.config.algorithm import OffPolicyConfig
from verl.trainer.ppo.core_algos import AdvantageEstimator


def _make_trainer(
    adv_estimator: str,
    rollout_n: int = 8,
    use_critic: bool = False,
    use_dynamic_bsz: bool = True,
    actor_mini_batch_size: int = 2,
    n_gpus_per_node: int = 8,
):
    trainer = RayAAPOTrainer.__new__(RayAAPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": adv_estimator,
                "off_policy": {
                    "enable": True,
                    "capacity": 32,
                    "warmup_steps": 1,
                    "replay_mini_batch_multiplier": 1,
                    "replay_schedule_type": "cosine_decay",
                    "replay_start_ratio": 1.0,
                    "replay_end_ratio": 0.25,
                    "replay_anneal_steps": 100,
                    "quality_metric": "seq_reward",
                    "quality_alpha": 0.5,
                    "late_quality_alpha_multiplier": 2.0,
                    "staleness_horizon": 10,
                    "uniform_mix": 0.2,
                    "late_uniform_mix": 0.05,
                    "max_age_steps": 10,
                    "zero_adv_epsilon": 1e-6,
                    "cpu_offload": True,
                },
            },
            "actor_rollout_ref": {
                "rollout": {"n": rollout_n},
                "actor": {"ppo_mini_batch_size": actor_mini_batch_size, "use_dynamic_bsz": use_dynamic_bsz},
            },
            "data": {"train_batch_size": 2},
            "trainer": {"balance_batch": False, "total_training_steps": 100, "n_gpus_per_node": n_gpus_per_node},
        }
    )
    trainer.use_critic = use_critic
    trainer.off_policy_config = None
    trainer.off_policy_buffer = None
    trainer.global_steps = 3
    return trainer


def test_aapo_replay_initializes_for_gae_with_critic():
    trainer = _make_trainer(AdvantageEstimator.GAE, use_critic=True)

    trainer._init_off_policy_replay()

    assert trainer.off_policy_config is not None
    assert isinstance(trainer.off_policy_buffer, OffPolicyReplayBuffer)


def test_aapo_replay_rejects_unsupported_estimator():
    trainer = _make_trainer("reinforce_plus_plus")

    with pytest.raises(ValueError, match="supports algorithm.adv_estimator in \\{grpo, gae\\}"):
        trainer._init_off_policy_replay()


def test_sync_mix_plan_uses_group_replacement_only_for_grpo():
    grpo_trainer = _make_trainer(AdvantageEstimator.GRPO, rollout_n=4)
    grpo_replay_size, grpo_replace_indices = grpo_trainer._compute_sync_replay_mix_plan(
        batch_size=16, requested_replay_size=12
    )

    assert grpo_replay_size == 12
    assert len(grpo_replace_indices) == 12
    assert grpo_replay_size % 4 == 0

    gae_trainer = _make_trainer(AdvantageEstimator.GAE, rollout_n=4)
    gae_replay_size, gae_replace_indices = gae_trainer._compute_sync_replay_mix_plan(
        batch_size=16, requested_replay_size=15
    )

    assert gae_replay_size == 15
    assert len(gae_replace_indices) == 15


def test_off_policy_schedule_decays_replay_ratio_and_strengthens_quality_bias():
    trainer = _make_trainer(AdvantageEstimator.GAE, rollout_n=4)

    trainer.global_steps = 0
    early_schedule = trainer._compute_off_policy_schedule()
    trainer.global_steps = 100
    late_schedule = trainer._compute_off_policy_schedule()

    assert early_schedule.effective_replay_batch_size > late_schedule.effective_replay_batch_size
    assert early_schedule.effective_replay_ratio > late_schedule.effective_replay_ratio
    assert early_schedule.effective_quality_alpha < late_schedule.effective_quality_alpha
    assert early_schedule.effective_uniform_mix > late_schedule.effective_uniform_mix


def test_off_policy_schedule_keeps_grpo_replay_batch_aligned_to_rollout_n():
    trainer = _make_trainer(AdvantageEstimator.GRPO, rollout_n=4)
    trainer.global_steps = 100

    late_schedule = trainer._compute_off_policy_schedule()

    assert late_schedule.effective_replay_batch_size % 4 == 0
    assert late_schedule.effective_replay_batch_size >= 4


def test_off_policy_schedule_keeps_static_ppo_replay_batch_aligned_to_dp_size():
    trainer = _make_trainer(
        AdvantageEstimator.GAE,
        rollout_n=1,
        use_dynamic_bsz=False,
        actor_mini_batch_size=16,
        n_gpus_per_node=8,
    )
    trainer.global_steps = 100

    late_schedule = trainer._compute_off_policy_schedule()

    assert late_schedule.effective_replay_batch_size == 8
    assert late_schedule.effective_replay_batch_size % 8 == 0


def test_off_policy_buffer_selects_ppo_related_fields():
    buffer = OffPolicyReplayBuffer(OffPolicyConfig(enable=True, capacity=8, quality_metric="seq_reward"))
    batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.ones(2, 5, dtype=torch.long),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
            "position_ids": torch.arange(5, dtype=torch.long).repeat(2, 1),
            "prompts": torch.ones(2, 3, dtype=torch.long),
            "responses": torch.ones(2, 2, dtype=torch.long),
            "response_mask": torch.ones(2, 2, dtype=torch.long),
            "old_log_probs": torch.zeros(2, 2),
            "advantages": torch.ones(2, 2),
            "returns": torch.full((2, 2), 2.0),
            "values": torch.full((2, 2), 1.5),
            "token_level_scores": torch.ones(2, 2),
            "token_level_rewards": torch.ones(2, 2),
            "rm_scores": torch.tensor([1.0, 2.0]),
            "dummy_tensor": torch.zeros(2, 1),
        },
        non_tensors={
            "uid": np.array(["a", "b"], dtype=object),
        },
    )

    selected = buffer._select_replay_fields(batch)

    assert "prompts" in selected.batch.keys()
    assert "returns" in selected.batch.keys()
    assert "values" in selected.batch.keys()
    assert "rm_scores" in selected.batch.keys()
    assert "dummy_tensor" in selected.batch.keys()
