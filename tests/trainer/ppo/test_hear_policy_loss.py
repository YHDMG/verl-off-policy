import pytest
import torch

pytest.importorskip("ray")

from verl.trainer.ppo import core_algos as core
from verl.trainer.ppo.core_algos import compute_policy_loss_hear, get_hear_metrics, get_ratio_history
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import ActorConfig, PolicyLossConfig


@pytest.fixture(autouse=True)
def clear_hear_state():
    core._hear_ratio_history.clear()
    core._hear_metric_storage.clear()
    core._hear_entropy_history.clear()
    core._hear_entropy_ema_state.clear()
    core._hear_entropy_control_state.clear()
    yield
    core._hear_ratio_history.clear()
    core._hear_metric_storage.clear()
    core._hear_entropy_history.clear()
    core._hear_entropy_ema_state.clear()
    core._hear_entropy_control_state.clear()


def _make_actor_config(**policy_loss_overrides) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        use_dynamic_bsz=True,
        rollout_n=1,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        ppo_epochs=2,
        high_entropy_ratio=0.5,
        high_entropy_last_epoch_enabled=False,
        policy_loss=PolicyLossConfig(loss_mode="hear", **policy_loss_overrides),
    )


def test_hear_applies_correction_before_guard():
    config = _make_actor_config(
        enable_correction=True,
        correction_history_size=4,
        correction_beta=0.0,
        correction_lambda=4.0,
        enable_high_entropy_guard=True,
        high_entropy_guard_min_ratio=1.0,
        high_entropy_guard_select_ratio=1.0,
        high_entropy_guard_max_iters=3,
        high_entropy_guard_high_step=0.1,
        high_entropy_guard_upper_max=1.5,
    )
    get_ratio_history("cpu", max_size=4).append((0.0, 0.0, 16, 0.0))

    ratio = torch.tensor([[1.6]], dtype=torch.float32)
    old_log_prob = torch.zeros_like(ratio)
    log_prob = torch.log(ratio)
    advantages = torch.ones_like(ratio)
    response_mask = torch.ones_like(ratio)
    entropy = torch.ones_like(ratio)

    compute_policy_loss_hear(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        entropy=entropy,
    )
    hear_metrics = get_hear_metrics(config.policy_loss)

    assert hear_metrics["actor/hear_clip_high"] == pytest.approx(1.2)
    assert hear_metrics["actor/hear_high_entropy_guard/coverage_before"] == pytest.approx(1.0)
    assert hear_metrics["actor/hear_high_entropy_guard/coverage_after"] == pytest.approx(1.0)


def test_hear_history_update_ignores_guard_widened_clip():
    config = _make_actor_config(
        enable_correction=True,
        correction_history_size=4,
        correction_beta=0.0,
        correction_lambda=0.0,
        enable_high_entropy_guard=True,
        high_entropy_guard_min_ratio=1.0,
        high_entropy_guard_select_ratio=0.5,
        high_entropy_guard_max_iters=3,
        high_entropy_guard_high_step=0.1,
        high_entropy_guard_upper_max=1.4,
    )

    ratio = torch.tensor([[1.3, 1.1]], dtype=torch.float32)
    old_log_prob = torch.zeros_like(ratio)
    log_prob = torch.log(ratio)
    advantages = torch.ones_like(ratio)
    response_mask = torch.ones_like(ratio)
    entropy = torch.tensor([[1.0, 0.1]], dtype=torch.float32)

    compute_policy_loss_hear(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        entropy=entropy,
    )

    history_queue = get_ratio_history("cpu", max_size=4)
    assert len(history_queue) == 1
    history_mean, history_std, history_count, history_max = history_queue[-1]

    assert history_mean == pytest.approx(torch.log(torch.tensor(1.1)).item())
    assert history_std == pytest.approx(0.0)
    assert history_count == 1
    assert history_max == pytest.approx(torch.log(torch.tensor(1.1)).item())

    hear_metrics = get_hear_metrics(config.policy_loss)
    assert hear_metrics["actor/hear_clip_high"] > 1.2


def test_hear_metrics_export_stable_ratio_and_guard_keys():
    config = _make_actor_config(
        enable_correction=True,
        correction_history_size=4,
        correction_beta=0.0,
        correction_lambda=2.0,
        enable_high_entropy_guard=True,
        high_entropy_guard_min_ratio=1.0,
        high_entropy_guard_select_ratio=0.5,
        high_entropy_guard_max_iters=2,
        high_entropy_guard_high_step=0.1,
        high_entropy_guard_upper_max=1.4,
    )
    get_ratio_history("cpu", max_size=4).append((0.0, 0.0, 8, 0.0))

    ratio = torch.tensor([[1.5, 0.7]], dtype=torch.float32)
    old_log_prob = torch.zeros_like(ratio)
    log_prob = torch.log(ratio)
    advantages = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    response_mask = torch.ones_like(ratio)
    entropy = torch.tensor([[0.9, 0.1]], dtype=torch.float32)

    compute_policy_loss_hear(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        entropy=entropy,
    )
    hear_metrics = get_hear_metrics(config.policy_loss)

    expected_keys = {
        "actor/hear_clip_low",
        "actor/hear_clip_high",
        "actor/hear_entropy_mean",
        "actor/hear_entropy_ema_mean",
        "actor/hear_entropy_ema_std",
        "actor/hear_high_entropy_coverage",
        "actor/hear_high_entropy/selected_ratio",
        "actor/hear_high_entropy/coverage_after_clip",
        "actor/hear_high_entropy_guard/coverage_before",
        "actor/hear_high_entropy_guard/coverage_after",
        "actor/hear_ratio/pos_adv_mean",
        "actor/hear_ratio/pos_adv_std",
        "actor/hear_ratio/pos_adv_max",
        "actor/hear_ratio/pos_adv_min",
        "actor/hear_ratio/pos_adv_exceed_high",
        "actor/hear_ratio/pos_adv_count",
        "actor/hear_ratio/neg_adv_mean",
        "actor/hear_ratio/neg_adv_std",
        "actor/hear_ratio/neg_adv_max",
        "actor/hear_ratio/neg_adv_min",
        "actor/hear_ratio/neg_adv_below_low",
        "actor/hear_ratio/neg_adv_count",
        "actor/hear_ratio/correction_count",
        "actor/hear_ratio/correction_mean_diff",
        "actor/hear_ratio/correction_trigger_count",
        "actor/hear_ratio/correction_success_count",
        "actor/hear_ratio/history_update_frequency",
        "actor/hear_ratio/history_mean_log_ratio",
        "actor/hear_ratio/history_max_log_ratio",
    }

    assert expected_keys.issubset(hear_metrics.keys())


def test_high_entropy_last_epoch_requires_explicit_enablement():
    disabled_config = _make_actor_config(enable_correction=False)
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.config = disabled_config
    actor.high_entropy_ratio = 0.5
    actor.high_entropy_last_epoch_enabled = False

    assert actor._should_use_high_entropy_last_epoch(epoch_idx=1) is False

    enabled_config = _make_actor_config(enable_correction=False)
    enabled_actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    enabled_actor.config = enabled_config
    enabled_actor.high_entropy_ratio = 0.5
    enabled_actor.high_entropy_last_epoch_enabled = True

    assert enabled_actor._should_use_high_entropy_last_epoch(epoch_idx=0) is False
    assert enabled_actor._should_use_high_entropy_last_epoch(epoch_idx=1) is True


def test_hear_guard_keeps_global_ratio_constant_during_reuse_epoch():
    config = _make_actor_config(
        enable_correction=False,
        enable_high_entropy_guard=True,
        high_entropy_guard_min_ratio=1.0,
        high_entropy_guard_select_ratio=0.25,
        high_entropy_guard_max_iters=0,
    )
    config._temp_hear_reuse_epoch = True
    config._temp_hear_reuse_ratio = 0.5

    ratio = torch.ones((1, 4), dtype=torch.float32)
    old_log_prob = torch.zeros_like(ratio)
    log_prob = torch.log(ratio)
    advantages = torch.ones_like(ratio)
    response_mask = torch.ones_like(ratio)
    entropy = torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float32)

    compute_policy_loss_hear(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        entropy=entropy,
    )

    hear_metrics = get_hear_metrics(config.policy_loss)
    selected_guard_tokens = (
        hear_metrics["actor/hear_high_entropy_guard/covered_token_count_before"]
        + hear_metrics["actor/hear_high_entropy_guard/uncovered_token_count_before"]
    )

    # Reuse epoch only sees half of the original tokens; with a global guard ratio of 0.25,
    # the guard budget should therefore cover 0.25 / 0.5 = 0.5 of the reused subset => 2 tokens.
    assert selected_guard_tokens == pytest.approx(2.0)
