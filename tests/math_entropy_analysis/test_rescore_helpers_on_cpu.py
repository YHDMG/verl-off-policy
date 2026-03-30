import math

from math_entropy_analysis.rescore import (
    completion_prediction_slice,
    entropy_from_logit_row,
    rolling_mean,
    selected_logprob_from_logit_row,
    top1_prob_and_margin,
)


def test_completion_prediction_slice_aligns_first_completion_token():
    assert completion_prediction_slice(prompt_length=5, completion_length=3) == (4, 7)


def test_entropy_and_selected_logprob_match_manual_distribution():
    logits = [0.0, math.log(3.0)]
    entropy = entropy_from_logit_row(logits)
    selected = selected_logprob_from_logit_row(logits, selected_index=1)

    assert math.isclose(entropy, -0.25 * math.log(0.25) - 0.75 * math.log(0.75), rel_tol=1e-6)
    assert math.isclose(selected, math.log(0.75), rel_tol=1e-6)


def test_top1_margin_and_rolling_mean_are_stable():
    top1, margin = top1_prob_and_margin([0.0, math.log(3.0), math.log(1.5)])
    smoothed = rolling_mean([1.0, 3.0, 5.0, 7.0], window_size=2)

    assert top1 > 0.5
    assert margin > 0.0
    assert smoothed == [1.0, 2.0, 4.0, 6.0]
