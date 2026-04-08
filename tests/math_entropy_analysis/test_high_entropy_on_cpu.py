import math

from math_entropy_analysis.high_entropy import analyze_high_entropy_tokens, resolve_high_entropy_threshold


def test_resolve_high_entropy_threshold_supports_quantile_mode():
    threshold, mode = resolve_high_entropy_threshold([1.0, 2.0, 3.0], entropy_quantile=0.5)

    assert mode == "quantile"
    assert math.isclose(threshold, 2.0, rel_tol=1e-6)


def test_analyze_high_entropy_tokens_reports_global_and_correctness_breakdown():
    generations = [
        {"sample_id": "a", "decode_id": 0, "is_correct": True},
        {"sample_id": "b", "decode_id": 0, "is_correct": False},
    ]
    token_rows = [
        {"sample_id": "a", "decode_id": 0, "position": 0, "relative_position": 0.0, "token_text": "Let", "entropy": 1.0, "logprob": -0.1, "top1_prob": 0.9, "margin_1_2": 0.6, "segment_tag": "reasoning"},
        {"sample_id": "a", "decode_id": 0, "position": 1, "relative_position": 0.33, "token_text": " x", "entropy": 2.5, "logprob": -1.2, "top1_prob": 0.4, "margin_1_2": 0.1, "segment_tag": "reasoning"},
        {"sample_id": "a", "decode_id": 0, "position": 2, "relative_position": 0.66, "token_text": " =", "entropy": 0.2, "logprob": -0.2, "top1_prob": 0.95, "margin_1_2": 0.7, "segment_tag": "reasoning"},
        {"sample_id": "a", "decode_id": 0, "position": 3, "relative_position": 1.0, "token_text": " 5", "entropy": 3.0, "logprob": -1.8, "top1_prob": 0.3, "margin_1_2": 0.05, "segment_tag": "answer"},
        {"sample_id": "b", "decode_id": 0, "position": 0, "relative_position": 0.0, "token_text": "Let", "entropy": 2.1, "logprob": -1.1, "top1_prob": 0.45, "margin_1_2": 0.08, "segment_tag": "reasoning"},
        {"sample_id": "b", "decode_id": 0, "position": 1, "relative_position": 0.33, "token_text": " y", "entropy": 2.2, "logprob": -1.3, "top1_prob": 0.42, "margin_1_2": 0.09, "segment_tag": "reasoning"},
        {"sample_id": "b", "decode_id": 0, "position": 2, "relative_position": 0.66, "token_text": " =", "entropy": 0.4, "logprob": -0.3, "top1_prob": 0.91, "margin_1_2": 0.65, "segment_tag": "reasoning"},
        {"sample_id": "b", "decode_id": 0, "position": 3, "relative_position": 1.0, "token_text": " 7", "entropy": 2.3, "logprob": -1.5, "top1_prob": 0.37, "margin_1_2": 0.06, "segment_tag": "answer"},
    ]

    analysis = analyze_high_entropy_tokens(
        generation_rows=generations,
        token_rows=token_rows,
        entropy_threshold=2.0,
        position_bins=4,
        min_token_frequency=1,
        top_k_token_texts=20,
    )

    summary = analysis["summary"]
    assert math.isclose(summary["threshold"]["value"], 2.0, rel_tol=1e-6)
    assert summary["global"]["high_entropy_tokens"] == 5
    assert math.isclose(summary["correctness"]["correct"]["high_entropy_ratio"], 0.5, rel_tol=1e-6)
    assert math.isclose(summary["correctness"]["incorrect"]["high_entropy_ratio"], 0.75, rel_tol=1e-6)
    assert math.isclose(summary["correctness_gap"]["high_entropy_ratio_incorrect_minus_correct"], 0.25, rel_tol=1e-6)

    sample_rows = {row["sample_id"]: row for row in analysis["sample_stats"]}
    assert sample_rows["a"]["run_count"] == 2
    assert sample_rows["b"]["run_count"] == 2
    assert sample_rows["b"]["longest_run_length"] == 2
    assert math.isclose(sample_rows["a"]["answer_high_entropy_ratio"], 1.0, rel_tol=1e-6)

    token_text_rows = {row["token_text_display"]: row for row in analysis["token_text_stats"]}
    assert token_text_rows[repr("Let")]["total_count"] == 2
    assert token_text_rows[repr("Let")]["high_entropy_count"] == 1
    assert token_text_rows[repr(" =")]["high_entropy_count"] == 0

    runs = analysis["run_rows"]
    assert len(runs) == 4
    assert max(row["run_length"] for row in runs) == 2
    assert any(row["token_preview"] == "Let y" for row in runs)