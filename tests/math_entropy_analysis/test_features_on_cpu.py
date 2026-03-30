import math

from math_entropy_analysis.features import compute_sequence_features


def test_compute_sequence_features_aggregates_entropy_statistics():
    generations = [
        {
            "sample_id": "a",
            "decode_id": 0,
            "is_correct": True,
            "response_length": 3,
            "verification_method": "gsm8k_rule",
            "extraction_method": "hash",
        }
    ]
    token_rows = [
        {"sample_id": "a", "decode_id": 0, "position": 0, "entropy": 1.0, "logprob": -0.1, "segment_tag": "reasoning"},
        {"sample_id": "a", "decode_id": 0, "position": 1, "entropy": 2.0, "logprob": -0.2, "segment_tag": "reasoning"},
        {"sample_id": "a", "decode_id": 0, "position": 2, "entropy": 0.5, "logprob": -0.3, "segment_tag": "answer"},
    ]

    features = compute_sequence_features(generations, token_rows)

    assert len(features) == 1
    feature = features[0]
    assert math.isclose(feature["mean_entropy"], (1.0 + 2.0 + 0.5) / 3, rel_tol=1e-6)
    assert math.isclose(feature["sequence_nll"], 0.6, rel_tol=1e-6)
    assert math.isclose(feature["answer_span_entropy_mean"], 0.5, rel_tol=1e-6)
