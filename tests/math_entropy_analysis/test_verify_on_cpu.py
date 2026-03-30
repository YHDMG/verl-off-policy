from math_entropy_analysis.verify import classify_token_segments, extract_answer_candidate, verify_math_completion


def test_extract_answer_candidate_prefers_boxed_answer():
    text = "We compute carefully.\nFinal answer: \\boxed{\\frac{1}{2}}"
    extraction = extract_answer_candidate(text)

    assert extraction.answer == "\\frac{1}{2}"
    assert extraction.method == "boxed"


def test_verify_math_completion_uses_rule_fallback_for_gsm8k_style_answers():
    result = verify_math_completion("Reasoning here\n#### 42", "42")

    assert result.is_correct is True
    assert result.extracted_answer == "42"
    assert result.extraction_method == "hash"


def test_classify_token_segments_marks_answer_overlap():
    token_spans = [(0, 4), (4, 9), (9, 12)]
    token_texts = ["calc", " =42", "\n"]
    tags = classify_token_segments(token_spans, token_texts, answer_span=(5, 8))

    assert tags == ["reasoning", "answer", "other"]
