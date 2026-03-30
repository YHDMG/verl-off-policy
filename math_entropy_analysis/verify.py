"""Math answer extraction, verification, and segment tagging."""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass


ANSWER_PHRASE_RE = re.compile(
    r"(?:the answer is|final answer(?: is)?|answer(?:\s*[:=])?)\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
HASH_ANSWER_RE = re.compile(r"####\s*(.+)")
INLINE_MATH_RE = re.compile(r"\$(.+?)\$")
TRAILING_PUNCTUATION = " \n\t\r.,;:!?"


@dataclass(slots=True)
class AnswerExtraction:
    answer: str | None
    start: int | None
    end: int | None
    method: str


@dataclass(slots=True)
class VerificationResult:
    is_correct: bool
    extracted_answer: str | None
    extraction_method: str
    verification_method: str
    answer_start: int | None
    answer_end: int | None


def _strip_wrapping(answer: str) -> str:
    stripped = answer.strip().strip(TRAILING_PUNCTUATION)
    if "=" in stripped:
        stripped = stripped.split("=")[-1].strip()
    return stripped.strip(TRAILING_PUNCTUATION)


def _find_boxed_answer(text: str) -> AnswerExtraction | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None

    cursor = start + len(marker)
    depth = 1
    while cursor < len(text) and depth > 0:
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        cursor += 1

    if depth != 0:
        return None

    answer_start = start + len(marker)
    answer_end = cursor - 1
    answer = text[answer_start:answer_end]
    return AnswerExtraction(answer=_strip_wrapping(answer), start=answer_start, end=answer_end, method="boxed")


def _find_hash_answer(text: str) -> AnswerExtraction | None:
    matches = list(HASH_ANSWER_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    answer = _strip_wrapping(match.group(1))
    start = match.start(1)
    end = start + len(match.group(1))
    return AnswerExtraction(answer=answer, start=start, end=end, method="hash")


def _find_answer_phrase(text: str) -> AnswerExtraction | None:
    matches = list(ANSWER_PHRASE_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    raw = match.group(1).strip().splitlines()[0]
    answer = _strip_wrapping(raw)
    start = match.start(1)
    end = start + len(raw)
    return AnswerExtraction(answer=answer, start=start, end=end, method="answer_phrase")


def _find_inline_math(text: str) -> AnswerExtraction | None:
    matches = list(INLINE_MATH_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    answer = _strip_wrapping(match.group(1))
    return AnswerExtraction(answer=answer, start=match.start(1), end=match.end(1), method="inline_math")


def _find_last_expression(text: str) -> AnswerExtraction | None:
    tail = text[-240:]
    offset = len(text) - len(tail)
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not lines:
        return None
    candidate_line = lines[-1]
    answer = _strip_wrapping(candidate_line)
    if not answer:
        return None
    relative_start = tail.rfind(candidate_line)
    start = offset + relative_start
    end = start + len(candidate_line)
    return AnswerExtraction(answer=answer, start=start, end=end, method="tail")


def extract_answer_candidate(text: str) -> AnswerExtraction:
    for extractor in (
        _find_boxed_answer,
        _find_hash_answer,
        _find_answer_phrase,
        _find_inline_math,
        _find_last_expression,
    ):
        extraction = extractor(text)
        if extraction and extraction.answer:
            return extraction
    return AnswerExtraction(answer=None, start=None, end=None, method="none")


def _can_use_math_verify() -> bool:
    return importlib.util.find_spec("math_verify") is not None


def _verify_with_math_verify(model_output: str, ground_truth: str) -> bool | None:
    if not _can_use_math_verify():
        return None
    try:
        from verl.utils.reward_score.math_verify import compute_score
    except Exception:
        return None
    try:
        return bool(compute_score(model_output, ground_truth))
    except Exception:
        return None


def _verify_with_prime_math(candidate: str, ground_truth: str) -> bool | None:
    try:
        from verl.utils.reward_score.prime_math.grader import math_equal
    except Exception:
        return None
    try:
        return bool(math_equal(candidate, ground_truth))
    except Exception:
        return None


def _verify_with_gsm8k_rules(model_output: str, ground_truth: str) -> bool:
    from verl.utils.reward_score.gsm8k import compute_score

    return bool(
        compute_score(model_output, ground_truth, method="strict")
        or compute_score(model_output, ground_truth, method="flexible")
    )


def _answers_match(candidate: str | None, ground_truth: str) -> bool:
    if candidate is None:
        return False
    return _strip_wrapping(candidate) == _strip_wrapping(ground_truth)


def verify_math_completion(completion_text: str, ground_truth: str) -> VerificationResult:
    extraction = extract_answer_candidate(completion_text)

    verified = _verify_with_math_verify(completion_text, ground_truth)
    if verified is True:
        return VerificationResult(
            is_correct=True,
            extracted_answer=extraction.answer,
            extraction_method=extraction.method,
            verification_method="math_verify",
            answer_start=extraction.start,
            answer_end=extraction.end,
        )

    if extraction.answer is not None:
        prime_math = _verify_with_prime_math(extraction.answer, ground_truth)
        if prime_math is not None:
            return VerificationResult(
                is_correct=bool(prime_math),
                extracted_answer=extraction.answer,
                extraction_method=extraction.method,
                verification_method="prime_math",
                answer_start=extraction.start,
                answer_end=extraction.end,
            )

        if _answers_match(extraction.answer, ground_truth):
            return VerificationResult(
                is_correct=True,
                extracted_answer=extraction.answer,
                extraction_method=extraction.method,
                verification_method="exact_string",
                answer_start=extraction.start,
                answer_end=extraction.end,
            )

    gsm8k_verified = _verify_with_gsm8k_rules(completion_text, ground_truth)
    return VerificationResult(
        is_correct=bool(gsm8k_verified),
        extracted_answer=extraction.answer,
        extraction_method=extraction.method,
        verification_method="gsm8k_rule",
        answer_start=extraction.start,
        answer_end=extraction.end,
    )


def classify_token_segments(
    token_char_spans: list[tuple[int, int]],
    token_texts: list[str],
    answer_span: tuple[int, int] | None,
) -> list[str]:
    tags: list[str] = []
    for (start, end), token_text in zip(token_char_spans, token_texts, strict=True):
        stripped = token_text.strip()
        if not stripped:
            tags.append("other")
            continue
        if stripped.startswith("<|") and stripped.endswith("|>"):
            tags.append("other")
            continue
        if answer_span is not None and not (end <= answer_span[0] or start >= answer_span[1]):
            tags.append("answer")
            continue
        tags.append("reasoning")
    return tags
