"""Teacher-forcing rescoring for math-task entropy analysis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from .dataset import load_records, prepare_artifact_dir, write_parquet_records
from .verify import classify_token_segments, verify_math_completion


def completion_prediction_slice(prompt_length: int, completion_length: int) -> tuple[int, int]:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive for teacher-forcing alignment")
    return prompt_length - 1, prompt_length - 1 + completion_length


def softmax_row(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    shifted = [math.exp(value - max_logit) for value in logits]
    total = sum(shifted)
    return [value / total for value in shifted]


def entropy_from_logit_row(logits: list[float]) -> float:
    probabilities = softmax_row(logits)
    return -sum(probability * math.log(probability) for probability in probabilities if probability > 0)


def selected_logprob_from_logit_row(logits: list[float], selected_index: int) -> float:
    probabilities = softmax_row(logits)
    return math.log(probabilities[selected_index])


def top1_prob_and_margin(logits: list[float]) -> tuple[float, float]:
    probabilities = sorted(softmax_row(logits), reverse=True)
    if len(probabilities) == 1:
        return probabilities[0], probabilities[0]
    return probabilities[0], probabilities[0] - probabilities[1]


def rolling_mean(values: list[float], window_size: int = 5) -> list[float]:
    if not values:
        return []
    outputs = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window_size:
            running_sum -= values[index - window_size]
        divisor = min(index + 1, window_size)
        outputs.append(running_sum / divisor)
    return outputs


def render_token_pieces(tokenizer, token_ids: list[int]) -> tuple[list[str], list[tuple[int, int]], str]:
    token_texts: list[str] = []
    token_spans: list[tuple[int, int]] = []
    accumulated = ""
    for index in range(len(token_ids)):
        decoded = tokenizer.decode(token_ids[: index + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        piece = decoded[len(accumulated) :]
        start = len(accumulated)
        end = len(decoded)
        token_texts.append(piece)
        token_spans.append((start, end))
        accumulated = decoded
    return token_texts, token_spans, accumulated


def _load_transformers_model(model_name_or_path: str, device: str | None = None, trust_remote_code: bool = True):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Running rescoring requires torch and transformers to be installed") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(resolved_device)
    model.eval()
    return model, tokenizer, torch, resolved_device


def rescore_records(
    model_name_or_path: str,
    generations: list[dict[str, Any]],
    device: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model, tokenizer, torch, resolved_device = _load_transformers_model(model_name_or_path, device=device)
    import torch.nn.functional as F

    enriched_generations: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []

    for record in generations:
        prompt_ids = list(record.get("prompt_token_ids") or [])
        completion_ids = list(record.get("completion_token_ids") or [])
        if not prompt_ids:
            prompt_ids = tokenizer(str(record["prompt"]), add_special_tokens=True)["input_ids"]
        if not completion_ids:
            completion_ids = tokenizer(
                str(record.get("completion_text", "")),
                add_special_tokens=False,
            )["input_ids"]

        verification = verify_math_completion(str(record.get("completion_text", "")), str(record["ground_truth"]))
        enriched = dict(record)
        enriched.update(
            {
                "extracted_answer": verification.extracted_answer,
                "extraction_method": verification.extraction_method,
                "verification_method": verification.verification_method,
                "is_correct": verification.is_correct,
                "answer_span_start": verification.answer_start,
                "answer_span_end": verification.answer_end,
            }
        )
        enriched_generations.append(enriched)

        if not completion_ids:
            continue

        sequence_ids = prompt_ids + completion_ids
        input_tensor = torch.tensor([sequence_ids], dtype=torch.long, device=resolved_device)
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=attention_mask)

        logits = outputs.logits[:, :-1, :]
        start, end = completion_prediction_slice(len(prompt_ids), len(completion_ids))
        completion_logits = logits[:, start:end, :].float()
        labels = input_tensor[:, len(prompt_ids) : len(prompt_ids) + len(completion_ids)]

        log_probs = F.log_softmax(completion_logits, dim=-1)
        selected_logprobs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)[0].cpu().tolist()
        probabilities = torch.softmax(completion_logits, dim=-1)
        entropy = (torch.logsumexp(completion_logits, dim=-1) - (probabilities * completion_logits).sum(dim=-1))[0]
        top2_prob = torch.topk(probabilities[0], k=min(2, probabilities.shape[-1]), dim=-1).values.cpu()
        token_texts, token_spans, rendered_text = render_token_pieces(tokenizer, completion_ids)
        if not enriched.get("completion_text"):
            enriched["completion_text"] = rendered_text
        answer_span = None
        if verification.answer_start is not None and verification.answer_end is not None:
            answer_span = (verification.answer_start, verification.answer_end)
        segment_tags = classify_token_segments(token_spans, token_texts, answer_span)
        rolling_entropy = rolling_mean(entropy.cpu().tolist())

        for position, token_id in enumerate(completion_ids):
            token_rows.append(
                {
                    "sample_id": enriched["sample_id"],
                    "decode_id": enriched["decode_id"],
                    "position": position,
                    "relative_position": 0.0 if len(completion_ids) == 1 else position / (len(completion_ids) - 1),
                    "token_id": token_id,
                    "token_text": token_texts[position],
                    "logprob": selected_logprobs[position],
                    "entropy": float(entropy[position].item()),
                    "top1_prob": float(top2_prob[position, 0].item()),
                    "margin_1_2": float(
                        top2_prob[position, 0].item()
                        if top2_prob.shape[1] == 1
                        else (top2_prob[position, 0] - top2_prob[position, 1]).item()
                    ),
                    "rolling_entropy_mean": rolling_entropy[position],
                    "segment_tag": segment_tags[position],
                }
            )

    return enriched_generations, token_rows


def run_rescoring(
    model_name_or_path: str,
    problems_path: str | Path,
    generations_path: str | Path,
    output_dir: str | Path,
    device: str | None = None,
    run_name: str | None = None,
) -> Path:
    _ = load_records(problems_path)
    generations = load_records(generations_path)
    artifact_dir = prepare_artifact_dir(output_dir, run_name=run_name)
    enriched_generations, token_rows = rescore_records(
        model_name_or_path=model_name_or_path,
        generations=generations,
        device=device,
    )
    write_parquet_records(enriched_generations, artifact_dir / "generations.parquet")
    write_parquet_records(token_rows, artifact_dir / "token_metrics.parquet")
    return artifact_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teacher-force generated completions and emit token-level metrics.")
    parser.add_argument("--model", required=True, help="Model name or local path")
    parser.add_argument("--problems", required=True, help="Path to problems.parquet")
    parser.add_argument("--generations", required=True, help="Path to generations.parquet")
    parser.add_argument("--output-dir", required=True, help="Artifact directory")
    parser.add_argument("--run-name", default=None, help="Optional run subdirectory")
    parser.add_argument("--device", default=None, help="Override model device, e.g. cpu or cuda")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_rescoring(
        model_name_or_path=args.model,
        problems_path=args.problems,
        generations_path=args.generations,
        output_dir=args.output_dir,
        device=args.device,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
