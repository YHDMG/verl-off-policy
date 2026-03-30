"""Generation stage for math-task entropy analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .dataset import load_problem_records, prepare_artifact_dir, write_parquet_records


@dataclass(slots=True)
class DecodeConfig:
    decode_mode: str = "greedy"
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    do_sample: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DecodeConfig":
        decode_mode = args.decode_mode.lower()
        if decode_mode not in {"greedy", "sampling"}:
            raise ValueError(f"Unsupported decode mode: {args.decode_mode}")
        return cls(
            decode_mode=decode_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0 if decode_mode == "greedy" else args.temperature,
            top_p=1.0 if decode_mode == "greedy" else args.top_p,
            num_return_sequences=args.num_return_sequences,
            repetition_penalty=args.repetition_penalty,
            do_sample=decode_mode == "sampling",
        )

    def generation_kwargs(self) -> dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_return_sequences": self.num_return_sequences,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
        }


def _load_transformers_model(model_name_or_path: str, device: str | None = None, trust_remote_code: bool = True):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Running generation requires torch and transformers to be installed") from exc

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


def _trim_special_tokens(tokenizer, token_ids: list[int], logprobs: list[float]) -> tuple[list[int], list[float]]:
    trimmed_ids = list(token_ids)
    trimmed_logprobs = list(logprobs)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    while trimmed_ids and trimmed_ids[-1] in special_ids:
        trimmed_ids.pop()
        if trimmed_logprobs:
            trimmed_logprobs.pop()
    return trimmed_ids, trimmed_logprobs


def generate_with_transformers(
    model_name_or_path: str,
    problems: list[dict[str, Any]],
    decode_config: DecodeConfig,
    batch_size: int = 1,
    device: str | None = None,
    trust_remote_code: bool = True,
) -> list[dict[str, Any]]:
    model, tokenizer, torch, resolved_device = _load_transformers_model(
        model_name_or_path=model_name_or_path,
        device=device,
        trust_remote_code=trust_remote_code,
    )

    generation_rows: list[dict[str, Any]] = []
    generation_kwargs = decode_config.generation_kwargs()

    for batch_start in range(0, len(problems), batch_size):
        batch = problems[batch_start : batch_start + batch_size]
        prompts = [str(record["prompt"]) for record in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(resolved_device)
        attention_mask = encoded["attention_mask"].to(resolved_device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **generation_kwargs,
            )

        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        sequences = outputs.sequences.cpu()
        transition_scores = transition_scores.cpu()
        prompt_width = input_ids.shape[1]

        expanded_batch_size = len(batch) * decode_config.num_return_sequences
        for row_index in range(expanded_batch_size):
            problem_index = row_index // decode_config.num_return_sequences
            decode_index = row_index % decode_config.num_return_sequences
            problem = batch[problem_index]

            prompt_ids = input_ids[problem_index][attention_mask[problem_index].bool()].detach().cpu().tolist()
            completion_ids = sequences[row_index, prompt_width:].tolist()
            completion_logprobs = transition_scores[row_index].tolist()
            completion_ids, completion_logprobs = _trim_special_tokens(tokenizer, completion_ids, completion_logprobs)
            completion_text = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            generation_rows.append(
                {
                    "sample_id": str(problem["sample_id"]),
                    "decode_id": decode_index,
                    "prompt": str(problem["prompt"]),
                    "ground_truth": str(problem["ground_truth"]),
                    "prompt_token_ids": prompt_ids,
                    "completion_token_ids": completion_ids,
                    "completion_text": completion_text,
                    "sampled_token_logprobs": completion_logprobs[: len(completion_ids)],
                    "response_length": len(completion_ids),
                    "stop_reason": "length" if len(completion_ids) >= decode_config.max_new_tokens else "eos_or_stop",
                    "backend": "transformers",
                    "decode_config": json.dumps(asdict(decode_config), ensure_ascii=False, sort_keys=True),
                }
            )

    return generation_rows


def generate_with_vllm(
    model_name_or_path: str,
    problems: list[dict[str, Any]],
    decode_config: DecodeConfig,
) -> list[dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError("Running vLLM generation requires vllm and transformers to be installed") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    llm = LLM(model=model_name_or_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        max_tokens=decode_config.max_new_tokens,
        temperature=decode_config.temperature,
        top_p=decode_config.top_p,
        repetition_penalty=decode_config.repetition_penalty,
        n=decode_config.num_return_sequences,
        logprobs=1,
    )

    prompts = [str(record["prompt"]) for record in problems]
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

    generation_rows: list[dict[str, Any]] = []
    for problem, output in zip(problems, outputs, strict=True):
        prompt_token_ids = tokenizer(str(problem["prompt"]), add_special_tokens=True)["input_ids"]
        for decode_index, candidate in enumerate(output.outputs):
            completion_ids = list(candidate.token_ids)
            completion_logprobs = []
            if getattr(candidate, "logprobs", None) is not None:
                for token_id, token_logprob in zip(completion_ids, candidate.logprobs, strict=True):
                    completion_logprobs.append(token_logprob[token_id].logprob)
            completion_ids, completion_logprobs = _trim_special_tokens(tokenizer, completion_ids, completion_logprobs)
            completion_text = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            generation_rows.append(
                {
                    "sample_id": str(problem["sample_id"]),
                    "decode_id": decode_index,
                    "prompt": str(problem["prompt"]),
                    "ground_truth": str(problem["ground_truth"]),
                    "prompt_token_ids": prompt_token_ids,
                    "completion_token_ids": completion_ids,
                    "completion_text": completion_text,
                    "sampled_token_logprobs": completion_logprobs[: len(completion_ids)],
                    "response_length": len(completion_ids),
                    "stop_reason": str(candidate.finish_reason),
                    "backend": "vllm",
                    "decode_config": json.dumps(asdict(decode_config), ensure_ascii=False, sort_keys=True),
                }
            )
    return generation_rows


def run_generation(
    model_name_or_path: str,
    problems_path: str | Path,
    output_dir: str | Path,
    decode_config: DecodeConfig,
    backend: str = "transformers",
    batch_size: int = 1,
    device: str | None = None,
    run_name: str | None = None,
    max_samples: int | None = None,
) -> Path:
    problems = load_problem_records(problems_path)
    if max_samples is not None:
        if max_samples < 0:
            raise ValueError(f"max_samples must be >= 0, got {max_samples}")
        problems = problems[:max_samples]
    artifact_dir = prepare_artifact_dir(output_dir, run_name=run_name)

    if backend == "transformers":
        generations = generate_with_transformers(
            model_name_or_path=model_name_or_path,
            problems=problems,
            decode_config=decode_config,
            batch_size=batch_size,
            device=device,
        )
    elif backend == "vllm":
        generations = generate_with_vllm(
            model_name_or_path=model_name_or_path,
            problems=problems,
            decode_config=decode_config,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    write_parquet_records(problems, artifact_dir / "problems.parquet")
    write_parquet_records(generations, artifact_dir / "generations.parquet")
    return artifact_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate math-task completions and save raw trajectories.")
    parser.add_argument("--model", required=True, help="Model name or local path")
    parser.add_argument("--problems", required=True, help="Path to normalized problems file")
    parser.add_argument("--output-dir", required=True, help="Artifact directory")
    parser.add_argument("--run-name", default=None, help="Optional run subdirectory")
    parser.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    parser.add_argument("--decode-mode", default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None, help="Optionally limit the number of problems to process")
    parser.add_argument("--device", default=None, help="Override model device, e.g. cpu or cuda")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    decode_config = DecodeConfig.from_args(args)
    run_generation(
        model_name_or_path=args.model,
        problems_path=args.problems,
        output_dir=args.output_dir,
        decode_config=decode_config,
        backend=args.backend,
        batch_size=args.batch_size,
        device=args.device,
        run_name=args.run_name,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
