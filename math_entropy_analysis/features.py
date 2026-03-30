"""Sequence-level feature aggregation for math-task entropy analysis."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from .dataset import load_records, prepare_artifact_dir, write_parquet_records


def _safe_mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _safe_std(values: list[float]) -> float | None:
    return pstdev(values) if len(values) > 1 else (0.0 if values else None)


def _partition_by_thirds(values: list[float]) -> tuple[list[float], list[float], list[float]]:
    if not values:
        return [], [], []
    early: list[float] = []
    middle: list[float] = []
    late: list[float] = []
    length = len(values)
    for index, value in enumerate(values):
        relative = index / max(length - 1, 1)
        if relative < (1 / 3):
            early.append(value)
        elif relative < (2 / 3):
            middle.append(value)
        else:
            late.append(value)
    return early, middle, late


def compute_sequence_features(
    generation_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped_tokens: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        grouped_tokens[(str(row["sample_id"]), int(row["decode_id"]))].append(dict(row))

    features: list[dict[str, Any]] = []
    for generation in generation_rows:
        key = (str(generation["sample_id"]), int(generation["decode_id"]))
        rows = sorted(grouped_tokens.get(key, []), key=lambda item: int(item["position"]))
        entropies = [float(row["entropy"]) for row in rows]
        logprobs = [float(row["logprob"]) for row in rows]
        answer_entropies = [float(row["entropy"]) for row in rows if row["segment_tag"] == "answer"]
        answer_logprobs = [float(row["logprob"]) for row in rows if row["segment_tag"] == "answer"]
        early, middle, late = _partition_by_thirds(entropies)

        mean_logprob = _safe_mean(logprobs)
        sequence_nll = -sum(logprobs) if logprobs else None
        perplexity = math.exp(-mean_logprob) if mean_logprob is not None else None
        mean_entropy = _safe_mean(entropies)
        late_entropy_mean = _safe_mean(late)
        early_entropy_mean = _safe_mean(early)
        entropy_collapse_ratio = None
        if early_entropy_mean not in (None, 0.0) and late_entropy_mean is not None:
            entropy_collapse_ratio = late_entropy_mean / early_entropy_mean

        features.append(
            {
                "sample_id": generation["sample_id"],
                "decode_id": generation["decode_id"],
                "is_correct": generation.get("is_correct"),
                "response_length": generation.get("response_length", len(rows)),
                "mean_entropy": mean_entropy,
                "early_entropy_mean": early_entropy_mean,
                "middle_entropy_mean": _safe_mean(middle),
                "late_entropy_mean": late_entropy_mean,
                "entropy_std": _safe_std(entropies),
                "entropy_collapse_ratio": entropy_collapse_ratio,
                "mean_logprob": mean_logprob,
                "sequence_nll": sequence_nll,
                "perplexity": perplexity,
                "answer_span_entropy_mean": _safe_mean(answer_entropies),
                "answer_span_logprob_mean": _safe_mean(answer_logprobs),
                "verification_method": generation.get("verification_method"),
                "extraction_method": generation.get("extraction_method"),
            }
        )

    return features


def run_feature_aggregation(
    generations_path: str | Path,
    token_metrics_path: str | Path,
    output_dir: str | Path,
    run_name: str | None = None,
) -> Path:
    generation_rows = load_records(generations_path)
    token_rows = load_records(token_metrics_path)
    sequence_features = compute_sequence_features(generation_rows, token_rows)
    artifact_dir = prepare_artifact_dir(output_dir, run_name=run_name)
    write_parquet_records(sequence_features, artifact_dir / "sequence_features.parquet")
    return artifact_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate token-level metrics into sequence-level math features.")
    parser.add_argument("--generations", required=True, help="Path to generations.parquet")
    parser.add_argument("--token-metrics", required=True, help="Path to token_metrics.parquet")
    parser.add_argument("--output-dir", required=True, help="Artifact directory")
    parser.add_argument("--run-name", default=None, help="Optional run subdirectory")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_feature_aggregation(
        generations_path=args.generations,
        token_metrics_path=args.token_metrics,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
