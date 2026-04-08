"""Detailed high-entropy token analysis for math-task generations."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable

from .dataset import load_records, prepare_artifact_dir, write_parquet_records


def _safe_mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _safe_median(values: list[float]) -> float | None:
    return median(values) if values else None


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile from an empty entropy list")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be between 0 and 1, got {quantile}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sample_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["sample_id"]), int(row.get("decode_id", 0))


def _correctness_label(value: Any) -> str:
    if value is True:
        return "correct"
    if value is False:
        return "incorrect"
    return "unknown"


def _display_token_text(token_text: Any, limit: int = 80) -> str:
    raw = "" if token_text is None else str(token_text)
    display = repr(raw)
    if len(display) > limit:
        return display[: limit - 3] + "..."
    return display


def _preview_text(parts: Iterable[str], limit: int = 120) -> str:
    preview = "".join(parts).replace("\n", "\\n")
    if len(preview) > limit:
        return preview[: limit - 3] + "..."
    return preview


def _position_bucket(relative_position: float, bins: int) -> str:
    if bins <= 0:
        raise ValueError(f"position bins must be positive, got {bins}")
    clipped = min(max(relative_position, 0.0), 1.0)
    bucket_index = min(int(clipped * bins), bins - 1)
    start = int((bucket_index / bins) * 100)
    end = int((((bucket_index + 1) / bins) * 100) - 1)
    return f"{bucket_index:02d}:{start:02d}-{end:02d}%"


def resolve_high_entropy_threshold(
    entropies: list[float],
    entropy_threshold: float | None = None,
    entropy_quantile: float = 0.9,
) -> tuple[float, str]:
    if entropy_threshold is not None:
        return float(entropy_threshold), "absolute"
    return _quantile(entropies, entropy_quantile), "quantile"


def annotate_token_rows(
    generation_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    threshold_value: float,
    position_bins: int = 10,
) -> list[dict[str, Any]]:
    correctness_map = {
        (str(row["sample_id"]), int(row.get("decode_id", 0))): row.get("is_correct")
        for row in generation_rows
    }

    annotated_rows: list[dict[str, Any]] = []
    for row in token_rows:
        entropy = row.get("entropy")
        if entropy is None:
            continue
        sample_id = str(row["sample_id"])
        decode_id = int(row.get("decode_id", 0))
        relative_position = float(row.get("relative_position", 0.0) or 0.0)
        token_text = "" if row.get("token_text") is None else str(row.get("token_text"))
        correctness = correctness_map.get((sample_id, decode_id), row.get("is_correct"))

        annotated = dict(row)
        annotated.update(
            {
                "sample_id": sample_id,
                "decode_id": decode_id,
                "entropy": float(entropy),
                "relative_position": relative_position,
                "is_correct": correctness,
                "correctness_label": _correctness_label(correctness),
                "is_high_entropy": float(entropy) >= threshold_value,
                "token_text": token_text,
                "token_text_display": _display_token_text(token_text),
                "position_bucket": _position_bucket(relative_position, position_bins),
                "segment_tag": str(row.get("segment_tag", "unknown")),
            }
        )
        if annotated.get("logprob") is not None:
            annotated["logprob"] = float(annotated["logprob"])
        if annotated.get("top1_prob") is not None:
            annotated["top1_prob"] = float(annotated["top1_prob"])
        if annotated.get("margin_1_2") is not None:
            annotated["margin_1_2"] = float(annotated["margin_1_2"])
        if annotated.get("rolling_entropy_mean") is not None:
            annotated["rolling_entropy_mean"] = float(annotated["rolling_entropy_mean"])
        annotated_rows.append(annotated)

    annotated_rows.sort(key=lambda row: (_sample_key(row), int(row.get("position", 0))))
    return annotated_rows


def summarize_token_group(rows: list[dict[str, Any]], group_type: str, group_value: str) -> dict[str, Any]:
    sample_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_groups[_sample_key(row)].append(row)

    high_rows = [row for row in rows if bool(row.get("is_high_entropy"))]
    entropies = [float(row["entropy"]) for row in rows]
    high_entropies = [float(row["entropy"]) for row in high_rows]
    high_logprobs = [float(row["logprob"]) for row in high_rows if row.get("logprob") is not None]
    high_top1 = [float(row["top1_prob"]) for row in high_rows if row.get("top1_prob") is not None]
    high_margin = [float(row["margin_1_2"]) for row in high_rows if row.get("margin_1_2") is not None]

    sample_high_ratios: list[float] = []
    sample_high_counts: list[float] = []
    samples_with_high_entropy = 0
    for sample_rows in sample_groups.values():
        high_count = sum(1 for row in sample_rows if bool(row.get("is_high_entropy")))
        total_count = len(sample_rows)
        sample_high_counts.append(float(high_count))
        sample_high_ratios.append(high_count / total_count if total_count else 0.0)
        if high_count > 0:
            samples_with_high_entropy += 1

    return {
        "group_type": group_type,
        "group_value": group_value,
        "total_tokens": len(rows),
        "high_entropy_tokens": len(high_rows),
        "high_entropy_ratio": _safe_ratio(len(high_rows), len(rows)),
        "sample_count": len(sample_groups),
        "samples_with_high_entropy": samples_with_high_entropy,
        "sample_incidence_ratio": _safe_ratio(samples_with_high_entropy, len(sample_groups)),
        "unique_high_entropy_token_texts": len({row["token_text_display"] for row in high_rows}),
        "mean_entropy": _safe_mean(entropies),
        "median_entropy": _safe_median(entropies),
        "p90_entropy": _quantile(entropies, 0.90) if entropies else None,
        "p95_entropy": _quantile(entropies, 0.95) if entropies else None,
        "mean_high_entropy": _safe_mean(high_entropies),
        "median_high_entropy": _safe_median(high_entropies),
        "mean_high_entropy_logprob": _safe_mean(high_logprobs),
        "mean_high_entropy_top1_prob": _safe_mean(high_top1),
        "mean_high_entropy_margin_1_2": _safe_mean(high_margin),
        "avg_high_entropy_tokens_per_sample": _safe_mean(sample_high_counts),
        "avg_high_entropy_ratio_per_sample": _safe_mean(sample_high_ratios),
    }


def compute_group_stats(annotated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_rows: list[dict[str, Any]] = []
    grouping_specs = [
        ("global", lambda row: "all"),
        ("correctness", lambda row: row["correctness_label"]),
        ("segment", lambda row: row["segment_tag"]),
        ("correctness_segment", lambda row: f"{row['correctness_label']}::{row['segment_tag']}"),
        ("position_bucket", lambda row: row["position_bucket"]),
        ("correctness_position_bucket", lambda row: f"{row['correctness_label']}::{row['position_bucket']}"),
    ]

    for group_type, key_fn in grouping_specs:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in annotated_rows:
            buckets[key_fn(row)].append(row)
        for group_value, rows in sorted(buckets.items(), key=lambda item: item[0]):
            grouped_rows.append(summarize_token_group(rows, group_type=group_type, group_value=group_value))

    return grouped_rows


def extract_high_entropy_runs(annotated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in annotated_rows:
        sample_groups[_sample_key(row)].append(row)

    run_rows: list[dict[str, Any]] = []
    for (sample_id, decode_id), sample_rows in sample_groups.items():
        ordered = sorted(sample_rows, key=lambda row: int(row.get("position", 0)))
        run_index = 0
        current_run: list[dict[str, Any]] = []
        previous_position: int | None = None

        def flush_run() -> None:
            nonlocal current_run, run_index
            if not current_run:
                return
            entropies = [float(row["entropy"]) for row in current_run]
            segment_tags = {row["segment_tag"] for row in current_run}
            run_rows.append(
                {
                    "sample_id": sample_id,
                    "decode_id": decode_id,
                    "is_correct": current_run[0].get("is_correct"),
                    "correctness_label": current_run[0]["correctness_label"],
                    "run_index": run_index,
                    "start_position": int(current_run[0].get("position", 0)),
                    "end_position": int(current_run[-1].get("position", 0)),
                    "run_length": len(current_run),
                    "start_relative_position": float(current_run[0].get("relative_position", 0.0)),
                    "end_relative_position": float(current_run[-1].get("relative_position", 0.0)),
                    "mean_run_entropy": _safe_mean(entropies),
                    "max_run_entropy": max(entropies),
                    "segment_tag": current_run[0]["segment_tag"] if len(segment_tags) == 1 else "mixed",
                    "token_preview": _preview_text(str(row.get("token_text", "")) for row in current_run),
                }
            )
            run_index += 1
            current_run = []

        for row in ordered:
            position = int(row.get("position", 0))
            if bool(row.get("is_high_entropy")):
                if current_run and previous_position is not None and position != previous_position + 1:
                    flush_run()
                current_run.append(row)
            else:
                flush_run()
            previous_position = position
        flush_run()

    run_rows.sort(key=lambda row: (row["sample_id"], row["decode_id"], row["run_index"]))
    return run_rows


def compute_sample_stats(annotated_rows: list[dict[str, Any]], run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in annotated_rows:
        sample_groups[_sample_key(row)].append(row)

    run_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        run_groups[(str(row["sample_id"]), int(row["decode_id"]))].append(row)

    sample_stats: list[dict[str, Any]] = []
    for key, rows in sorted(sample_groups.items(), key=lambda item: item[0]):
        ordered = sorted(rows, key=lambda row: int(row.get("position", 0)))
        entropies = [float(row["entropy"]) for row in ordered]
        high_rows = [row for row in ordered if bool(row.get("is_high_entropy"))]
        high_positions = [int(row.get("position", 0)) for row in high_rows]
        sample_run_rows = run_groups.get(key, [])
        run_lengths = [int(row["run_length"]) for row in sample_run_rows]

        stats_row: dict[str, Any] = {
            "sample_id": key[0],
            "decode_id": key[1],
            "is_correct": ordered[0].get("is_correct"),
            "correctness_label": ordered[0]["correctness_label"],
            "token_count": len(ordered),
            "high_entropy_count": len(high_rows),
            "high_entropy_ratio": _safe_ratio(len(high_rows), len(ordered)),
            "mean_entropy": _safe_mean(entropies),
            "max_entropy": max(entropies) if entropies else None,
            "mean_high_entropy": _safe_mean([float(row["entropy"]) for row in high_rows]),
            "run_count": len(sample_run_rows),
            "longest_run_length": max(run_lengths) if run_lengths else 0,
            "mean_run_length": _safe_mean([float(length) for length in run_lengths]),
            "first_high_entropy_position": min(high_positions) if high_positions else None,
            "last_high_entropy_position": max(high_positions) if high_positions else None,
        }

        for segment_tag in ("reasoning", "answer", "other"):
            segment_rows = [row for row in ordered if row.get("segment_tag") == segment_tag]
            segment_high_rows = [row for row in segment_rows if bool(row.get("is_high_entropy"))]
            stats_row[f"{segment_tag}_token_count"] = len(segment_rows)
            stats_row[f"{segment_tag}_high_entropy_count"] = len(segment_high_rows)
            stats_row[f"{segment_tag}_high_entropy_ratio"] = _safe_ratio(len(segment_high_rows), len(segment_rows))

        sample_stats.append(stats_row)

    return sample_stats


def compute_token_text_stats(
    annotated_rows: list[dict[str, Any]],
    min_token_frequency: int = 2,
    top_k_token_texts: int = 50,
) -> list[dict[str, Any]]:
    text_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_text_map: dict[str, str] = {}
    for row in annotated_rows:
        key = row["token_text_display"]
        text_groups[key].append(row)
        raw_text_map.setdefault(key, row["token_text"])

    text_stats: list[dict[str, Any]] = []
    for display_text, rows in text_groups.items():
        if len(rows) < min_token_frequency:
            continue
        high_rows = [row for row in rows if bool(row.get("is_high_entropy"))]
        text_stats.append(
            {
                "token_text": raw_text_map[display_text],
                "token_text_display": display_text,
                "total_count": len(rows),
                "high_entropy_count": len(high_rows),
                "high_entropy_ratio": _safe_ratio(len(high_rows), len(rows)),
                "mean_entropy": _safe_mean([float(row["entropy"]) for row in rows]),
                "median_entropy": _safe_median([float(row["entropy"]) for row in rows]),
                "mean_high_entropy": _safe_mean([float(row["entropy"]) for row in high_rows]),
                "sample_count": len({_sample_key(row) for row in rows}),
                "correct_total_count": sum(1 for row in rows if row["correctness_label"] == "correct"),
                "incorrect_total_count": sum(1 for row in rows if row["correctness_label"] == "incorrect"),
                "correct_high_entropy_count": sum(1 for row in high_rows if row["correctness_label"] == "correct"),
                "incorrect_high_entropy_count": sum(1 for row in high_rows if row["correctness_label"] == "incorrect"),
            }
        )

    text_stats.sort(
        key=lambda row: (
            -(row["high_entropy_count"] or 0),
            -(row["high_entropy_ratio"] or 0.0),
            -(row["total_count"] or 0),
            row["token_text_display"],
        )
    )
    return text_stats[:top_k_token_texts]


def summarize_runs(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_lengths = [int(row["run_length"]) for row in run_rows]
    return {
        "total_runs": len(run_rows),
        "mean_run_length": _safe_mean([float(value) for value in run_lengths]),
        "median_run_length": _safe_median([float(value) for value in run_lengths]),
        "p90_run_length": _quantile([float(value) for value in run_lengths], 0.90) if run_lengths else None,
        "max_run_length": max(run_lengths) if run_lengths else 0,
        "correct_run_count": sum(1 for row in run_rows if row["correctness_label"] == "correct"),
        "incorrect_run_count": sum(1 for row in run_rows if row["correctness_label"] == "incorrect"),
    }


def _find_group_row(group_rows: list[dict[str, Any]], group_type: str, group_value: str) -> dict[str, Any] | None:
    for row in group_rows:
        if row["group_type"] == group_type and row["group_value"] == group_value:
            return row
    return None


def _top_records(rows: list[dict[str, Any]], key_names: list[str], limit: int = 10) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows[:limit]:
        output.append({key: row.get(key) for key in key_names})
    return output


def build_summary(
    group_rows: list[dict[str, Any]],
    sample_stats: list[dict[str, Any]],
    token_text_stats: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    threshold_value: float,
    threshold_mode: str,
    entropy_quantile: float,
) -> dict[str, Any]:
    correct_row = _find_group_row(group_rows, "correctness", "correct") or {}
    incorrect_row = _find_group_row(group_rows, "correctness", "incorrect") or {}
    high_entropy_token_text_stats = [row for row in token_text_stats if (row.get("high_entropy_count") or 0) > 0]

    return {
        "threshold": {
            "mode": threshold_mode,
            "value": threshold_value,
            "quantile": entropy_quantile if threshold_mode == "quantile" else None,
        },
        "global": _find_group_row(group_rows, "global", "all") or {},
        "correctness": {
            "correct": correct_row,
            "incorrect": incorrect_row,
            "unknown": _find_group_row(group_rows, "correctness", "unknown") or {},
        },
        "correctness_gap": {
            "high_entropy_ratio_incorrect_minus_correct": (
                (incorrect_row.get("high_entropy_ratio") or 0.0) - (correct_row.get("high_entropy_ratio") or 0.0)
            ),
            "sample_incidence_ratio_incorrect_minus_correct": (
                (incorrect_row.get("sample_incidence_ratio") or 0.0) - (correct_row.get("sample_incidence_ratio") or 0.0)
            ),
            "mean_entropy_incorrect_minus_correct": (
                (incorrect_row.get("mean_entropy") or 0.0) - (correct_row.get("mean_entropy") or 0.0)
            ),
        },
        "runs": summarize_runs(run_rows),
        "top_samples_by_high_entropy_ratio": _top_records(
            sorted(sample_stats, key=lambda row: (-(row.get("high_entropy_ratio") or 0.0), -(row.get("high_entropy_count") or 0))),
            ["sample_id", "decode_id", "correctness_label", "high_entropy_count", "high_entropy_ratio", "longest_run_length"],
        ),
        "top_samples_by_high_entropy_count": _top_records(
            sorted(sample_stats, key=lambda row: (-(row.get("high_entropy_count") or 0), -(row.get("high_entropy_ratio") or 0.0))),
            ["sample_id", "decode_id", "correctness_label", "high_entropy_count", "high_entropy_ratio", "longest_run_length"],
        ),
        "top_token_texts_by_high_entropy_count": _top_records(
            high_entropy_token_text_stats,
            ["token_text_display", "total_count", "high_entropy_count", "high_entropy_ratio", "mean_entropy", "mean_high_entropy"],
        ),
        "top_token_texts_by_mean_high_entropy": _top_records(
            sorted(
                high_entropy_token_text_stats,
                key=lambda row: (
                    -(row.get("mean_high_entropy") if row.get("mean_high_entropy") is not None else float("-inf")),
                    -(row.get("high_entropy_count") or 0),
                    row.get("token_text_display") or "",
                ),
                reverse=False,
            ),
            ["token_text_display", "total_count", "high_entropy_count", "high_entropy_ratio", "mean_entropy", "mean_high_entropy"],
        ),
        "top_runs_by_length": _top_records(
            sorted(run_rows, key=lambda row: (-(row.get("run_length") or 0), -(row.get("mean_run_entropy") or 0.0))),
            ["sample_id", "decode_id", "correctness_label", "run_length", "start_position", "end_position", "token_preview"],
        ),
    }


def analyze_high_entropy_tokens(
    generation_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    entropy_threshold: float | None = None,
    entropy_quantile: float = 0.9,
    position_bins: int = 10,
    min_token_frequency: int = 2,
    top_k_token_texts: int = 50,
) -> dict[str, Any]:
    entropies = [float(row["entropy"]) for row in token_rows if row.get("entropy") is not None]
    threshold_value, threshold_mode = resolve_high_entropy_threshold(
        entropies,
        entropy_threshold=entropy_threshold,
        entropy_quantile=entropy_quantile,
    )
    annotated_rows = annotate_token_rows(
        generation_rows=generation_rows,
        token_rows=token_rows,
        threshold_value=threshold_value,
        position_bins=position_bins,
    )
    group_rows = compute_group_stats(annotated_rows)
    run_rows = extract_high_entropy_runs(annotated_rows)
    sample_stats = compute_sample_stats(annotated_rows, run_rows)
    token_text_stats = compute_token_text_stats(
        annotated_rows,
        min_token_frequency=min_token_frequency,
        top_k_token_texts=top_k_token_texts,
    )
    summary = build_summary(
        group_rows=group_rows,
        sample_stats=sample_stats,
        token_text_stats=token_text_stats,
        run_rows=run_rows,
        threshold_value=threshold_value,
        threshold_mode=threshold_mode,
        entropy_quantile=entropy_quantile,
    )
    return {
        "summary": summary,
        "annotated_token_rows": annotated_rows,
        "group_rows": group_rows,
        "sample_stats": sample_stats,
        "token_text_stats": token_text_stats,
        "run_rows": run_rows,
    }


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("High-entropy plots require matplotlib to be installed") from exc
    return plt


def _plot_high_entropy_ratio_by_correctness(group_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    plt = _require_matplotlib()
    output_path = output_dir / "high_entropy_ratio_by_correctness.png"
    labels = ["global", "correct", "incorrect"]
    values = []
    for group_type, group_value in (("global", "all"), ("correctness", "correct"), ("correctness", "incorrect")):
        row = _find_group_row(group_rows, group_type, group_value) or {}
        values.append(row.get("high_entropy_ratio") or 0.0)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=["tab:gray", "tab:blue", "tab:red"])
    plt.ylabel("High-entropy token ratio")
    plt.title("High-entropy ratio by correctness")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _plot_sample_ratio_histogram(sample_stats: list[dict[str, Any]], output_dir: Path) -> Path:
    plt = _require_matplotlib()
    output_path = output_dir / "sample_high_entropy_ratio_histogram.png"
    correct = [row["high_entropy_ratio"] for row in sample_stats if row["correctness_label"] == "correct" and row.get("high_entropy_ratio") is not None]
    incorrect = [row["high_entropy_ratio"] for row in sample_stats if row["correctness_label"] == "incorrect" and row.get("high_entropy_ratio") is not None]

    plt.figure(figsize=(8, 5))
    if correct:
        plt.hist(correct, bins=20, alpha=0.6, label="correct")
    if incorrect:
        plt.hist(incorrect, bins=20, alpha=0.6, label="incorrect")
    plt.xlabel("Sample-level high-entropy ratio")
    plt.ylabel("Count")
    plt.title("Distribution of sample high-entropy ratios")
    if correct or incorrect:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _plot_position_ratio(group_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    plt = _require_matplotlib()
    output_path = output_dir / "high_entropy_ratio_by_position.png"

    plt.figure(figsize=(9, 5))
    for label, color in (("correct", "tab:blue"), ("incorrect", "tab:red")):
        rows = [
            row
            for row in group_rows
            if row["group_type"] == "correctness_position_bucket" and row["group_value"].startswith(f"{label}::")
        ]
        rows.sort(key=lambda row: row["group_value"])
        x_values = [row["group_value"].split("::", 1)[1] for row in rows]
        y_values = [row.get("high_entropy_ratio") or 0.0 for row in rows]
        if x_values:
            plt.plot(x_values, y_values, marker="o", label=label, color=color)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("High-entropy token ratio")
    plt.title("High-entropy ratio across relative positions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _plot_segment_ratio(group_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    plt = _require_matplotlib()
    output_path = output_dir / "high_entropy_ratio_by_segment.png"
    segments = ["reasoning", "answer", "other"]
    base_positions = list(range(len(segments)))

    plt.figure(figsize=(8, 5))
    for offset, (label, color) in enumerate((("correct", "tab:blue"), ("incorrect", "tab:red"))):
        values = []
        for segment in segments:
            row = _find_group_row(group_rows, "correctness_segment", f"{label}::{segment}") or {}
            values.append(row.get("high_entropy_ratio") or 0.0)
        shifted = [position + (offset * 0.3) for position in base_positions]
        plt.bar(shifted, values, width=0.28, label=label, color=color)

    plt.xticks([position + 0.15 for position in base_positions], segments)
    plt.ylabel("High-entropy token ratio")
    plt.title("High-entropy ratio by segment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def write_high_entropy_plots(group_rows: list[dict[str, Any]], sample_stats: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return [
        _plot_high_entropy_ratio_by_correctness(group_rows, plot_dir),
        _plot_sample_ratio_histogram(sample_stats, plot_dir),
        _plot_position_ratio(group_rows, plot_dir),
        _plot_segment_ratio(group_rows, plot_dir),
    ]


def _write_json_object(value: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _format_optional_float(value: Any, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def render_console_summary(summary: dict[str, Any]) -> str:
    threshold = summary.get("threshold", {})

    lines = [
        "========== High-Entropy Token Text Tops ==========",
        (
            "Threshold: "
            f"mode={threshold.get('mode', 'n/a')}, "
            f"value={_format_optional_float(threshold.get('value'))}, "
            f"quantile={_format_optional_float(threshold.get('quantile'))}"
        ),
        "Top Token Texts By High-Entropy Count:",
    ]

    for row in summary.get("top_token_texts_by_high_entropy_count", [])[:10]:
        lines.append(
            (
                f"  token={row.get('token_text_display')}, total_count={row.get('total_count')}, "
                f"high_entropy_count={row.get('high_entropy_count')}, "
                f"high_entropy_ratio={_format_optional_float(row.get('high_entropy_ratio'))}, "
                f"mean_high_entropy={_format_optional_float(row.get('mean_high_entropy'))}"
            )
        )

    lines.append("Top Token Texts By Mean High Entropy:")
    for row in summary.get("top_token_texts_by_mean_high_entropy", [])[:10]:
        lines.append(
            (
                f"  token={row.get('token_text_display')}, total_count={row.get('total_count')}, "
                f"high_entropy_count={row.get('high_entropy_count')}, "
                f"mean_high_entropy={_format_optional_float(row.get('mean_high_entropy'))}, "
                f"mean_entropy={_format_optional_float(row.get('mean_entropy'))}"
            )
        )

    lines.append("===================================================")
    return "\n".join(lines)


def run_high_entropy_analysis(
    generations_path: str | Path,
    token_metrics_path: str | Path,
    output_dir: str | Path,
    run_name: str | None = None,
    entropy_threshold: float | None = None,
    entropy_quantile: float = 0.9,
    position_bins: int = 10,
    min_token_frequency: int = 2,
    top_k_token_texts: int = 50,
    write_plots: bool = True,
    print_summary: bool = True,
) -> Path:
    generation_rows = load_records(generations_path)
    token_rows = load_records(token_metrics_path)
    artifact_dir = prepare_artifact_dir(output_dir, run_name=run_name)
    analysis = analyze_high_entropy_tokens(
        generation_rows=generation_rows,
        token_rows=token_rows,
        entropy_threshold=entropy_threshold,
        entropy_quantile=entropy_quantile,
        position_bins=position_bins,
        min_token_frequency=min_token_frequency,
        top_k_token_texts=top_k_token_texts,
    )

    write_parquet_records(analysis["annotated_token_rows"], artifact_dir / "high_entropy_token_metrics.parquet")
    write_parquet_records(analysis["group_rows"], artifact_dir / "high_entropy_group_stats.parquet")
    write_parquet_records(analysis["sample_stats"], artifact_dir / "high_entropy_sample_stats.parquet")
    write_parquet_records(analysis["token_text_stats"], artifact_dir / "high_entropy_token_text_stats.parquet")
    write_parquet_records(analysis["run_rows"], artifact_dir / "high_entropy_runs.parquet")
    _write_json_object(analysis["summary"], artifact_dir / "high_entropy_summary.json")

    if write_plots:
        try:
            write_high_entropy_plots(
                group_rows=analysis["group_rows"],
                sample_stats=analysis["sample_stats"],
                output_dir=artifact_dir / "high_entropy_plots",
            )
        except ImportError:
            pass

    if print_summary:
        print(render_console_summary(analysis["summary"]))

    return artifact_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze high-entropy token behavior in math generations.")
    parser.add_argument("--generations", required=True, help="Path to generations.parquet")
    parser.add_argument("--token-metrics", required=True, help="Path to token_metrics.parquet")
    parser.add_argument("--output-dir", required=True, help="Artifact directory")
    parser.add_argument("--run-name", default=None, help="Optional run subdirectory")
    parser.add_argument("--entropy-threshold", type=float, default=None, help="Absolute entropy threshold for high-entropy tokens")
    parser.add_argument("--entropy-quantile", type=float, default=0.9, help="Use the global entropy quantile when --entropy-threshold is not set")
    parser.add_argument("--position-bins", type=int, default=10, help="Number of relative-position buckets")
    parser.add_argument("--min-token-frequency", type=int, default=2, help="Minimum token-text frequency for token_text stats")
    parser.add_argument("--top-k-token-texts", type=int, default=50, help="Maximum number of token_text stats rows to keep")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation even if matplotlib is installed")
    parser.add_argument("--quiet-summary", action="store_true", help="Do not print the human-readable summary to stdout")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_high_entropy_analysis(
        generations_path=args.generations,
        token_metrics_path=args.token_metrics,
        output_dir=args.output_dir,
        run_name=args.run_name,
        entropy_threshold=args.entropy_threshold,
        entropy_quantile=args.entropy_quantile,
        position_bins=args.position_bins,
        min_token_frequency=args.min_token_frequency,
        top_k_token_texts=args.top_k_token_texts,
        write_plots=not args.skip_plots,
        print_summary=not args.quiet_summary,
    )


if __name__ == "__main__":
    main()
