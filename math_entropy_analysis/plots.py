"""Basic plots for math-task entropy studies."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .dataset import load_records


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib to be installed") from exc
    return plt


def _split_by_correctness(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    correct = [row for row in rows if bool(row.get("is_correct"))]
    incorrect = [row for row in rows if not bool(row.get("is_correct"))]
    return correct, incorrect


def plot_entropy_distribution(sequence_features: list[dict], output_dir: Path) -> Path:
    plt = _require_matplotlib()
    correct, incorrect = _split_by_correctness(sequence_features)
    output_path = output_dir / "entropy_distribution.png"

    plt.figure(figsize=(8, 5))
    plt.hist([row["mean_entropy"] for row in correct if row.get("mean_entropy") is not None], bins=20, alpha=0.65, label="correct")
    plt.hist(
        [row["mean_entropy"] for row in incorrect if row.get("mean_entropy") is not None],
        bins=20,
        alpha=0.65,
        label="incorrect",
    )
    plt.xlabel("Mean entropy")
    plt.ylabel("Count")
    plt.title("Correct vs incorrect mean entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_length_vs_entropy(sequence_features: list[dict], output_dir: Path) -> Path:
    plt = _require_matplotlib()
    output_path = output_dir / "length_vs_entropy.png"
    colors = ["tab:blue" if bool(row.get("is_correct")) else "tab:red" for row in sequence_features]

    plt.figure(figsize=(8, 5))
    plt.scatter(
        [row.get("response_length", 0) for row in sequence_features],
        [row.get("mean_entropy", 0.0) for row in sequence_features],
        c=colors,
        alpha=0.75,
    )
    plt.xlabel("Response length")
    plt.ylabel("Mean entropy")
    plt.title("Length vs entropy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_entropy_trajectory(token_rows: list[dict], output_dir: Path, bins: int = 20) -> Path:
    plt = _require_matplotlib()
    output_path = output_dir / "entropy_trajectory.png"
    grouped: dict[bool, dict[int, list[float]]] = {True: defaultdict(list), False: defaultdict(list)}

    for row in token_rows:
        correctness = bool(row.get("is_correct", False))
        relative_position = float(row.get("relative_position", 0.0))
        bucket = min(int(relative_position * bins), bins - 1)
        grouped[correctness][bucket].append(float(row["entropy"]))

    plt.figure(figsize=(8, 5))
    for correctness, label, color in ((True, "correct", "tab:blue"), (False, "incorrect", "tab:red")):
        x_values = []
        y_values = []
        for bucket in range(bins):
            values = grouped[correctness].get(bucket, [])
            if not values:
                continue
            x_values.append(bucket / max(bins - 1, 1))
            y_values.append(sum(values) / len(values))
        if x_values:
            plt.plot(x_values, y_values, marker="o", label=label, color=color)

    plt.xlabel("Relative position")
    plt.ylabel("Mean entropy")
    plt.title("Entropy trajectory by correctness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_answer_vs_reasoning(sequence_features: list[dict], output_dir: Path) -> Path:
    plt = _require_matplotlib()
    output_path = output_dir / "answer_vs_reasoning_entropy.png"

    labels = ["overall", "answer_span"]
    correct, incorrect = _split_by_correctness(sequence_features)
    series = {
        "correct": [
            [row["mean_entropy"] for row in correct if row.get("mean_entropy") is not None],
            [row["answer_span_entropy_mean"] for row in correct if row.get("answer_span_entropy_mean") is not None],
        ],
        "incorrect": [
            [row["mean_entropy"] for row in incorrect if row.get("mean_entropy") is not None],
            [row["answer_span_entropy_mean"] for row in incorrect if row.get("answer_span_entropy_mean") is not None],
        ],
    }

    plt.figure(figsize=(8, 5))
    positions = [0, 1]
    for offset, (label, color) in enumerate((("correct", "tab:blue"), ("incorrect", "tab:red"))):
        means = []
        for values in series[label]:
            means.append(sum(values) / len(values) if values else 0.0)
        shifted = [position + (offset * 0.25) for position in positions]
        plt.bar(shifted, means, width=0.22, label=label, color=color)

    plt.xticks([position + 0.125 for position in positions], labels)
    plt.ylabel("Mean entropy")
    plt.title("Overall vs answer-span entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def run_plotting(sequence_features_path: str | Path, token_metrics_path: str | Path, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sequence_features = load_records(sequence_features_path)
    token_rows = load_records(token_metrics_path)

    correct_map = {(row["sample_id"], row["decode_id"]): bool(row.get("is_correct")) for row in sequence_features}
    enriched_token_rows = []
    for row in token_rows:
        enriched = dict(row)
        enriched["is_correct"] = correct_map.get((row["sample_id"], row["decode_id"]), False)
        enriched_token_rows.append(enriched)

    plot_entropy_distribution(sequence_features, output_path)
    plot_length_vs_entropy(sequence_features, output_path)
    plot_entropy_trajectory(enriched_token_rows, output_path)
    plot_answer_vs_reasoning(sequence_features, output_path)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create basic math entropy study plots.")
    parser.add_argument("--sequence-features", required=True, help="Path to sequence_features.parquet")
    parser.add_argument("--token-metrics", required=True, help="Path to token_metrics.parquet")
    parser.add_argument("--output-dir", required=True, help="Plot output directory")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_plotting(
        sequence_features_path=args.sequence_features,
        token_metrics_path=args.token_metrics,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
