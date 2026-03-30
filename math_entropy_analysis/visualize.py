"""Single-sample entropy heatmap rendering."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

from .dataset import load_records


def _normalize(value: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.0
    clipped = min(max(value, vmin), vmax)
    return (clipped - vmin) / (vmax - vmin)


def entropy_to_colors(value: float, vmin: float, vmax: float) -> tuple[str, str]:
    normalized = _normalize(value, vmin, vmax)
    lightness = 96.0 - normalized * 60.0
    background = f"hsl(14 85% {lightness:.1f}%)"
    foreground = "#111827" if lightness >= 58 else "#f9fafb"
    return background, foreground


def html_escape_preserve_whitespace(text: str) -> str:
    escaped = html.escape(text)
    escaped = escaped.replace(" ", "&nbsp;")
    escaped = escaped.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
    escaped = escaped.replace("\n", "<br/>\n")
    return escaped if escaped else "&nbsp;"


def render_heatmap_token(token_row: dict[str, Any], vmin: float, vmax: float) -> str:
    token_text = str(token_row.get("token_text", ""))
    entropy = float(token_row.get("entropy", 0.0))
    logprob = token_row.get("logprob")
    position = token_row.get("position")
    segment_tag = token_row.get("segment_tag", "")
    background, foreground = entropy_to_colors(entropy, vmin=vmin, vmax=vmax)
    logprob_text = "n/a" if logprob is None else f"{float(logprob):.4f}"
    title_text = f"pos={position} | entropy={entropy:.4f} | logprob={logprob_text} | segment={segment_tag}"
    title = html.escape(title_text)
    return (
        f'<span class="token" title="{title}" '
        f'style="background:{background};color:{foreground};">{html_escape_preserve_whitespace(token_text)}</span>'
    )


def render_entropy_heatmap_html(
    token_rows: list[dict[str, Any]],
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    if not token_rows:
        raise ValueError("token_rows cannot be empty")

    sorted_rows = sorted(token_rows, key=lambda row: int(row.get("position", 0)))
    entropies = [float(row.get("entropy", 0.0)) for row in sorted_rows]
    resolved_vmin = min(entropies) if vmin is None else vmin
    resolved_vmax = max(entropies) if vmax is None else vmax
    token_html = "".join(render_heatmap_token(row, resolved_vmin, resolved_vmax) for row in sorted_rows)

    prompt_block = ""
    summary_rows = ""
    if metadata:
        prompt = metadata.get("prompt")
        if prompt:
            prompt_block = (
                '<div class="card"><div class="label">Prompt</div>'
                f'<div class="prompt">{html_escape_preserve_whitespace(str(prompt))}</div></div>'
            )
        fields = [
            ("sample_id", metadata.get("sample_id")),
            ("decode_id", metadata.get("decode_id")),
            ("is_correct", metadata.get("is_correct")),
            ("extracted_answer", metadata.get("extracted_answer")),
            ("verification_method", metadata.get("verification_method")),
        ]
        parts = []
        for key, value in fields:
            if value is None:
                continue
            parts.append(f"<div><span class=\"meta-key\">{html.escape(str(key))}</span>: {html.escape(str(value))}</div>")
        summary_rows = "".join(parts)

    page_title = html.escape(title or "Entropy Heatmap")
    legend_min = f"{resolved_vmin:.3f}"
    legend_max = f"{resolved_vmax:.3f}"
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{page_title}</title>
  <style>
    body {{ font-family: 'Segoe UI', 'PingFang SC', sans-serif; margin: 24px; background: #fffaf5; color: #1f2937; }}
    h1 {{ margin: 0 0 16px 0; font-size: 28px; }}
    .layout {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; }}
    .card {{ background: white; border: 1px solid #f2d3b4; border-radius: 16px; padding: 16px 18px; box-shadow: 0 8px 24px rgba(120, 53, 15, 0.06); }}
    .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: #9a3412; margin-bottom: 10px; }}
    .prompt {{ line-height: 1.6; white-space: normal; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 14px 18px; font-size: 14px; }}
    .meta-key {{ color: #9a3412; font-weight: 600; }}
    .legend {{ display: flex; align-items: center; gap: 12px; font-size: 13px; }}
    .legend-bar {{ width: 280px; height: 14px; border-radius: 999px; background: linear-gradient(90deg, hsl(14 85% 96%), hsl(14 85% 36%)); border: 1px solid rgba(154, 52, 18, 0.2); }}
    .tokens {{ line-height: 2.1; font-size: 20px; word-break: break-word; }}
    .token {{ display: inline; padding: 0.12em 0.18em; border-radius: 6px; box-decoration-break: clone; -webkit-box-decoration-break: clone; }}
  </style>
</head>
<body>
  <div class=\"layout\">
    <div>
      <h1>{page_title}</h1>
    </div>
    {prompt_block}
    <div class=\"card\">
      <div class=\"label\">Summary</div>
      <div class=\"meta\">{summary_rows}</div>
    </div>
    <div class=\"card\">
      <div class=\"label\">Entropy Scale</div>
      <div class=\"legend\"><span>{legend_min}</span><div class=\"legend-bar\"></div><span>{legend_max}</span></div>
    </div>
    <div class=\"card\">
      <div class=\"label\">Token Heatmap</div>
      <div class=\"tokens\">{token_html}</div>
    </div>
  </div>
</body>
</html>
"""


def _select_sample_rows(token_rows: list[dict[str, Any]], sample_id: str | None, decode_id: int | None) -> tuple[list[dict[str, Any]], tuple[str, int]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in token_rows:
        key = (str(row["sample_id"]), int(row["decode_id"]))
        grouped.setdefault(key, []).append(row)

    if not grouped:
        raise ValueError("No token rows found")

    if sample_id is None:
        key = sorted(grouped.keys())[0]
        return grouped[key], key

    resolved_key = (str(sample_id), 0 if decode_id is None else int(decode_id))
    if resolved_key not in grouped:
        raise KeyError(f"Sample not found in token rows: {resolved_key}")
    return grouped[resolved_key], resolved_key


def write_entropy_heatmap(
    token_metrics_path: str | Path,
    output_path: str | Path,
    sample_id: str | None = None,
    decode_id: int | None = None,
    generations_path: str | Path | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    token_rows = load_records(token_metrics_path)
    selected_rows, key = _select_sample_rows(token_rows, sample_id=sample_id, decode_id=decode_id)

    metadata = {"sample_id": key[0], "decode_id": key[1]}
    if generations_path is not None:
        for row in load_records(generations_path):
            if str(row["sample_id"]) == key[0] and int(row["decode_id"]) == key[1]:
                metadata.update(row)
                break

    html_text = render_entropy_heatmap_html(
        token_rows=selected_rows,
        title=title or f"Entropy Heatmap | sample={key[0]} decode={key[1]}",
        metadata=metadata,
        vmin=vmin,
        vmax=vmax,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html_text, encoding="utf-8")
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a single-sample entropy heatmap as HTML.")
    parser.add_argument("--token-metrics", required=True, help="Path to token_metrics.parquet")
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument("--sample-id", default=None, help="Sample id to render; defaults to the first available sample")
    parser.add_argument("--decode-id", type=int, default=None, help="Decode id within the sample")
    parser.add_argument("--generations", default=None, help="Optional generations.parquet for prompt and correctness metadata")
    parser.add_argument("--title", default=None, help="Optional page title")
    parser.add_argument("--vmin", type=float, default=None, help="Lower bound for entropy color scaling")
    parser.add_argument("--vmax", type=float, default=None, help="Upper bound for entropy color scaling")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    write_entropy_heatmap(
        token_metrics_path=args.token_metrics,
        output_path=args.output,
        sample_id=args.sample_id,
        decode_id=args.decode_id,
        generations_path=args.generations,
        title=args.title,
        vmin=args.vmin,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
