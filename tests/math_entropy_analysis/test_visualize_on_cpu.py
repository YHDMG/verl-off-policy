from pathlib import Path

from math_entropy_analysis.visualize import entropy_to_colors, render_entropy_heatmap_html, write_entropy_heatmap


def test_entropy_to_colors_darkens_with_higher_entropy():
    low_bg, _ = entropy_to_colors(0.1, vmin=0.0, vmax=1.0)
    high_bg, _ = entropy_to_colors(0.9, vmin=0.0, vmax=1.0)

    assert low_bg != high_bg


def test_render_entropy_heatmap_html_contains_token_spans():
    html_text = render_entropy_heatmap_html(
        token_rows=[
            {"position": 0, "token_text": "Hello", "entropy": 0.1, "logprob": -0.2, "segment_tag": "reasoning"},
            {"position": 1, "token_text": " world", "entropy": 0.8, "logprob": -0.7, "segment_tag": "answer"},
        ],
        title="Demo",
        metadata={"sample_id": "s1", "decode_id": 0},
    )

    assert "Token Heatmap" in html_text
    assert "Hello" in html_text
    assert "world" in html_text


def test_write_entropy_heatmap_reads_records_and_writes_html(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    token_metrics = [
        {"sample_id": "s1", "decode_id": 0, "position": 0, "token_text": "A", "entropy": 0.2, "logprob": -0.1, "segment_tag": "reasoning"},
        {"sample_id": "s1", "decode_id": 0, "position": 1, "token_text": " B", "entropy": 0.9, "logprob": -0.9, "segment_tag": "answer"},
    ]
    generations = [
        {"sample_id": "s1", "decode_id": 0, "prompt": "Q", "is_correct": True, "extracted_answer": "B", "verification_method": "rule"}
    ]

    token_path = tmp_path / "token_metrics.json"
    gen_path = tmp_path / "generations.json"
    token_path.write_text(__import__("json").dumps(token_metrics, ensure_ascii=False), encoding="utf-8")
    gen_path.write_text(__import__("json").dumps(generations, ensure_ascii=False), encoding="utf-8")

    output_path = tmp_path / "heatmap.html"
    written = write_entropy_heatmap(token_path, output_path, sample_id="s1", decode_id=0, generations_path=gen_path)

    assert written == output_path
    assert output_path.exists()
    assert "Entropy Heatmap" in output_path.read_text(encoding="utf-8")
