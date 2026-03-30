# Math Entropy Analysis

Independent math-task entropy analysis tools for:

- sampling completions from a local model
- rescoring completions with teacher forcing
- verifying math answers
- aggregating token-level uncertainty metrics into sequence-level features
- producing basic diagnostic plots
- rendering a single-sample entropy heatmap as HTML

Typical workflow:

```bash
python -m math_entropy_analysis.generate \
  --model /path/to/model \
  --problems /path/to/problems.parquet \
  --output-dir artifacts/run_a \
  --decode-mode greedy \
  --max-samples 3

python -m math_entropy_analysis.rescore \
  --model /path/to/model \
  --problems artifacts/run_a/problems.parquet \
  --generations artifacts/run_a/generations.parquet \
  --output-dir artifacts/run_a

python -m math_entropy_analysis.features \
  --generations artifacts/run_a/generations.parquet \
  --token-metrics artifacts/run_a/token_metrics.parquet \
  --output-dir artifacts/run_a

python -m math_entropy_analysis.plots \
  --sequence-features artifacts/run_a/sequence_features.parquet \
  --token-metrics artifacts/run_a/token_metrics.parquet \
  --output-dir artifacts/run_a/plots

python -m math_entropy_analysis.visualize \
  --token-metrics artifacts/run_a/token_metrics.parquet \
  --generations artifacts/run_a/generations.parquet \
  --sample-id <sample_id> \
  --decode-id 0 \
  --output artifacts/run_a/plots/sample_entropy_heatmap.html
```

Required problem schema:

- `sample_id`
- `prompt`
- `ground_truth`

Optional fields:

- `difficulty`
- `source`
- `split`
- `metadata`


Tip: set `--max-samples 3` (or use `MAX_SAMPLES="3"` in the shell script) when you only want to run a small subset for debugging.
