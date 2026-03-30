"""Dataset utilities for math-task entropy analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

REQUIRED_COLUMNS = ("sample_id", "prompt", "ground_truth")
OPTIONAL_COLUMNS = ("difficulty", "source", "split", "metadata")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_dumps_stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def _to_python(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, list, tuple, dict)):
        value = value.tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes, list, tuple, dict)):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def _normalize_metadata(value: Any) -> dict[str, Any] | None:
    value = _to_python(value)
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            loaded = json.loads(stripped)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            return {"raw": value}
    return {"raw": value}


def _extract_prompt_text(prompt_value: Any) -> str:
    prompt_value = _to_python(prompt_value)
    if isinstance(prompt_value, dict):
        if "content" in prompt_value:
            return str(prompt_value["content"])
        return str(prompt_value)
    if isinstance(prompt_value, (list, tuple)):
        parts = []
        for item in prompt_value:
            item = _to_python(item)
            if isinstance(item, Mapping) and "content" in item:
                parts.append(str(item["content"]))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(prompt_value)


def _extract_ground_truth(record: Mapping[str, Any]) -> Any:
    if "ground_truth" in record:
        return record["ground_truth"]
    reward_model = _to_python(record.get("reward_model"))
    if isinstance(reward_model, Mapping):
        return reward_model.get("ground_truth")
    return None


def _extract_sample_id(record: Mapping[str, Any], fallback_index: int | None) -> str:
    if "sample_id" in record:
        return str(record["sample_id"])
    extra_info = _to_python(record.get("extra_info"))
    if isinstance(extra_info, Mapping):
        if "index" in extra_info:
            return str(extra_info["index"])
        if "question" in extra_info:
            return str(extra_info["question"])
    if fallback_index is not None:
        return str(fallback_index)
    raise ValueError("Unable to infer sample_id from record")


def _normalize_verl_style_record(record: Mapping[str, Any], fallback_index: int | None) -> dict[str, Any] | None:
    if "prompt" not in record or "reward_model" not in record:
        return None

    ground_truth = _extract_ground_truth(record)
    if ground_truth is None:
        raise ValueError("verl-style record is missing reward_model.ground_truth")

    extra_info = _to_python(record.get("extra_info"))
    if not isinstance(extra_info, Mapping):
        extra_info = {} if extra_info is None else {"raw_extra_info": extra_info}

    metadata = dict(extra_info)
    metadata["reward_model"] = _to_python(record.get("reward_model"))
    metadata["raw_prompt"] = _to_python(record.get("prompt"))
    if "ability" in record:
        metadata["ability"] = _to_python(record.get("ability"))
    if "data_source" in record:
        metadata["data_source"] = _to_python(record.get("data_source"))

    normalized = {
        "sample_id": _extract_sample_id(record, fallback_index=fallback_index),
        "prompt": _extract_prompt_text(record.get("prompt")),
        "ground_truth": str(ground_truth),
        "source": _to_python(record.get("data_source")),
        "split": extra_info.get("split"),
        "metadata": metadata,
    }

    if "difficulty" in record:
        normalized["difficulty"] = _to_python(record.get("difficulty"))

    for key, value in record.items():
        if key not in normalized and key not in {"prompt", "reward_model", "extra_info", "data_source"}:
            normalized[key] = _to_python(value)

    return normalized


def normalize_problem_record(record: Mapping[str, Any], fallback_index: int | None = None) -> dict[str, Any]:
    record = {key: _to_python(value) for key, value in dict(record).items()}

    if all(column in record for column in REQUIRED_COLUMNS):
        normalized = {
            "sample_id": str(record["sample_id"]),
            "prompt": _extract_prompt_text(record["prompt"]),
            "ground_truth": str(record["ground_truth"]),
        }
    else:
        normalized = _normalize_verl_style_record(record, fallback_index=fallback_index)
        if normalized is None:
            missing = [column for column in REQUIRED_COLUMNS if column not in record]
            raise ValueError(f"Problem record is missing required columns: {missing}")

    for key in OPTIONAL_COLUMNS:
        if key not in normalized and key in record:
            value = record[key]
            if key == "metadata":
                normalized[key] = _normalize_metadata(value)
            elif value is not None:
                normalized[key] = value

    for key, value in record.items():
        if key not in normalized and key not in REQUIRED_COLUMNS and key not in OPTIONAL_COLUMNS:
            normalized[key] = value

    if "metadata" in normalized:
        normalized["metadata"] = _normalize_metadata(normalized["metadata"])

    return normalized


def normalize_problem_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_problem_record(record, fallback_index=index) for index, record in enumerate(records)]


def _load_json_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            return [dict(item) for item in data["records"]]
        raise ValueError("JSON input must be a list of records or contain a top-level 'records' list")
    raise ValueError("Unsupported JSON structure for problem records")


def _load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _load_csv_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_parquet_file(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Reading parquet files requires either pyarrow or pandas to be installed") from exc
        dataframe = pd.read_parquet(path)
        return dataframe.to_dict(orient="records")

    table = pq.read_table(path)
    return table.to_pylist()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl_file(file_path)
    if suffix == ".json":
        return _load_json_file(file_path)
    if suffix == ".csv":
        return _load_csv_file(file_path)
    if suffix == ".parquet":
        return _load_parquet_file(file_path)
    raise ValueError(f"Unsupported input format: {file_path.suffix}")


def load_problem_records(path: str | Path) -> list[dict[str, Any]]:
    return normalize_problem_records(load_records(path))


def write_parquet_records(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(record) for record in records]

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Writing parquet files requires either pyarrow or pandas to be installed") from exc
        dataframe = pd.DataFrame.from_records(rows)
        dataframe.to_parquet(output_path, index=False)
        return output_path

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)
    return output_path


def prepare_artifact_dir(output_dir: str | Path, run_name: str | None = None) -> Path:
    base_dir = Path(output_dir)
    artifact_dir = base_dir / run_name if run_name else base_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def write_json_records(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump([dict(record) for record in records], handle, ensure_ascii=False, indent=2, default=_json_default)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a math benchmark file into the required problem schema.")
    parser.add_argument("--input", required=True, help="Path to the raw problem file")
    parser.add_argument("--output", required=True, help="Path to the normalized parquet file")
    args = parser.parse_args()

    records = load_problem_records(args.input)
    write_parquet_records(records, args.output)


if __name__ == "__main__":
    main()
