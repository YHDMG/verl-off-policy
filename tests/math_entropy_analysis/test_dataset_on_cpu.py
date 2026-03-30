from math_entropy_analysis.dataset import normalize_problem_records


class FakeArray:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


def test_normalize_problem_records_preserves_required_fields():
    records = normalize_problem_records(
        [
            {
                "sample_id": 7,
                "prompt": "What is 1 + 1?",
                "ground_truth": "2",
                "metadata": '{"split": "test"}',
            }
        ]
    )

    assert records == [
        {
            "sample_id": "7",
            "prompt": "What is 1 + 1?",
            "ground_truth": "2",
            "metadata": {"split": "test"},
        }
    ]


def test_normalize_problem_records_supports_verl_style_math_schema():
    records = normalize_problem_records(
        [
            {
                "data_source": "DeepscalerDataset",
                "ability": "math",
                "prompt": FakeArray([
                    {
                        "role": "user",
                        "content": "Find x. Let's think step by step and output the final answer after \"####\".",
                    }
                ]),
                "reward_model": {"ground_truth": "70", "style": "rule"},
                "extra_info": {"index": 0, "question": "Find x.", "split": "train", "answer": "70"},
            }
        ]
    )

    assert records[0]["sample_id"] == "0"
    assert records[0]["ground_truth"] == "70"
    assert records[0]["prompt"].startswith("Find x.")
    assert records[0]["source"] == "DeepscalerDataset"
    assert records[0]["split"] == "train"
    assert records[0]["metadata"]["reward_model"]["ground_truth"] == "70"
