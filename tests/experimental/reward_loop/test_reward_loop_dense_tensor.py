import torch
import pytest

pytest.importorskip("ray")
from verl import DataProto
from verl.experimental.reward_loop.reward_loop import build_reward_dataproto


def test_build_reward_dataproto_supports_dense_reward_tensor():
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "responses": torch.zeros((2, 6), dtype=torch.long),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 0, 0, 0],
                ],
                dtype=torch.long,
            ),
        },
        non_tensors={},
    )
    outputs_flat = [
        {
            "reward_score": 1.0,
            "reward_tensor": torch.tensor([0.1, 0.2, 0.0, 0.0, 0.0, 1.0]),
            "reward_extra_info": {"acc": 0.0},
        },
        {
            "reward_score": 0.5,
            "reward_extra_info": {"acc": 1.0},
        },
    ]

    reward_dp = build_reward_dataproto(data, outputs_flat)

    assert torch.allclose(
        reward_dp.batch["rm_scores"][0],
        torch.tensor([0.1, 0.2, 0.0, 0.0, 0.0, 1.0]),
    )
    assert torch.allclose(
        reward_dp.batch["rm_scores"][1],
        torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
    )
    assert reward_dp.non_tensor_batch["acc"].tolist() == [0.0, 1.0]
