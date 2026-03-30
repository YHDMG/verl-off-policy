import asyncio

import pytest
import torch

OmegaConf = pytest.importorskip("omegaconf").OmegaConf
pytest.importorskip("ray")

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.incorrect_only_dense_step import IncorrectOnlyDenseStepRewardManager


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(int(tok)) for tok in token_ids)


def _make_sample(prompt: str, response: str) -> DataProto:
    prompt_ids = torch.tensor([[ord(ch) for ch in prompt]], dtype=torch.long)
    response_ids = torch.tensor([[ord(ch) for ch in response]], dtype=torch.long)
    attention_mask = torch.ones((1, prompt_ids.size(1) + response_ids.size(1)), dtype=torch.long)
    return DataProto.from_dict(
        tensors={
            "prompts": prompt_ids,
            "responses": response_ids,
            "attention_mask": attention_mask,
        },
        non_tensors={
            "data_source": ["math"],
            "reward_model": [{"ground_truth": "4"}],
            "extra_info": [{}],
        },
    )


def test_correct_response_stays_outcome_only():
    async def compute_score(**kwargs):
        return {
            "score": 1.0,
            "acc": 1.0,
            "step_scores": [1.0],
            "step_texts": ["Step 1: add 2 and 2\n"],
        }

    async def _run():
        manager = IncorrectOnlyDenseStepRewardManager(
            config=OmegaConf.create({"reward": {"reward_kwargs": {"process_reward_weight": 0.2}}}),
            tokenizer=CharTokenizer(),
            compute_score=compute_score,
        )
        return await manager.run_single(_make_sample("2+2=", "Step 1: add 2 and 2\nFinal Answer: 4"))

    result = asyncio.run(_run())

    reward_tensor = result["reward_tensor"]
    assert reward_tensor[:-1].sum().item() == pytest.approx(0.0)
    assert reward_tensor[-1].item() == pytest.approx(1.0)
    assert result["reward_extra_info"]["process_reward_applied"] is False


def test_incorrect_response_gets_dense_step_reward():
    step_1 = "Step 1: compute 2+2=4\n"
    step_2 = "Step 2: guess the answer is 5\n"
    response = step_1 + step_2 + "Final Answer: 5"

    async def compute_score(**kwargs):
        return {
            "score": 0.0,
            "acc": 0.0,
            "step_scores": [1.0, 0.5],
            "step_texts": [step_1, step_2],
        }

    async def _run():
        manager = IncorrectOnlyDenseStepRewardManager(
            config=OmegaConf.create({"reward": {"reward_kwargs": {"process_reward_weight": 0.2}}}),
            tokenizer=CharTokenizer(),
            compute_score=compute_score,
        )
        return await manager.run_single(_make_sample("2+2=", response))

    result = asyncio.run(_run())

    reward_tensor = result["reward_tensor"]
    assert reward_tensor.sum().item() == pytest.approx(0.3, rel=1e-5)
    assert reward_tensor[-1].item() == pytest.approx(0.0)
    assert result["reward_extra_info"]["process_reward_applied"] is True
    assert result["reward_extra_info"]["num_step_scores"] == 2
