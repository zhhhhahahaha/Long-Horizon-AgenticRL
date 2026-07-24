"""CPU tests for BC+ token-level summary-turn advantages."""

from __future__ import annotations

import ast
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401
import pytest
import torch
from megatron.core import mpu

from examples.supo_browsecomp.summary_advantage import compute_summary_aware_advantages


NUM_GPUS = 0
GENERATE_PATH = Path(__file__).parents[1] / "examples/supo_browsecomp/generate_with_bcplus.py"


def _rollout_data(*, base_reward=2.5, source="fallback", start=3, end=7, response_length=8):
    return {
        "rewards": [base_reward],
        "kl": [torch.zeros(response_length, dtype=torch.float32)],
        "metadata": [
            {
                "summary_source": source,
                "summary_turn_start": start,
                "summary_turn_end": end,
            }
        ],
        "total_lengths": [response_length + 4],
        "response_lengths": [response_length],
    }


@pytest.mark.unit
def test_fallback_overrides_only_summary_turn(monkeypatch):
    monkeypatch.setenv("BCPLUS_COMPRESS_PENALTY", "0.5")
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    data = _rollout_data()

    compute_summary_aware_advantages(SimpleNamespace(), data)

    assert data["advantages"][0].tolist() == pytest.approx([2.5, 2.5, 2.5, -0.5, -0.5, -0.5, -0.5, 2.5])
    assert torch.equal(data["returns"][0], data["advantages"][0])


@pytest.mark.unit
@pytest.mark.parametrize("source", ["extracted", "empty", ""])
def test_non_fallback_summary_keeps_base_advantage(monkeypatch, source):
    monkeypatch.setenv("BCPLUS_COMPRESS_PENALTY", "0.5")
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    data = _rollout_data(base_reward=-0.25, source=source)

    compute_summary_aware_advantages(SimpleNamespace(), data)

    assert data["advantages"][0].tolist() == pytest.approx([-0.25] * 8)


@pytest.mark.unit
def test_zero_penalty_disables_fallback_override(monkeypatch):
    monkeypatch.setenv("BCPLUS_COMPRESS_PENALTY", "0")
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    data = _rollout_data(base_reward=1.25)

    compute_summary_aware_advantages(SimpleNamespace(), data)

    assert data["advantages"][0].tolist() == pytest.approx([1.25] * 8)


@pytest.mark.unit
def test_fallback_span_is_sliced_consistently_across_cp_ranks(monkeypatch):
    monkeypatch.setenv("BCPLUS_COMPRESS_PENALTY", "0.5")
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 2)
    overridden = 0
    base = 0

    from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

    for cp_rank in range(2):
        monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda rank=cp_rank: rank)
        full = torch.zeros(8, dtype=torch.float32)
        local = slice_log_prob_with_cp(full, total_length=12, response_length=8)
        data = _rollout_data()
        data["kl"] = [local]

        compute_summary_aware_advantages(SimpleNamespace(), data)

        overridden += int((data["advantages"][0] == -0.5).sum().item())
        base += int((data["advantages"][0] == 2.5).sum().item())

    assert overridden == 4  # response positions [3, 7)
    assert base == 4


@pytest.mark.unit
@pytest.mark.parametrize("start,end", [(None, None), (-1, 2), (3, 3), (2, 9)])
def test_fallback_requires_valid_response_relative_span(monkeypatch, start, end):
    monkeypatch.setenv("BCPLUS_COMPRESS_PENALTY", "0.5")
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    data = _rollout_data(start=start, end=end)

    with pytest.raises(ValueError, match="summary-turn span|invalid response-relative span"):
        compute_summary_aware_advantages(SimpleNamespace(), data)


class _FakeSample:
    def __init__(self, *, rollout_id, group_index, final, source="", score=0.0):
        self.rollout_id = rollout_id
        self.group_index = group_index
        self.reward = {"score": score}
        self.metadata = {
            "_bcplus_sibling": {"is_final": final},
            "_bcplus": {"summary_source": source},
        }
        self.train_metadata = {}

    def get_reward_value(self, args):
        return float(self.reward["score"])


def _load_reward_post_process():
    tree = ast.parse(GENERATE_PATH.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "reward_post_process"
    )
    namespace = {"copy": copy}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(GENERATE_PATH), "exec"), namespace)
    return namespace["reward_post_process"]


def _load_do_compression(output):
    tree = ast.parse(GENERATE_PATH.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "_do_compression")

    async def fake_post(url, payload):
        return output

    namespace = {
        "Sample": object,
        "_COMPRESS_PROMPT": "compress",
        "_wrap_summary_request_and_reopen_assistant": lambda prompt: "request",
        "_clamp_max_new_tokens": lambda args, input_length: 16,
        "_extract_summary": lambda text: "handover" if "<summary>" in text else None,
        "post": fake_post,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(GENERATE_PATH), "exec"), namespace)
    return namespace["_do_compression"]


@pytest.mark.unit
def test_compression_span_excludes_request_and_covers_all_generated_tokens():
    output = {
        "text": "<think>reason</think><summary>handover</summary>",
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [(-0.1, 20), (-0.2, 21), (-0.3, 22)],
        },
    }
    do_compression = _load_do_compression(output)

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            assert text == "request"
            return {"input_ids": [10, 11]}

    class Sample:
        def __init__(self):
            self.appends = []

        def append_response_tokens(self, args, **kwargs):
            self.appends.append(kwargs)

    response_tokens = [1, 2, 3]
    sample = Sample()
    summary, source, added, span = asyncio.run(
        do_compression(
            SimpleNamespace(),
            Tokenizer(),
            "http://router/generate",
            sample,
            {},
            [100, 101],
            response_tokens,
        )
    )

    assert (summary, source, added, span) == ("handover", "extracted", 5, (5, 8))
    assert response_tokens == [1, 2, 3, 10, 11, 20, 21, 22]
    assert [append["trainable"] for append in sample.appends] == [False, True]


@pytest.mark.unit
def test_zero_std_group_stays_finite_before_token_level_override():
    reward_post_process = _load_reward_post_process()
    samples = [
        _FakeSample(rollout_id=1, group_index=10, final=False, source="fallback"),
        _FakeSample(rollout_id=1, group_index=10, final=True),
        _FakeSample(rollout_id=2, group_index=10, final=True),
    ]
    args = SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
    )

    raw_rewards, normalized = reward_post_process(args, samples)

    assert raw_rewards == [0.0, 0.0, 0.0]
    assert normalized == [0.0, 0.0, 0.0]
    assert all(sample.reward["score"] == 0.0 for sample in samples)
