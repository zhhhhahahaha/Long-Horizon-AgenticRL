"""CPU tests for BC+ token-level summary-turn advantages."""

from __future__ import annotations

import ast
import asyncio
import copy
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import _cp_dist_helpers  # noqa: F401
import pytest
import torch
from megatron.core import mpu

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.supo_browsecomp.dynamic_sampling import (
    _CandidateGroup,
    _build_metrics,
    _candidate_group,
    _choose_topup_group_count,
    _sampling_targets,
    _select_candidates,
)
from examples.supo_browsecomp.summary_advantage import compute_summary_aware_advantages


NUM_GPUS = 0
GENERATE_PATH = REPO_ROOT / "examples/supo_browsecomp/generate_with_bcplus.py"


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

    def extract_summary(text):
        matches = re.findall(r"<summary>(.*?)</summary>", text, re.DOTALL)
        return matches[-1].strip() if matches else None

    namespace = {
        "Sample": object,
        "_COMPRESS_PROMPT": "compress",
        "_wrap_summary_request_and_reopen_assistant": lambda prompt: "request",
        "_clamp_max_new_tokens": lambda args, input_length: 16,
        "_extract_summary": extract_summary,
        "post": fake_post,
        "re": re,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(GENERATE_PATH), "exec"), namespace)
    return namespace["_do_compression"]


def _load_generate(run_one_sub_trajectory, sample_type):
    tree = ast.parse(GENERATE_PATH.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate")
    namespace = {
        "Sample": sample_type,
        "BCPLUS_CONFIGS": {"max_sub_trajs": 5},
        "_run_one_sub_trajectory": run_one_sub_trajectory,
        "_build_continuation_chat": lambda prompt, summary: (_ for _ in ()).throw(
            AssertionError("an empty handover must not create a continuation prompt")
        ),
        "copy": copy,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(GENERATE_PATH), "exec"), namespace)
    return namespace["generate"]


def _load_dump_rollout_data_postprocess(dump_dir: Path):
    tree = ast.parse(GENERATE_PATH.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "dump_rollout_data_postprocess"
    )
    namespace = {
        "BCPLUS_CONFIGS": {"dump_dir": str(dump_dir), "dump_train_old": False},
        "os": os,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(GENERATE_PATH), "exec"), namespace)
    return namespace["dump_rollout_data_postprocess"]


def _load_log_bcplus():
    tree = ast.parse(GENERATE_PATH.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "log_bcplus")
    namespace = {
        "Sample": object,
        "BCPLUS_CONFIGS": {"compress_penalty": 0.5},
        "_BCPLUS_METRIC_DEFINED": False,
        "_evidence_docids": lambda value: set(),
        "math": math,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(GENERATE_PATH), "exec"), namespace)
    return namespace["log_bcplus"]


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
@pytest.mark.parametrize(
    "generated_text",
    [
        "done</think><summary>\n  </summary>",
        "<think>done</think><summary>\n  </summary>",
    ],
)
def test_empty_summary_block_salvages_thinking_as_handover(generated_text):
    output = {
        "text": generated_text,
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [(-0.1, 20), (-0.2, 21)],
        },
    }
    do_compression = _load_do_compression(output)

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [10]}

    class Sample:
        def append_response_tokens(self, args, **kwargs):
            pass

    response_tokens = [1, 2]
    summary, source, added, span = asyncio.run(
        do_compression(
            SimpleNamespace(),
            Tokenizer(),
            "http://router/generate",
            Sample(),
            {},
            [100],
            response_tokens,
        )
    )

    assert (summary, source, added, span) == ("done", "fallback", 3, (3, 5))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("generated_text", "expected_summary"),
    [
        ("reasoning</think>plain handover", "plain handover"),
        ("reasoning</think>", "reasoning"),
        ("plain handover", "plain handover"),
    ],
)
def test_missing_summary_block_uses_best_available_fallback(generated_text, expected_summary):
    output = {
        "text": generated_text,
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [(-0.1, 20)],
        },
    }
    do_compression = _load_do_compression(output)

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [10]}

    class Sample:
        def append_response_tokens(self, args, **kwargs):
            pass

    summary, source, added, span = asyncio.run(
        do_compression(
            SimpleNamespace(),
            Tokenizer(),
            "http://router/generate",
            Sample(),
            {},
            [100],
            [],
        )
    )

    assert (summary, source, added, span) == (expected_summary, "fallback", 2, (1, 2))


@pytest.mark.unit
def test_tag_only_summary_cannot_be_used_as_handover():
    output = {
        "text": "<summary>\n  </summary>",
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [(-0.1, 20)],
        },
    }
    do_compression = _load_do_compression(output)

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [10]}

    class Sample:
        def append_response_tokens(self, args, **kwargs):
            pass

    summary, source, added, span = asyncio.run(
        do_compression(
            SimpleNamespace(),
            Tokenizer(),
            "http://router/generate",
            Sample(),
            {},
            [100],
            [],
        )
    )

    assert (summary, source, added, span) == (None, "empty", 2, None)


@pytest.mark.unit
def test_generate_rejects_empty_handover_and_propagates_query_id():
    class Sample:
        class Status:
            TRUNCATED = "truncated"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.metadata = kwargs.get("metadata", {})
            self.status = None
            self.rollout_id = None

    async def run_one_sub_trajectory(args, sample, sampling_params):
        sample.metadata["_bcplus"] = {
            "outcome": "compressed",
            "summary": "",
            "summary_source": "extracted",
        }
        return "compressed"

    generate = _load_generate(run_one_sub_trajectory, Sample)
    sample = Sample(
        index=7,
        group_index=3,
        prompt=[{"role": "user", "content": "question"}],
        label="answer",
        metadata={"query_id": "query-7"},
    )

    sub_trajs = asyncio.run(generate(SimpleNamespace(partial_rollout=False), sample, {}))

    assert sub_trajs == [sample]
    assert sample.status == Sample.Status.TRUNCATED
    assert sample.metadata["_bcplus"]["outcome"] == "compress_failed"
    assert sample.metadata["_bcplus"]["summary"] is None
    assert sample.train_metadata["query_id"] == "query-7"


@pytest.mark.unit
def test_rollout_dump_persists_non_nullable_query_id(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    from slime.backends.megatron_utils import cp_utils

    monkeypatch.setattr(cp_utils, "all_gather_with_cp", lambda values, *_args: values)
    cpu_device = torch.device("cpu")
    monkeypatch.setattr(torch, "device", lambda *_args, **_kwargs: cpu_device)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    dump = _load_dump_rollout_data_postprocess(tmp_path)
    rollout_data = {
        "tokens": [torch.tensor([10, 11, 20, 21])],
        "response_lengths": [2],
        "total_lengths": [4],
        "loss_masks": [torch.tensor([1, 1])],
        "rollout_log_probs": [torch.tensor([-0.1, -0.2])],
        "advantages": [torch.tensor([0.5, 0.5])],
        "rollout_ids": [7],
        "metadata": [
            {
                "query_id": "query-7",
                "group_index": 3,
                "sub_traj_index": 0,
                "total_sub_trajs": 1,
                "is_final": True,
            }
        ],
    }

    dump(SimpleNamespace(), 4, rollout_data)

    table = pq.read_table(tmp_path / "rollouts_iter_00004_dp0.parquet")
    assert table.column("query_id").to_pylist() == ["query-7"]
    assert table.schema.field("query_id").nullable is False

    rollout_data["metadata"] = [{}]
    with pytest.raises(ValueError, match="missing train_metadata.query_id"):
        dump(SimpleNamespace(), 5, rollout_data)


@pytest.mark.unit
def test_bcplus_wandb_sections_and_summary_content_length(monkeypatch):
    defined = []
    logged = []
    fake_wandb = ModuleType("wandb")
    fake_wandb.run = object()
    fake_wandb.define_metric = lambda *args, **kwargs: defined.append((args, kwargs))
    fake_wandb.log = logged.append
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    class Sample:
        def __init__(self, rollout_id, source, content_len):
            self.rollout_id = rollout_id
            self.group_index = rollout_id
            self.reward = {"score": 0.0}
            self.metadata = {
                "_bcplus_sibling": {"is_final": True},
                "_bcplus": {
                    "outcome": "compressed",
                    "summary_source": source,
                    "summary_content_len_tokens": content_len,
                    "response_len_tokens": 1,
                },
            }

    samples = [
        Sample(1, "extracted", 3),
        Sample(2, "extracted", 7),
        Sample(3, "fallback", 100),
    ]

    _load_log_bcplus()(42, SimpleNamespace(), samples, {}, 0.0)

    assert logged[-1]["bcplus_compression/summary_extracted_count"] == 2
    assert logged[-1]["bcplus_compression/summary_content_len_tokens_mean"] == 5.0
    bcplus_sections = {key.split("/", 1)[0] for key in logged[-1] if key.startswith("bcplus")}
    assert bcplus_sections == {
        "bcplus_compression",
        "bcplus_evidence",
        "bcplus_health",
        "bcplus_reward",
        "bcplus_sub_traj",
        "bcplus_trajectory",
    }
    assert {section: sum(key.startswith(f"{section}/") for key in logged[-1]) for section in bcplus_sections} == {
        "bcplus_compression": 9,
        "bcplus_evidence": 8,
        "bcplus_health": 4,
        "bcplus_reward": 5,
        "bcplus_sub_traj": 8,
        "bcplus_trajectory": 11,
    }
    assert not any(key.startswith("bcplus/") for key in logged[-1])
    assert defined == [
        (("bcplus_health/*",), {"step_metric": "rollout/step"}),
        (("bcplus_sub_traj/*",), {"step_metric": "rollout/step"}),
        (("bcplus_trajectory/*",), {"step_metric": "rollout/step"}),
        (("bcplus_reward/*",), {"step_metric": "rollout/step"}),
        (("bcplus_compression/*",), {"step_metric": "rollout/step"}),
        (("bcplus_evidence/*",), {"step_metric": "rollout/step"}),
        (("dynamic_sampling/*",), {"step_metric": "rollout/step"}),
    ]


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


@pytest.mark.unit
@pytest.mark.parametrize(
    "first_valid,expected_topup",
    [
        (18, 64),
        (19, 64),
        (20, 56),
        (21, 48),
        (22, 40),
        (23, 40),
        (24, 32),
        (25, 24),
        (26, 24),
        (27, 16),
        (28, 16),
        (29, 8),
        (30, 0),
    ],
)
def test_dynamic_sampling_beta_binomial_topup_table(first_valid, expected_topup):
    assert (
        _choose_topup_group_count(
            first_pool_valid_count=first_valid,
            first_pool_group_count=64,
            target_valid_count=30,
            max_topup_group_count=64,
        )
        == expected_topup
    )


@pytest.mark.unit
def test_dynamic_sampling_targets_scale_with_rollout_batch_size():
    assert _sampling_targets(32) == (64, 30, 64)
    assert _sampling_targets(16) == (32, 14, 32)


class _DynamicSample:
    def __init__(self, index, score, *, final=False):
        self.index = index
        self.reward = {"score": score}
        self.metadata = {"_bcplus_sibling": {"is_final": final}}

    def get_reward_value(self, args):
        return self.reward[args.reward_key]


@pytest.mark.unit
def test_dynamic_sampling_uses_only_final_fanout_sibling_rewards():
    expected_rewards = [0.0, 1.0] * 4
    group = [
        [
            _DynamicSample(index, 0.0),
            _DynamicSample(index, reward, final=True),
        ]
        for index, reward in enumerate(expected_rewards)
    ]

    candidate = _candidate_group(
        SimpleNamespace(n_samples_per_prompt=8, reward_key="score"),
        group,
        from_first_pool=True,
    )

    assert candidate.rewards == tuple(expected_rewards)
    assert candidate.has_nonzero_std


def _make_candidate(index, rewards, *, first_pool=True):
    return _CandidateGroup(
        group=[index],
        from_first_pool=first_pool,
        index=index,
        rewards=tuple(rewards),
    )


@pytest.mark.unit
def test_dynamic_sampling_selection_and_metrics_cover_all_candidates():
    first_valid = [_make_candidate(index, (0.0, 1.0)) for index in range(20)]
    first_all_correct = [_make_candidate(index, (1.0, 1.0)) for index in range(20, 42)]
    first_all_wrong = [_make_candidate(index, (0.0, 0.0)) for index in range(42, 64)]
    topup_valid = [
        _make_candidate(index, (0.0, 1.0), first_pool=False) for index in range(64, 76)
    ]
    topup_all_correct = [
        _make_candidate(index, (1.0, 1.0), first_pool=False) for index in range(76, 98)
    ]
    topup_all_wrong = [
        _make_candidate(index, (0.0, 0.0), first_pool=False) for index in range(98, 120)
    ]
    candidates = (
        first_valid
        + first_all_correct
        + first_all_wrong
        + topup_valid
        + topup_all_correct
        + topup_all_wrong
    )

    selected = _select_candidates(candidates, batch_size=32)
    metrics = _build_metrics(candidates, selected, topup_group_count=56)

    assert [candidate.index for candidate in selected] == list(range(20)) + list(range(64, 76))
    assert metrics == {
        "dynamic_sampling/candidate_zero_std_1_count": 44,
        "dynamic_sampling/candidate_zero_std_0_count": 44,
        "dynamic_sampling/topup_requested_group_count": 56,
        "dynamic_sampling/first_pool_kept_group_count": 20,
        "dynamic_sampling/selected_group_count": 32,
    }


@pytest.mark.unit
def test_dynamic_sampling_fills_two_zero_std_groups_without_topup():
    candidates = [_make_candidate(index, (0.0, 1.0)) for index in range(30)]
    candidates += [_make_candidate(index, (1.0, 1.0)) for index in range(30, 47)]
    candidates += [_make_candidate(index, (0.0, 0.0)) for index in range(47, 64)]

    selected = _select_candidates(candidates, batch_size=32)
    metrics = _build_metrics(candidates, selected, topup_group_count=0)

    assert [candidate.index for candidate in selected] == list(range(32))
    assert metrics["dynamic_sampling/candidate_zero_std_1_count"] == 17
    assert metrics["dynamic_sampling/candidate_zero_std_0_count"] == 17
    assert metrics["dynamic_sampling/first_pool_kept_group_count"] == 32
    assert metrics["dynamic_sampling/selected_group_count"] == 32


@pytest.mark.unit
def test_search_client_connection_pool_tracks_search_concurrency(monkeypatch):
    from examples.supo_browsecomp import local_search_client

    captured = {}
    limits_sentinel = object()

    def fake_limits(**kwargs):
        captured["limits"] = kwargs
        return limits_sentinel

    def fake_async_client(**kwargs):
        captured["client"] = kwargs
        return object()

    monkeypatch.setattr(local_search_client.httpx, "Limits", fake_limits)
    monkeypatch.setattr(local_search_client.httpx, "AsyncClient", fake_async_client)

    local_search_client.AsyncSearchClient("http://search", max_connections=512)

    assert captured["limits"] == {
        "max_connections": 512,
        "max_keepalive_connections": 128,
        "keepalive_expiry": 30.0,
    }
    assert captured["client"] == {
        "base_url": "http://search",
        "limits": limits_sentinel,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
