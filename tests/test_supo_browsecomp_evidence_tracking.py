"""CPU tests for BC+ evidence tracking at the observation commit boundary."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


NUM_GPUS = 0
GENERATE_PATH = Path(__file__).parents[1] / "examples/supo_browsecomp/generate_with_bcplus.py"


class _FakeSample:
    class Status:
        ABORTED = "aborted"
        COMPLETED = "completed"
        TRUNCATED = "truncated"

    def __init__(self):
        self.prompt = [{"role": "user", "content": "question"}]
        self.metadata = {}
        self.status = None
        self.appended = []

    def append_response_tokens(self, args, **kwargs):
        self.appended.append(kwargs)


class _FakeTokenizer:
    def apply_chat_template(self, prompt, **kwargs):
        return "prompt"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1]}


def _load_run_one_sub_trajectory(*, compress: bool):
    tree = ast.parse(GENERATE_PATH.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_one_sub_trajectory"
    )

    async def fake_post(url, payload):
        return {
            "text": "tool call",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [(-0.1, 2)],
            },
        }

    async def fake_run_action(fn_calls, visited):
        return "tool observation", None, True, {"evidence-doc"}, {"evidence-doc"}

    async def fake_do_compression(*args, **kwargs):
        return "handover", "extracted", 0, None

    def fake_stash(sample, stats):
        sample.metadata["_bcplus"] = dict(stats)

    namespace = {
        "Sample": _FakeSample,
        "GenerateState": lambda args: SimpleNamespace(tokenizer=_FakeTokenizer()),
        "TOOLS": [],
        "BCPLUS_CONFIGS": {
            "max_turns": 1,
            "compress_length_threshold": 0.5 if compress else 1.0,
        },
        "_clamp_max_new_tokens": lambda args, input_length: 16,
        "_extract_fn_call": lambda text: [{"function": "open_page", "arguments": {"docid": "opened-doc"}}],
        "_run_action": fake_run_action,
        "_wrap_observation_and_reopen_assistant": lambda observation: observation,
        "_do_compression": fake_do_compression,
        "_stash_bcplus": fake_stash,
        "post": fake_post,
        "asyncio": asyncio,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(GENERATE_PATH), "exec"), namespace)
    return namespace["_run_one_sub_trajectory"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "compress,context_length,expected_outcome,expected_docids",
    [
        (False, 100, "truncated", ["evidence-doc"]),
        (True, 4, "compressed", []),
    ],
)
def test_evidence_docids_are_recorded_only_after_observation_commit(
    compress, context_length, expected_outcome, expected_docids
):
    run_one_sub_trajectory = _load_run_one_sub_trajectory(compress=compress)
    sample = _FakeSample()
    args = SimpleNamespace(
        partial_rollout=False,
        rollout_max_context_len=context_length,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )

    outcome = asyncio.run(run_one_sub_trajectory(args, sample, {}))

    assert outcome == expected_outcome
    assert sample.metadata["_bcplus"]["retrieved_docids"] == expected_docids
    assert sample.metadata["_bcplus"]["opened_docids"] == expected_docids
