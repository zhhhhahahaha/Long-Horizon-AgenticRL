"""CPU tests for BC+ judge failure reporting."""

from __future__ import annotations

import ast
import asyncio
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


NUM_GPUS = 0
GENERATE_PATH = Path(__file__).parents[1] / "examples/supo_browsecomp/generate_with_bcplus.py"


def _async_function(name: str):
    tree = ast.parse(GENERATE_PATH.read_text())
    return next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == name
    )


@pytest.mark.unit
def test_call_openai_raises_after_api_retries(monkeypatch):
    calls = 0

    class FakeCompletions:
        async def create(self, **kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError("gateway unavailable")

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    openai = ModuleType("openai")
    openai.AsyncOpenAI = FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setenv("LLAMA_API_KEY", "test-key")

    namespace = {
        "asyncio": asyncio,
        "os": os,
        "BCPLUS_CONFIGS": {"judge_base_url": "https://judge.invalid/v1"},
    }
    function = _async_function("_call_openai")
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(GENERATE_PATH), "exec"), namespace)

    with pytest.raises(RuntimeError, match="gateway unavailable"):
        asyncio.run(namespace["_call_openai"]("prompt", "judge-model", max_retries=3))

    assert calls == 3


@pytest.mark.unit
def test_judge_one_raises_after_unparseable_responses():
    calls = 0

    async def fake_call_openai(messages, model):
        nonlocal calls
        calls += 1
        return "response without a verdict"

    namespace = {
        "_patch_browsecomp_typos": lambda gold, pred: (gold, pred),
        "_em_score": lambda gold, pred: False,
        "_GRADER_TEMPLATE": "{question} {response} {correct_answer}",
        "_JUDGE_SEM": asyncio.Semaphore(1),
        "BCPLUS_CONFIGS": {"judge_max_retries": 3, "judge_model": "judge-model"},
        "_call_openai": fake_call_openai,
        "_parse_judge_response": lambda response: {"correct": None, "parse_error": True},
    }
    function = _async_function("_judge_one")
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(GENERATE_PATH), "exec"), namespace)

    with pytest.raises(RuntimeError, match="unparseable after 3 attempts"):
        asyncio.run(namespace["_judge_one"]("question", "gold", "prediction"))

    assert calls == 3


@pytest.mark.unit
def test_reward_marks_judge_exception_as_failed():
    class FakeSample:
        def __init__(self):
            self.metadata = {
                "query": "question",
                "_bcplus": {"finished": True, "finish_answer": "prediction"},
            }
            self.label = "gold"

    async def failed_judge(question, gold, prediction):
        raise RuntimeError("judge failed")

    namespace = {"Sample": FakeSample, "_judge": failed_judge}
    function = _async_function("_reward_one")
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(GENERATE_PATH), "exec"), namespace)

    reward = asyncio.run(namespace["_reward_one"](SimpleNamespace(), FakeSample()))

    assert reward["score"] == 0.0
    assert reward["judge_failed"] == 1.0
