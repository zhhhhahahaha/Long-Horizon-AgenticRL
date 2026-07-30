"""CPU tests for configurable BrowseComp-Plus tool variants."""

from __future__ import annotations

import ast
import asyncio
import os
from pathlib import Path

import pytest

from examples.supo_browsecomp.tool_schemas import build_tools


NUM_GPUS = 0
GENERATE_PATH = Path(__file__).parents[1] / "examples/supo_browsecomp/generate_with_bcplus.py"


def _function(name: str):
    tree = ast.parse(GENERATE_PATH.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _search_function(tools: list[dict]) -> dict:
    return next(tool["function"] for tool in tools if tool["function"]["name"] == "search")


@pytest.mark.unit
def test_baseline_schema_keeps_model_controlled_topk():
    search = _search_function(build_tools())

    assert search["parameters"]["properties"]["topk"]["default"] == 10
    assert "default 10" in search["description"]


@pytest.mark.unit
def test_fixed_topk_schema_hides_model_argument():
    search = _search_function(build_tools(5))

    assert "topk" not in search["parameters"]["properties"]
    assert "top 5" in search["description"]


@pytest.mark.unit
def test_search_topk_resolution_preserves_baseline_and_supports_fixed_mode():
    namespace = {
        "Mapping": dict,
        "BCPLUS_CONFIGS": {
            "fixed_search_topk": None,
            "search_topk_default": 10,
            "search_topk_cap": 20,
        },
    }
    function = _function("_resolve_search_topk")
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(GENERATE_PATH), "exec"), namespace)
    resolve = namespace["_resolve_search_topk"]

    assert resolve({}) == 10
    assert resolve({"topk": 100}) == 20
    assert resolve({"topk": "invalid"}) == 10

    namespace["BCPLUS_CONFIGS"]["fixed_search_topk"] = 5
    assert resolve({"topk": 20}) == 5


@pytest.mark.unit
def test_open_page_word_limit_env_defaults_and_overrides(monkeypatch):
    namespace = {"os": os}
    optional_function = _function("_optional_positive_int_env")
    positive_function = _function("_positive_int_env")
    exec(
        compile(
            ast.Module(body=[optional_function, positive_function], type_ignores=[]),
            str(GENERATE_PATH),
            "exec",
        ),
        namespace,
    )
    read_limit = namespace["_positive_int_env"]

    monkeypatch.delenv("BCPLUS_DOC_WORDS_FULL", raising=False)
    assert read_limit("BCPLUS_DOC_WORDS_FULL", 4096) == 4096

    monkeypatch.setenv("BCPLUS_DOC_WORDS_FULL", "10000")
    assert read_limit("BCPLUS_DOC_WORDS_FULL", 4096) == 10000

    monkeypatch.setenv("BCPLUS_DOC_WORDS_FULL", "0")
    with pytest.raises(ValueError, match="positive integer"):
        read_limit("BCPLUS_DOC_WORDS_FULL", 4096)


@pytest.mark.unit
@pytest.mark.parametrize(("word_limit", "expected_words"), [(4096, 4096), (10000, 6000)])
def test_open_page_execution_honors_word_limit(word_limit, expected_words):
    page_text = " ".join(f"word-{index}" for index in range(6000))

    class FakeSearchClient:
        async def open(self, **kwargs):
            return [{"docid": "doc-1", "url": "https://example.test", "text": page_text}]

    namespace = {
        "_search_client": lambda: FakeSearchClient(),
        "_SEARCH_SEM": asyncio.Semaphore(1),
        "_resolve_search_topk": lambda args: 5,
        "_keep_first_n_words": lambda text, limit: " ".join(text.split()[:limit]),
        "BCPLUS_CONFIGS": {"doc_words_snippet": 512, "doc_words_full": word_limit},
    }
    function = next(
        node
        for node in ast.parse(GENERATE_PATH.read_text()).body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_action"
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(GENERATE_PATH), "exec"), namespace)

    observation, *_ = asyncio.run(
        namespace["_run_action"](
            [{"function": "open_page", "arguments": {"docid": "doc-1"}}],
            set(),
        )
    )
    content = observation.split("content: ", 1)[1].split("\n\n", 1)[0]

    assert len(content.split()) == expected_words
