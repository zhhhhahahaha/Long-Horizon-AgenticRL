"""CPU contract tests for the supported BC+ summary-retention analysis."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "examples/supo_browsecomp/eval/analysis/summary_retention.py"


def _load():
    spec = importlib.util.spec_from_file_location("bcplus_summary_retention", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


retention = _load()


def _sample(
    rollout_id: int,
    sub_index: int,
    total: int,
    *,
    response: str = "",
    summary: str | None = None,
) -> dict:
    final = sub_index == total - 1
    return {
        "rollout_id": rollout_id,
        "response": response,
        "metadata": {
            "_bcplus_sibling": {
                "sub_traj_index": sub_index,
                "total_sub_trajs": total,
                "is_final": final,
            },
            "_bcplus": {
                "outcome": "finished" if final else "compressed",
                "summary_source": "" if final else "extracted",
                "summary": summary,
            },
        },
    }


def _rollout_row(rollout_id: int, *, correct: bool = False, judge_failed: bool = False) -> dict:
    return {
        "rollout_id": rollout_id,
        "query_id": f"q{rollout_id}",
        "question": "Which paired names are supported by the source?",
        "gold": "Alpha; Beta",
        "finish_answer": "Gamma",
        "correct": correct,
        "judge_failed": judge_failed,
    }


def _judge_response(
    evidence_ids: list[str], *, evidence_verdict: str = "yes", summary_retention: str = "dropped"
) -> dict:
    return {
        "evidence_assessments": [
            {
                "evidence_id": evidence_id,
                "verdict": evidence_verdict,
                "rationale": "The matched answer has the intended meaning in this context.",
            }
            for evidence_id in evidence_ids
        ],
        "summary_retention": summary_retention,
        "summary_rationale": "The final handover does not preserve the answer.",
    }


@pytest.mark.unit
def test_source_point_identity_is_checked_once_from_completion_manifest():
    success = {"status": "ok"}
    manifest = {"point": "iter04", "load_verification": {"actual_step": 4}}

    assert retention.validate_source_point(success, manifest) == ("iter04", 4)
    with pytest.raises(ValueError, match="expected 4"):
        retention.validate_source_point(success, {**manifest, "load_verification": {"actual_step": 24}})


@pytest.mark.unit
def test_prefilter_uses_tool_observations_and_excludes_judge_failures():
    hit = "<tool_response>Search result: the paired names are Alpha and Beta.</tool_response>"
    samples = [
        _sample(0, 0, 2, response=hit, summary="Continue with an unrelated lead."),
        _sample(0, 1, 2),
        _sample(1, 0, 2, response=hit, summary="Alpha and Beta are the answer."),
        _sample(1, 1, 2),
        _sample(2, 0, 2, response=hit, summary="Alpha and Beta are the answer."),
        _sample(2, 1, 2),
        _sample(3, 0, 1, response=hit),
    ]
    rows = [
        _rollout_row(0),
        _rollout_row(1, judge_failed=True),
        _rollout_row(2, correct=True),
        _rollout_row(3),
    ]

    candidates, counts = retention.build_candidates(samples, rows, point="iter04")

    assert counts == {
        "n_rollouts": 4,
        "n_incorrect": 3,
        "n_judge_failures_excluded": 1,
        "n_model_failures": 2,
        "n_failures_with_gold_docs": 0,
        "n_failures_retrieved_gold_doc": 0,
        "n_failures_opened_gold_doc": 0,
        "n_failures_with_evidence_docs": 0,
        "n_failures_retrieved_evidence_doc": 0,
        "n_failures_opened_evidence_doc": 0,
        "n_compressed_model_failures": 1,
        "n_prefilter_candidates": 1,
    }
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["candidate_id"] == "iter04:0"
    assert candidate["gold_parts"] == ["alpha", "beta"]
    assert candidate["final_handover"]["summary"] == "Continue with an unrelated lead."
    assert len(candidate["matching_tool_responses"]) == 1
    assert candidate["matching_tool_responses"][0]["evidence_id"] == "evidence-001"
    assert candidate["matching_tool_responses"][0]["matched_gold_parts"] == ["alpha", "beta"]


@pytest.mark.unit
def test_prefilter_does_not_match_model_reasoning_outside_tool_observations():
    samples = [
        _sample(
            0,
            0,
            2,
            response="I think Alpha and Beta are likely. <tool_response>Nothing useful.</tool_response>",
            summary="No answer found.",
        ),
        _sample(0, 1, 2),
    ]

    candidates, counts = retention.build_candidates(samples, [_rollout_row(0)], point="base")

    assert candidates == []
    assert counts["n_prefilter_candidates"] == 0


@pytest.mark.unit
def test_candidate_gives_judge_complete_matching_responses_only():
    row = {**_rollout_row(0), "gold": "Demi Mabry", "question": "Who designed the invitation?"}
    search = """<tool_call><function=search><parameter=query>invitation designer</parameter></function></tool_call>
    <tool_response>[Search Results] The invitation designer worked in Atlanta.</tool_response>"""
    opened = """<tool_call><function=open_page><parameter=docid>42</parameter></function></tool_call>
    <tool_response>[Opened Page Content] docid: 42\nDemi Mabry is an Atlanta designer.</tool_response>"""
    samples = [
        _sample(0, 0, 2, response="UNTRUSTED PRIVATE REASONING " + search + opened, summary="No answer."),
        _sample(0, 1, 2),
    ]

    candidates, _ = retention.build_candidates(samples, [row], point="base")

    candidate = candidates[0]
    evidence = candidate["matching_tool_responses"]
    assert len(evidence) == 1
    assert evidence[0]["tool"] == "open_page"
    assert evidence[0]["content"] == "[Opened Page Content] docid: 42\nDemi Mabry is an Atlanta designer."
    assert evidence[0]["content_truncated"] is False
    prompt = retention._candidate_prompt(candidate)
    assert "UNTRUSTED PRIVATE REASONING" not in prompt
    assert "invitation designer" not in prompt
    assert "Demi Mabry" in prompt


@pytest.mark.unit
def test_matching_response_content_has_no_secondary_length_cap():
    long_content = "[Opened Page Content] Alpha " + "context " * 2_000
    samples = [
        _sample(
            0,
            0,
            2,
            response=f"<tool_response>{long_content}</tool_response>",
            summary="No answer.",
        ),
        _sample(0, 1, 2),
    ]
    row = {**_rollout_row(0), "gold": "Alpha"}

    candidates, _ = retention.build_candidates(samples, [row], point="base")

    evidence = candidates[0]["matching_tool_responses"][0]
    assert evidence["content"] == long_content.strip()
    assert len(evidence["content"]) > 12_000
    assert evidence["content_truncated"] is False


@pytest.mark.unit
def test_unsplit_two_character_gold_remains_prefilterable():
    row = {**_rollout_row(0), "gold": "U2"}
    samples = [
        _sample(0, 0, 2, response="<tool_response>The band is U2.</tool_response>", summary="No answer."),
        _sample(0, 1, 2),
    ]

    candidates, _ = retention.build_candidates(samples, [row], point="base")

    assert candidates[0]["gold_parts"] == ["u2"]


@pytest.mark.unit
def test_matching_evidence_keeps_distinct_search_and_open_records_and_deduplicates_repeats():
    search_call = """<tool_call><function=search><parameter=query>alpha source</parameter></function></tool_call>"""
    search_response = """<tool_response>[Search Results]
--- #1: 42---
docid: 42
url: https://example.test/alpha
content: --- title: Alpha Source date: 2020-01-01 --- Alpha is the answer.
--- #2: 99---
docid: 99
url: https://example.test/other
content: --- title: Other date: 2020-01-01 --- Nothing relevant.
</tool_response>"""
    open_call = """<tool_call><function=open_page><parameter=docid>42</parameter></function></tool_call>"""
    open_response = """<tool_response>[Opened Page Content]
docid: 42
url: https://example.test/alpha
content: --- title: Alpha Source date: 2020-01-01 --- Alpha is the answer with fuller context.
</tool_response>"""
    retitled_search_response = search_response.replace("Alpha Source", "Alternate Alpha Source")
    response = search_call + search_response + search_call + retitled_search_response + open_call + open_response

    records, covered = retention._matching_tool_response_records(response, ["alpha"], sub_traj_index=0)
    deduplicated = retention._deduplicate_evidence(records)

    assert covered == {"alpha"}
    assert [record["tool"] for record in deduplicated] == ["search", "open_page"]
    assert len(deduplicated[0]["occurrences"]) == 2
    assert deduplicated[0]["queries"] == ["alpha source"]
    assert len(deduplicated[1]["occurrences"]) == 1
    assert "fuller context" in deduplicated[1]["content"]


@pytest.mark.unit
def test_failure_doc_coverage_uses_all_sub_trajectories_and_excludes_judge_failures():
    evidence_only = [_sample(0, 0, 2), _sample(0, 1, 2)]
    evidence_only[0]["metadata"]["_bcplus"].update({"retrieved_docids": ["support-2"], "opened_docids": ["support-2"]})
    evidence_only[1]["metadata"].update(
        {
            "gold_docs": [{"docid": "gold-1"}],
            "evidence_docs": [{"docid": "gold-1"}, {"docid": "support-2"}],
        }
    )
    strict_hit = _sample(1, 0, 1)
    strict_hit["metadata"]["_bcplus"].update({"retrieved_docids": ["gold-3"], "opened_docids": []})
    strict_hit["metadata"].update({"gold_docs": [{"docid": "gold-3"}], "evidence_docs": [{"docid": "gold-3"}]})
    excluded = _sample(2, 0, 1)
    excluded["metadata"]["_bcplus"].update({"retrieved_docids": ["gold-4"], "opened_docids": ["gold-4"]})
    excluded["metadata"].update({"gold_docs": [{"docid": "gold-4"}], "evidence_docs": [{"docid": "gold-4"}]})

    failure_rows = []
    _, counts = retention.build_candidates(
        [*evidence_only, strict_hit, excluded],
        [_rollout_row(0), _rollout_row(1), _rollout_row(2, judge_failed=True)],
        point="base",
        failure_retrieval_rows=failure_rows,
    )

    assert counts["n_model_failures"] == 2
    assert counts["n_failures_with_gold_docs"] == 2
    assert counts["n_failures_retrieved_gold_doc"] == 1
    assert counts["n_failures_opened_gold_doc"] == 0
    assert counts["n_failures_with_evidence_docs"] == 2
    assert counts["n_failures_retrieved_evidence_doc"] == 2
    assert counts["n_failures_opened_evidence_doc"] == 1
    assert [row["candidate_id"] for row in failure_rows] == ["base:0", "base:1"]
    assert failure_rows[0]["retrieved_gold_doc"] is False
    assert failure_rows[0]["retrieved_evidence_doc"] is True


@pytest.mark.unit
def test_verdict_parser_validates_evidence_coverage_and_computes_aggregate():
    response = _judge_response(["evidence-001", "evidence-002"], evidence_verdict="no")
    response["evidence_assessments"][1]["verdict"] = "unclear"

    verdict = retention.parse_verdict(f"```json\n{json.dumps(response)}\n```", ["evidence-001", "evidence-002"])

    assert verdict["early_retrieval"] == "unclear"
    assert verdict["summary_retention"] == "dropped"
    with pytest.raises(ValueError, match="IDs do not match"):
        retention.parse_verdict(json.dumps(response), ["evidence-001"])

    inconsistent = {**verdict, "early_retrieval": "yes"}
    with pytest.raises(ValueError, match="does not match evidence assessments"):
        retention.parse_verdict(json.dumps(inconsistent), ["evidence-001", "evidence-002"])


@pytest.mark.unit
def test_early_retrieval_requires_every_gold_part_to_be_semantically_confirmed():
    response = _judge_response(["evidence-001", "evidence-002"])
    response["evidence_assessments"][1]["verdict"] = "no"
    contract = {
        "evidence_gold_parts": {"evidence-001": ["alpha"], "evidence-002": ["beta"]},
        "required_gold_parts": ["alpha", "beta"],
    }

    verdict = retention.parse_verdict(json.dumps(response), ["evidence-001", "evidence-002"], **contract)
    assert verdict["early_retrieval"] == "no"

    response["evidence_assessments"][1]["verdict"] = "unclear"
    verdict = retention.parse_verdict(json.dumps(response), ["evidence-001", "evidence-002"], **contract)
    assert verdict["early_retrieval"] == "unclear"

    response["evidence_assessments"][1]["verdict"] = "yes"
    verdict = retention.parse_verdict(json.dumps(response), ["evidence-001", "evidence-002"], **contract)
    assert verdict["early_retrieval"] == "yes"


@pytest.mark.unit
def test_judge_stage_is_resumable_and_report_formula_is_deterministic(tmp_path):
    candidate = {
        "schema_version": retention.SCHEMA_VERSION,
        "prefilter_version": retention.PREFILTER_VERSION,
        "candidate_id": "base:0",
        "point": "base",
        "rollout_id": 0,
        "query_id": "q0",
        "question": "Question",
        "gold_answer": "Alpha",
        "gold_parts": ["alpha"],
        "final_answer": "Gamma",
        "n_sub_trajs": 2,
        "matching_tool_responses": [
            {
                "evidence_id": "evidence-001",
                "tool": "search",
                "docid": "1",
                "matched_gold_parts": ["alpha"],
                "content": "the answer is alpha",
                "occurrences": [{"sub_traj_index": 0, "tool_response_index": 0}],
            }
        ],
        "final_handover": {
            "sub_traj_index": 0,
            "outcome": "compressed",
            "summary_source": "extracted",
            "summary": "No answer yet.",
        },
    }
    candidates_path = tmp_path / "candidates.jsonl"
    retention._write_jsonl(candidates_path, [candidate])
    retention._write_jsonl(
        tmp_path / "failure_retrieval.jsonl",
        [
            {
                "schema_version": retention.SCHEMA_VERSION,
                "candidate_id": "base:0",
                "point": "base",
                "rollout_id": 0,
                "query_id": "q0",
                "has_gold_docs": True,
                "retrieved_gold_doc": True,
                "opened_gold_doc": False,
                "has_evidence_docs": True,
                "retrieved_evidence_doc": True,
                "opened_evidence_doc": False,
            }
        ],
    )
    stage_manifest = {
        "schema_version": retention.SCHEMA_VERSION,
        "prefilter_version": retention.PREFILTER_VERSION,
        "point": "base",
        "counts": {
            "n_rollouts": 1,
            "n_incorrect": 1,
            "n_judge_failures_excluded": 0,
            "n_model_failures": 1,
            "n_compressed_model_failures": 1,
            "n_prefilter_candidates": 1,
        },
    }
    retention._write_json(tmp_path / "stage_manifest.json", stage_manifest)
    retention._write_json(tmp_path / "_STAGED", {"status": "ok"})
    calls = []

    async def fake_call(messages, model):
        calls.append((messages, model))
        return json.dumps(_judge_response(["evidence-001"]))

    asyncio.run(
        retention.judge_stage(
            tmp_path,
            tmp_path,
            model="judge-v1",
            base_url="https://judge.invalid/v1",
            api_key=None,
            concurrency=2,
            max_retries=1,
            keep_raw_responses=False,
            call_model=fake_call,
        )
    )
    assert len(calls) == 1
    assert "do NOT require" in calls[0][0][0]["content"]
    assert "different wrong" in calls[0][0][0]["content"]
    assert "Do not use distorted when the gold" in calls[0][0][0]["content"]
    assert "answer is entirely absent" in calls[0][0][0]["content"]
    assert "matching_tool_responses" in calls[0][0][1]["content"]
    assert "raw_response" not in retention._load_jsonl(tmp_path / "judgments.jsonl")[0]

    # A compatible rerun consumes the completed judgment without another API call.
    asyncio.run(
        retention.judge_stage(
            tmp_path,
            tmp_path,
            model="judge-v1",
            base_url="https://judge.invalid/v1",
            api_key=None,
            concurrency=2,
            max_retries=1,
            keep_raw_responses=False,
            call_model=fake_call,
        )
    )
    assert len(calls) == 1

    with pytest.raises(ValueError, match="incompatible"):
        asyncio.run(
            retention.judge_stage(
                tmp_path,
                tmp_path,
                model="judge-v2",
                base_url="https://judge.invalid/v1",
                api_key=None,
                concurrency=2,
                max_retries=1,
                keep_raw_responses=False,
                call_model=fake_call,
            )
        )
    assert (tmp_path / "_JUDGED").is_file()
    assert json.loads((tmp_path / "judge_manifest.json").read_text())["judge_model"] == "judge-v1"

    output = tmp_path / "report"
    report = retention.build_report([tmp_path], output)
    metrics = report["points"][0]["metrics"]
    assert metrics["drop_count"] == 1
    assert metrics["drop_rate"] == 1.0
    assert metrics["summary_loss_count"] == 1
    assert metrics["summary_loss_rate"] == 1.0
    assert metrics["retention_coverage"] == 1.0
    assert (output / "summary_retention_report.md").is_file()
    assert (output / "summary_retention_metrics.csv").is_file()


@pytest.mark.unit
def test_judge_canary_limits_new_candidates_and_remains_resumable(tmp_path):
    candidates = []
    for rollout_id in range(2):
        candidates.append(
            {
                "schema_version": retention.SCHEMA_VERSION,
                "prefilter_version": retention.PREFILTER_VERSION,
                "candidate_id": f"base:{rollout_id}",
                "point": "base",
                "rollout_id": rollout_id,
                "query_id": f"q{rollout_id}",
                "question": "Question",
                "gold_answer": "Alpha",
                "gold_parts": ["alpha"],
                "final_answer": "Gamma",
                "n_sub_trajs": 2,
                "matching_tool_responses": [
                    {
                        "evidence_id": "evidence-001",
                        "tool": "search",
                        "docid": "1",
                        "matched_gold_parts": ["alpha"],
                        "content": "alpha occurs in an unrelated context",
                        "occurrences": [{"sub_traj_index": 0, "tool_response_index": 0}],
                    }
                ],
                "final_handover": {"summary": "No answer."},
            }
        )
    retention._write_jsonl(tmp_path / "candidates.jsonl", candidates)
    retention._write_json(
        tmp_path / "stage_manifest.json",
        {
            "schema_version": retention.SCHEMA_VERSION,
            "prefilter_version": retention.PREFILTER_VERSION,
            "point": "base",
            "counts": {"n_prefilter_candidates": 2},
        },
    )
    retention._write_json(tmp_path / "_STAGED", {"status": "ok"})

    async def fake_call(messages, model):
        return json.dumps(_judge_response(["evidence-001"], evidence_verdict="no"))

    first = asyncio.run(
        retention.judge_stage(
            tmp_path,
            tmp_path,
            model="judge-v1",
            base_url="https://judge.invalid/v1",
            api_key=None,
            concurrency=1,
            max_retries=1,
            keep_raw_responses=True,
            max_new_candidates=1,
            call_model=fake_call,
        )
    )
    assert first["n_judgments"] == 1
    assert first["n_remaining"] == 1
    assert first["complete"] is False
    assert not (tmp_path / "_JUDGED").exists()

    second = asyncio.run(
        retention.judge_stage(
            tmp_path,
            tmp_path,
            model="judge-v1",
            base_url="https://judge.invalid/v1",
            api_key=None,
            concurrency=1,
            max_retries=1,
            keep_raw_responses=True,
            max_new_candidates=1,
            call_model=fake_call,
        )
    )
    assert second["n_judgments"] == 2
    assert second["n_resumed"] == 1
    assert second["complete"] is True
    assert (tmp_path / "_JUDGED").is_file()


@pytest.mark.unit
def test_large_candidate_batches_evidence_and_merges_one_judgment():
    evidence = [
        {
            "evidence_id": f"evidence-{index:03d}",
            "matched_gold_parts": ["alpha"],
            "content": f"Alpha context {index}",
        }
        for index in range(1, 42)
    ]
    candidate = {
        "candidate_id": "base:0",
        "point": "base",
        "rollout_id": 0,
        "question": "Question",
        "gold_answer": "Alpha",
        "gold_parts": ["alpha"],
        "matching_tool_responses": evidence,
        "final_handover": {"summary": "No answer."},
    }
    calls = []

    async def fake_call(messages, model):
        payload = json.loads(messages[1]["content"].split("CANDIDATE:\n", 1)[1])
        ids = [record["evidence_id"] for record in payload["matching_tool_responses"]]
        calls.append(ids)
        return json.dumps(_judge_response(ids))

    judgment = asyncio.run(
        retention.judge_candidate(
            candidate,
            model="judge-v1",
            call_model=fake_call,
            max_retries=1,
            keep_raw_response=False,
        )
    )

    assert [len(batch) for batch in calls] == [16, 16, 9]
    assert judgment["request_batches"] == 3
    assert judgment["attempts"] == 3
    assert len(judgment["verdict"]["evidence_assessments"]) == 41
    assert judgment["verdict"]["early_retrieval"] == "yes"
    assert judgment["verdict"]["summary_retention"] == "dropped"


@pytest.mark.unit
def test_drop_rate_denominator_excludes_unclear_retention():
    def judgment(candidate_id, retrieval_verdict, retention_verdict):
        return {
            "candidate_id": candidate_id,
            "verdict": {
                "early_retrieval": retrieval_verdict,
                "summary_retention": retention_verdict,
            },
        }

    judgments = [
        judgment("a", "yes", "carried"),
        judgment("b", "yes", "dropped"),
        judgment("c", "yes", "distorted"),
        judgment("d", "yes", "unclear"),
        judgment("e", "no", "carried"),
    ]

    metrics = retention.summarize_verdicts(judgments)

    assert metrics["n_confirmed_early_retrieval"] == 4
    assert metrics["n_resolved_confirmed_retrieval"] == 3
    assert metrics["retention_coverage"] == 0.75
    assert metrics["drop_count"] == 1
    assert metrics["drop_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert metrics["distorted_count"] == 1
    assert metrics["summary_loss_count"] == 2
    assert metrics["summary_loss_rate"] == pytest.approx(2 / 3, abs=1e-6)

    headroom = retention.summarize_failure_headroom(
        {
            "n_rollouts": 10,
            "n_model_failures": 5,
            "n_failures_with_gold_docs": 4,
            "n_failures_retrieved_gold_doc": 1,
            "n_failures_opened_gold_doc": 1,
            "n_failures_with_evidence_docs": 5,
            "n_failures_retrieved_evidence_doc": 3,
            "n_failures_opened_evidence_doc": 2,
        },
        metrics,
    )
    assert headroom["failure_no_gold_doc_rate"] == 0.75
    assert headroom["failure_no_evidence_doc_rate"] == 0.4
    assert headroom["drop_share_of_model_failures"] == 0.2
    assert headroom["optimistic_drop_uplift_all_rollouts"] == 0.1


@pytest.mark.unit
def test_failure_cause_table_is_mutually_exclusive_and_summary_verdict_takes_priority():
    def failure(candidate_id, *, retrieved):
        return {
            "candidate_id": candidate_id,
            "has_evidence_docs": True,
            "retrieved_evidence_doc": retrieved,
        }

    failure_rows = [
        failure("a", retrieved=False),
        failure("b", retrieved=True),
        failure("c", retrieved=False),
        failure("d", retrieved=True),
        failure("e", retrieved=True),
    ]
    judgments = {
        "c": {"verdict": {"early_retrieval": "yes", "summary_retention": "dropped"}},
        "d": {"verdict": {"early_retrieval": "yes", "summary_retention": "carried"}},
        "e": {"verdict": {"early_retrieval": "unclear", "summary_retention": "distorted"}},
    }

    metrics = retention.summarize_failure_causes(failure_rows, judgments)

    assert metrics["n_exclusive_failure_causes"] == 5
    assert sum(metrics["failure_cause_counts"].values()) == 5
    assert metrics["failure_cause_counts"] == {
        "no_evidence_doc_retrieved": 1,
        "evidence_doc_retrieved_answer_not_confirmed": 1,
        "summary_dropped": 1,
        "summary_distorted": 0,
        "summary_carried_final_wrong": 1,
        "unresolved": 1,
    }
