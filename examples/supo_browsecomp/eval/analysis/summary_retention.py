#!/usr/bin/env python3
"""Auditable semantic evaluation of answer retention across BC+ compression.

The pipeline has three explicit stages:

1. ``stage`` applies a high-recall lexical pre-filter to validated point data.
2. ``judge`` independently judges every gold-part/response match, then judges the
   final handover only for candidates with confirmed early retrieval.
3. ``report`` validates one-to-one coverage and computes deterministic metrics.

The lexical match is candidate generation only. A rollout contributes to the
summary-loss denominator only after the semantic judge confirms that the gold
answer was genuinely present in an earlier tool observation.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 5
PREFILTER_VERSION = "gold-parts-across-early-responses-v4"
JUDGE_PROTOCOL_VERSION = "part-semantic-summary-v7.2"
API_REQUEST_TIMEOUT_SECONDS = 180.0
API_MAX_OUTPUT_TOKENS = 4_096
OBSERVATION_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL | re.IGNORECASE)
PARAMETER_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE)
SEARCH_RESULT_RE = re.compile(r"(?=---\s*#\d+\s*:)", re.IGNORECASE)
MULTI_ANSWER_RE = re.compile(r"<q\d+>(.*?)</q\d+>", re.DOTALL | re.IGNORECASE)
EVIDENCE_VERDICTS = {"yes", "no", "unclear"}
EARLY_RETRIEVAL_VERDICTS = {"yes", "no", "unclear"}
RETENTION_VERDICTS = {"carried", "dropped", "distorted", "unclear"}
SUMMARY_NOT_JUDGED = "not_judged"
FAILURE_CAUSES = (
    "no_evidence_doc_retrieved",
    "evidence_doc_retrieved_answer_not_confirmed",
    "summary_dropped",
    "summary_distorted",
    "summary_carried_final_wrong",
    "unresolved",
)

MATCH_JUDGE_INSTRUCTIONS = """You judge one lexical match between one gold-answer part and one earlier tool response.

Decide only whether the matched occurrence in the supplied full tool response is
genuinely the same GOLD_PART that appears within GOLD_ANSWER for this QUESTION.

- yes: this occurrence is genuinely the same answer part.
- no: it is an unrelated same-name mention, numeric/date collision, different work,
  different role, or another false lexical match.
- unclear: the response lacks enough context to distinguish yes from no.

The gold answer is authoritative. Do not solve the question, challenge the gold, or
require this response to prove that GOLD_PART satisfies the question's requested
relation, full answer, or every clue.

Apply the following type-specific rule strictly:
- For a distinctive named person, institution, place, or work, judge entity identity
  only. If the response refers to the same named entity, verdict=yes even when it
  mentions that entity in a relation unrelated to the question. NEVER require the
  response to assert that the entity is the requested winner, creator, location, etc.
  Example: if GOLD_PART is "Yemi Alade," a response saying Spice collaborated with
  Yemi Alade is yes because it is the same person, even if it says nothing about the
  award asked in the question.
- For an ambiguous common word, number, or date, its local semantic role must match.
  Example: 2015 as the requested award year is yes, but 2015 as an article publication
  date is no; ordinary weather "Rain" is not a work titled "Rain".

Search-query text may explain retrieval intent but is not factual evidence.

Return one JSON object and no markdown, with exactly these fields:
{
  "match_id": "the supplied match_id",
  "verdict": "yes|no|unclear",
  "rationale": "brief reason grounded in this response"
}
"""

SUMMARY_JUDGE_INSTRUCTIONS = """You judge whether a final handover summary retained a supplied gold answer.

Judge only GOLD_ANSWER against FINAL_HANDOVER_SUMMARY:

- carried: the gold answer is preserved accurately, including an unambiguous
  paraphrase or clearly named usable candidate. A competing wrong candidate does not
  erase a still-usable gold candidate.
- dropped: the gold answer itself is absent. Use dropped when the summary instead
  gives another answer, retains only related clues or a parent entity, or follows a
  wrong lead. Treat the supplied summary as complete.
- distorted: the gold remains recognizably present but is materially corrupted,
  contradicted, rejected, or only partly retained when the gold answer has multiple
  required parts.
- unclear: the supplied summary is insufficient to distinguish these outcomes.

Do not infer omitted content and do not use any earlier retrieval evidence.
Return one JSON object and no markdown, with exactly these fields:
{
  "summary_retention": "carried|dropped|distorted|unclear",
  "summary_rationale": "brief reason grounded only in the handover summary"
}
"""
ModelCaller = Callable[[list[dict[str, str]], str], Awaitable[str]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _gold_parts(gold: Any) -> list[str]:
    text = _norm(gold)
    tagged = [_norm(value) for value in MULTI_ANSWER_RE.findall(text) if _norm(value)]
    raw_parts = tagged or re.split(r"\s*(?:[;,]|\band\b)\s*", text)
    parts = [part.strip(" \t\r\n\"'") for part in raw_parts]
    minimum_length = 3 if tagged or len(raw_parts) > 1 else 2
    parts = [part for part in parts if len(part) >= minimum_length]
    return list(dict.fromkeys(parts))


def _docids(value: Any) -> set[str]:
    """Normalize dataset and rollout docid metadata into strings."""
    if value is None:
        return set()
    if isinstance(value, str):
        value = value.strip()
        return {value} if value else set()
    if isinstance(value, dict):
        for key in ("docid", "docids", "doc_id", "doc_ids", "document_id", "document_ids"):
            if key in value:
                return _docids(value[key])
        return set()
    if isinstance(value, (list, tuple, set)):
        return set().union(*(_docids(item) for item in value)) if value else set()
    if isinstance(value, int) and not isinstance(value, bool):
        return {str(value)}
    return set()


def _trajectory_doc_coverage(siblings: list[dict[str, Any]]) -> dict[str, set[str]]:
    final_metadata = siblings[-1].get("metadata")
    final_metadata = final_metadata if isinstance(final_metadata, dict) else {}
    retrieved: set[str] = set()
    opened: set[str] = set()
    for sample in siblings:
        bcplus = _bc(sample)
        retrieved.update(_docids(bcplus.get("retrieved_docids")))
        opened.update(_docids(bcplus.get("opened_docids")))
    return {
        "gold_docs": _docids(final_metadata.get("gold_docs")),
        "evidence_docs": _docids(final_metadata.get("evidence_docs")),
        "retrieved": retrieved,
        "opened": opened,
    }


def _match_span(text: str, phrase: str) -> re.Match[str] | None:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.IGNORECASE)


def _tool_call_details(block: str | None) -> tuple[str, dict[str, str]]:
    if not block:
        return "unknown", {}
    functions = FUNCTION_RE.findall(block)
    if not functions:
        return "unknown", {}
    name, body = functions[-1]
    parameters = {key.strip().lower(): value.strip() for key, value in PARAMETER_RE.findall(body)}
    return name.strip().lower(), parameters


def _metadata_value(text: str, key: str) -> str | None:
    if key == "title":
        match = re.search(r"\btitle:\s*(.*?)(?=\s+date:|\n|---|$)", text, re.IGNORECASE | re.DOTALL)
    else:
        match = re.search(rf"(?im)^\s*{re.escape(key)}:\s*(.*?)\s*$", text)
    return match.group(1).strip() if match and match.group(1).strip() else None


def _matching_tool_response_records(
    response: Any, parts: list[str], *, sub_traj_index: int
) -> tuple[list[dict[str, Any]], set[str]]:
    """Return response-level evidence records and the gold parts they cover."""
    text = str(response or "")
    calls = list(TOOL_CALL_RE.finditer(text))
    call_index = 0
    latest_call: re.Match[str] | None = None
    records = []
    covered_parts: set[str] = set()

    for response_index, observation_match in enumerate(OBSERVATION_RE.finditer(text)):
        while call_index < len(calls) and calls[call_index].end() <= observation_match.start():
            latest_call = calls[call_index]
            call_index += 1
        tool, parameters = _tool_call_details(latest_call.group(1) if latest_call else None)
        observation = observation_match.group(1)
        if tool == "unknown":
            if "[search results" in observation.lower():
                tool = "search"
            elif "[opened page content]" in observation.lower():
                tool = "open_page"

        units = [observation]
        if tool == "search":
            split_units = [
                value for value in SEARCH_RESULT_RE.split(observation) if re.match(r"---\s*#\d+\s*:", value)
            ]
            if split_units:
                units = split_units

        for unit in units:
            normalized = _norm(unit)
            matched_parts = [part for part in parts if _match_span(normalized, part) is not None]
            if not matched_parts:
                continue
            covered_parts.update(matched_parts)
            content = unit.strip()
            docid = _metadata_value(unit, "docid") or parameters.get("docid")
            record = {
                "tool": tool,
                "docid": docid,
                "url": _metadata_value(unit, "url"),
                "title": _metadata_value(unit, "title"),
                "matched_gold_parts": matched_parts,
                "content": content,
                "content_truncated": False,
                "original_content_chars": len(content),
                "occurrences": [
                    {
                        "sub_traj_index": sub_traj_index,
                        "tool_response_index": response_index,
                    }
                ],
            }
            if tool == "search" and parameters.get("query"):
                record["queries"] = [parameters["query"]]
            records.append(record)
    return records, covered_parts


def _deduplicate_evidence(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        identity = str(record.get("docid") or record.get("url") or record.get("content", ""))
        key = (str(record.get("tool")), identity, str(record.get("content", "")))
        existing = deduplicated.get(key)
        if existing is None:
            deduplicated[key] = record
            continue
        existing["matched_gold_parts"] = list(
            dict.fromkeys([*existing["matched_gold_parts"], *record["matched_gold_parts"]])
        )
        existing["occurrences"].extend(record["occurrences"])
        if record.get("queries"):
            existing["queries"] = list(dict.fromkeys([*existing.get("queries", []), *record["queries"]]))
    return list(deduplicated.values())


def _bc(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata")
    value = metadata.get("_bcplus") if isinstance(metadata, dict) else None
    return value if isinstance(value, dict) else {}


def _sibling(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata")
    value = metadata.get("_bcplus_sibling") if isinstance(metadata, dict) else None
    return value if isinstance(value, dict) else {}


def _validated_groups(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        rollout_id = sample.get("rollout_id")
        if rollout_id is None:
            raise ValueError("raw sample is missing rollout_id")
        grouped[str(rollout_id)].append(sample)

    for rollout_id, siblings in grouped.items():
        indices = [_sibling(sample).get("sub_traj_index") for sample in siblings]
        if any(not isinstance(index, int) for index in indices):
            raise ValueError(f"rollout {rollout_id} has invalid sibling indices: {indices}")
        if sorted(indices) != list(range(len(siblings))):
            raise ValueError(f"rollout {rollout_id} has non-contiguous sibling indices: {indices}")
        totals = {_sibling(sample).get("total_sub_trajs") for sample in siblings}
        if totals != {len(siblings)}:
            raise ValueError(f"rollout {rollout_id} has invalid total_sub_trajs: {totals}")
        finals = [sample for sample in siblings if _sibling(sample).get("is_final") is True]
        if len(finals) != 1 or _sibling(finals[0]).get("sub_traj_index") != len(siblings) - 1:
            raise ValueError(f"rollout {rollout_id} does not have one final sibling at the last index")
        grouped[rollout_id] = sorted(siblings, key=lambda sample: _sibling(sample)["sub_traj_index"])
    return dict(grouped)


def build_candidates(
    samples: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    *,
    point: str,
    failure_retrieval_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build high-recall candidates from current point artifacts and raw samples."""
    groups = _validated_groups(samples)
    rows_by_id = {str(row.get("rollout_id")): row for row in rollout_rows}
    if len(rows_by_id) != len(rollout_rows):
        raise ValueError("rollouts.jsonl contains duplicate rollout_id values")
    if groups.keys() != rows_by_id.keys():
        raise ValueError("raw dump and rollouts.jsonl contain different rollout ids")

    counts = Counter(
        {
            "n_rollouts": len(rollout_rows),
            "n_incorrect": 0,
            "n_judge_failures_excluded": 0,
            "n_model_failures": 0,
            "n_failures_with_gold_docs": 0,
            "n_failures_retrieved_gold_doc": 0,
            "n_failures_opened_gold_doc": 0,
            "n_failures_with_evidence_docs": 0,
            "n_failures_retrieved_evidence_doc": 0,
            "n_failures_opened_evidence_doc": 0,
            "n_compressed_model_failures": 0,
            "n_prefilter_candidates": 0,
            "n_semantic_match_tasks": 0,
        }
    )
    candidates = []
    for rollout_id, row in rows_by_id.items():
        if bool(row.get("correct")):
            continue
        counts["n_incorrect"] += 1
        if bool(row.get("judge_failed")):
            counts["n_judge_failures_excluded"] += 1
            continue
        counts["n_model_failures"] += 1
        siblings = groups[rollout_id]
        doc_coverage = _trajectory_doc_coverage(siblings)
        if failure_retrieval_rows is not None:
            failure_retrieval_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "candidate_id": f"{point}:{rollout_id}",
                    "point": point,
                    "rollout_id": row.get("rollout_id"),
                    "query_id": row.get("query_id"),
                    "has_gold_docs": bool(doc_coverage["gold_docs"]),
                    "retrieved_gold_doc": bool(doc_coverage["gold_docs"] & doc_coverage["retrieved"]),
                    "opened_gold_doc": bool(doc_coverage["gold_docs"] & doc_coverage["opened"]),
                    "has_evidence_docs": bool(doc_coverage["evidence_docs"]),
                    "retrieved_evidence_doc": bool(doc_coverage["evidence_docs"] & doc_coverage["retrieved"]),
                    "opened_evidence_doc": bool(doc_coverage["evidence_docs"] & doc_coverage["opened"]),
                }
            )
        for label in ("gold", "evidence"):
            annotated = doc_coverage[f"{label}_docs"]
            if not annotated:
                continue
            counts[f"n_failures_with_{label}_docs"] += 1
            if annotated & doc_coverage["retrieved"]:
                counts[f"n_failures_retrieved_{label}_doc"] += 1
            if annotated & doc_coverage["opened"]:
                counts[f"n_failures_opened_{label}_doc"] += 1
        if len(siblings) < 2:
            continue
        counts["n_compressed_model_failures"] += 1

        parts = _gold_parts(row.get("gold"))
        evidence_records = []
        covered_parts_across_sub_trajs: set[str] = set()
        for index, sample in enumerate(siblings[:-1]):
            records, covered_parts = _matching_tool_response_records(
                sample.get("response", ""),
                parts,
                sub_traj_index=index,
            )
            evidence_records.extend(records)
            covered_parts_across_sub_trajs.update(covered_parts)
        if not parts or not set(parts).issubset(covered_parts_across_sub_trajs):
            continue
        evidence = _deduplicate_evidence(evidence_records)
        for evidence_index, record in enumerate(evidence, start=1):
            record["evidence_id"] = f"evidence-{evidence_index:03d}"

        gold_part_records = [
            {"gold_part_id": f"part-{index:03d}", "text": part}
            for index, part in enumerate(parts, start=1)
        ]
        part_ids = {record["text"]: record["gold_part_id"] for record in gold_part_records}
        semantic_match_tasks = []
        for evidence_record in evidence:
            for part in evidence_record["matched_gold_parts"]:
                semantic_match_tasks.append(
                    {
                        "match_id": f"match-{len(semantic_match_tasks) + 1:04d}",
                        "evidence_id": evidence_record["evidence_id"],
                        "gold_part_id": part_ids[part],
                        "gold_part": part,
                    }
                )

        handover_index = len(siblings) - 2
        handover_bc = _bc(siblings[handover_index])
        candidate = {
            "schema_version": SCHEMA_VERSION,
            "prefilter_version": PREFILTER_VERSION,
            "candidate_id": f"{point}:{rollout_id}",
            "point": point,
            "rollout_id": row.get("rollout_id"),
            "query_id": row.get("query_id"),
            "question": row.get("question"),
            "gold_answer": row.get("gold"),
            "gold_parts": parts,
            "gold_part_records": gold_part_records,
            "final_answer": row.get("finish_answer", ""),
            "n_sub_trajs": len(siblings),
            "matching_tool_responses": evidence,
            "semantic_match_tasks": semantic_match_tasks,
            "final_handover": {
                "sub_traj_index": handover_index,
                "outcome": handover_bc.get("outcome", ""),
                "summary_source": handover_bc.get("summary_source", ""),
                "summary": handover_bc.get("summary"),
            },
        }
        candidates.append(candidate)
        counts["n_prefilter_candidates"] += 1
        counts["n_semantic_match_tasks"] += len(semantic_match_tasks)

    candidates.sort(key=lambda row: str(row["candidate_id"]))
    if failure_retrieval_rows is not None:
        failure_retrieval_rows.sort(key=lambda row: str(row["candidate_id"]))
    return candidates, dict(counts)


def validate_source_point(success: dict[str, Any], manifest: dict[str, Any]) -> tuple[str, int | str]:
    """Trust a completed point after checking its recorded checkpoint identity once."""
    if success.get("status") != "ok":
        raise ValueError("point does not have a successful completion marker")
    point = str(manifest.get("point") or "")
    if not point:
        raise ValueError("point manifest is missing point")
    expected_step: int | str
    if point == "base":
        expected_step = "base"
    else:
        match = re.fullmatch(r"iter0*(\d+)", point)
        if match is None:
            raise ValueError(f"invalid point name in manifest: {point!r}")
        expected_step = int(match.group(1))
    actual_step = manifest.get("load_verification", {}).get("actual_step")
    if actual_step != expected_step:
        raise ValueError(f"point {point} says it loaded {actual_step!r}, expected {expected_step!r}")
    return point, actual_step


def stage_point(point_dir: Path, output_dir: Path) -> dict[str, Any]:
    required = ("_SUCCESS", "manifest.json", "rollouts.jsonl")
    missing = [name for name in required if not (point_dir / name).is_file()]
    if missing:
        raise ValueError(f"point directory {point_dir} is missing artifacts: {missing}")
    dump = point_dir / "rollout_data/eval_0.pt"
    if not dump.is_file():
        raise ValueError(f"point directory {point_dir} is missing raw dump {dump}")

    success = json.loads((point_dir / "_SUCCESS").read_text())
    manifest = json.loads((point_dir / "manifest.json").read_text())
    point, actual_step = validate_source_point(success, manifest)
    rollout_rows = _load_jsonl(point_dir / "rollouts.jsonl")
    import torch

    blob = torch.load(dump, weights_only=False)
    samples = blob.get("samples", []) if isinstance(blob, dict) else []
    failure_retrieval_rows: list[dict[str, Any]] = []
    candidates, counts = build_candidates(
        samples,
        rollout_rows,
        point=point,
        failure_retrieval_rows=failure_retrieval_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_STAGED").unlink(missing_ok=True)
    candidates_path = output_dir / "candidates.jsonl"
    existing_candidates = _load_jsonl(candidates_path) if candidates_path.is_file() else None
    if existing_candidates is not None and existing_candidates != candidates and (output_dir / "_JUDGED").is_file():
        raise ValueError("staged candidates changed after judging; use a fresh analysis directory")
    if existing_candidates is not None and existing_candidates != candidates:
        for stale_name in (
            "judge_manifest.json",
            "match_judgments.jsonl",
            "summary_judgments.jsonl",
            "judgments.jsonl",
        ):
            (output_dir / stale_name).unlink(missing_ok=True)
    _write_jsonl(candidates_path, candidates)
    _write_jsonl(output_dir / "failure_retrieval.jsonl", failure_retrieval_rows)
    stage_manifest = {
        "schema_version": SCHEMA_VERSION,
        "prefilter_version": PREFILTER_VERSION,
        "created_at": _utc_now(),
        "point": point,
        "source_point_dir": str(point_dir.resolve()),
        "source_actual_step": actual_step,
        "counts": counts,
        "definitions": {
            "model_failure": "correct is false and judge_failed is false",
            "retrieved_gold_doc": "at least one dataset gold_docs docid appeared in any search response",
            "retrieved_evidence_doc": (
                "at least one broader dataset evidence_docs docid appeared in any search response"
            ),
            "opened_gold_or_evidence_doc": "the matching retrieved docid was also opened in any sub-trajectory",
            "compressed_model_failure": "model failure with at least two sub-trajectories",
            "prefilter_candidate": (
                "all normalized gold parts occur somewhere across tool observations of non-final sub-trajectories"
            ),
            "semantic_match_task": (
                "one independently judged lexical (gold_part, full tool response) match"
            ),
        },
    }
    _write_json(output_dir / "stage_manifest.json", stage_manifest)
    _write_json(output_dir / "_STAGED", {"status": "ok", "n_candidates": len(candidates)})
    return stage_manifest


def _match_prompt(
    candidate: dict[str, Any], task: dict[str, Any], evidence: dict[str, Any]
) -> str:
    payload = {
        "candidate_id": candidate["candidate_id"],
        "match_id": task["match_id"],
        "question": candidate["question"],
        "gold_answer": candidate["gold_answer"],
        "gold_part": task["gold_part"],
        "matching_tool_response": evidence,
    }
    return "Judge this one lexical match.\n\nMATCH:\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def _summary_prompt(candidate: dict[str, Any]) -> str:
    payload = {
        "candidate_id": candidate["candidate_id"],
        "gold_answer": candidate["gold_answer"],
        "final_handover_summary": candidate["final_handover"].get("summary"),
    }
    return "Judge this final handover.\n\nSUMMARY:\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def _extract_json_object(response: str) -> dict[str, Any]:
    text = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge response does not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge response JSON is not an object")
    return value


def parse_match_verdict(response: str, expected_match_id: str) -> dict[str, str]:
    value = _extract_json_object(response)
    required = {"match_id", "verdict", "rationale"}
    if not required.issubset(value):
        raise ValueError(f"match verdict is missing required fields: {sorted(required - set(value))}")
    if value["match_id"] != expected_match_id:
        raise ValueError(f"match_id mismatch: expected {expected_match_id!r}, received {value['match_id']!r}")
    if value["verdict"] not in EVIDENCE_VERDICTS:
        raise ValueError(f"invalid match verdict: {value['verdict']!r}")
    if not isinstance(value["rationale"], str) or not value["rationale"].strip():
        raise ValueError("match rationale must be a non-empty string")
    return {
        "match_id": expected_match_id,
        "verdict": value["verdict"],
        "rationale": value["rationale"].strip(),
    }


def parse_summary_verdict(response: str) -> dict[str, str]:
    value = _extract_json_object(response)
    required = {"summary_retention", "summary_rationale"}
    if not required.issubset(value):
        raise ValueError(f"summary verdict is missing required fields: {sorted(required - set(value))}")
    if value["summary_retention"] not in RETENTION_VERDICTS:
        raise ValueError(f"invalid summary_retention verdict: {value['summary_retention']!r}")
    if not isinstance(value["summary_rationale"], str) or not value["summary_rationale"].strip():
        raise ValueError("summary_rationale must be a non-empty string")
    return {
        "summary_retention": value["summary_retention"],
        "summary_rationale": value["summary_rationale"].strip(),
    }


def _validate_candidate(candidate: dict[str, Any]) -> None:
    if candidate.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"candidate {candidate.get('candidate_id')} uses an unsupported schema")
    part_records = candidate.get("gold_part_records", [])
    part_ids = [record.get("gold_part_id") for record in part_records]
    part_text = {record.get("gold_part_id"): record.get("text") for record in part_records}
    if not part_ids or len(part_ids) != len(set(part_ids)) or list(part_text.values()) != candidate.get("gold_parts"):
        raise ValueError(f"candidate {candidate.get('candidate_id')} has invalid gold part records")
    evidence = candidate.get("matching_tool_responses", [])
    evidence_ids = [record.get("evidence_id") for record in evidence]
    if not evidence_ids or any(not isinstance(value, str) or not value for value in evidence_ids):
        raise ValueError(f"candidate {candidate.get('candidate_id')} has invalid evidence IDs")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError(f"candidate {candidate.get('candidate_id')} has duplicate evidence IDs")
    evidence_by_id = {record["evidence_id"]: record for record in evidence}
    tasks = candidate.get("semantic_match_tasks", [])
    match_ids = [task.get("match_id") for task in tasks]
    if not match_ids or len(match_ids) != len(set(match_ids)):
        raise ValueError(f"candidate {candidate.get('candidate_id')} has invalid match IDs")
    actual_pairs = set()
    for task in tasks:
        if set(task) != {"match_id", "evidence_id", "gold_part_id", "gold_part"}:
            raise ValueError(f"candidate {candidate.get('candidate_id')} has a malformed match task")
        evidence_record = evidence_by_id.get(task["evidence_id"])
        if evidence_record is None or part_text.get(task["gold_part_id"]) != task["gold_part"]:
            raise ValueError(f"candidate {candidate.get('candidate_id')} has an invalid match task reference")
        if task["gold_part"] not in evidence_record["matched_gold_parts"]:
            raise ValueError(f"candidate {candidate.get('candidate_id')} match task is not a lexical match")
        actual_pairs.add((task["evidence_id"], task["gold_part_id"]))
    expected_pairs = {
        (record["evidence_id"], part_record["gold_part_id"])
        for record in evidence
        for part_record in part_records
        if part_record["text"] in record["matched_gold_parts"]
    }
    if actual_pairs != expected_pairs or len(actual_pairs) != len(tasks):
        raise ValueError(f"candidate {candidate.get('candidate_id')} match tasks do not cover lexical matches")


def aggregate_candidate_verdict(
    candidate: dict[str, Any],
    *,
    match_judgments: dict[str, dict[str, Any]],
    summary_judgment: dict[str, Any] | None,
    require_summary: bool = True,
) -> dict[str, Any]:
    _validate_candidate(candidate)
    tasks = candidate["semantic_match_tasks"]
    expected_ids = {task["match_id"] for task in tasks}
    if set(match_judgments) != expected_ids:
        raise ValueError(f"match judgment coverage mismatch for {candidate['candidate_id']}")
    evidence_by_id = {record["evidence_id"]: record for record in candidate["matching_tool_responses"]}
    required_parts = {record["gold_part_id"] for record in candidate["gold_part_records"]}
    sub_traj_parts: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: {"yes": set(), "unclear": set()}
    )
    part_match_ids: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"yes": [], "no": [], "unclear": []}
    )
    for task in tasks:
        judgment = match_judgments[task["match_id"]]
        verdict = parse_match_verdict(json.dumps(judgment["verdict"]), task["match_id"])["verdict"]
        part_match_ids[task["gold_part_id"]][verdict].append(task["match_id"])
        if verdict not in {"yes", "unclear"}:
            continue
        for occurrence in evidence_by_id[task["evidence_id"]]["occurrences"]:
            sub_traj_parts[int(occurrence["sub_traj_index"])][verdict].add(task["gold_part_id"])

    sub_traj_assessments = []
    confirmed_sub_trajs = []
    for sub_traj_index, values in sorted(sub_traj_parts.items()):
        if required_parts.issubset(values["yes"]):
            verdict = "yes"
            confirmed_sub_trajs.append(sub_traj_index)
        elif required_parts.issubset(values["yes"] | values["unclear"]):
            verdict = "unclear"
        else:
            verdict = "no"
        sub_traj_assessments.append(
            {
                "sub_traj_index": sub_traj_index,
                "verdict": verdict,
                "confirmed_gold_part_ids": sorted(values["yes"]),
                "unclear_gold_part_ids": sorted(values["unclear"]),
                "missing_gold_part_ids": sorted(required_parts - values["yes"] - values["unclear"]),
            }
        )
    part_assessments = []
    confirmed_parts = set()
    unclear_parts = set()
    for part in candidate["gold_part_records"]:
        match_ids = part_match_ids[part["gold_part_id"]]
        if match_ids["yes"]:
            verdict = "yes"
            confirmed_parts.add(part["gold_part_id"])
        elif match_ids["unclear"]:
            verdict = "unclear"
            unclear_parts.add(part["gold_part_id"])
        else:
            verdict = "no"
        part_assessments.append({**part, "verdict": verdict, "match_ids_by_verdict": match_ids})

    if required_parts.issubset(confirmed_parts):
        early_retrieval = "yes"
    elif required_parts.issubset(confirmed_parts | unclear_parts):
        early_retrieval = "unclear"
    else:
        early_retrieval = "no"
    if early_retrieval == "yes":
        if summary_judgment is None and require_summary:
            raise ValueError(f"confirmed candidate {candidate['candidate_id']} is missing its summary judgment")
        if summary_judgment is None:
            summary_verdict = {
                "summary_retention": SUMMARY_NOT_JUDGED,
                "summary_rationale": "Summary judge has not run yet.",
            }
        else:
            summary_verdict = parse_summary_verdict(json.dumps(summary_judgment["verdict"]))
    else:
        if summary_judgment is not None:
            raise ValueError(f"unconfirmed candidate {candidate['candidate_id']} has a summary judgment")
        summary_verdict = {
            "summary_retention": SUMMARY_NOT_JUDGED,
            "summary_rationale": "Not judged because early retrieval was not semantically confirmed.",
        }
    return {
        "part_assessments": part_assessments,
        "sub_traj_assessments": sub_traj_assessments,
        "individually_confirmed_sub_traj_indices": confirmed_sub_trajs,
        "early_retrieval": early_retrieval,
        **summary_verdict,
    }


async def _call_with_retries(
    messages: list[dict[str, str]],
    *,
    model: str,
    call_model: ModelCaller,
    max_retries: int,
    parse: Callable[[str], dict[str, str]],
    label: str,
) -> tuple[dict[str, str], str, int]:
    errors = []
    for attempt in range(1, max_retries + 1):
        try:
            raw_response = await call_model(messages, model)
            return parse(raw_response), raw_response, attempt
        except Exception as error:
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            if attempt < max_retries:
                await asyncio.sleep(min(attempt, 3))
    raise RuntimeError(f"judge failed for {label}: " + "; ".join(errors))


async def judge_match(
    candidate: dict[str, Any],
    task: dict[str, Any],
    evidence: dict[str, Any],
    *,
    model: str,
    call_model: ModelCaller,
    max_retries: int,
    keep_raw_response: bool,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": MATCH_JUDGE_INSTRUCTIONS},
        {"role": "user", "content": _match_prompt(candidate, task, evidence)},
    ]
    verdict, raw_response, attempts = await _call_with_retries(
        messages,
        model=model,
        call_model=call_model,
        max_retries=max_retries,
        parse=lambda response: parse_match_verdict(response, task["match_id"]),
        label=f"{candidate['candidate_id']}:{task['match_id']}",
    )
    judgment = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate["candidate_id"],
        "point": candidate["point"],
        "rollout_id": candidate["rollout_id"],
        "match_id": task["match_id"],
        "evidence_id": task["evidence_id"],
        "gold_part_id": task["gold_part_id"],
        "gold_part": task["gold_part"],
        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
        "judge_model": model,
        "attempts": attempts,
        "judged_at": _utc_now(),
        "verdict": verdict,
    }
    if keep_raw_response:
        judgment["raw_response"] = raw_response
    return judgment


async def judge_summary(
    candidate: dict[str, Any],
    *,
    model: str,
    call_model: ModelCaller,
    max_retries: int,
    keep_raw_response: bool,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SUMMARY_JUDGE_INSTRUCTIONS},
        {"role": "user", "content": _summary_prompt(candidate)},
    ]
    verdict, raw_response, attempts = await _call_with_retries(
        messages,
        model=model,
        call_model=call_model,
        max_retries=max_retries,
        parse=parse_summary_verdict,
        label=f"{candidate['candidate_id']}:summary",
    )
    judgment = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate["candidate_id"],
        "point": candidate["point"],
        "rollout_id": candidate["rollout_id"],
        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
        "judge_model": model,
        "attempts": attempts,
        "judged_at": _utc_now(),
        "verdict": verdict,
    }
    if keep_raw_response:
        judgment["raw_response"] = raw_response
    return judgment


def _openai_caller(base_url: str, api_key: str) -> ModelCaller:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=API_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    async def call(messages: list[dict[str, str]], model: str) -> str:
        output_limit = (
            {"max_tokens": API_MAX_OUTPUT_TOKENS}
            if "claude" in model.lower()
            else {"max_completion_tokens": API_MAX_OUTPUT_TOKENS}
        )
        response = await client.chat.completions.create(model=model, messages=messages, **output_limit)
        return response.choices[0].message.content or ""

    return call


def _validate_resumed_match_judgments(
    judgments: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    model: str,
    *,
    keep_raw_responses: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    by_id = {}
    for judgment in judgments:
        candidate_id = str(judgment.get("candidate_id"))
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ValueError(f"resumed match judgment has unknown candidate {candidate_id}")
        match_id = str(judgment.get("match_id"))
        key = (candidate_id, match_id)
        if key in by_id:
            raise ValueError(f"duplicate resumed match judgment for {candidate_id}:{match_id}")
        task = next((value for value in candidate["semantic_match_tasks"] if value["match_id"] == match_id), None)
        if task is None:
            raise ValueError(f"resumed match judgment has unknown match {candidate_id}:{match_id}")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
            "judge_model": model,
            "evidence_id": task["evidence_id"],
            "gold_part_id": task["gold_part_id"],
            "gold_part": task["gold_part"],
        }
        mismatches = {
            key: (expected_value, judgment.get(key))
            for key, expected_value in expected.items()
            if judgment.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(f"resumed match judgment {candidate_id}:{match_id} is incompatible: {mismatches}")
        parse_match_verdict(json.dumps(judgment.get("verdict"), ensure_ascii=False), match_id)
        if keep_raw_responses and "raw_response" not in judgment:
            raise ValueError(f"resumed match judgment {candidate_id}:{match_id} lacks its raw response")
        if not keep_raw_responses:
            judgment = {key: value for key, value in judgment.items() if key != "raw_response"}
        by_id[key] = judgment
    return by_id


def _validate_resumed_summary_judgments(
    judgments: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    model: str,
    *,
    keep_raw_responses: bool,
) -> dict[str, dict[str, Any]]:
    by_id = {}
    for judgment in judgments:
        candidate_id = str(judgment.get("candidate_id"))
        if candidate_id in by_id:
            raise ValueError(f"duplicate resumed summary judgment for {candidate_id}")
        if candidate_id not in candidates:
            raise ValueError(f"resumed summary judgment has unknown candidate {candidate_id}")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
            "judge_model": model,
        }
        mismatches = {
            key: (expected_value, judgment.get(key))
            for key, expected_value in expected.items()
            if judgment.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(f"resumed summary judgment {candidate_id} is incompatible: {mismatches}")
        parse_summary_verdict(json.dumps(judgment.get("verdict"), ensure_ascii=False))
        if keep_raw_responses and "raw_response" not in judgment:
            raise ValueError(f"resumed summary judgment {candidate_id} lacks its raw response")
        if not keep_raw_responses:
            judgment = {key: value for key, value in judgment.items() if key != "raw_response"}
        by_id[candidate_id] = judgment
    return by_id


async def judge_stage(
    stage_dir: Path,
    output_dir: Path,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    concurrency: int,
    max_retries: int,
    keep_raw_responses: bool,
    max_new_candidates: int | None = None,
    call_model: ModelCaller | None = None,
) -> dict[str, Any]:
    if concurrency < 1 or max_retries < 1:
        raise ValueError("concurrency and max_retries must be positive")
    if max_new_candidates is not None and max_new_candidates < 1:
        raise ValueError("max_new_candidates must be positive")
    if stage_dir.resolve() != output_dir.resolve():
        raise ValueError("judge output_dir must equal stage_dir so the analysis artifact remains self-contained")
    for required in ("_STAGED", "stage_manifest.json", "candidates.jsonl"):
        if not (stage_dir / required).is_file():
            raise ValueError(f"stage directory {stage_dir} is missing {required}")
    stage_manifest = json.loads((stage_dir / "stage_manifest.json").read_text())
    if stage_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported staged candidate schema {stage_manifest.get('schema_version')!r}; rerun the stage command"
        )
    if stage_manifest.get("prefilter_version") != PREFILTER_VERSION:
        raise ValueError(f"unsupported prefilter version {stage_manifest.get('prefilter_version')!r}")
    candidates_path = stage_dir / "candidates.jsonl"
    candidate_rows = _load_jsonl(candidates_path)
    for candidate in candidate_rows:
        _validate_candidate(candidate)
    candidates = {str(row["candidate_id"]): row for row in candidate_rows}
    if len(candidates) != len(candidate_rows):
        raise ValueError("candidates.jsonl contains duplicate candidate_id values")
    output_dir.mkdir(parents=True, exist_ok=True)

    match_path = output_dir / "match_judgments.jsonl"
    summary_path = output_dir / "summary_judgments.jsonl"
    judgment_path = output_dir / "judgments.jsonl"
    resumed_match_rows = _load_jsonl(match_path) if match_path.is_file() else []
    resumed_summary_rows = _load_jsonl(summary_path) if summary_path.is_file() else []
    match_judgments = _validate_resumed_match_judgments(
        resumed_match_rows,
        candidates,
        model,
        keep_raw_responses=keep_raw_responses,
    )
    summary_judgments = _validate_resumed_summary_judgments(
        resumed_summary_rows,
        candidates,
        model,
        keep_raw_responses=keep_raw_responses,
    )

    def candidate_matches(candidate: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
        values = {
            task["match_id"]: match_judgments[(candidate["candidate_id"], task["match_id"])]
            for task in candidate["semantic_match_tasks"]
            if (candidate["candidate_id"], task["match_id"]) in match_judgments
        }
        return values if len(values) == len(candidate["semantic_match_tasks"]) else None

    def is_complete(candidate: dict[str, Any]) -> bool:
        values = candidate_matches(candidate)
        if values is None:
            return False
        preliminary = aggregate_candidate_verdict(
            candidate,
            match_judgments=values,
            summary_judgment=None,
            require_summary=False,
        )
        return preliminary["early_retrieval"] != "yes" or candidate["candidate_id"] in summary_judgments

    n_resumed_candidates = sum(is_complete(candidate) for candidate in candidate_rows)
    all_pending = [candidate for candidate in candidate_rows if not is_complete(candidate)]
    pending = all_pending[:max_new_candidates] if max_new_candidates is not None else all_pending
    if pending:
        (output_dir / "_JUDGED").unlink(missing_ok=True)
    if pending and call_model is None:
        if not api_key:
            raise ValueError("semantic judging requires an API key")
        call_model = _openai_caller(base_url, api_key)
    assert call_model is not None or not pending

    semaphore = asyncio.Semaphore(concurrency)

    async def run_match(
        candidate: dict[str, Any], task: dict[str, Any], evidence: dict[str, Any]
    ) -> dict[str, Any]:
        async with semaphore:
            assert call_model is not None
            return await judge_match(
                candidate,
                task,
                evidence,
                model=model,
                call_model=call_model,
                max_retries=max_retries,
                keep_raw_response=keep_raw_responses,
            )

    def ordered_match_rows() -> list[dict[str, Any]]:
        return [
            match_judgments[(candidate["candidate_id"], task["match_id"])]
            for candidate in candidate_rows
            for task in candidate["semantic_match_tasks"]
            if (candidate["candidate_id"], task["match_id"]) in match_judgments
        ]

    evidence_by_candidate = {
        candidate["candidate_id"]: {
            record["evidence_id"]: record for record in candidate["matching_tool_responses"]
        }
        for candidate in pending
    }
    match_tasks = [
        asyncio.create_task(
            run_match(candidate, task, evidence_by_candidate[candidate["candidate_id"]][task["evidence_id"]])
        )
        for candidate in pending
        for task in candidate["semantic_match_tasks"]
        if (candidate["candidate_id"], task["match_id"]) not in match_judgments
    ]
    try:
        for task in asyncio.as_completed(match_tasks):
            judgment = await task
            match_judgments[(judgment["candidate_id"], judgment["match_id"])] = judgment
            _write_jsonl(match_path, ordered_match_rows())
    except Exception:
        for task in match_tasks:
            task.cancel()
        await asyncio.gather(*match_tasks, return_exceptions=True)
        raise
    _write_jsonl(match_path, ordered_match_rows())

    needs_summary = []
    for candidate in pending:
        values = candidate_matches(candidate)
        assert values is not None
        preliminary = aggregate_candidate_verdict(
            candidate,
            match_judgments=values,
            summary_judgment=None,
            require_summary=False,
        )
        if preliminary["early_retrieval"] == "yes" and candidate["candidate_id"] not in summary_judgments:
            needs_summary.append(candidate)

    async def run_summary(candidate: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            assert call_model is not None
            return await judge_summary(
                candidate,
                model=model,
                call_model=call_model,
                max_retries=max_retries,
                keep_raw_response=keep_raw_responses,
            )

    def ordered_summary_rows() -> list[dict[str, Any]]:
        return [
            summary_judgments[candidate["candidate_id"]]
            for candidate in candidate_rows
            if candidate["candidate_id"] in summary_judgments
        ]

    summary_tasks = [asyncio.create_task(run_summary(candidate)) for candidate in needs_summary]
    try:
        for task in asyncio.as_completed(summary_tasks):
            judgment = await task
            summary_judgments[judgment["candidate_id"]] = judgment
            _write_jsonl(summary_path, ordered_summary_rows())
    except Exception:
        for task in summary_tasks:
            task.cancel()
        await asyncio.gather(*summary_tasks, return_exceptions=True)
        raise
    _write_jsonl(summary_path, ordered_summary_rows())

    judgments = {}
    for candidate in candidate_rows:
        values = candidate_matches(candidate)
        if values is None:
            continue
        preliminary = aggregate_candidate_verdict(
            candidate,
            match_judgments=values,
            summary_judgment=None,
            require_summary=False,
        )
        summary_judgment = summary_judgments.get(candidate["candidate_id"])
        if preliminary["early_retrieval"] == "yes" and summary_judgment is None:
            continue
        verdict = aggregate_candidate_verdict(
            candidate,
            match_judgments=values,
            summary_judgment=summary_judgment,
        )
        judgments[candidate["candidate_id"]] = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": candidate["candidate_id"],
            "point": candidate["point"],
            "rollout_id": candidate["rollout_id"],
            "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
            "judge_model": model,
            "derived_at": _utc_now(),
            "verdict": verdict,
        }

    complete = len(judgments) == len(candidate_rows)
    if not complete and max_new_candidates is None:
        raise RuntimeError(f"judged {len(judgments)} of {len(candidate_rows)} candidates")
    ordered = [judgments[row["candidate_id"]] for row in candidate_rows if row["candidate_id"] in judgments]
    _write_jsonl(judgment_path, ordered)
    judge_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "point": stage_manifest["point"],
        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
        "judge_model": model,
        "judge_base_url": base_url,
        "concurrency": concurrency,
        "max_retries": max_retries,
        "max_new_candidates": max_new_candidates,
        "keep_raw_responses": keep_raw_responses,
        "n_candidates": len(candidate_rows),
        "n_judgments": len(judgments),
        "n_resumed": n_resumed_candidates,
        "n_remaining": len(candidate_rows) - len(judgments),
        "n_match_tasks": sum(len(candidate["semantic_match_tasks"]) for candidate in candidate_rows),
        "n_match_judgments": len(match_judgments),
        "n_match_resumed": len(resumed_match_rows),
        "n_summary_judgments": len(summary_judgments),
        "n_summary_resumed": len(resumed_summary_rows),
        "complete": complete,
    }
    _write_json(output_dir / "judge_manifest.json", judge_manifest)
    if complete:
        _write_json(output_dir / "_JUDGED", {"status": "ok", "n_judgments": len(judgments)})
    else:
        (output_dir / "_JUDGED").unlink(missing_ok=True)
    return judge_manifest


def summarize_verdicts(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval = Counter(row["verdict"]["early_retrieval"] for row in judgments)
    confirmed_rows = [row for row in judgments if row["verdict"]["early_retrieval"] == "yes"]
    retention = Counter(row["verdict"]["summary_retention"] for row in confirmed_rows)
    resolved = retention["carried"] + retention["dropped"] + retention["distorted"]
    dropped = retention["dropped"]
    distorted = retention["distorted"]
    loss = retention["dropped"] + retention["distorted"]

    def rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 6) if denominator else None

    return {
        "n_candidates": len(judgments),
        "early_retrieval_verdict_counts": {key: retrieval[key] for key in sorted(EARLY_RETRIEVAL_VERDICTS)},
        "n_confirmed_early_retrieval": len(confirmed_rows),
        "confirmed_early_retrieval_rate": rate(len(confirmed_rows), len(judgments)),
        "prefilter_false_positive_rate": rate(retrieval["no"], len(judgments)),
        "early_retrieval_unclear_rate": rate(retrieval["unclear"], len(judgments)),
        "retention_verdict_counts": {key: retention[key] for key in ("carried", "dropped", "distorted", "unclear")},
        "n_resolved_confirmed_retrieval": resolved,
        "retention_coverage": rate(resolved, len(confirmed_rows)),
        "drop_count": dropped,
        "drop_rate": rate(dropped, resolved),
        "drop_candidate_ids": [
            row["candidate_id"] for row in confirmed_rows if row["verdict"]["summary_retention"] == "dropped"
        ],
        "distorted_count": distorted,
        "distorted_rate": rate(distorted, resolved),
        "distorted_candidate_ids": [
            row["candidate_id"] for row in confirmed_rows if row["verdict"]["summary_retention"] == "distorted"
        ],
        "summary_loss_count": loss,
        "summary_loss_rate": rate(loss, resolved),
        "carried_rate": rate(retention["carried"], resolved),
        "loss_candidate_ids": [
            row["candidate_id"]
            for row in confirmed_rows
            if row["verdict"]["summary_retention"] in {"dropped", "distorted"}
        ],
    }


def summarize_failure_headroom(stage_counts: dict[str, int], retention: dict[str, Any]) -> dict[str, Any]:
    def rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 6) if denominator else None

    n_failures = int(stage_counts["n_model_failures"])
    n_rollouts = int(stage_counts["n_rollouts"])
    gold_annotated = int(stage_counts.get("n_failures_with_gold_docs", 0))
    gold_retrieved = int(stage_counts.get("n_failures_retrieved_gold_doc", 0))
    gold_opened = int(stage_counts.get("n_failures_opened_gold_doc", 0))
    evidence_annotated = int(stage_counts.get("n_failures_with_evidence_docs", 0))
    evidence_retrieved = int(stage_counts.get("n_failures_retrieved_evidence_doc", 0))
    evidence_opened = int(stage_counts.get("n_failures_opened_evidence_doc", 0))
    drop_count = int(retention["drop_count"])
    return {
        "n_failures_with_gold_docs": gold_annotated,
        "n_failures_retrieved_gold_doc": gold_retrieved,
        "failure_gold_doc_retrieved_rate": rate(gold_retrieved, gold_annotated),
        "n_failures_without_retrieved_gold_doc": gold_annotated - gold_retrieved,
        "failure_no_gold_doc_rate": rate(gold_annotated - gold_retrieved, gold_annotated),
        "n_failures_opened_gold_doc": gold_opened,
        "failure_gold_doc_opened_rate": rate(gold_opened, gold_annotated),
        "n_failures_with_evidence_docs": evidence_annotated,
        "n_failures_retrieved_evidence_doc": evidence_retrieved,
        "failure_evidence_doc_retrieved_rate": rate(evidence_retrieved, evidence_annotated),
        "n_failures_without_retrieved_evidence_doc": evidence_annotated - evidence_retrieved,
        "failure_no_evidence_doc_rate": rate(evidence_annotated - evidence_retrieved, evidence_annotated),
        "n_failures_opened_evidence_doc": evidence_opened,
        "failure_evidence_doc_opened_rate": rate(evidence_opened, evidence_annotated),
        "drop_share_of_model_failures": rate(drop_count, n_failures),
        "optimistic_drop_uplift_all_rollouts": rate(drop_count, n_rollouts),
    }


def summarize_failure_causes(
    failure_rows: list[dict[str, Any]], judgments: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    ids: dict[str, list[str]] = {cause: [] for cause in FAILURE_CAUSES}
    for row in failure_rows:
        candidate_id = str(row["candidate_id"])
        judgment = judgments.get(candidate_id)
        verdict = judgment.get("verdict", {}) if judgment is not None else {}
        early_retrieval = verdict.get("early_retrieval")
        retention = verdict.get("summary_retention")
        if early_retrieval == "yes" and retention == "dropped":
            cause = "summary_dropped"
        elif early_retrieval == "yes" and retention == "distorted":
            cause = "summary_distorted"
        elif early_retrieval == "yes" and retention == "carried":
            cause = "summary_carried_final_wrong"
        elif early_retrieval == "yes" or early_retrieval == "unclear" or not row.get("has_evidence_docs"):
            cause = "unresolved"
        elif not row.get("retrieved_evidence_doc"):
            cause = "no_evidence_doc_retrieved"
        else:
            cause = "evidence_doc_retrieved_answer_not_confirmed"
        counts[cause] += 1
        ids[cause].append(candidate_id)

    total = len(failure_rows)
    rates = {cause: round(counts[cause] / total, 6) if total else None for cause in FAILURE_CAUSES}
    result = {
        "n_exclusive_failure_causes": total,
        "failure_cause_counts": {cause: counts[cause] for cause in FAILURE_CAUSES},
        "failure_cause_rates": rates,
        "failure_cause_candidate_ids": ids,
    }
    for cause in FAILURE_CAUSES:
        result[f"failure_cause_{cause}_count"] = counts[cause]
        result[f"failure_cause_{cause}_rate"] = rates[cause]
    return result


def _fmt_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def build_report(analysis_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    if not analysis_dirs:
        raise ValueError("at least one analysis directory is required")
    points = []
    seen_points = set()
    comparison_contracts = set()
    for directory in analysis_dirs:
        for required in (
            "_STAGED",
            "_JUDGED",
            "stage_manifest.json",
            "judge_manifest.json",
            "candidates.jsonl",
            "failure_retrieval.jsonl",
            "match_judgments.jsonl",
            "summary_judgments.jsonl",
            "judgments.jsonl",
        ):
            if not (directory / required).is_file():
                raise ValueError(f"analysis directory {directory} is missing {required}")
        stage_manifest = json.loads((directory / "stage_manifest.json").read_text())
        judge_manifest = json.loads((directory / "judge_manifest.json").read_text())
        if stage_manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported stage schema in {directory}")
        if judge_manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported judge schema in {directory}")
        point = str(stage_manifest["point"])
        if point in seen_points:
            raise ValueError(f"duplicate analysis point: {point}")
        seen_points.add(point)
        if judge_manifest.get("point") != point:
            raise ValueError(f"stage/judge point mismatch in {directory}")
        if judge_manifest.get("judge_protocol_version") != JUDGE_PROTOCOL_VERSION:
            raise ValueError(f"judge protocol mismatch in {directory}")
        comparison_contracts.add(
            (
                stage_manifest.get("prefilter_version"),
                judge_manifest.get("judge_protocol_version"),
                judge_manifest.get("judge_model"),
            )
        )

        candidates_path = directory / "candidates.jsonl"
        failure_retrieval_path = directory / "failure_retrieval.jsonl"
        match_judgments_path = directory / "match_judgments.jsonl"
        summary_judgments_path = directory / "summary_judgments.jsonl"
        judgments_path = directory / "judgments.jsonl"
        candidate_rows = _load_jsonl(candidates_path)
        candidates = {row["candidate_id"]: row for row in candidate_rows}
        failure_rows = _load_jsonl(failure_retrieval_path)
        failures = {row["candidate_id"]: row for row in failure_rows}
        match_rows = _load_jsonl(match_judgments_path)
        match_judgments = _validate_resumed_match_judgments(
            match_rows,
            candidates,
            judge_manifest["judge_model"],
            keep_raw_responses=bool(judge_manifest["keep_raw_responses"]),
        )
        summary_rows = _load_jsonl(summary_judgments_path)
        summary_judgments = _validate_resumed_summary_judgments(
            summary_rows,
            candidates,
            judge_manifest["judge_model"],
            keep_raw_responses=bool(judge_manifest["keep_raw_responses"]),
        )
        judgment_rows = _load_jsonl(judgments_path)
        judgments = {row["candidate_id"]: row for row in judgment_rows}
        if (
            len(candidates) != len(candidate_rows)
            or len(judgments) != len(judgment_rows)
            or candidates.keys() != judgments.keys()
        ):
            raise ValueError(f"candidate/judgment coverage mismatch in {directory}")
        if len(failures) != len(failure_rows) or len(failure_rows) != stage_manifest["counts"]["n_model_failures"]:
            raise ValueError(f"failure retrieval coverage mismatch in {directory}")
        expected_count = len(candidate_rows)
        manifest_counts = (
            stage_manifest.get("counts", {}).get("n_prefilter_candidates"),
            judge_manifest.get("n_candidates"),
            judge_manifest.get("n_judgments"),
        )
        if manifest_counts != (expected_count, expected_count, expected_count):
            raise ValueError(f"manifest candidate counts do not match artifacts in {directory}: {manifest_counts}")
        for candidate_id in candidates:
            judgment = judgments[candidate_id]
            expected_judge_contract = {
                "schema_version": SCHEMA_VERSION,
                "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
                "judge_model": judge_manifest["judge_model"],
            }
            mismatches = {
                key: (value, judgment.get(key))
                for key, value in expected_judge_contract.items()
                if judgment.get(key) != value
            }
            if mismatches:
                raise ValueError(f"judgment contract mismatch for {candidate_id}: {mismatches}")
            candidate = candidates[candidate_id]
            candidate_matches = {
                task["match_id"]: match_judgments[(candidate_id, task["match_id"])]
                for task in candidate["semantic_match_tasks"]
            }
            expected_verdict = aggregate_candidate_verdict(
                candidate,
                match_judgments=candidate_matches,
                summary_judgment=summary_judgments.get(candidate_id),
            )
            if judgment.get("verdict") != expected_verdict:
                raise ValueError(f"derived verdict does not match pair-level artifacts for {candidate_id}")
        expected_match_count = sum(len(candidate["semantic_match_tasks"]) for candidate in candidate_rows)
        confirmed_count = sum(
            judgment["verdict"]["early_retrieval"] == "yes" for judgment in judgment_rows
        )
        pair_manifest_counts = (
            stage_manifest.get("counts", {}).get("n_semantic_match_tasks"),
            judge_manifest.get("n_match_tasks"),
            judge_manifest.get("n_match_judgments"),
            len(match_rows),
            judge_manifest.get("n_summary_judgments"),
            len(summary_rows),
        )
        if pair_manifest_counts != (
            expected_match_count,
            expected_match_count,
            expected_match_count,
            expected_match_count,
            confirmed_count,
            confirmed_count,
        ):
            raise ValueError(f"pair/summary counts do not match artifacts in {directory}: {pair_manifest_counts}")

        retention_metrics = summarize_verdicts(judgment_rows)
        retention_metrics.update(summarize_failure_headroom(stage_manifest["counts"], retention_metrics))
        cause_metrics = summarize_failure_causes(failure_rows, judgments)
        if cause_metrics["failure_cause_counts"]["summary_dropped"] != retention_metrics["drop_count"]:
            raise ValueError(f"exclusive summary-drop count does not match judged drops in {directory}")
        retention_metrics.update(cause_metrics)
        points.append(
            {
                "point": point,
                "stage_counts": stage_manifest["counts"],
                "judge_model": judge_manifest["judge_model"],
                "metrics": retention_metrics,
            }
        )

    if len(comparison_contracts) > 1:
        raise ValueError(f"analysis points use incompatible prefilter/judge contracts: {comparison_contracts}")

    def point_key(row: dict[str, Any]) -> tuple[int, int | str]:
        if row["point"] == "base":
            return (0, 0)
        match = re.fullmatch(r"iter0*(\d+)", row["point"])
        return (1, int(match.group(1))) if match else (2, row["point"])

    points.sort(key=point_key)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
        "primary_metric_definition": (
            "drop_rate=dropped/(carried+dropped+distorted) among semantically confirmed early retrievals; "
            "confirmation requires every gold part somewhere across non-final sub-trajectories; unclear retention "
            "is excluded and exposed by retention_coverage"
        ),
        "secondary_metric_definition": (
            "summary_loss_rate=(dropped+distorted)/(carried+dropped+distorted); distorted is retained as a "
            "separate diagnostic because it is less reliable than strict dropped"
        ),
        "points": points,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary_retention_metrics.json", report)

    csv_fields = [
        "point",
        "n_rollouts",
        "n_model_failures",
        "n_compressed_model_failures",
        "n_candidates",
        "n_confirmed_early_retrieval",
        "confirmed_early_retrieval_rate",
        "prefilter_false_positive_rate",
        "retention_coverage",
        "failure_no_gold_doc_rate",
        "failure_no_evidence_doc_rate",
        "failure_gold_doc_opened_rate",
        "failure_evidence_doc_opened_rate",
        "drop_count",
        "drop_rate",
        "drop_share_of_model_failures",
        "optimistic_drop_uplift_all_rollouts",
        "failure_cause_no_evidence_doc_retrieved_count",
        "failure_cause_no_evidence_doc_retrieved_rate",
        "failure_cause_evidence_doc_retrieved_answer_not_confirmed_count",
        "failure_cause_evidence_doc_retrieved_answer_not_confirmed_rate",
        "failure_cause_summary_dropped_count",
        "failure_cause_summary_dropped_rate",
        "failure_cause_summary_distorted_count",
        "failure_cause_summary_distorted_rate",
        "failure_cause_summary_carried_final_wrong_count",
        "failure_cause_summary_carried_final_wrong_rate",
        "failure_cause_unresolved_count",
        "failure_cause_unresolved_rate",
        "distorted_count",
        "distorted_rate",
        "summary_loss_count",
        "summary_loss_rate",
        "carried_rate",
    ]
    with (output_dir / "summary_retention_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for point in points:
            values = {"point": point["point"], **point["stage_counts"], **point["metrics"]}
            writer.writerow({key: values.get(key) for key in csv_fields})

    lines = [
        "# Summary-retention evaluation",
        "",
        "Primary metric: `drop_rate = dropped / (carried + dropped + distorted)` among confirmed early retrievals.",
        "Early retrieval requires every gold part to be semantically confirmed somewhere across non-final sub-trajectories.",
        "`summary_loss_rate = (dropped + distorted) / resolved` is retained only as a broader diagnostic.",
        "Unclear retention judgments are excluded from both denominators and reported through coverage.",
        "",
        "| point | model failures | compressed failures | candidates | confirmed retrieval | coverage | carried | dropped | distorted | expanded failure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        stage, metrics = point["stage_counts"], point["metrics"]
        lines.append(
            f"| {point['point']} | {stage['n_model_failures']} | {stage['n_compressed_model_failures']} | "
            f"{metrics['n_candidates']} | {metrics['n_confirmed_early_retrieval']} "
            f"({_fmt_rate(metrics['confirmed_early_retrieval_rate'])}) | "
            f"{_fmt_rate(metrics['retention_coverage'])} | {_fmt_rate(metrics['carried_rate'])} | "
            f"{metrics['drop_count']} ({_fmt_rate(metrics['drop_rate'])}) | "
            f"{metrics['distorted_count']} ({_fmt_rate(metrics['distorted_rate'])}) | "
            f"{metrics['summary_loss_count']} ({_fmt_rate(metrics['summary_loss_rate'])}) |"
        )
    lines += [
        "",
        "The lexical pre-filter is not a metric. `no` and `unclear` early-retrieval verdicts do not enter the retention denominator.",
        "Judge failures from the original answer scorer are excluded before candidate staging.",
        "",
        "## Mutually exclusive failure decomposition",
        "",
        "Each model failure appears in exactly one column. A confirmed pre-handover semantic answer is assigned to its summary outcome first; remaining failures are divided by broader evidence-document retrieval.",
        "These are hierarchical failure modes, not proof that fixing the assigned stage alone would make the answer correct.",
        "`No confirmed early answer` is a residual bucket, not proof that the answer never appeared: it can include truncated pages, answers seen only in a final or uncompressed trajectory, and surface-form misses.",
        "",
        "| point | failures | no evidence doc retrieved | evidence doc retrieved, no confirmed early answer | summary dropped | summary distorted | summary carried, final wrong | unresolved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        stage, metrics = point["stage_counts"], point["metrics"]
        cause_counts = metrics["failure_cause_counts"]
        cause_rates = metrics["failure_cause_rates"]
        lines.append(
            f"| {point['point']} | {stage['n_model_failures']} | "
            f"{cause_counts['no_evidence_doc_retrieved']} "
            f"({_fmt_rate(cause_rates['no_evidence_doc_retrieved'])}) | "
            f"{cause_counts['evidence_doc_retrieved_answer_not_confirmed']} "
            f"({_fmt_rate(cause_rates['evidence_doc_retrieved_answer_not_confirmed'])}) | "
            f"{cause_counts['summary_dropped']} ({_fmt_rate(cause_rates['summary_dropped'])}) | "
            f"{cause_counts['summary_distorted']} ({_fmt_rate(cause_rates['summary_distorted'])}) | "
            f"{cause_counts['summary_carried_final_wrong']} "
            f"({_fmt_rate(cause_rates['summary_carried_final_wrong'])}) | "
            f"{cause_counts['unresolved']} ({_fmt_rate(cause_rates['unresolved'])}) |"
        )
    lines += [
        "",
        "The optimistic rollout uplift from observed summary drops is: "
        + ", ".join(
            f"{point['point']}={_fmt_rate(point['metrics']['optimistic_drop_uplift_all_rollouts'])}"
            for point in points
        )
        + ".",
        "",
        "## Overlapping retrieval diagnostics",
        "",
        "`No gold doc` uses the strict dataset `gold_docs`; `no evidence doc` uses the broader supporting `evidence_docs` set.",
        "These columns overlap and must not be added together. They diagnose retrieval coverage rather than assign one cause per failure.",
        "",
        "| point | model failures | no gold doc returned | no evidence doc returned | evidence doc opened |",
        "|---|---:|---:|---:|---:|",
    ]
    for point in points:
        stage, metrics = point["stage_counts"], point["metrics"]
        lines.append(
            f"| {point['point']} | {stage['n_model_failures']} | "
            f"{metrics['n_failures_without_retrieved_gold_doc']} "
            f"({_fmt_rate(metrics['failure_no_gold_doc_rate'])}) | "
            f"{metrics['n_failures_without_retrieved_evidence_doc']} "
            f"({_fmt_rate(metrics['failure_no_evidence_doc_rate'])}) | "
            f"{metrics['n_failures_opened_evidence_doc']} "
            f"({_fmt_rate(metrics['failure_evidence_doc_opened_rate'])}) |"
        )
    lines.append("")
    (output_dir / "summary_retention_report.md").write_text("\n".join(lines))
    _write_json(output_dir / "_SUMMARY_RETENTION_SUCCESS", {"status": "ok", "n_points": len(points)})
    return report


def _load_comparison_side(analysis_dirs: list[Path]) -> tuple[str, dict[str, dict[str, Any]]]:
    points = {}
    models = set()
    for directory in analysis_dirs:
        for required in (
            "_STAGED",
            "_JUDGED",
            "stage_manifest.json",
            "judge_manifest.json",
            "candidates.jsonl",
            "failure_retrieval.jsonl",
            "match_judgments.jsonl",
            "summary_judgments.jsonl",
            "judgments.jsonl",
        ):
            if not (directory / required).is_file():
                raise ValueError(f"comparison input {directory} is missing {required}")
        stage_manifest = json.loads((directory / "stage_manifest.json").read_text())
        judge_manifest = json.loads((directory / "judge_manifest.json").read_text())
        point = str(stage_manifest["point"])
        if point in points:
            raise ValueError(f"duplicate comparison point {point}")
        if stage_manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported comparison stage schema in {directory}")
        if judge_manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported comparison judge schema in {directory}")
        if stage_manifest.get("prefilter_version") != PREFILTER_VERSION:
            raise ValueError(f"comparison prefilter mismatch in {directory}")
        if judge_manifest.get("judge_protocol_version") != JUDGE_PROTOCOL_VERSION:
            raise ValueError(f"comparison judge protocol mismatch in {directory}")
        if judge_manifest.get("point") != point:
            raise ValueError(f"comparison stage/judge point mismatch in {directory}")
        models.add(str(judge_manifest["judge_model"]))
        candidate_rows = _load_jsonl(directory / "candidates.jsonl")
        failure_rows = _load_jsonl(directory / "failure_retrieval.jsonl")
        judgment_rows = _load_jsonl(directory / "judgments.jsonl")
        match_rows = _load_jsonl(directory / "match_judgments.jsonl")
        summary_rows = _load_jsonl(directory / "summary_judgments.jsonl")
        candidates = {row["candidate_id"]: row for row in candidate_rows}
        judgments = {row["candidate_id"]: row for row in judgment_rows}
        if len(candidates) != len(candidate_rows) or len(judgments) != len(judgment_rows):
            raise ValueError(f"duplicate comparison candidate or judgment in {directory}")
        match_judgments = _validate_resumed_match_judgments(
            match_rows,
            candidates,
            judge_manifest["judge_model"],
            keep_raw_responses=bool(judge_manifest["keep_raw_responses"]),
        )
        summary_judgments = _validate_resumed_summary_judgments(
            summary_rows,
            candidates,
            judge_manifest["judge_model"],
            keep_raw_responses=bool(judge_manifest["keep_raw_responses"]),
        )
        candidate_ids = list(candidates)
        judgment_ids = [row["candidate_id"] for row in judgment_rows]
        if candidate_ids != judgment_ids:
            raise ValueError(f"comparison candidate/judgment order mismatch in {directory}")
        match_verdicts = {
            key: row["verdict"]["verdict"] for key, row in match_judgments.items()
        }
        expected_match_ids = {
            (candidate["candidate_id"], task["match_id"])
            for candidate in candidate_rows
            for task in candidate["semantic_match_tasks"]
        }
        if set(match_verdicts) != expected_match_ids:
            raise ValueError(f"comparison match coverage mismatch in {directory}")
        for candidate_id, candidate in candidates.items():
            candidate_matches = {
                task["match_id"]: match_judgments[(candidate_id, task["match_id"])]
                for task in candidate["semantic_match_tasks"]
            }
            expected_verdict = aggregate_candidate_verdict(
                candidate,
                match_judgments=candidate_matches,
                summary_judgment=summary_judgments.get(candidate_id),
            )
            if judgments[candidate_id].get("verdict") != expected_verdict:
                raise ValueError(
                    f"comparison derived verdict does not match pair-level artifacts for {candidate_id}"
                )
        metrics = summarize_verdicts(judgment_rows)
        metrics.update(
            summarize_failure_causes(
                failure_rows,
                {row["candidate_id"]: row for row in judgment_rows},
            )
        )
        points[point] = {
            "stage_counts": stage_manifest["counts"],
            "candidates": candidate_rows,
            "judgments": judgments,
            "match_verdicts": match_verdicts,
            "metrics": metrics,
        }
    if len(models) != 1:
        raise ValueError(f"one comparison side must use exactly one judge model: {models}")
    return models.pop(), points


def build_model_comparison(
    model_a_dirs: list[Path],
    model_b_dirs: list[Path],
    output_dir: Path,
    *,
    model_a_name: str,
    model_b_name: str,
) -> dict[str, Any]:
    model_a, side_a = _load_comparison_side(model_a_dirs)
    model_b, side_b = _load_comparison_side(model_b_dirs)
    if side_a.keys() != side_b.keys():
        raise ValueError("comparison sides contain different points")

    def point_key(point: str) -> tuple[int, int | str]:
        if point == "base":
            return (0, 0)
        match = re.fullmatch(r"iter0*(\d+)", point)
        return (1, int(match.group(1))) if match else (2, point)

    def agreement_rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 6) if denominator else None

    rows = []
    totals = Counter()
    summary_cross_totals: Counter[tuple[str, str]] = Counter()
    for point in sorted(side_a, key=point_key):
        a, b = side_a[point], side_b[point]
        if a["candidates"] != b["candidates"] or a["stage_counts"] != b["stage_counts"]:
            raise ValueError(f"comparison sides use different staged data for {point}")
        if a["match_verdicts"].keys() != b["match_verdicts"].keys():
            raise ValueError(f"comparison match IDs differ for {point}")
        pair_total = len(a["match_verdicts"])
        pair_agree = sum(
            verdict == b["match_verdicts"][match_id]
            for match_id, verdict in a["match_verdicts"].items()
        )
        candidate_ids = set(a["judgments"])
        if candidate_ids != set(b["judgments"]):
            raise ValueError(f"comparison candidate IDs differ for {point}")
        early_agree = sum(
            a["judgments"][candidate_id]["verdict"]["early_retrieval"]
            == b["judgments"][candidate_id]["verdict"]["early_retrieval"]
            for candidate_id in candidate_ids
        )
        common_confirmed = {
            candidate_id
            for candidate_id in candidate_ids
            if a["judgments"][candidate_id]["verdict"]["early_retrieval"] == "yes"
            and b["judgments"][candidate_id]["verdict"]["early_retrieval"] == "yes"
        }
        summary_agree = sum(
            a["judgments"][candidate_id]["verdict"]["summary_retention"]
            == b["judgments"][candidate_id]["verdict"]["summary_retention"]
            for candidate_id in common_confirmed
        )
        summary_cross = Counter(
            (
                a["judgments"][candidate_id]["verdict"]["summary_retention"],
                b["judgments"][candidate_id]["verdict"]["summary_retention"],
            )
            for candidate_id in common_confirmed
        )
        summary_cross_totals.update(summary_cross)
        agreement = {
            "n_match_pairs": pair_total,
            "n_match_pair_agreements": pair_agree,
            "match_pair_agreement_rate": agreement_rate(pair_agree, pair_total),
            "n_candidates": len(candidate_ids),
            "n_early_retrieval_agreements": early_agree,
            "early_retrieval_agreement_rate": agreement_rate(early_agree, len(candidate_ids)),
            "n_common_confirmed_retrieval": len(common_confirmed),
            "n_summary_agreements": summary_agree,
            "summary_agreement_rate": agreement_rate(summary_agree, len(common_confirmed)),
            "summary_label_cross_tab": {
                label_a: {label_b: summary_cross[(label_a, label_b)] for label_b in sorted(RETENTION_VERDICTS)}
                for label_a in sorted(RETENTION_VERDICTS)
            },
        }
        for key in (
            "n_match_pairs",
            "n_match_pair_agreements",
            "n_candidates",
            "n_early_retrieval_agreements",
            "n_common_confirmed_retrieval",
            "n_summary_agreements",
        ):
            totals[key] += agreement[key]
        rows.append(
            {
                "point": point,
                "stage_counts": a["stage_counts"],
                "model_a_metrics": a["metrics"],
                "model_b_metrics": b["metrics"],
                "agreement": agreement,
            }
        )
    total_agreement = {
        **totals,
        "match_pair_agreement_rate": agreement_rate(
            totals["n_match_pair_agreements"], totals["n_match_pairs"]
        ),
        "early_retrieval_agreement_rate": agreement_rate(
            totals["n_early_retrieval_agreements"], totals["n_candidates"]
        ),
        "summary_agreement_rate": agreement_rate(
            totals["n_summary_agreements"], totals["n_common_confirmed_retrieval"]
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
        "model_a": {"name": model_a_name, "model": model_a},
        "model_b": {"name": model_b_name, "model": model_b},
        "overall_agreement": total_agreement,
        "summary_label_cross_tab": {
            label_a: {
                label_b: summary_cross_totals[(label_a, label_b)] for label_b in sorted(RETENTION_VERDICTS)
            }
            for label_a in sorted(RETENTION_VERDICTS)
        },
        "points": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary_retention_model_comparison.json", report)

    lines = [
        "# Summary-retention judge comparison",
        "",
        f"Protocol: `{JUDGE_PROTOCOL_VERSION}`. The staged candidates and match IDs are identical for both judges.",
        "The primary metric is strict `drop_rate = dropped / (carried + dropped + distorted)` among confirmed retrievals.",
        "",
        "## Primary metrics",
        "",
        f"| point | confirmed {model_a_name} | confirmed {model_b_name} | drops {model_a_name} | drops {model_b_name} | drop rate {model_a_name} | drop rate {model_b_name} | distorted {model_a_name} | distorted {model_b_name} |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        a, b = row["model_a_metrics"], row["model_b_metrics"]
        lines.append(
            f"| {row['point']} | {a['n_confirmed_early_retrieval']} | {b['n_confirmed_early_retrieval']} | "
            f"{a['drop_count']} | {b['drop_count']} | {_fmt_rate(a['drop_rate'])} | {_fmt_rate(b['drop_rate'])} | "
            f"{a['distorted_count']} | {b['distorted_count']} |"
        )
    lines += [
        "",
        "## Judge agreement",
        "",
        "Pair agreement compares every independent `(gold_part, tool_response)` verdict. Summary agreement is restricted to candidates both judges confirmed as early retrieval.",
        "",
        "| point | pair agreement | early-retrieval agreement | common confirmed | summary agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        agreement = row["agreement"]
        lines.append(
            f"| {row['point']} | {agreement['n_match_pair_agreements']}/{agreement['n_match_pairs']} "
            f"({_fmt_rate(agreement['match_pair_agreement_rate'])}) | "
            f"{agreement['n_early_retrieval_agreements']}/{agreement['n_candidates']} "
            f"({_fmt_rate(agreement['early_retrieval_agreement_rate'])}) | "
            f"{agreement['n_common_confirmed_retrieval']} | "
            f"{agreement['n_summary_agreements']}/{agreement['n_common_confirmed_retrieval']} "
            f"({_fmt_rate(agreement['summary_agreement_rate'])}) |"
        )
    lines.append(
        f"| **all** | **{totals['n_match_pair_agreements']}/{totals['n_match_pairs']} "
        f"({_fmt_rate(total_agreement['match_pair_agreement_rate'])})** | "
        f"**{totals['n_early_retrieval_agreements']}/{totals['n_candidates']} "
        f"({_fmt_rate(total_agreement['early_retrieval_agreement_rate'])})** | "
        f"**{totals['n_common_confirmed_retrieval']}** | "
        f"**{totals['n_summary_agreements']}/{totals['n_common_confirmed_retrieval']} "
        f"({_fmt_rate(total_agreement['summary_agreement_rate'])})** |"
    )
    labels = ("carried", "dropped", "distorted", "unclear")
    lines += [
        "",
        "### Summary-label cross-tab",
        "",
        f"Rows are {model_a_name}; columns are {model_b_name}. Only the {totals['n_common_confirmed_retrieval']} common confirmed candidates are included.",
        "",
        "| | " + " | ".join(labels) + " |",
        "|---|---:|---:|---:|---:|",
    ]
    for label_a in labels:
        lines.append(
            f"| {label_a} | "
            + " | ".join(str(summary_cross_totals[(label_a, label_b)]) for label_b in labels)
            + " |"
        )
    for side_key, name in (("model_a_metrics", model_a_name), ("model_b_metrics", model_b_name)):
        lines += [
            "",
            f"## Mutually exclusive failure decomposition: {name}",
            "",
            "| point | failures | no evidence doc | evidence doc, no confirmed answer | summary dropped | summary distorted | summary carried, final wrong | unresolved |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            counts = row[side_key]["failure_cause_counts"]
            rates = row[side_key]["failure_cause_rates"]
            values = [
                f"{counts[cause]} ({_fmt_rate(rates[cause])})"
                for cause in FAILURE_CAUSES
            ]
            lines.append(
                f"| {row['point']} | {row['stage_counts']['n_model_failures']} | " + " | ".join(values) + " |"
            )
    lines += [
        "",
        "The failure columns are mutually exclusive within each model and sum to all strict model failures. Model-specific confirmed-retrieval denominators can differ, so drop rates should be compared together with agreement and counts.",
        "",
        "## Interpretation",
        "",
        f"Across all points, the judges agree on {_fmt_rate(total_agreement['match_pair_agreement_rate'])} of pair verdicts, {_fmt_rate(total_agreement['early_retrieval_agreement_rate'])} of candidate retrieval verdicts, and {_fmt_rate(total_agreement['summary_agreement_rate'])} of summary labels on their common confirmed set.",
        f"At {rows[-1]['point']}, strict conditional drop rate is {_fmt_rate(rows[-1]['model_a_metrics']['drop_rate'])} for {model_a_name} and {_fmt_rate(rows[-1]['model_b_metrics']['drop_rate'])} for {model_b_name}. The corresponding observed-drop upper bound over all rollouts is {_fmt_rate(rows[-1]['model_a_metrics']['drop_count'] / rows[-1]['stage_counts']['n_rollouts'])} and {_fmt_rate(rows[-1]['model_b_metrics']['drop_count'] / rows[-1]['stage_counts']['n_rollouts'])}.",
        "The conditional drop rate can rise while absolute drop headroom falls because later checkpoints have fewer model failures and a different confirmed-retrieval subset. Use the all-rollout upper bound for potential pass@1 headroom and the conditional drop rate to diagnose summary retention within retrieved cases.",
        "Because the judges still disagree on some summary boundaries, report both model estimates rather than silently merging them. Their range is a useful sensitivity interval, not a statistical confidence interval.",
        "",
    ]
    (output_dir / "summary_retention_model_comparison.md").write_text("\n".join(lines))
    _write_json(output_dir / "_SUMMARY_RETENTION_COMPARISON_SUCCESS", {"status": "ok", "n_points": len(rows)})
    return report


def _stage_command(args: argparse.Namespace) -> None:
    result = stage_point(Path(args.point_dir), Path(args.output_dir))
    print(json.dumps(result["counts"], indent=2))


def _judge_command(args: argparse.Namespace) -> None:
    api_key = os.environ.get(args.api_key_env)
    result = asyncio.run(
        judge_stage(
            Path(args.stage_dir),
            Path(args.stage_dir),
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
            keep_raw_responses=args.keep_raw_responses,
            max_new_candidates=args.max_new_candidates,
        )
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("point", "n_candidates", "n_judgments", "n_resumed", "n_remaining", "complete")
            },
            indent=2,
        )
    )


def _report_command(args: argparse.Namespace) -> None:
    report = build_report([Path(value) for value in args.analysis_dir], Path(args.output_dir))
    print(json.dumps({"n_points": len(report["points"]), "output_dir": args.output_dir}, indent=2))


def _compare_command(args: argparse.Namespace) -> None:
    report = build_model_comparison(
        [Path(value) for value in args.model_a_analysis_dir],
        [Path(value) for value in args.model_b_analysis_dir],
        Path(args.output_dir),
        model_a_name=args.model_a_name,
        model_b_name=args.model_b_name,
    )
    print(json.dumps({"n_points": len(report["points"]), "output_dir": args.output_dir}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    stage = subparsers.add_parser("stage", help="generate high-recall candidates from one validated point")
    stage.add_argument("--point-dir", required=True)
    stage.add_argument("--output-dir", required=True)
    stage.set_defaults(func=_stage_command)

    judge = subparsers.add_parser("judge", help="semantically judge every staged candidate")
    judge.add_argument("--stage-dir", required=True)
    judge.add_argument(
        "--model",
        default=os.environ.get(
            "BCPLUS_SUMMARY_RETENTION_JUDGE_MODEL", os.environ.get("BCPLUS_JUDGE_MODEL", "gpt-5-4-genai-dss4")
        ),
    )
    judge.add_argument(
        "--base-url",
        default=os.environ.get("BCPLUS_JUDGE_BASE_URL", "https://api.llama.com/compat/v1/"),
    )
    judge.add_argument("--api-key-env", default="LLAMA_API_KEY")
    judge.add_argument("--concurrency", type=int, default=8)
    judge.add_argument("--max-retries", type=int, default=3)
    judge.add_argument("--max-new-candidates", type=int)
    judge.add_argument("--keep-raw-responses", action="store_true")
    judge.set_defaults(func=_judge_command)

    report = subparsers.add_parser("report", help="validate judgments and compute deterministic metrics")
    report.add_argument("--analysis-dir", action="append", required=True)
    report.add_argument("--output-dir", required=True)
    report.set_defaults(func=_report_command)

    compare = subparsers.add_parser("compare", help="compare two completed judge-model analyses")
    compare.add_argument("--model-a-analysis-dir", action="append", required=True)
    compare.add_argument("--model-b-analysis-dir", action="append", required=True)
    compare.add_argument("--model-a-name", required=True)
    compare.add_argument("--model-b-name", required=True)
    compare.add_argument("--output-dir", required=True)
    compare.set_defaults(func=_compare_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
