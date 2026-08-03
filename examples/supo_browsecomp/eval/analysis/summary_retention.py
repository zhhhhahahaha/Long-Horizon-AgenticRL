#!/usr/bin/env python3
"""Auditable semantic evaluation of answer retention across BC+ compression.

The pipeline has three explicit stages:

1. ``stage`` applies a high-recall lexical pre-filter to validated point data.
2. ``judge`` sends one candidate at a time, batching only unusually large evidence sets.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

SCHEMA_VERSION = 4
PREFILTER_VERSION = "gold-parts-in-one-early-observation-v2"
JUDGE_PROTOCOL_VERSION = "summary-retention-judge-v6"
API_REQUEST_TIMEOUT_SECONDS = 180.0
API_MAX_OUTPUT_TOKENS = 4_096
API_LARGE_MAX_OUTPUT_TOKENS = 8_192
MAX_EVIDENCE_PER_SINGLE_REQUEST = 40
EVIDENCE_BATCH_SIZE = 16
OBSERVATION_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL | re.IGNORECASE)
PARAMETER_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE)
SEARCH_RESULT_RE = re.compile(r"(?=---\s*#\d+\s*:)", re.IGNORECASE)
MULTI_ANSWER_RE = re.compile(r"<q\d+>(.*?)</q\d+>", re.DOTALL | re.IGNORECASE)
EVIDENCE_VERDICTS = {"yes", "no", "unclear"}
EARLY_RETRIEVAL_VERDICTS = {"yes", "no", "unclear"}
RETENTION_VERDICTS = {"carried", "dropped", "distorted", "unclear"}
FAILURE_CAUSES = (
    "no_evidence_doc_retrieved",
    "evidence_doc_retrieved_answer_not_confirmed",
    "summary_dropped",
    "summary_distorted",
    "summary_carried_final_wrong",
    "unresolved",
)

JUDGE_INSTRUCTIONS = """You are a semantic identity and summary-retention judge for one failed BrowseComp-Plus rollout.

The search agent periodically discards its context and hands a summary to a fresh
sub-trajectory. The candidate was selected by a lenient lexical pre-filter because
all parts of the gold answer appeared in tool observations before the final
sub-trajectory. The supplied evidence contains each distinct matching tool response,
with repeated occurrences deduplicated. No non-matching tool responses or model
reasoning are supplied. A lexical match may be coincidental, so do not assume that
the matched words have the meaning intended by the gold answer.

Judge two separate claims:

1. For EACH matching tool response, decide whether the matched answer word, name,
   value, or fact refers to the same answer intended by the gold answer in this
   question.
   - yes: it has the intended answer meaning in this context.
   - no: it is an unrelated same-name mention, numeric collision, or other false match.
   - unclear: the supplied response does not provide enough context to decide.
   Treat the supplied gold answer as the reference identity; do not independently
   solve the question or challenge whether the gold answer is correct. Judge semantic
   identity only. In particular, do NOT require this response, page, or all supplied
   matching responses to prove every clue in the question or to establish a complete
   evidence chain. An exact distinctive name or answer string used for the same kind of entity
   should be yes unless the matching responses affirmatively show a different
   referent. You may use the other supplied matching responses only to resolve
   identity ambiguity. Search queries show retrieval intent but are not themselves
   factual evidence. Do not decide whether the agent fully solved the question,
   whether the source was a search result or opened page, or whether this evidence
   caused the final answer.

   Apply these identity rules consistently:
   - If the gold is a person, institution, place, or work title and the response
     refers to that same named entity, return yes even when the response discusses it
     in a relation unrelated to the question. The relation need not be proven. For
     example, if the gold is "Amherst College," a response about an unrelated person
     at Amherst College is yes because it names the same institution.
   - If the same surface string denotes a different semantic object or fact, return
     no. For example, a year used as a publication date is not that year used as an
     award date, and ordinary weather "rain" is not a work or episode titled "Rain".
   - Use unclear rather than yes when a match could have the intended identity but the
     response does not affirmatively establish it. Possibility alone is not yes.

2. Decide whether the FINAL handover summary preserved the gold answer in a form a
   later sub-trajectory could use.
   - carried: preserved accurately, including an unambiguous paraphrase or a clearly
     named usable candidate. The gold answer need not be the summary's preferred or
     final conclusion; a competing wrong candidate does not erase a still-usable gold
     candidate.
   - dropped: the gold answer itself is absent from the final handover summary. Use
     dropped even when the summary proposes or confidently asserts a different wrong
     answer, preserves related clues, mentions only a related or parent entity, or
     continues along a wrong lead. Judge the supplied summary as complete: placeholders
     such as "..." do not imply hidden answer content.
   - distorted: the gold answer itself remains recognizably present, but its content
     is materially changed, contradicted, corrupted, explicitly rejected, or attached
     to the wrong role/entity. For a compound gold answer, use distorted when only part
     remains and another part is missing or wrong. Do not use distorted when the gold
     answer is entirely absent.
   - unclear: insufficient evidence to distinguish the above.

Assess summary retention independently even if every earlier evidence item is no or
unclear. Return exactly one assessment for every supplied evidence_id, and no others.

Return one JSON object and no markdown. It must have exactly these fields:
{
  "evidence_assessments": [
    {
      "evidence_id": "the supplied evidence_id",
      "verdict": "yes|no|unclear",
      "rationale": "brief reason based on this tool response's context"
    }
  ],
  "summary_retention": "carried|dropped|distorted|unclear",
  "summary_rationale": "brief reason based on the final handover summary"
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
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        identity = str(record.get("docid") or record.get("url") or record.get("content", ""))
        key = (str(record.get("tool")), identity)
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
        if len(record["content"]) > len(existing["content"]):
            for field in ("content", "content_truncated", "original_content_chars"):
                existing[field] = record[field]
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
        has_complete_sub_traj_match = False
        for index, sample in enumerate(siblings[:-1]):
            records, covered_parts = _matching_tool_response_records(
                sample.get("response", ""),
                parts,
                sub_traj_index=index,
            )
            evidence_records.extend(records)
            if parts and set(parts).issubset(covered_parts):
                has_complete_sub_traj_match = True
        if not has_complete_sub_traj_match:
            continue
        evidence = _deduplicate_evidence(evidence_records)
        for evidence_index, record in enumerate(evidence, start=1):
            record["evidence_id"] = f"evidence-{evidence_index:03d}"

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
            "final_answer": row.get("finish_answer", ""),
            "n_sub_trajs": len(siblings),
            "matching_tool_responses": evidence,
            "final_handover": {
                "sub_traj_index": handover_index,
                "outcome": handover_bc.get("outcome", ""),
                "summary_source": handover_bc.get("summary_source", ""),
                "summary": handover_bc.get("summary"),
            },
        }
        candidates.append(candidate)
        counts["n_prefilter_candidates"] += 1

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
        for stale_name in ("judge_manifest.json", "judgments.jsonl"):
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
                "all normalized gold parts occur in tool observations of at least one non-final sub-trajectory"
            ),
        },
    }
    _write_json(output_dir / "stage_manifest.json", stage_manifest)
    _write_json(output_dir / "_STAGED", {"status": "ok", "n_candidates": len(candidates)})
    return stage_manifest


def _candidate_prompt(candidate: dict[str, Any]) -> str:
    payload = {
        "candidate_id": candidate["candidate_id"],
        "question": candidate["question"],
        "gold_answer": candidate["gold_answer"],
        "matching_tool_responses": candidate["matching_tool_responses"],
        "final_handover_summary": candidate["final_handover"].get("summary"),
    }
    return "Judge this candidate according to the protocol.\n\nCANDIDATE:\n" + json.dumps(
        payload, indent=2, ensure_ascii=False
    )


def parse_verdict(
    response: str,
    expected_evidence_ids: list[str],
    *,
    evidence_gold_parts: dict[str, list[str]] | None = None,
    required_gold_parts: list[str] | None = None,
) -> dict[str, Any]:
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
    required = {"evidence_assessments", "summary_retention", "summary_rationale"}
    normalized_fields = required | {"early_retrieval"}
    if set(value) not in (required, normalized_fields):
        raise ValueError(
            f"judge response fields must be {sorted(required)}; stored verdicts may also contain early_retrieval"
        )
    assessments = value["evidence_assessments"]
    if not isinstance(assessments, list):
        raise ValueError("evidence_assessments must be a list")
    by_id: dict[str, dict[str, str]] = {}
    for assessment in assessments:
        if not isinstance(assessment, dict) or set(assessment) != {"evidence_id", "verdict", "rationale"}:
            raise ValueError("each evidence assessment must contain exactly evidence_id, verdict, and rationale")
        evidence_id = assessment["evidence_id"]
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("evidence_id must be a non-empty string")
        if evidence_id in by_id:
            raise ValueError(f"duplicate evidence assessment: {evidence_id}")
        if assessment["verdict"] not in EVIDENCE_VERDICTS:
            raise ValueError(f"invalid evidence verdict for {evidence_id}: {assessment['verdict']!r}")
        if not isinstance(assessment["rationale"], str) or not assessment["rationale"].strip():
            raise ValueError(f"evidence rationale must be non-empty for {evidence_id}")
        by_id[evidence_id] = {
            "evidence_id": evidence_id,
            "verdict": assessment["verdict"],
            "rationale": assessment["rationale"].strip(),
        }

    if len(expected_evidence_ids) != len(set(expected_evidence_ids)):
        raise ValueError("candidate contains duplicate evidence_id values")
    if set(by_id) != set(expected_evidence_ids):
        raise ValueError(
            "evidence assessment IDs do not match candidate: "
            f"expected {expected_evidence_ids}, received {list(by_id)}"
        )
    if value["summary_retention"] not in RETENTION_VERDICTS:
        raise ValueError(f"invalid summary_retention verdict: {value['summary_retention']!r}")
    if not isinstance(value["summary_rationale"], str) or not value["summary_rationale"].strip():
        raise ValueError("summary_rationale must be a non-empty string")

    ordered_assessments = [by_id[evidence_id] for evidence_id in expected_evidence_ids]
    if (evidence_gold_parts is None) != (required_gold_parts is None):
        raise ValueError("evidence_gold_parts and required_gold_parts must be supplied together")
    if evidence_gold_parts is not None and required_gold_parts is not None:
        if set(evidence_gold_parts) != set(expected_evidence_ids):
            raise ValueError("evidence gold-part IDs do not match candidate evidence IDs")
        required_parts = set(required_gold_parts)
        represented_parts = {
            part for evidence_id in expected_evidence_ids for part in evidence_gold_parts[evidence_id]
        }
        if not required_parts or not required_parts.issubset(represented_parts):
            raise ValueError("candidate evidence does not represent every required gold part")
        confirmed_parts = {
            part
            for assessment in ordered_assessments
            if assessment["verdict"] == "yes"
            for part in evidence_gold_parts[assessment["evidence_id"]]
        }
        uncertain_parts = {
            part
            for assessment in ordered_assessments
            if assessment["verdict"] == "unclear"
            for part in evidence_gold_parts[assessment["evidence_id"]]
        }
        missing_parts = required_parts - confirmed_parts
        if not missing_parts:
            early_retrieval = "yes"
        elif missing_parts & uncertain_parts:
            early_retrieval = "unclear"
        else:
            early_retrieval = "no"
    else:
        evidence_verdicts = {assessment["verdict"] for assessment in ordered_assessments}
        if "yes" in evidence_verdicts:
            early_retrieval = "yes"
        elif "unclear" in evidence_verdicts:
            early_retrieval = "unclear"
        else:
            early_retrieval = "no"
    supplied_aggregate = value.get("early_retrieval")
    if supplied_aggregate is not None and supplied_aggregate != early_retrieval:
        raise ValueError(
            f"early_retrieval does not match evidence assessments: {supplied_aggregate!r} != {early_retrieval!r}"
        )
    return {
        "evidence_assessments": ordered_assessments,
        "early_retrieval": early_retrieval,
        "summary_retention": value["summary_retention"],
        "summary_rationale": value["summary_rationale"].strip(),
    }


async def judge_candidate(
    candidate: dict[str, Any],
    *,
    model: str,
    call_model: ModelCaller,
    max_retries: int,
    keep_raw_response: bool,
) -> dict[str, Any]:
    evidence = candidate["matching_tool_responses"]
    if len(evidence) <= MAX_EVIDENCE_PER_SINGLE_REQUEST:
        batches = [evidence]
    else:
        batches = [
            evidence[index : index + EVIDENCE_BATCH_SIZE] for index in range(0, len(evidence), EVIDENCE_BATCH_SIZE)
        ]

    batch_verdicts = []
    raw_responses = []
    total_attempts = 0
    for batch_index, batch in enumerate(batches, start=1):
        batch_candidate = {**candidate, "matching_tool_responses": batch}
        messages = [
            {"role": "system", "content": JUDGE_INSTRUCTIONS},
            {"role": "user", "content": _candidate_prompt(batch_candidate)},
        ]
        errors = []
        for attempt in range(1, max_retries + 1):
            total_attempts += 1
            try:
                raw_response = await call_model(messages, model)
                evidence_ids = [record["evidence_id"] for record in batch]
                if len(batches) == 1:
                    batch_verdict = parse_verdict(
                        raw_response,
                        evidence_ids,
                        evidence_gold_parts={record["evidence_id"]: record["matched_gold_parts"] for record in batch},
                        required_gold_parts=candidate["gold_parts"],
                    )
                else:
                    batch_verdict = parse_verdict(raw_response, evidence_ids)
                batch_verdicts.append(batch_verdict)
                raw_responses.append(raw_response)
                break
            except Exception as error:
                errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
                if attempt < max_retries:
                    await asyncio.sleep(min(attempt, 3))
        else:
            raise RuntimeError(
                f"judge failed for {candidate['candidate_id']} batch {batch_index}/{len(batches)}: "
                + "; ".join(errors)
            )

    if len(batch_verdicts) == 1:
        verdict = batch_verdicts[0]
    else:
        retention_counts = Counter(value["summary_retention"] for value in batch_verdicts)
        top_count = max(retention_counts.values())
        top_labels = [label for label, count in retention_counts.items() if count == top_count]
        summary_retention = top_labels[0] if len(top_labels) == 1 else "unclear"
        if summary_retention == "unclear":
            summary_rationale = "Evidence-batch summary judgments did not produce a unique majority."
        else:
            summary_rationale = next(
                value["summary_rationale"]
                for value in batch_verdicts
                if value["summary_retention"] == summary_retention
            )
        merged_response = {
            "evidence_assessments": [
                assessment for value in batch_verdicts for assessment in value["evidence_assessments"]
            ],
            "summary_retention": summary_retention,
            "summary_rationale": summary_rationale,
        }
        evidence_ids = [record["evidence_id"] for record in evidence]
        verdict = parse_verdict(
            json.dumps(merged_response, ensure_ascii=False),
            evidence_ids,
            evidence_gold_parts={record["evidence_id"]: record["matched_gold_parts"] for record in evidence},
            required_gold_parts=candidate["gold_parts"],
        )

    judgment = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate["candidate_id"],
        "point": candidate["point"],
        "rollout_id": candidate["rollout_id"],
        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
        "judge_model": model,
        "attempts": total_attempts,
        "request_batches": len(batches),
        "judged_at": _utc_now(),
        "verdict": verdict,
    }
    if keep_raw_response:
        judgment["raw_response"] = raw_responses[0] if len(raw_responses) == 1 else raw_responses
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
        evidence_count = messages[-1]["content"].count('"evidence_id"')
        max_output_tokens = API_LARGE_MAX_OUTPUT_TOKENS if evidence_count > 32 else API_MAX_OUTPUT_TOKENS
        output_limit = (
            {"max_tokens": max_output_tokens}
            if "claude" in model.lower()
            else {"max_completion_tokens": max_output_tokens}
        )
        response = await client.chat.completions.create(model=model, messages=messages, **output_limit)
        return response.choices[0].message.content or ""

    return call


def _validate_resumed_judgments(
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
            raise ValueError(f"duplicate resumed judgment for {candidate_id}")
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ValueError(f"resumed judgment has unknown candidate {candidate_id}")
        expected = {
            "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
            "judge_model": model,
        }
        mismatches = {
            key: (expected_value, judgment.get(key))
            for key, expected_value in expected.items()
            if judgment.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(f"resumed judgment {candidate_id} is incompatible: {mismatches}")
        evidence_ids = [record["evidence_id"] for record in candidate["matching_tool_responses"]]
        parse_verdict(
            json.dumps(judgment.get("verdict"), ensure_ascii=False),
            evidence_ids,
            evidence_gold_parts={
                record["evidence_id"]: record["matched_gold_parts"] for record in candidate["matching_tool_responses"]
            },
            required_gold_parts=candidate["gold_parts"],
        )
        if keep_raw_responses and "raw_response" not in judgment:
            raise ValueError(f"resumed judgment {candidate_id} did not retain its raw response")
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
        if candidate.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"candidate {candidate.get('candidate_id')} uses an unsupported schema")
        evidence_ids = [record.get("evidence_id") for record in candidate.get("matching_tool_responses", [])]
        if not evidence_ids or any(
            not isinstance(evidence_id, str) or not evidence_id for evidence_id in evidence_ids
        ):
            raise ValueError(f"candidate {candidate.get('candidate_id')} has invalid evidence IDs")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(f"candidate {candidate.get('candidate_id')} has duplicate evidence IDs")
    candidates = {str(row["candidate_id"]): row for row in candidate_rows}
    if len(candidates) != len(candidate_rows):
        raise ValueError("candidates.jsonl contains duplicate candidate_id values")
    output_dir.mkdir(parents=True, exist_ok=True)
    judgment_path = output_dir / "judgments.jsonl"
    resumed = _load_jsonl(judgment_path) if judgment_path.is_file() else []
    judgments = _validate_resumed_judgments(
        resumed,
        candidates,
        model,
        keep_raw_responses=keep_raw_responses,
    )
    all_pending = [candidate for candidate in candidate_rows if candidate["candidate_id"] not in judgments]
    pending = all_pending[:max_new_candidates] if max_new_candidates is not None else all_pending
    if pending:
        (output_dir / "_JUDGED").unlink(missing_ok=True)
        (output_dir / "judge_manifest.json").unlink(missing_ok=True)
    if pending and call_model is None:
        if not api_key:
            raise ValueError("semantic judging requires an API key")
        call_model = _openai_caller(base_url, api_key)
    assert call_model is not None or not pending

    semaphore = asyncio.Semaphore(concurrency)

    async def run(candidate: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            assert call_model is not None
            return await judge_candidate(
                candidate,
                model=model,
                call_model=call_model,
                max_retries=max_retries,
                keep_raw_response=keep_raw_responses,
            )

    tasks = [asyncio.create_task(run(candidate)) for candidate in pending]
    try:
        for task in asyncio.as_completed(tasks):
            judgment = await task
            judgments[judgment["candidate_id"]] = judgment
            ordered = [judgments[row["candidate_id"]] for row in candidate_rows if row["candidate_id"] in judgments]
            _write_jsonl(judgment_path, ordered)
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

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
        "n_resumed": len(resumed),
        "n_remaining": len(candidate_rows) - len(judgments),
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
        judgments_path = directory / "judgments.jsonl"
        candidate_rows = _load_jsonl(candidates_path)
        candidates = {row["candidate_id"]: row for row in candidate_rows}
        failure_rows = _load_jsonl(failure_retrieval_path)
        failures = {row["candidate_id"]: row for row in failure_rows}
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
            evidence_ids = [record["evidence_id"] for record in candidates[candidate_id]["matching_tool_responses"]]
            candidate = candidates[candidate_id]
            parse_verdict(
                json.dumps(judgment.get("verdict"), ensure_ascii=False),
                evidence_ids,
                evidence_gold_parts={
                    record["evidence_id"]: record["matched_gold_parts"]
                    for record in candidate["matching_tool_responses"]
                },
                required_gold_parts=candidate["gold_parts"],
            )

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
            "unclear retention is excluded and exposed by retention_coverage"
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
