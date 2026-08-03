#!/usr/bin/env python3
"""Intersect strict 8/8 successes across the configured model points."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def aggregate(batch_root: Path) -> dict[str, Any]:
    config = read_json(batch_root / "config.json")
    point_names = [point["name"] for point in config["points"]]
    expected_questions = int(config["evaluation"]["expected_questions"])
    expected_samples = int(config["evaluation"]["samples_per_question"])

    manifests: dict[str, dict[str, Any]] = {}
    question_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for point in config["points"]:
        point_root = batch_root / "points" / point["name"]
        for filename in ("_SUCCESS", "manifest.json", "point_metrics.json", "questions.jsonl"):
            if not (point_root / filename).is_file():
                raise RuntimeError(f"{point_root} is missing {filename}")
        manifest = read_json(point_root / "manifest.json")
        metrics = read_json(point_root / "point_metrics.json")
        rows = read_jsonl(point_root / "questions.jsonl")
        if metrics.get("n_questions") != expected_questions:
            raise RuntimeError(
                f"{point['name']} has {metrics.get('n_questions')} questions, expected {expected_questions}"
            )
        if metrics.get("samples_per_question") != expected_samples:
            raise RuntimeError(
                f"{point['name']} has n={metrics.get('samples_per_question')}, expected {expected_samples}"
            )
        if manifest.get("load_verification", {}).get("actual_step") != point["step"]:
            raise RuntimeError(
                f"{point['name']} loaded {manifest.get('load_verification')}, expected {point['step']}"
            )
        by_id = {str(row["query_id"]): row for row in rows}
        if len(by_id) != expected_questions:
            raise RuntimeError(f"{point['name']} has {len(by_id)} unique query ids")
        manifests[point["name"]] = manifest
        question_maps[point["name"]] = by_id

    reference_manifest = manifests[point_names[0]]
    if reference_manifest.get("dataset_sha256") != config["dataset"]["sha256"]:
        raise RuntimeError(
            "evaluated dataset hash does not match the source hash frozen in config.json"
        )
    reference_ids = set(question_maps[point_names[0]])
    for point_name in point_names[1:]:
        mismatches = {
            field: {"reference": reference_manifest.get(field), "point": manifests[point_name].get(field)}
            for field in ("model_name", "dataset_sha256", "judge_model", "sampling")
            if reference_manifest.get(field) != manifests[point_name].get(field)
        }
        if mismatches:
            raise RuntimeError(f"{point_name} protocol does not match base: {mismatches}")
        if set(question_maps[point_name]) != reference_ids:
            raise RuntimeError(f"{point_name} query ids do not match base")

    candidates = []
    for query_id in sorted(reference_ids):
        rows = {point_name: question_maps[point_name][query_id] for point_name in point_names}
        successes = {point_name: int(row["successes"]) for point_name, row in rows.items()}
        if all(value == expected_samples for value in successes.values()):
            candidates.append(
                {
                    "query_id": query_id,
                    "question": rows[point_names[0]]["question"],
                    "successes": successes,
                    "n_samples_per_point": expected_samples,
                    "criterion": f"all {expected_samples} samples succeeded at every model point",
                }
            )

    summary = {
        "run_name": config["run_name"],
        "points": point_names,
        "n_questions": expected_questions,
        "samples_per_question": expected_samples,
        "n_filter_candidates": len(candidates),
        "candidate_rate": round(len(candidates) / expected_questions, 6),
        "dataset_sha256": reference_manifest["dataset_sha256"],
        "criterion": {
            "score": "score == 1",
            "within_point": f"{expected_samples}/{expected_samples}",
            "across_points": "base and every requested checkpoint",
        },
    }
    write_jsonl(batch_root / "filter_candidates.jsonl", candidates)
    (batch_root / "filter_candidate_query_ids.txt").write_text(
        "".join(f"{row['query_id']}\n" for row in candidates)
    )
    write_json(batch_root / "filter_summary.json", summary)
    lines = [
        "# BC+ training-data filter candidates",
        "",
        f"- Run: `{config['run_name']}`",
        f"- Points: {', '.join(point_names)}",
        f"- Questions: {expected_questions}",
        f"- Samples per question and point: {expected_samples}",
        f"- Strict candidates: {len(candidates)} ({summary['candidate_rate']:.2%})",
        "",
        "A candidate has `score == 1` for every sample at base and every requested checkpoint.",
        "Use `filter_candidates.jsonl` as the auditable source of query ids and per-point counts.",
    ]
    (batch_root / "filter_summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.batch_root), indent=2))


if __name__ == "__main__":
    main()
