#!/usr/bin/env python3
"""Strict point analysis and per-run reporting for BC+ checkpoint evals."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ATTEMPT_OUTCOMES = {"compressed", "compress_failed", "compressed_capped"}
SUMMARY_SOURCES = {"extracted", "fallback", "empty"}


def _bc(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    value = metadata.get("_bcplus")
    return value if isinstance(value, dict) else {}


def _sibling(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    value = metadata.get("_bcplus_sibling")
    return value if isinstance(value, dict) else {}


def _score(reward: Any) -> float:
    if isinstance(reward, dict):
        value = reward.get("score", 0.0)
    else:
        value = reward
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _reward_metric(reward: Any, key: str) -> float:
    if not isinstance(reward, dict):
        return 0.0
    try:
        return float(reward.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _question(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("query") or metadata.get("problem_statement")
        if value:
            return str(value)
    prompt = sample.get("prompt")
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
    return str(prompt or "")


def _query_id(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("query_id") is None:
        raise ValueError("eval sample is missing metadata.query_id")
    return str(metadata["query_id"])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[int | float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _length_stats(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "median": None, "p90": None}
    return {
        "count": len(values),
        "mean": round(_mean([float(value) for value in values]), 3),
        "min": min(values),
        "max": max(values),
        "median": _percentile(values, 0.5),
        "p90": _percentile(values, 0.9),
    }


def _bootstrap_ci(question_successes: list[int], n_samples: int, metric: str, iterations: int = 10_000) -> list[float]:
    if not question_successes:
        return [0.0, 0.0]

    def compute(values: list[int]) -> float:
        if metric == "pass@1":
            return _mean([value / n_samples for value in values])
        if metric == "pass@n":
            return _mean([float(value > 0) for value in values])
        if metric == "all_correct":
            return _mean([float(value == n_samples) for value in values])
        raise ValueError(f"unknown bootstrap metric: {metric}")

    rng = random.Random(0)
    size = len(question_successes)
    estimates = [compute([question_successes[rng.randrange(size)] for _ in range(size)]) for _ in range(iterations)]
    return [round(_percentile(estimates, 0.025) or 0.0, 6), round(_percentile(estimates, 0.975) or 0.0, 6)]


def verify_load_log(log_text: str, requested_step: str) -> dict[str, Any]:
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in log_text.splitlines()]
    if requested_step == "base":
        evidence = [
            line.strip()
            for line in lines
            if re.search(r"loading release(?: distributed)? checkpoint", line, re.I)
            or (
                re.search(r"successfully loaded checkpoint", line, re.I)
                and "torch_dist" in line
                and re.search(r"\bat iteration\s+0\b", line, re.I)
            )
        ]
        if not evidence:
            raise ValueError("base eval log does not confirm loading a release checkpoint")
        return {"requested_step": "base", "actual_step": "base", "evidence": evidence[-3:]}

    expected = int(requested_step)
    pattern = re.compile(rf"\bat iteration\s+0*{expected}\b", re.I)
    load_lines = [line for line in lines if re.search(r"\bload(?:ed|ing)?\b", line, re.I)]
    evidence = [line.strip() for line in load_lines if pattern.search(line)]
    if not evidence:
        observed = sorted(
            {int(value) for value in re.findall(r"\bat iteration\s+(\d+)\b", "\n".join(load_lines), re.I)}
        )
        raise ValueError(f"checkpoint log does not confirm iteration {expected}; observed={observed}")
    release_evidence = [line.strip() for line in load_lines if re.search(r"\brelease checkpoint\b", line, re.I)]
    if release_evidence:
        raise ValueError(
            f"trained checkpoint eval also logged a release checkpoint load: {release_evidence[-3:]}"
        )
    return {"requested_step": expected, "actual_step": expected, "evidence": evidence[-3:]}


def analyze_samples(
    samples: list[dict[str, Any]], *, expected_questions: int, samples_per_question: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        rollout_id = sample.get("rollout_id")
        if rollout_id is None:
            raise ValueError("every BC+ eval sample must have rollout_id")
        grouped[rollout_id].append(sample)

    expected_rollouts = expected_questions * samples_per_question
    if len(grouped) != expected_rollouts:
        raise ValueError(f"expected {expected_rollouts} parent rollouts, found {len(grouped)}")

    rollout_rows: list[dict[str, Any]] = []
    all_extracted_lengths: list[int] = []
    for rollout_id, siblings in grouped.items():
        sibling_metadata = [_sibling(sample) for sample in siblings]
        if any(not metadata for metadata in sibling_metadata):
            raise ValueError(f"rollout {rollout_id} has sibling without _bcplus_sibling metadata")
        indices = [metadata.get("sub_traj_index") for metadata in sibling_metadata]
        if any(not isinstance(index, int) for index in indices):
            raise ValueError(f"rollout {rollout_id} has invalid sub_traj_index values: {indices}")
        expected_indices = list(range(len(siblings)))
        if sorted(indices) != expected_indices:
            raise ValueError(f"rollout {rollout_id} has non-contiguous sub_traj_index values: {indices}")
        totals = {metadata.get("total_sub_trajs") for metadata in sibling_metadata}
        if totals != {len(siblings)}:
            raise ValueError(f"rollout {rollout_id} total_sub_trajs mismatch: {totals} vs {len(siblings)}")
        finals = [sample for sample in siblings if _sibling(sample).get("is_final") is True]
        if len(finals) != 1:
            raise ValueError(f"rollout {rollout_id} must have exactly one final sibling, found {len(finals)}")
        siblings = sorted(siblings, key=lambda sample: _sibling(sample)["sub_traj_index"])
        final = finals[0]
        query_ids = {_query_id(sample) for sample in siblings}
        if len(query_ids) != 1:
            raise ValueError(f"rollout {rollout_id} spans multiple query_ids: {query_ids}")

        score = _score(final.get("reward"))
        source_counts: Counter[str] = Counter()
        extracted_lengths: list[int] = []
        compression_attempts = 0
        for sample in siblings:
            bc = _bc(sample)
            outcome = str(bc.get("outcome", ""))
            source = str(bc.get("summary_source", "") or "")
            if outcome in ATTEMPT_OUTCOMES:
                compression_attempts += 1
                if source not in SUMMARY_SOURCES:
                    raise ValueError(f"rollout {rollout_id} compression outcome {outcome!r} has source {source!r}")
                source_counts[source] += 1
                if source == "extracted":
                    value = bc.get("summary_content_len_tokens")
                    if not isinstance(value, int) or value < 0:
                        raise ValueError(f"rollout {rollout_id} extracted summary is missing token length")
                    extracted_lengths.append(value)
                    all_extracted_lengths.append(value)

        final_bc = _bc(final)
        rollout_rows.append(
            {
                "rollout_id": rollout_id,
                "query_id": next(iter(query_ids)),
                "question": _question(final),
                "gold": final.get("label"),
                "finish_answer": final_bc.get("finish_answer", ""),
                "score": score,
                "correct": score == 1.0,
                "judge_failed": bool(_reward_metric(final.get("reward"), "judge_failed")),
                "finished": bool(final_bc.get("finished", False)),
                "outcome": final_bc.get("outcome", ""),
                "final_stop_reason": final_bc.get("final_stop_reason", ""),
                "n_sub_trajs": len(siblings),
                "n_turns": sum(int(_bc(sample).get("n_turns_used", 0) or 0) for sample in siblings),
                "n_search": sum(int(_bc(sample).get("n_search", 0) or 0) for sample in siblings),
                "n_open": sum(int(_bc(sample).get("n_open", 0) or 0) for sample in siblings),
                "n_bad_tool_calls": sum(int(_bc(sample).get("n_bad_tool_calls", 0) or 0) for sample in siblings),
                "n_search_server_error": sum(
                    int(_bc(sample).get("n_search_server_error", 0) or 0) for sample in siblings
                ),
                "response_tokens": sum(int(_bc(sample).get("response_len_tokens", 0) or 0) for sample in siblings),
                "generated_tokens": sum(sum(1 for value in (sample.get("loss_mask") or []) if value) for sample in siblings),
                "compression_attempts": compression_attempts,
                "summary_extracted": source_counts["extracted"],
                "summary_fallback": source_counts["fallback"],
                "summary_empty": source_counts["empty"],
                "extracted_summary_lengths": extracted_lengths,
            }
        )

    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        by_question[row["query_id"]].append(row)
    if len(by_question) != expected_questions:
        raise ValueError(f"expected {expected_questions} unique query_ids, found {len(by_question)}")
    wrong_counts = {query_id: len(rows) for query_id, rows in by_question.items() if len(rows) != samples_per_question}
    if wrong_counts:
        raise ValueError(f"queries do not have exactly {samples_per_question} rollouts: {wrong_counts}")

    question_rows = []
    for query_id, rows in sorted(by_question.items()):
        successes = sum(bool(row["correct"]) for row in rows)
        question_rows.append(
            {
                "query_id": query_id,
                "question": rows[0]["question"],
                "successes": successes,
                "n_samples": samples_per_question,
                "pass@1": successes / samples_per_question,
                "pass@n": int(successes > 0),
                "all_correct": int(successes == samples_per_question),
            }
        )

    question_successes = [row["successes"] for row in question_rows]
    attempts = sum(row["compression_attempts"] for row in rollout_rows)
    summary_counts = {
        source: sum(row[f"summary_{source}"] for row in rollout_rows) for source in ("extracted", "fallback", "empty")
    }
    numeric_averages = {}
    for key in (
        "n_sub_trajs",
        "n_turns",
        "n_search",
        "n_open",
        "n_bad_tool_calls",
        "n_search_server_error",
        "response_tokens",
        "generated_tokens",
    ):
        numeric_averages[f"avg_{key}"] = round(_mean([float(row[key]) for row in rollout_rows]), 4)

    metrics = {
        "n_questions": len(question_rows),
        "n_rollouts": len(rollout_rows),
        "samples_per_question": samples_per_question,
        "score_mean": round(_mean([row["score"] for row in rollout_rows]), 6),
        "pass@1": round(_mean([row["pass@1"] for row in question_rows]), 6),
        "pass@n": round(_mean([float(row["pass@n"]) for row in question_rows]), 6),
        "all_correct_rate": round(_mean([float(row["all_correct"]) for row in question_rows]), 6),
        "pass@1_ci95": _bootstrap_ci(question_successes, samples_per_question, "pass@1"),
        "pass@n_ci95": _bootstrap_ci(question_successes, samples_per_question, "pass@n"),
        "all_correct_ci95": _bootstrap_ci(question_successes, samples_per_question, "all_correct"),
        "question_success_hist": {str(key): value for key, value in sorted(Counter(question_successes).items())},
        "finish_rate": round(_mean([float(row["finished"]) for row in rollout_rows]), 6),
        "judge_failed_count": sum(row["judge_failed"] for row in rollout_rows),
        "judge_failed_rate": round(_mean([float(row["judge_failed"]) for row in rollout_rows]), 6),
        "search_error_rollout_count": sum(row["n_search_server_error"] > 0 for row in rollout_rows),
        "search_error_rollout_rate": round(
            _mean([float(row["n_search_server_error"] > 0) for row in rollout_rows]), 6
        ),
        "outcome_dist": dict(Counter(str(row["outcome"] or "?") for row in rollout_rows)),
        "compression_attempts": attempts,
        "compressed_rollout_count": sum(row["compression_attempts"] > 0 for row in rollout_rows),
        "compressed_rollout_rate": round(_mean([float(row["compression_attempts"] > 0) for row in rollout_rows]), 6),
        "summary_counts": summary_counts,
        "summary_rates": {
            source: round(summary_counts[source] / attempts, 6) if attempts else 0.0 for source in summary_counts
        },
        "fallback_rollout_count": sum(row["summary_fallback"] > 0 for row in rollout_rows),
        "fallback_rollout_rate": round(_mean([float(row["summary_fallback"] > 0) for row in rollout_rows]), 6),
        "extracted_summary_content_tokens": _length_stats(all_extracted_lengths),
        **numeric_averages,
    }
    return metrics, sorted(rollout_rows, key=lambda row: int(row["rollout_id"])), question_rows


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def analyze_point(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    load_log = Path(args.load_log).read_text(errors="replace")
    load_verification = verify_load_log(load_log, args.requested_step)

    import torch

    blob = torch.load(args.dump, weights_only=False)
    samples = blob.get("samples", []) if isinstance(blob, dict) else []
    metrics, rollout_rows, question_rows = analyze_samples(
        samples, expected_questions=args.expected_questions, samples_per_question=args.samples_per_question
    )
    manifest = {
        "run_name": args.run_name,
        "point": args.point,
        "model_name": args.model_name,
        "load_verification": load_verification,
        "checkpoint_root": args.checkpoint_root,
        "checkpoint_metadata_sha256": args.checkpoint_metadata_sha256,
        "code_archive_sha256": args.code_archive_sha256,
        "dataset_sha256": args.dataset_sha256,
        "mast_job_name": args.mast_job_name,
        "judge_model": args.judge_model,
        "search_url": args.search_url,
        "sampling": {
            "samples_per_question": args.samples_per_question,
            "rollout_seed": args.rollout_seed,
            "sampling_seeds": list(range(args.rollout_seed, args.rollout_seed + args.samples_per_question)),
            "deterministic": True,
            "temperature": args.temperature,
            "max_response_len": args.max_response_len,
            "max_context_len": args.max_context_len,
            "max_turns": args.max_turns,
            "max_sub_trajs": args.max_sub_trajs,
            "compression_threshold": args.compression_threshold,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "point_metrics.json", metrics)
    _write_jsonl(output_dir / "rollouts.jsonl", rollout_rows)
    _write_jsonl(output_dir / "questions.jsonl", question_rows)
    _write_json(output_dir / "_SUCCESS", {"status": "ok", "n_rollouts": metrics["n_rollouts"]})
    print(json.dumps(metrics, indent=2))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _point_sort_key(name: str) -> tuple[int, int]:
    if name == "base":
        return (0, 0)
    match = re.fullmatch(r"iter0*(\d+)", name)
    if match is None:
        raise ValueError(f"invalid eval point name: {name}")
    return (1, int(match.group(1)))


def _manifest_model_name(manifest: dict[str, Any]) -> str | None:
    if manifest.get("model_name"):
        return str(manifest["model_name"])
    checkpoint_root = str(manifest.get("checkpoint_root", "")).lower()
    if "9b" in checkpoint_root:
        return "Qwen3.5-9B"
    if "4b" in checkpoint_root:
        return "Qwen3.5-4B"
    return None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_curve_svg(path: Path, points: list[tuple[str, float]], *, title: str) -> None:
    width, height, margin = 760, 360, 55
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    values = [value for _, value in points]
    y_min = max(0.0, min(values) - 0.05) if values else 0.0
    y_max = min(1.0, max(values) + 0.05) if values else 1.0
    if y_max <= y_min:
        y_max = min(1.0, y_min + 0.1)

    def xy(index: int, value: float) -> tuple[float, float]:
        x = margin + (plot_width * index / max(1, len(points) - 1))
        y = margin + plot_height * (y_max - value) / max(1e-9, y_max - y_min)
        return x, y

    coordinates = [xy(index, value) for index, (_, value) in enumerate(points)]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
    labels = []
    for (name, value), (x, y) in zip(points, coordinates, strict=True):
        labels.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#166534"/>')
        labels.append(
            f'<text x="{x:.1f}" y="{height - 22}" text-anchor="middle" font-size="12">{html.escape(name)}</text>'
        )
        labels.append(f'<text x="{x:.1f}" y="{y - 9:.1f}" text-anchor="middle" font-size="11">{value:.3f}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-size="16" font-weight="600">{html.escape(title)}</text>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#555"/>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#555"/>
<text x="18" y="{margin}" font-size="11">{y_max:.3f}</text><text x="18" y="{height-margin}" font-size="11">{y_min:.3f}</text>
<polyline points="{polyline}" fill="none" stroke="#166534" stroke-width="2"/>
{''.join(labels)}
</svg>\n'''
    path.write_text(svg)


def build_run_report(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    base_root = Path(args.base_root)
    point_dirs = [path for path in run_root.iterdir() if path.is_dir() and re.fullmatch(r"iter\d+", path.name)]
    point_dirs = [base_root, *sorted(point_dirs, key=lambda path: _point_sort_key(path.name))]
    if not point_dirs:
        raise ValueError(f"no eval points found under {run_root}")

    points = []
    question_maps = {}
    for point_dir in point_dirs:
        for required in ("_SUCCESS", "manifest.json", "point_metrics.json", "questions.jsonl"):
            if not (point_dir / required).is_file():
                raise ValueError(f"{point_dir} is missing {required}")
        name = point_dir.name
        metrics = json.loads((point_dir / "point_metrics.json").read_text())
        manifest = json.loads((point_dir / "manifest.json").read_text())
        points.append({"name": name, "metrics": metrics, "manifest": manifest})
        question_maps[name] = {row["query_id"]: row for row in _load_jsonl(point_dir / "questions.jsonl")}

    base_manifest = points[0]["manifest"]
    for point in points[1:]:
        mismatches = {
            field: {"base": base_manifest.get(field), "checkpoint": point["manifest"].get(field)}
            for field in ("dataset_sha256", "judge_model", "sampling")
            if base_manifest.get(field) != point["manifest"].get(field)
        }
        base_model = _manifest_model_name(base_manifest)
        checkpoint_model = _manifest_model_name(point["manifest"])
        if base_model != checkpoint_model:
            mismatches["model_name"] = {"base": base_model, "checkpoint": checkpoint_model}
        if mismatches:
            raise ValueError(f"{point['name']} is incompatible with base: {mismatches}")

    base_questions = question_maps["base"]
    sample_counts = {point["metrics"]["samples_per_question"] for point in points}
    if len(sample_counts) != 1:
        raise ValueError(f"eval points use inconsistent samples_per_question values: {sample_counts}")
    samples_per_question = next(iter(sample_counts))
    changes = []
    for point in points[1:]:
        name = point["name"]
        if question_maps[name].keys() != base_questions.keys():
            raise ValueError(f"{name} and base have different query ids")
        for query_id, row in question_maps[name].items():
            base = base_questions[query_id]
            delta = row["successes"] - base["successes"]
            changes.append(
                {
                    "point": name,
                    "query_id": query_id,
                    "base_successes": base["successes"],
                    "checkpoint_successes": row["successes"],
                    "delta": delta,
                    "change": "gained" if delta > 0 else "lost" if delta < 0 else "unchanged",
                    "question": row["question"],
                }
            )

    trained_points = points[1:]
    best = max(
        trained_points,
        key=lambda point: (
            point["metrics"]["pass@1"],
            point["metrics"]["pass@n"],
            -_point_sort_key(point["name"])[1],
        ),
    )
    output_dir = Path(args.output_dir or run_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_columns = [
        "point",
        "pass@1",
        "pass@n",
        "all_correct_rate",
        "finish_rate",
        "judge_failed_count",
        "search_error_rollout_count",
        "avg_n_turns",
        "avg_n_search",
        "avg_n_open",
        "avg_n_sub_trajs",
        "avg_response_tokens",
        "compressed_rollout_rate",
        "summary_extracted_rate",
        "summary_fallback_rate",
        "summary_empty_rate",
        "extracted_summary_len_mean",
        "extracted_summary_len_min",
        "extracted_summary_len_max",
        "extracted_summary_len_median",
        "extracted_summary_len_p90",
    ]
    metric_rows = []
    for point in points:
        metrics = point["metrics"]
        lengths = metrics["extracted_summary_content_tokens"]
        metric_rows.append(
            {
                "point": point["name"],
                "pass@1": metrics["pass@1"],
                "pass@n": metrics["pass@n"],
                "all_correct_rate": metrics["all_correct_rate"],
                "finish_rate": metrics["finish_rate"],
                "judge_failed_count": metrics["judge_failed_count"],
                "search_error_rollout_count": metrics["search_error_rollout_count"],
                "avg_n_turns": metrics["avg_n_turns"],
                "avg_n_search": metrics["avg_n_search"],
                "avg_n_open": metrics["avg_n_open"],
                "avg_n_sub_trajs": metrics["avg_n_sub_trajs"],
                "avg_response_tokens": metrics["avg_response_tokens"],
                "compressed_rollout_rate": metrics["compressed_rollout_rate"],
                "summary_extracted_rate": metrics["summary_rates"]["extracted"],
                "summary_fallback_rate": metrics["summary_rates"]["fallback"],
                "summary_empty_rate": metrics["summary_rates"]["empty"],
                "extracted_summary_len_mean": lengths["mean"],
                "extracted_summary_len_min": lengths["min"],
                "extracted_summary_len_max": lengths["max"],
                "extracted_summary_len_median": lengths["median"],
                "extracted_summary_len_p90": lengths["p90"],
            }
        )

    with (output_dir / "checkpoint_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_columns)
        writer.writeheader()
        writer.writerows(metric_rows)
    with (output_dir / "question_changes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changes[0].keys()))
        writer.writeheader()
        writer.writerows(changes)

    lines = [
        f"# BC+ checkpoint evaluation: {args.run_name}",
        "",
        f"Best observed test checkpoint: **{best['name']}** (pass@1={best['metrics']['pass@1']:.3f}).",
        "This is test-set checkpoint selection and is not an unbiased model-selection estimate.",
        "",
        "## Core metrics",
        "",
        f"| point | pass@1 (95% CI) | pass@{samples_per_question} | "
        f"{samples_per_question}/{samples_per_question} | finish | judge fail | search-error rollouts |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        metrics = point["metrics"]
        ci = metrics["pass@1_ci95"]
        lines.append(
            f"| {point['name']} | {metrics['pass@1']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{metrics['pass@n']:.3f} | {metrics['all_correct_rate']:.3f} | {metrics['finish_rate']:.3f} | "
            f"{metrics['judge_failed_count']} | {metrics['search_error_rollout_count']} |"
        )
    lines += [
        "",
        "## Accuracy curves",
        "",
        "![pass@1](accuracy_curve.svg)",
        "",
        f"![pass@{samples_per_question}](pass_at_n_curve.svg)",
        "",
        "## Behavior and efficiency",
        "",
        "| point | turns | search | open | sub-trajs | response tokens | bad tools | compressed rollouts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        metrics = point["metrics"]
        lines.append(
            f"| {point['name']} | {metrics['avg_n_turns']:.2f} | {metrics['avg_n_search']:.2f} | "
            f"{metrics['avg_n_open']:.2f} | {metrics['avg_n_sub_trajs']:.2f} | "
            f"{metrics['avg_response_tokens']:.1f} | {metrics['avg_n_bad_tool_calls']:.3f} | "
            f"{metrics['compressed_rollout_rate']:.3f} |"
        )
    lines += [
        "",
        "## Summary behavior",
        "",
        "| point | attempts | extracted | fallback | empty | fallback rollouts | extracted length mean/min/max/median/P90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        metrics = point["metrics"]
        rates = metrics["summary_rates"]
        lengths = metrics["extracted_summary_content_tokens"]
        length_text = "/".join(_fmt(lengths[key], 1) for key in ("mean", "min", "max", "median", "p90"))
        lines.append(
            f"| {point['name']} | {metrics['compression_attempts']} | {rates['extracted']:.3f} | "
            f"{rates['fallback']:.3f} | {rates['empty']:.3f} | {metrics['fallback_rollout_rate']:.3f} | "
            f"{length_text} |"
        )
    lines += ["", "## Question-level changes from base", "", "| point | gained | lost | unchanged |", "|---|---:|---:|---:|"]
    for point in trained_points:
        point_changes = Counter(row["change"] for row in changes if row["point"] == point["name"])
        lines.append(
            f"| {point['name']} | {point_changes['gained']} | {point_changes['lost']} | {point_changes['unchanged']} |"
        )
    lines += [
        "",
        "See `question_changes.csv` for per-question success-count deltas.",
        "",
        "## Load verification",
        "",
        "| point | actual load | checkpoint metadata SHA-256 | MAST job |",
        "|---|---:|---|---|",
    ]
    for point in points:
        manifest = point["manifest"]
        actual_step = manifest["load_verification"]["actual_step"]
        lines.append(
            f"| {point['name']} | {actual_step} | `{manifest['checkpoint_metadata_sha256']}` | "
            f"`{manifest['mast_job_name']}` |"
        )
    sampling = points[0]["manifest"]["sampling"]
    lines += [
        "",
        f"Deterministic sampling seeds: `{sampling['sampling_seeds']}`. "
        f"Judge: `{points[0]['manifest']['judge_model']}`.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))
    _write_curve_svg(
        output_dir / "accuracy_curve.svg",
        [(row["point"], float(row["pass@1"])) for row in metric_rows],
        title="pass@1",
    )
    _write_curve_svg(
        output_dir / "pass_at_n_curve.svg",
        [(row["point"], float(row["pass@n"])) for row in metric_rows],
        title=f"pass@{samples_per_question}",
    )
    _write_json(
        output_dir / "metrics.json",
        {"run_name": args.run_name, "best_point": best["name"], "points": points, "question_changes": changes},
    )
    print(output_dir / "report.md")


def _point_parser(subparsers) -> None:
    parser = subparsers.add_parser("point")
    parser.add_argument("--dump", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--load-log", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--point", required=True)
    parser.add_argument("--requested-step", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-metadata-sha256", required=True)
    parser.add_argument("--code-archive-sha256", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--mast-job-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--search-url", required=True)
    parser.add_argument("--expected-questions", type=int, default=150)
    parser.add_argument("--samples-per-question", type=int, default=4)
    parser.add_argument("--rollout-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-response-len", type=int, default=32768)
    parser.add_argument("--max-context-len", type=int, default=65536)
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument("--max-sub-trajs", type=int, default=5)
    parser.add_argument("--compression-threshold", type=float, default=0.85)
    parser.set_defaults(func=analyze_point)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    _point_parser(subparsers)
    report = subparsers.add_parser("report")
    report.add_argument("--run-root", required=True)
    report.add_argument("--base-root", required=True)
    report.add_argument("--run-name", required=True)
    report.add_argument("--output-dir")
    report.set_defaults(func=build_run_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
