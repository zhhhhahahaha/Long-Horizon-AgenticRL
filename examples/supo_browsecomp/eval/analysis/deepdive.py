#!/usr/bin/env python3
"""Reproducible run-level orchestration for BC+ deep-dive evaluation."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from . import summary_retention as retention
except ImportError:
    import summary_retention as retention


CONFIG_SCHEMA_VERSION = 1
PIPELINE_VERSION = "bcplus-deepdive-v1"
POINT_NAME_RE = re.compile(r"(?:base|iter\d+)")
SAFE_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
STAGE_FILES = (
    "_STAGED",
    "stage_manifest.json",
    "candidates.jsonl",
    "failure_retrieval.jsonl",
)
JUDGE_RESULT_FILES = (
    "_JUDGED",
    "judge_manifest.json",
    "match_judgments.jsonl",
    "summary_judgments.jsonl",
    "judgments.jsonl",
)


@dataclass(frozen=True)
class PointConfig:
    name: str
    source_dir: Path


@dataclass(frozen=True)
class JudgeConfig:
    name: str
    display_name: str
    model: str
    base_url: str
    api_key_env: str
    concurrency: int
    max_retries: int
    keep_raw_responses: bool


@dataclass(frozen=True)
class ComparisonConfig:
    name: str
    model_a: str
    model_b: str


@dataclass(frozen=True)
class DeepDiveConfig:
    config_path: Path
    run_name: str
    output_dir: Path
    points: tuple[PointConfig, ...]
    judges: tuple[JudgeConfig, ...]
    comparisons: tuple[ComparisonConfig, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _require_fields(value: dict[str, Any], allowed: set[str], required: set[str], label: str) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown or missing:
        raise ValueError(f"invalid {label} fields: unknown={sorted(unknown)}, missing={sorted(missing)}")


def _resolve_path(value: Any, base_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def load_config(path: Path) -> DeepDiveConfig:
    config_path = path.expanduser().resolve()
    value = json.loads(config_path.read_text())
    if not isinstance(value, dict):
        raise ValueError("deep-dive config must be a JSON object")
    _require_fields(
        value,
        {"schema_version", "run_name", "output_dir", "points", "judges", "comparisons"},
        {"schema_version", "run_name", "output_dir", "points", "judges"},
        "top-level config",
    )
    if value["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"unsupported deep-dive config schema: {value['schema_version']!r}")
    run_name = value["run_name"]
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be a non-empty string")
    output_dir = _resolve_path(value["output_dir"], config_path.parent, "output_dir")

    raw_points = value["points"]
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError("points must be a non-empty list")
    points = []
    for index, raw_point in enumerate(raw_points):
        if not isinstance(raw_point, dict):
            raise ValueError(f"points[{index}] must be an object")
        _require_fields(raw_point, {"name", "source_dir"}, {"name", "source_dir"}, f"points[{index}]")
        name = raw_point["name"]
        if not isinstance(name, str) or POINT_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"invalid point name: {name!r}")
        points.append(
            PointConfig(
                name=name,
                source_dir=_resolve_path(raw_point["source_dir"], config_path.parent, f"points[{index}].source_dir"),
            )
        )
    point_names = [point.name for point in points]
    if len(point_names) != len(set(point_names)):
        raise ValueError("point names must be unique")

    raw_judges = value["judges"]
    if not isinstance(raw_judges, list) or not raw_judges:
        raise ValueError("judges must be a non-empty list")
    judges = []
    judge_allowed = {
        "name",
        "display_name",
        "model",
        "base_url",
        "api_key_env",
        "concurrency",
        "max_retries",
        "keep_raw_responses",
    }
    for index, raw_judge in enumerate(raw_judges):
        if not isinstance(raw_judge, dict):
            raise ValueError(f"judges[{index}] must be an object")
        _require_fields(raw_judge, judge_allowed, {"name", "model"}, f"judges[{index}]")
        name = raw_judge["name"]
        model = raw_judge["model"]
        if not isinstance(name, str) or SAFE_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"invalid judge name: {name!r}")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"judge {name} model must be a non-empty string")
        display_name = raw_judge.get("display_name", name)
        base_url = raw_judge.get("base_url", "https://api.llama.com/compat/v1/")
        api_key_env = raw_judge.get("api_key_env", "LLAMA_API_KEY")
        concurrency = raw_judge.get("concurrency", 8)
        max_retries = raw_judge.get("max_retries", 3)
        keep_raw_responses = raw_judge.get("keep_raw_responses", False)
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError(f"judge {name} display_name must be a non-empty string")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError(f"judge {name} base_url must be a non-empty string")
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise ValueError(f"judge {name} api_key_env must be a non-empty string")
        if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
            raise ValueError(f"judge {name} concurrency must be a positive integer")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 1:
            raise ValueError(f"judge {name} max_retries must be a positive integer")
        if not isinstance(keep_raw_responses, bool):
            raise ValueError(f"judge {name} keep_raw_responses must be boolean")
        judges.append(
            JudgeConfig(
                name=name,
                display_name=display_name,
                model=model,
                base_url=base_url,
                api_key_env=api_key_env,
                concurrency=concurrency,
                max_retries=max_retries,
                keep_raw_responses=keep_raw_responses,
            )
        )
    judge_names = [judge.name for judge in judges]
    if len(judge_names) != len(set(judge_names)):
        raise ValueError("judge names must be unique")

    raw_comparisons = value.get("comparisons")
    if raw_comparisons is None:
        raw_comparisons = [
            {"name": f"{model_a}_vs_{model_b}", "model_a": model_a, "model_b": model_b}
            for model_a, model_b in itertools.combinations(judge_names, 2)
        ]
    if not isinstance(raw_comparisons, list):
        raise ValueError("comparisons must be a list")
    comparisons = []
    for index, raw_comparison in enumerate(raw_comparisons):
        if not isinstance(raw_comparison, dict):
            raise ValueError(f"comparisons[{index}] must be an object")
        _require_fields(
            raw_comparison,
            {"name", "model_a", "model_b"},
            {"name", "model_a", "model_b"},
            f"comparisons[{index}]",
        )
        name = raw_comparison["name"]
        model_a = raw_comparison["model_a"]
        model_b = raw_comparison["model_b"]
        if not isinstance(name, str) or SAFE_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"invalid comparison name: {name!r}")
        if model_a not in judge_names or model_b not in judge_names or model_a == model_b:
            raise ValueError(f"comparison {name} must reference two different configured judges")
        comparisons.append(ComparisonConfig(name=name, model_a=model_a, model_b=model_b))
    comparison_names = [comparison.name for comparison in comparisons]
    if len(comparison_names) != len(set(comparison_names)):
        raise ValueError("comparison names must be unique")

    return DeepDiveConfig(
        config_path=config_path,
        run_name=run_name,
        output_dir=output_dir,
        points=tuple(points),
        judges=tuple(judges),
        comparisons=tuple(comparisons),
    )


def _contract(config: DeepDiveConfig) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "summary_retention": {
            "schema_version": retention.SCHEMA_VERSION,
            "prefilter_version": retention.PREFILTER_VERSION,
            "judge_protocol_version": retention.JUDGE_PROTOCOL_VERSION,
        },
        "run_name": config.run_name,
        "points": [
            {"name": point.name, "source_dir": str(point.source_dir)} for point in config.points
        ],
        "judges": [
            {
                "name": judge.name,
                "model": judge.model,
                "base_url": judge.base_url,
                "keep_raw_responses": judge.keep_raw_responses,
            }
            for judge in config.judges
        ],
    }


def _resolved_config(config: DeepDiveConfig) -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "run_name": config.run_name,
        "output_dir": str(config.output_dir),
        "points": [
            {"name": point.name, "source_dir": str(point.source_dir)} for point in config.points
        ],
        "judges": [
            {
                "name": judge.name,
                "display_name": judge.display_name,
                "model": judge.model,
                "base_url": judge.base_url,
                "api_key_env": judge.api_key_env,
                "concurrency": judge.concurrency,
                "max_retries": judge.max_retries,
                "keep_raw_responses": judge.keep_raw_responses,
            }
            for judge in config.judges
        ],
        "comparisons": [
            {"name": value.name, "model_a": value.model_a, "model_b": value.model_b}
            for value in config.comparisons
        ],
    }


def prepare_output(config: DeepDiveConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "deepdive_manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    contract = _contract(config)
    if existing is not None and existing.get("contract") != contract:
        raise ValueError(
            f"deep-dive contract changed for {config.output_dir}; use a fresh output directory"
        )
    comparisons = [
        {"name": value.name, "model_a": value.model_a, "model_b": value.model_b}
        for value in config.comparisons
    ]
    if existing is not None and existing.get("comparisons") != comparisons:
        (config.output_dir / "_SUCCESS").unlink(missing_ok=True)
    created_at = existing.get("created_at") if existing is not None else _utc_now()
    _write_json(config.output_dir / "deepdive_config.resolved.json", _resolved_config(config))
    _write_json(
        manifest_path,
        {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "created_at": created_at,
            "updated_at": _utc_now(),
            "config_path": str(config.config_path),
            "contract": contract,
            "comparisons": comparisons,
        },
    )


def _validate_existing_stage(point: PointConfig, stage_dir: Path) -> dict[str, Any]:
    missing = [name for name in STAGE_FILES if not (stage_dir / name).is_file()]
    if missing:
        raise ValueError(f"incomplete staged point {stage_dir}: missing {missing}")
    manifest = json.loads((stage_dir / "stage_manifest.json").read_text())
    expected = {
        "schema_version": retention.SCHEMA_VERSION,
        "prefilter_version": retention.PREFILTER_VERSION,
        "point": point.name,
        "source_point_dir": str(point.source_dir),
    }
    mismatches = {
        key: (expected_value, manifest.get(key))
        for key, expected_value in expected.items()
        if manifest.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"staged point {stage_dir} is incompatible: {mismatches}")
    return manifest


def stage_all(config: DeepDiveConfig) -> dict[str, Any]:
    prepare_output(config)
    stage_root = config.output_dir / "stage"
    point_results = []
    staged_new_point = False
    for point in config.points:
        stage_dir = stage_root / point.name
        if (stage_dir / "_STAGED").is_file():
            manifest = _validate_existing_stage(point, stage_dir)
            resumed = True
        else:
            if any((config.output_dir / "judges" / judge.name / point.name).exists() for judge in config.judges):
                raise ValueError(f"cannot create missing stage for {point.name} after judge artifacts exist")
            manifest = retention.stage_point(point.source_dir, stage_dir)
            resumed = False
            staged_new_point = True
        point_results.append(
            {
                "point": point.name,
                "resumed": resumed,
                "n_candidates": manifest["counts"]["n_prefilter_candidates"],
                "n_match_tasks": manifest["counts"]["n_semantic_match_tasks"],
            }
        )
    result = {"status": "ok", "created_at": _utc_now(), "points": point_results}
    _write_json(stage_root / "_STAGED", result)
    if staged_new_point:
        (config.output_dir / "_SUCCESS").unlink(missing_ok=True)
    return result


def _same_file_content(source: Path, destination: Path) -> bool:
    try:
        if source.samefile(destination):
            return True
    except OSError:
        pass
    return source.stat().st_size == destination.stat().st_size and source.read_bytes() == destination.read_bytes()


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _same_file_content(source, destination):
            return
        raise ValueError(f"judge stage file differs from shared stage: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize_judge_stage(config: DeepDiveConfig, judge: JudgeConfig, point: PointConfig) -> Path:
    shared_stage = config.output_dir / "stage" / point.name
    _validate_existing_stage(point, shared_stage)
    judge_stage = config.output_dir / "judges" / judge.name / point.name
    for name in STAGE_FILES:
        _link_or_copy(shared_stage / name, judge_stage / name)
    return judge_stage


def _selected_judges(config: DeepDiveConfig, names: list[str] | None) -> list[JudgeConfig]:
    if not names:
        return list(config.judges)
    configured = {judge.name: judge for judge in config.judges}
    unknown = set(names) - set(configured)
    if unknown:
        raise ValueError(f"unknown judge names: {sorted(unknown)}")
    return [configured[name] for name in dict.fromkeys(names)]


async def judge_all(
    config: DeepDiveConfig,
    *,
    judge_names: list[str] | None = None,
    max_new_candidates_per_point: int | None = None,
) -> dict[str, Any]:
    if max_new_candidates_per_point is not None and max_new_candidates_per_point < 1:
        raise ValueError("max_new_candidates_per_point must be positive")
    selected_judges = _selected_judges(config, judge_names)
    stage_all(config)
    if any(
        not (config.output_dir / "judges" / judge.name / point.name / "_JUDGED").is_file()
        for judge in config.judges
        for point in config.points
    ):
        (config.output_dir / "_SUCCESS").unlink(missing_ok=True)
    results = []
    for judge in selected_judges:
        for point in config.points:
            stage_dir = materialize_judge_stage(config, judge, point)
            manifest = await retention.judge_stage(
                stage_dir,
                stage_dir,
                model=judge.model,
                base_url=judge.base_url,
                api_key=os.environ.get(judge.api_key_env),
                concurrency=judge.concurrency,
                max_retries=judge.max_retries,
                keep_raw_responses=judge.keep_raw_responses,
                max_new_candidates=max_new_candidates_per_point,
            )
            results.append(
                {
                    "judge": judge.name,
                    "point": point.name,
                    "n_judgments": manifest["n_judgments"],
                    "n_remaining": manifest["n_remaining"],
                    "complete": manifest["complete"],
                }
            )
    return {"status": "ok", "results": results}


def report_all(config: DeepDiveConfig) -> dict[str, Any]:
    prepare_output(config)
    (config.output_dir / "_SUCCESS").unlink(missing_ok=True)
    judge_by_name = {judge.name: judge for judge in config.judges}
    judge_dirs = {}
    reports = []
    for judge in config.judges:
        analysis_dirs = [config.output_dir / "judges" / judge.name / point.name for point in config.points]
        incomplete = [
            point.name
            for point, directory in zip(config.points, analysis_dirs, strict=True)
            if not (directory / "_JUDGED").is_file()
        ]
        if incomplete:
            raise ValueError(f"judge {judge.name} has incomplete points: {incomplete}")
        output_dir = config.output_dir / "judges" / judge.name / "report"
        report = retention.build_report(analysis_dirs, output_dir)
        judge_dirs[judge.name] = analysis_dirs
        reports.append({"judge": judge.name, "n_points": len(report["points"]), "output_dir": str(output_dir)})

    comparisons = []
    for comparison in config.comparisons:
        model_a = judge_by_name[comparison.model_a]
        model_b = judge_by_name[comparison.model_b]
        output_dir = config.output_dir / "comparisons" / comparison.name
        report = retention.build_model_comparison(
            judge_dirs[comparison.model_a],
            judge_dirs[comparison.model_b],
            output_dir,
            model_a_name=model_a.display_name,
            model_b_name=model_b.display_name,
        )
        comparisons.append(
            {"comparison": comparison.name, "n_points": len(report["points"]), "output_dir": str(output_dir)}
        )
    result = {
        "status": "ok",
        "created_at": _utc_now(),
        "run_name": config.run_name,
        "reports": reports,
        "comparisons": comparisons,
    }
    _write_json(config.output_dir / "_SUCCESS", result)
    return result


def status(config: DeepDiveConfig) -> dict[str, Any]:
    manifest_path = config.output_dir / "deepdive_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("contract") != _contract(config):
        raise ValueError(f"configured contract does not match {manifest_path}")
    stages = []
    for point in config.points:
        point_dir = config.output_dir / "stage" / point.name
        manifest_file = point_dir / "stage_manifest.json"
        manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else {}
        stages.append(
            {
                "point": point.name,
                "staged": all((point_dir / name).is_file() for name in STAGE_FILES),
                "n_candidates": manifest.get("counts", {}).get("n_prefilter_candidates"),
                "n_match_tasks": manifest.get("counts", {}).get("n_semantic_match_tasks"),
            }
        )
    judges = []
    for judge in config.judges:
        points = []
        for point in config.points:
            point_dir = config.output_dir / "judges" / judge.name / point.name
            manifest_file = point_dir / "judge_manifest.json"
            manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else {}
            points.append(
                {
                    "point": point.name,
                    "complete": all((point_dir / name).is_file() for name in JUDGE_RESULT_FILES),
                    "n_judgments": manifest.get("n_judgments", 0),
                    "n_remaining": manifest.get("n_remaining"),
                    "n_match_judgments": manifest.get("n_match_judgments", 0),
                    "n_summary_judgments": manifest.get("n_summary_judgments", 0),
                }
            )
        judges.append(
            {
                "name": judge.name,
                "model": judge.model,
                "complete": all(value["complete"] for value in points),
                "report_complete": (
                    config.output_dir
                    / "judges"
                    / judge.name
                    / "report"
                    / "_SUMMARY_RETENTION_SUCCESS"
                ).is_file(),
                "points": points,
            }
        )
    comparisons = [
        {
            "name": comparison.name,
            "complete": (
                config.output_dir
                / "comparisons"
                / comparison.name
                / "_SUMMARY_RETENTION_COMPARISON_SUCCESS"
            ).is_file(),
        }
        for comparison in config.comparisons
    ]
    complete = (
        manifest_path.is_file()
        and (config.output_dir / "_SUCCESS").is_file()
        and all(value["staged"] for value in stages)
        and all(value["complete"] for value in judges)
        and all(value["report_complete"] for value in judges)
        and all(value["complete"] for value in comparisons)
    )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "run_name": config.run_name,
        "output_dir": str(config.output_dir),
        "complete": complete,
        "stages": stages,
        "judges": judges,
        "comparisons": comparisons,
    }


def _load_cli_config(args: argparse.Namespace) -> DeepDiveConfig:
    return load_config(Path(args.config))


def _stage_command(args: argparse.Namespace) -> None:
    print(json.dumps(stage_all(_load_cli_config(args)), indent=2))


def _judge_command(args: argparse.Namespace) -> None:
    result = asyncio.run(
        judge_all(
            _load_cli_config(args),
            judge_names=args.judge,
            max_new_candidates_per_point=args.max_new_candidates_per_point,
        )
    )
    print(json.dumps(result, indent=2))


def _report_command(args: argparse.Namespace) -> None:
    print(json.dumps(report_all(_load_cli_config(args)), indent=2))


def _status_command(args: argparse.Namespace) -> None:
    print(json.dumps(status(_load_cli_config(args)), indent=2))


def _run_command(args: argparse.Namespace) -> None:
    config = _load_cli_config(args)
    asyncio.run(
        judge_all(
            config,
            judge_names=args.judge,
            max_new_candidates_per_point=args.max_new_candidates_per_point,
        )
    )
    if args.max_new_candidates_per_point is None and not args.judge:
        report_all(config)
    print(json.dumps(status(config), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    for name, help_text, function in (
        ("stage", "stage all configured points without API calls", _stage_command),
        ("judge", "judge staged points with resumable checkpoints", _judge_command),
        ("report", "build all model and comparison reports", _report_command),
        ("status", "show stage, judge, and report completion", _status_command),
        ("run", "stage, judge, and report the configured run", _run_command),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", required=True)
        if name in {"judge", "run"}:
            command.add_argument("--judge", action="append", help="run only this configured judge")
            command.add_argument("--max-new-candidates-per-point", type=int)
        command.set_defaults(func=function)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
