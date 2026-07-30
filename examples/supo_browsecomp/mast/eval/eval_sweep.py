#!/usr/bin/env python3
"""Prepare, submit, monitor, and report reusable BC+ MAST checkpoint sweeps."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.request
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


FAILED_STATES = {"FAILED", "DEAD"}
REPORT_ARTIFACTS = (
    "report.md",
    "checkpoint_metrics.csv",
    "question_changes.csv",
    "metrics.json",
    "accuracy_curve.svg",
    "pass_at_n_curve.svg",
)

DEV_STAGE = Path("/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime")
MAST_STAGE = Path("/mnt/wsfuse/hhzhang01/supo-slime")
CLI = Path("/data/users/hhzhang01/fbsource/genai/msl/rl/cli.sh")
IMAGE = "588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a"
WSF_SRC = "ws://ws.ai.eag0genai/genai_fair_llm"
SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+")


def validate_safe_name(value: str, label: str) -> str:
    if SAFE_NAME.fullmatch(value) is None or value in {".", ".."}:
        raise RuntimeError(f"invalid {label}: {value!r}")
    return value


@dataclass(frozen=True)
class RunConfig:
    name: str
    alias: str
    steps: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunConfig:
        return cls(name=str(value["name"]), alias=str(value["alias"]), steps=tuple(map(int, value["steps"])))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "alias": self.alias, "steps": list(self.steps)}


@dataclass(frozen=True)
class SweepConfig:
    runs: tuple[RunConfig, ...]
    model_size: str = "4B"
    base_source_batch: str | None = None
    expected_questions: int = 150
    samples_per_question: int = 4
    rollout_seed: int = 42
    search_concurrency: int = 64
    judge_concurrency: int = 16
    local_report_root: str = "/home/hhzhang01/bcplus-eval-reports"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SweepConfig:
        evaluation = value.get("evaluation", {})
        base = value.get("base", {})
        model = value.get("model", {})
        return cls(
            runs=tuple(RunConfig.from_dict(run) for run in value["runs"]),
            model_size=str(model.get("size", "4B")).upper(),
            base_source_batch=base.get("source_batch"),
            expected_questions=int(evaluation.get("expected_questions", 150)),
            samples_per_question=int(evaluation.get("samples_per_question", 4)),
            rollout_seed=int(evaluation.get("rollout_seed", 42)),
            search_concurrency=int(evaluation.get("search_concurrency", 64)),
            judge_concurrency=int(evaluation.get("judge_concurrency", 16)),
            local_report_root=str(value.get("local_report_root", "/home/hhzhang01/bcplus-eval-reports")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 4,
            "runs": [run.to_dict() for run in self.runs],
            "model": {"size": self.model_size, "name": self.model_name},
            "base": {
                "mode": "reuse" if self.base_source_batch else "evaluate",
                "source_batch": self.base_source_batch,
            },
            "evaluation": {
                "expected_questions": self.expected_questions,
                "samples_per_question": self.samples_per_question,
                "rollout_seed": self.rollout_seed,
                "sampling_seeds": list(
                    range(self.rollout_seed, self.rollout_seed + self.samples_per_question)
                ),
                "search_concurrency": self.search_concurrency,
                "judge_concurrency": self.judge_concurrency,
            },
            "local_report_root": self.local_report_root,
        }

    def run(self, run_name: str) -> RunConfig:
        return next(run for run in self.runs if run.name == run_name)

    @property
    def model_name(self) -> str:
        return f"Qwen3.5-{self.model_size}"


def point_name(step: int) -> str:
    return f"iter{step:02d}"


def infer_model_size(runs: tuple[RunConfig, ...]) -> str:
    sizes = set()
    for run in runs:
        name = run.name.lower()
        if "9b" in name:
            sizes.add("9B")
        elif "4b" in name:
            sizes.add("4B")
    if len(sizes) > 1:
        raise RuntimeError(f"a sweep cannot share base across model sizes: {sorted(sizes)}")
    return next(iter(sizes), "4B")


def sweep_points(config: SweepConfig) -> list[dict[str, Any]]:
    points = []
    if config.base_source_batch is None:
        points.append({"key": "base", "run_name": "shared-base", "point": "base", "step": "base"})
    for run in config.runs:
        for step in run.steps:
            points.append(
                {
                    "key": f"{run.name}/{point_name(step)}",
                    "run_name": run.name,
                    "point": point_name(step),
                    "step": step,
                }
            )
    return points


def output_dir(batch_root: Path, point: dict[str, Any]) -> Path:
    if point["point"] == "base":
        return batch_root / "base"
    return batch_root / "runs" / point["run_name"] / point["point"]


def base_output_dir(config: SweepConfig, batch_root: Path) -> Path:
    if config.base_source_batch:
        return batch_root.parent / config.base_source_batch / "base"
    return batch_root / "base"


def manifest_model_name(manifest: dict[str, Any]) -> str | None:
    if manifest.get("model_name"):
        return str(manifest["model_name"])
    checkpoint_root = str(manifest.get("checkpoint_root", "")).lower()
    if "9b" in checkpoint_root:
        return "Qwen3.5-9B"
    if "4b" in checkpoint_root:
        return "Qwen3.5-4B"
    return None


def validate_base_source(config: SweepConfig, batch_root: Path) -> None:
    if config.base_source_batch is None:
        return
    if config.base_source_batch == batch_root.name:
        raise RuntimeError("--reuse-base-from cannot refer to the new batch itself")

    root = base_output_dir(config, batch_root)
    required = ("_SUCCESS", "manifest.json", "point_metrics.json", "questions.jsonl")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"reused base {root} is missing artifacts: {missing}")

    manifest = _read_json(root / "manifest.json", {})
    metrics = _read_json(root / "point_metrics.json", {})
    if manifest.get("load_verification", {}).get("actual_step") != "base":
        raise RuntimeError(f"reused base {root} does not confirm a release checkpoint load")
    actual_model = manifest_model_name(manifest)
    if actual_model != config.model_name:
        raise RuntimeError(
            f"reused base {root} uses model {actual_model!r}, expected {config.model_name}"
        )
    expected_rollouts = config.expected_questions * config.samples_per_question
    expected_metrics = {
        "n_questions": config.expected_questions,
        "n_rollouts": expected_rollouts,
        "samples_per_question": config.samples_per_question,
    }
    metric_mismatches = {
        key: {"expected": expected, "actual": metrics.get(key)}
        for key, expected in expected_metrics.items()
        if metrics.get(key) != expected
    }
    expected_sampling = {
        "samples_per_question": config.samples_per_question,
        "rollout_seed": config.rollout_seed,
        "sampling_seeds": list(range(config.rollout_seed, config.rollout_seed + config.samples_per_question)),
        "deterministic": True,
    }
    sampling = manifest.get("sampling", {})
    sampling_mismatches = {
        key: {"expected": expected, "actual": sampling.get(key)}
        for key, expected in expected_sampling.items()
        if sampling.get(key) != expected
    }
    if metric_mismatches or sampling_mismatches:
        raise RuntimeError(
            f"reused base {root} is incompatible: "
            f"metrics={metric_mismatches}, sampling={sampling_mismatches}"
        )


def validate_base_against_point(
    config: SweepConfig, batch_root: Path, point: dict[str, Any]
) -> None:
    if config.base_source_batch is None:
        return
    base_manifest = _read_json(base_output_dir(config, batch_root) / "manifest.json", {})
    point_manifest = _read_json(output_dir(batch_root, point) / "manifest.json", {})
    fields = ("dataset_sha256", "judge_model", "sampling")
    mismatches = {
        field: {"base": base_manifest.get(field), "checkpoint": point_manifest.get(field)}
        for field in fields
        if base_manifest.get(field) != point_manifest.get(field)
    }
    base_model = manifest_model_name(base_manifest)
    point_model = manifest_model_name(point_manifest)
    if base_model != point_model:
        mismatches["model_name"] = {"base": base_model, "checkpoint": point_model}
    if mismatches:
        raise RuntimeError(
            f"reused base is incompatible with {point['key']}: {mismatches}"
        )
    print(
        f"[base] reused {base_output_dir(config, batch_root)}; protocol matches {point['key']}",
        flush=True,
    )


def _mast_path(path: Path) -> str:
    relative = path.relative_to(DEV_STAGE)
    return str(MAST_STAGE / relative)


def _run(command: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    # The controller already owns cleanup. A nested cgrouped.sh scope can stop
    # the calling process before the MAST CLI result has been collected.
    env = os.environ.copy()
    env["CGROUPED"] = "1"
    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=capture,
        start_new_session=True,
        env=env,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


@contextmanager
def controller_lock(batch_root: Path):
    batch_root.mkdir(parents=True, exist_ok=True)
    lock_path = batch_root / ".controller.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise RuntimeError(f"batch controller is already running: {owner}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_run(run_name: str, index: int) -> RunConfig:
    validate_safe_name(run_name, "run name")
    checkpoint_root = DEV_STAGE / "checkpoints"
    root = checkpoint_root / run_name
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise RuntimeError(f"{run_name} lacks checkpoint tracker: {tracker}")
    try:
        tracked_step = int(tracker.read_text().strip())
    except ValueError as error:
        raise RuntimeError(f"{run_name} has an invalid checkpoint tracker: {tracker}") from error
    steps = tuple(
        sorted(
            int(path.name.removeprefix("iter_"))
            for path in root.glob("iter_[0-9]*")
            if path.name.removeprefix("iter_").isdigit() and (path / ".metadata").is_file()
        )
    )
    if not steps:
        raise RuntimeError(f"{run_name} has no iter_*/.metadata checkpoints under {root}")
    if tracked_step > steps[-1]:
        raise RuntimeError(f"{run_name} tracker={tracked_step}, but latest complete checkpoint is {steps[-1]}")
    if tracked_step < steps[-1]:
        print(
            f"[discover] {run_name} tracker={tracked_step} lags complete checkpoint "
            f"{steps[-1]}; --ckpt-step will select it explicitly",
            flush=True,
        )
    return RunConfig(name=run_name, alias=f"r{index:02d}", steps=steps)


def extend_config(config: SweepConfig) -> tuple[SweepConfig, dict[str, list[int]]]:
    runs = []
    additions = {}
    for index, run in enumerate(config.runs, start=1):
        discovered = discover_run(run.name, index)
        if discovered.steps[-1] < run.steps[-1]:
            raise RuntimeError(
                f"{run.name} latest checkpoint moved backwards from "
                f"{run.steps[-1]} to {discovered.steps[-1]}"
            )
        new_steps = sorted(set(discovered.steps) - set(run.steps))
        if new_steps:
            additions[run.name] = new_steps
        runs.append(
            RunConfig(
                name=run.name,
                alias=run.alias,
                steps=tuple(sorted(set(run.steps) | set(discovered.steps))),
            )
        )
    return replace(config, runs=tuple(runs)), additions


def validate_checkpoints(config: SweepConfig) -> None:
    checkpoint_root = DEV_STAGE / "checkpoints"
    names = [run.name for run in config.runs]
    if len(names) != len(set(names)):
        raise RuntimeError(f"duplicate run names in sweep config: {names}")
    for run in config.runs:
        root = checkpoint_root / run.name
        tracker = root / "latest_checkpointed_iteration.txt"
        if not tracker.is_file():
            raise RuntimeError(f"{run.name} lacks checkpoint tracker: {tracker}")
        missing = [step for step in run.steps if not (root / f"iter_{step:07d}" / ".metadata").is_file()]
        if missing:
            raise RuntimeError(f"{run.name} is missing configured checkpoints: {missing}")


def _make_config(
    args: argparse.Namespace,
    runs: tuple[RunConfig, ...],
) -> SweepConfig:
    config = SweepConfig(
        runs=runs,
        model_size=(getattr(args, "model_size", None) or infer_model_size(runs)).upper(),
        base_source_batch=args.reuse_base_from,
        expected_questions=args.expected_questions,
        samples_per_question=args.eval_n,
        rollout_seed=args.eval_seed,
        search_concurrency=args.search_concurrency,
        judge_concurrency=args.judge_concurrency,
        local_report_root=args.local_report_root,
    )
    if config.model_size not in {"4B", "9B"}:
        raise RuntimeError(f"unsupported model size: {config.model_size}")
    for name, value in (
        ("expected_questions", config.expected_questions),
        ("samples_per_question", config.samples_per_question),
        ("search_concurrency", config.search_concurrency),
        ("judge_concurrency", config.judge_concurrency),
    ):
        if value <= 0:
            raise RuntimeError(f"{name} must be positive, got {value}")
    validate_checkpoints(config)
    return config


def _config_from_state(args: argparse.Namespace, state: dict[str, Any]) -> SweepConfig:
    run_steps: dict[str, list[int]] = {}
    for key in state.get("jobs", {}):
        if key == "base" or "/" not in key:
            continue
        run_name, point = key.rsplit("/", 1)
        if not point.startswith("iter") or not point.removeprefix("iter").isdigit():
            continue
        step = int(point.removeprefix("iter"))
        run_steps.setdefault(run_name, []).append(step)
    if not run_steps:
        raise RuntimeError("batch has no sweep_config.json; pass at least one --run to initialize it")
    runs = tuple(
        RunConfig(name=run_name, alias=f"r{index:02d}", steps=tuple(sorted(set(steps))))
        for index, (run_name, steps) in enumerate(run_steps.items(), start=1)
    )
    return _make_config(args, runs)


def load_or_create_config(
    args: argparse.Namespace, batch_root: Path, state: dict[str, Any]
) -> SweepConfig:
    path = batch_root / "sweep_config.json"
    if path.is_file():
        config = SweepConfig.from_dict(json.loads(path.read_text()))
        requested_runs = tuple(args.run_names or ())
        configured_runs = tuple(run.name for run in config.runs)
        if requested_runs and requested_runs != configured_runs:
            raise RuntimeError(
                f"batch is already configured for {configured_runs}; requested {requested_runs}"
            )
        if args.reuse_base_from and args.reuse_base_from != config.base_source_batch:
            raise RuntimeError(
                f"batch already uses base source {config.base_source_batch!r}; "
                f"requested {args.reuse_base_from!r}"
            )
        requested_model_size = getattr(args, "model_size", None)
        if requested_model_size and requested_model_size.upper() != config.model_size:
            raise RuntimeError(
                f"batch already uses model size {config.model_size}; requested {requested_model_size}"
            )
        if getattr(args, "extend_checkpoints", False):
            config, additions = extend_config(config)
            if additions:
                _write_json(path, config.to_dict())
                print(f"[extend] added checkpoints: {additions}", flush=True)
            else:
                print("[extend] no new complete checkpoints", flush=True)
        if args.command in {"dry-run", "orchestrate", "retry-point"}:
            validate_checkpoints(config)
        if args.command != "status":
            validate_base_source(config, batch_root)
        return config

    if getattr(args, "extend_checkpoints", False):
        raise RuntimeError("--extend-checkpoints requires an existing batch configuration")
    if args.run_names:
        if len(args.run_names) != len(set(args.run_names)):
            raise RuntimeError(f"duplicate --run values: {args.run_names}")
        runs = tuple(discover_run(run_name, index) for index, run_name in enumerate(args.run_names, start=1))
        config = _make_config(args, runs)
    else:
        config = _config_from_state(args, state)
    validate_base_source(config, batch_root)
    _write_json(path, config.to_dict())
    return config


def stage_code(repo_root: Path, batch_root: Path) -> tuple[Path, str]:
    archive_dir = DEV_STAGE / "eval-code"
    archive_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_dir:
        candidate = Path(temporary_dir) / "eval-code.tgz"
        command = [
            "tar",
            "-czf",
            str(candidate),
            "--exclude=.git",
            "--exclude=.pytest_cache",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=examples/supo_browsecomp/mast/wandb_ray_probe.py",
            "-C",
            str(repo_root),
            ".",
        ]
        result = _run(command)
        if result.returncode:
            raise RuntimeError(f"failed to build eval archive: {result.stderr}")
        digest = sha256(candidate)
        destination = archive_dir / f"eval-code-{digest[:16]}.tgz"
        if not destination.exists():
            temporary = destination.with_suffix(".tmp")
            shutil.copyfile(candidate, temporary)
            temporary.replace(destination)
    _write_json(batch_root / "code.json", {"archive": str(destination), "sha256": digest})
    return destination, digest


def build_mast_command(
    *,
    config: SweepConfig,
    batch_id: str,
    batch_root: Path,
    archive: Path,
    archive_sha256: str,
    point: dict[str, Any],
    dry_run: bool,
) -> list[str]:
    validate_safe_name(batch_id, "batch id")
    validate_safe_name(str(point["run_name"]), "run name")
    destination = output_dir(batch_root, point)
    alias = "base" if point["point"] == "base" else f"{config.run(point['run_name']).alias}-{point['point']}"
    batch_tag = "".join(character if character.isalnum() else "-" for character in batch_id).strip("-")[-12:]
    job_name = f"bcplus-eval-{batch_tag}-{alias}"[:120]
    custom_command = " && ".join(
        [
            "mkdir -p /slime-src",
            f"tar xzf {shlex.quote(_mast_path(archive))} -C /slime-src",
            "cd /slime-src",
            "export "
            + " ".join(
                [
                    f"{name}={shlex.quote(str(value))}"
                    for name, value in (
                        ("EVAL_RUN_NAME", point["run_name"]),
                        ("EVAL_POINT", point["point"]),
                        ("EVAL_REQUESTED_STEP", point["step"]),
                        ("EVAL_OUTPUT_DIR", _mast_path(destination)),
                        ("EVAL_CODE_ARCHIVE_SHA256", archive_sha256),
                        ("EVAL_N", config.samples_per_question),
                        ("EVAL_SEED", config.rollout_seed),
                        ("EVAL_EXPECTED_QUESTIONS", config.expected_questions),
                        ("BC_MODEL_SIZE", config.model_size),
                        ("BCPLUS_SEARCH_CONCURRENCY", config.search_concurrency),
                        ("BCPLUS_JUDGE_CONCURRENCY", config.judge_concurrency),
                        ("TORCH_NCCL_DUMP_ON_TIMEOUT", 0),
                    )
                ]
            ),
            "if [ -f /slime-src/examples/supo_browsecomp/mast/eval/run_eval.sh ]; then "
            "bash /slime-src/examples/supo_browsecomp/mast/eval/run_eval.sh; else "
            "bash /slime-src/examples/supo_browsecomp/mast/run_eval.sh; fi",
        ]
    )
    command = [
        str(CLI),
        "mast",
        "--json",
        "--tenant=rhea_assistant_interns",
        "--region=nha",
        "--job_priority=HIGH",
        "--workspace=None",
        "--main_package=xlformers_pretrain1:latest",
        "program",
        "avocado.rev1.rl.debug_80m",
        "--roles=trainer_0",
        f"--job_name={job_name}",
        "--enable_ttls=True",
        "--retries=3",
        "--use_conda_docker=True",
        f"--conda_docker_image={IMAGE}",
        "--docker_host_cmd=sh -c 'nohup python3 /mnt/wsfuse/hhzhang01/slime-sanity/connect_proxy.py 9080 >/tmp/relay.log 2>&1 &'",
        f"--docker_custom_cmd={custom_command}",
        "--host=zionex_80g",
        f"--wsf_src={WSF_SRC}",
        "--overrides=cluster_config.trainer_parallelism.data_parallel_size=1,cluster_config.trainer_parallelism.context_parallel_size=1",
    ]
    if dry_run:
        command.append("--dryrun")
    return command


def parse_submission(stdout: str) -> dict[str, Any]:
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"MAST submission did not return JSON: {stdout[-1000:]}") from error
    if response.get("status") != "ok":
        raise RuntimeError(f"MAST submission failed: {response}")
    return response


def submit_point(
    *,
    config: SweepConfig,
    state: dict[str, Any],
    state_path: Path,
    batch_id: str,
    batch_root: Path,
    archive: Path,
    archive_sha256: str,
    point: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    key = point["key"]
    destination = output_dir(batch_root, point)
    if (destination / "_SUCCESS").is_file():
        state.setdefault("jobs", {})[key] = {"state": "COMPLETE", "output": str(destination), "skipped": True}
        _write_json(state_path, state)
        return state["jobs"][key]
    existing = state.setdefault("jobs", {}).get(key)
    if existing and existing.get("job_name") and not dry_run:
        return existing

    command = build_mast_command(
        config=config,
        batch_id=batch_id,
        batch_root=batch_root,
        archive=archive,
        archive_sha256=archive_sha256,
        point=point,
        dry_run=dry_run,
    )
    result = _run(command)
    if result.returncode:
        raise RuntimeError(f"submission command failed rc={result.returncode}: {result.stderr[-3000:]}")
    response = parse_submission(result.stdout)
    if dry_run:
        return response
    job = response.get("job") or {}
    job_name = job.get("job_name")
    if not job_name:
        raise RuntimeError(f"MAST submission response lacks job.job_name: {response}")
    record = {
        "job_name": job_name,
        "mast_url": job.get("mast_url"),
        "state": "SUBMITTED",
        "output": str(destination),
        "submitted_at": time.time(),
    }
    state["jobs"][key] = record
    _write_json(state_path, state)
    print(f"[submit] {key} -> {job_name}", flush=True)
    return record


def mast_status(job_name: str) -> str:
    result = _run(["with-proxy", "mast", "--output", "json", "get-status", job_name])
    if result.returncode:
        return "STATUS_ERROR"
    try:
        return str(json.loads(result.stdout)["data"]["state"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return "STATUS_ERROR"


def search_stats() -> dict[str, Any]:
    address_file = DEV_STAGE / "search-server.addr"
    address = address_file.read_text().strip()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{address}/stats", timeout=5) as response:
        return json.load(response)


def update_statuses(state: dict[str, Any], state_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in state.get("jobs", {}).values():
        if record.get("skipped") or (Path(record["output"]) / "_SUCCESS").is_file():
            status = "COMPLETE"
        else:
            status = mast_status(record["job_name"])
        record["state"] = status
        record["last_checked_at"] = time.time()
        counts[status] += 1
    _write_json(state_path, state)
    return counts


def wait_for_keys(state: dict[str, Any], state_path: Path, keys: list[str], poll_seconds: int) -> None:
    while True:
        counts = update_statuses(state, state_path)
        stats = search_stats()
        queue_sizes = stats.get("queue_sizes", {})
        print(
            f"[monitor] jobs={dict(counts)} pending={stats.get('pending')} queues={queue_sizes}",
            flush=True,
        )
        selected = [state["jobs"][key] for key in keys]
        failed = [record for record in selected if record.get("state") in FAILED_STATES]
        if failed:
            raise RuntimeError(f"MAST eval jobs failed: {failed}")
        if all(record.get("state") == "COMPLETE" for record in selected):
            missing = [record["output"] for record in selected if not (Path(record["output"]) / "_SUCCESS").is_file()]
            if missing:
                time.sleep(30)
                missing = [path for path in missing if not (Path(path) / "_SUCCESS").is_file()]
            if missing:
                raise RuntimeError(f"jobs completed without validated _SUCCESS: {missing}")
            return
        time.sleep(poll_seconds)


def health_gate(samples: int = 3, interval: int = 30) -> None:
    pending_values = []
    for index in range(samples):
        stats = search_stats()
        pending = int(stats.get("pending", 0))
        pending_values.append(pending)
        print(f"[health] {stats}", flush=True)
        if index + 1 < samples:
            time.sleep(interval)
    if all(value > 1500 for value in pending_values):
        raise RuntimeError(f"search pending remained above 1500: {pending_values}")


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def sync_local_reports(config: SweepConfig, batch_root: Path) -> Path:
    local_root = Path(config.local_report_root).expanduser() / batch_root.name
    for name in ("sweep_config.json", "sweep_state.json"):
        _copy_atomic(batch_root / name, local_root / name)
    for run in config.runs:
        source_root = batch_root / "runs" / run.name
        destination_root = local_root / run.name
        for name in REPORT_ARTIFACTS:
            _copy_atomic(source_root / name, destination_root / name)
    print(f"[local-report] {local_root}", flush=True)
    return local_root


def create_reports(config: SweepConfig, repo_root: Path, batch_root: Path) -> None:
    pipeline = repo_root / "examples/supo_browsecomp/eval/eval_pipeline.py"
    for run in config.runs:
        run_root = batch_root / "runs" / run.name
        command = [
            "python3",
            str(pipeline),
            "report",
            "--run-root",
            str(run_root),
            "--base-root",
            str(base_output_dir(config, batch_root)),
            "--run-name",
            run.name,
            "--output-dir",
            str(run_root),
        ]
        result = _run(command)
        if result.returncode:
            raise RuntimeError(f"report failed for {run.name}: {result.stderr}")
        print(f"[report] {run_root / 'report.md'}", flush=True)
    sync_local_reports(config, batch_root)


def prepare(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any], SweepConfig, Path, str]:
    repo_root = Path(args.repo_root).resolve()
    batch_id = args.batch_id or time.strftime("bcplus-4b-eval-%Y%m%d-%H%M%S")
    validate_safe_name(batch_id, "batch id")
    if args.reuse_base_from:
        validate_safe_name(args.reuse_base_from, "base source batch")
    batch_root = DEV_STAGE / "evals" / batch_id
    batch_root.mkdir(parents=True, exist_ok=True)
    state_path = batch_root / "sweep_state.json"
    state = _read_json(state_path, {"batch_id": batch_id, "jobs": {}, "created_at": time.time()})
    config = load_or_create_config(args, batch_root, state)
    code = _read_json(batch_root / "code.json", None)
    if code:
        archive, archive_sha256 = Path(code["archive"]), code["sha256"]
    else:
        archive, archive_sha256 = stage_code(repo_root, batch_root)
    return repo_root, batch_root, state, config, archive, archive_sha256


def orchestrate(args: argparse.Namespace) -> None:
    repo_root, batch_root, state, config, archive, archive_sha256 = prepare(args)
    state_path = batch_root / "sweep_state.json"
    batch_id = state["batch_id"]
    points = sweep_points(config)
    unrecorded = [point for point in points if point["key"] not in state.get("jobs", {})]
    if any(not (output_dir(batch_root, point) / "_SUCCESS").is_file() for point in unrecorded):
        health_gate(samples=1, interval=0)
    for point in unrecorded:
        submit_point(
            config=config,
            state=state,
            state_path=state_path,
            batch_id=batch_id,
            batch_root=batch_root,
            archive=archive,
            archive_sha256=archive_sha256,
            point=point,
        )

    wait_for_keys(state, state_path, [point["key"] for point in points], args.poll_seconds)
    first_checkpoint = next(point for point in points if point["point"] != "base")
    validate_base_against_point(config, batch_root, first_checkpoint)
    create_reports(config, repo_root, batch_root)
    print(f"[done] batch={batch_id} root={batch_root}", flush=True)


def status_command(args: argparse.Namespace) -> None:
    _, batch_root, state, _, _, _ = prepare(args)
    counts = update_statuses(state, batch_root / "sweep_state.json")
    print(json.dumps({"batch_root": str(batch_root), "jobs": dict(counts), "search": search_stats()}, indent=2))


def report_command(args: argparse.Namespace) -> None:
    repo_root, batch_root, _, config, _, _ = prepare(args)
    create_reports(config, repo_root, batch_root)


def retry_point_command(args: argparse.Namespace) -> None:
    repo_root, batch_root, state, config, _, _ = prepare(args)
    point_by_key = {point["key"]: point for point in sweep_points(config)}
    if args.key not in point_by_key:
        raise RuntimeError(f"unknown eval point key: {args.key}")
    existing = state.get("jobs", {}).get(args.key)
    if not existing or not existing.get("job_name"):
        raise RuntimeError(f"no prior submitted job for {args.key}")
    status = mast_status(existing["job_name"])
    if status not in FAILED_STATES:
        raise RuntimeError(f"refusing to retry {args.key}: existing job is {status}")
    destination = output_dir(batch_root, point_by_key[args.key])
    if (destination / "rollout_data/eval_0.pt").is_file():
        raise RuntimeError(f"refusing to overwrite an existing eval dump: {destination}")

    archive, archive_sha256 = stage_code(repo_root, batch_root)
    state["jobs"].pop(args.key)
    state_path = batch_root / "sweep_state.json"
    _write_json(state_path, state)
    submit_point(
        config=config,
        state=state,
        state_path=state_path,
        batch_id=state["batch_id"],
        batch_root=batch_root,
        archive=archive,
        archive_sha256=archive_sha256,
        point=point_by_key[args.key],
    )


def dry_run_command(args: argparse.Namespace) -> None:
    _, batch_root, state, config, archive, archive_sha256 = prepare(args)
    point = sweep_points(config)[0]
    response = submit_point(
        config=config,
        state=state,
        state_path=batch_root / "sweep_state.json",
        batch_id=state["batch_id"],
        batch_root=batch_root,
        archive=archive,
        archive_sha256=archive_sha256,
        point=point,
        dry_run=True,
    )
    print(json.dumps(response, indent=2))


def dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command == "dry-run":
        dry_run_command(args)
    elif args.command == "orchestrate":
        orchestrate(args)
    elif args.command == "retry-point":
        if not args.batch_id or not args.key:
            parser.error("retry-point requires --batch-id and --key")
        retry_point_command(args)
    elif args.command == "status":
        if not args.batch_id:
            parser.error("status requires --batch-id")
        status_command(args)
    else:
        if not args.batch_id:
            parser.error("report requires --batch-id")
        report_command(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("dry-run", "orchestrate", "retry-point", "status", "report")
    )
    parser.add_argument("--run", dest="run_names", action="append", help="MAST run name; repeat for multiple runs")
    parser.add_argument("--batch-id", help="persistent batch id; generated automatically for a new sweep")
    parser.add_argument(
        "--model-size",
        type=str.upper,
        choices=("4B", "9B"),
        help="model family for this batch; inferred from new run names when omitted",
    )
    parser.add_argument(
        "--extend-checkpoints",
        action="store_true",
        help="append newly completed checkpoints to an existing batch",
    )
    parser.add_argument("--key", help="point key for retry-point, e.g. RUN/iter39")
    parser.add_argument(
        "--reuse-base-from",
        help="reuse the validated base artifacts from this earlier eval batch",
    )
    parser.add_argument("--expected-questions", type=int, default=150)
    parser.add_argument("--eval-n", type=int, default=4, help="deterministic samples per question")
    parser.add_argument("--eval-seed", type=int, default=42, help="first deterministic sampling seed")
    parser.add_argument("--search-concurrency", type=int, default=64)
    parser.add_argument("--judge-concurrency", type=int, default=16)
    parser.add_argument("--local-report-root", default="/home/hhzhang01/bcplus-eval-reports")
    parser.add_argument("--repo-root", default=str(Path(__file__).parents[4]))
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.extend_checkpoints and (args.command != "orchestrate" or not args.batch_id):
        parser.error("--extend-checkpoints requires orchestrate with --batch-id")
    if args.command in {"dry-run", "orchestrate"} and not args.batch_id and not args.run_names:
        parser.error(f"{args.command} requires at least one --run when starting a new batch")
    if args.command in {"retry-point", "status", "report"} and not args.batch_id:
        parser.error(f"{args.command} requires --batch-id")
    if args.command == "retry-point" and not args.key:
        parser.error("retry-point requires --key")
    if args.batch_id is None:
        args.batch_id = time.strftime("bcplus-4b-eval-%Y%m%d-%H%M%S")
    with controller_lock(DEV_STAGE / "evals" / args.batch_id):
        dispatch(args, parser)


if __name__ == "__main__":
    main()
