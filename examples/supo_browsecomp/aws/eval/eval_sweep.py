#!/usr/bin/env python3
"""Discover BC+ checkpoints and run an incremental AWS Slurm eval sweep."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


STAGE_ROOT = Path("/genai/fsx-llm/interns/hhzhang01")
CHECKPOINT_ROOT = STAGE_ROOT / "checkpoints"
EVAL_ROOT = STAGE_ROOT / "evals"
CODE_ROOT = STAGE_ROOT / "eval-code"
SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+")
TERMINAL_SLURM_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}
FAILED_SLURM_STATES = TERMINAL_SLURM_STATES - {"COMPLETED"}


def validate_safe_name(value: str, label: str) -> str:
    if SAFE_NAME.fullmatch(value) is None or value in {".", ".."}:
        raise RuntimeError(f"invalid {label}: {value!r}")
    return value


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


@dataclass(frozen=True)
class SweepConfig:
    run_name: str
    steps: tuple[int, ...] = ()
    model_size: str = "4B"
    expected_questions: int = 150
    samples_per_question: int = 4
    rollout_seed: int = 42
    fixed_search_topk: int | None = 5
    doc_words_full: int = 10000
    search_concurrency: int = 64
    judge_concurrency: int = 16
    dev_qos: str = "a100_dev"
    shared_qos: str = "a100_genai_shared"
    shared_lanes: int = 2
    walltime: str = "12:00:00"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SweepConfig:
        evaluation = value["evaluation"]
        scheduling = value["scheduling"]
        model = value.get("model", {})
        return cls(
            run_name=str(value["run_name"]),
            steps=tuple(map(int, value.get("steps", ()))),
            model_size=str(model.get("size", "4B")).upper(),
            expected_questions=int(evaluation.get("expected_questions", 150)),
            samples_per_question=int(evaluation.get("samples_per_question", 4)),
            rollout_seed=int(evaluation.get("rollout_seed", 42)),
            fixed_search_topk=evaluation.get("fixed_search_topk"),
            doc_words_full=int(evaluation.get("doc_words_full", 4096)),
            search_concurrency=int(evaluation.get("search_concurrency", 64)),
            judge_concurrency=int(evaluation.get("judge_concurrency", 16)),
            dev_qos=str(scheduling.get("dev_qos", "a100_dev")),
            shared_qos=str(scheduling.get("shared_qos", "a100_genai_shared")),
            shared_lanes=int(scheduling.get("shared_lanes", 2)),
            walltime=str(scheduling.get("walltime", "12:00:00")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "run_name": self.run_name,
            "steps": list(self.steps),
            "model": {"size": self.model_size, "name": f"Qwen3.5-{self.model_size}"},
            "evaluation": {
                "expected_questions": self.expected_questions,
                "samples_per_question": self.samples_per_question,
                "rollout_seed": self.rollout_seed,
                "sampling_seeds": list(
                    range(self.rollout_seed, self.rollout_seed + self.samples_per_question)
                ),
                "fixed_search_topk": self.fixed_search_topk,
                "doc_words_full": self.doc_words_full,
                "search_concurrency": self.search_concurrency,
                "judge_concurrency": self.judge_concurrency,
            },
            "scheduling": {
                "dev_qos": self.dev_qos,
                "shared_qos": self.shared_qos,
                "shared_lanes": self.shared_lanes,
                "walltime": self.walltime,
            },
        }


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
            raise RuntimeError(f"batch controller is already running: {batch_root.name}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_signature(path: Path) -> dict[str, int]:
    total_size = 0
    file_count = 0
    latest_mtime_ns = 0
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        stat = candidate.stat()
        total_size += stat.st_size
        file_count += 1
        latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
    return {
        "total_size": total_size,
        "file_count": file_count,
        "latest_mtime_ns": latest_mtime_ns,
    }


def discover_stable_steps(
    config: SweepConfig,
    state: dict[str, Any],
    *,
    stability_seconds: int,
    now: float | None = None,
) -> tuple[int, tuple[int, ...]]:
    now = time.time() if now is None else now
    root = CHECKPOINT_ROOT / config.run_name
    tracker_path = root / "latest_checkpointed_iteration.txt"
    if not tracker_path.is_file():
        raise RuntimeError(f"missing checkpoint tracker: {tracker_path}")
    try:
        tracker = int(tracker_path.read_text().strip())
    except ValueError as error:
        raise RuntimeError(f"invalid checkpoint tracker: {tracker_path}") from error

    observations = state.setdefault("checkpoint_observations", {})
    stable = []
    for checkpoint in sorted(root.glob("iter_[0-9]*")):
        suffix = checkpoint.name.removeprefix("iter_")
        if not suffix.isdigit() or not (checkpoint / ".metadata").is_file():
            continue
        step = int(suffix)
        if step > tracker:
            continue
        signature = checkpoint_signature(checkpoint)
        key = str(step)
        previous = observations.get(key)
        signature_matches = previous and all(
            previous.get(field) == signature[field]
            for field in ("total_size", "file_count", "latest_mtime_ns")
        )
        if signature_matches:
            unchanged_since = float(previous["unchanged_since"])
        else:
            latest_write = signature["latest_mtime_ns"] / 1_000_000_000
            unchanged_since = min(now, latest_write or now)
        observations[key] = {**signature, "unchanged_since": unchanged_since, "observed_at": now}
        if now - unchanged_since >= stability_seconds:
            stable.append(step)
    return tracker, tuple(stable)


def point_name(step: int) -> str:
    return f"iter{step:02d}"


def point_key(config: SweepConfig, step: int | None) -> str:
    return "base" if step is None else f"{config.run_name}/{point_name(step)}"


def point_output(batch_root: Path, config: SweepConfig, step: int | None) -> Path:
    if step is None:
        return batch_root / "base"
    return batch_root / "runs" / config.run_name / point_name(step)


def stage_code(repo_root: Path, batch_root: Path) -> tuple[Path, str, Path, str]:
    code_path = batch_root / "code.json"
    frozen_dir = batch_root / "frozen-code"
    existing = _read_json(code_path, None)
    if existing:
        archive = Path(existing["archive"])
        if not archive.is_file() or sha256(archive) != existing["sha256"]:
            raise RuntimeError(f"frozen code archive is missing or changed: {archive}")
        if not frozen_dir.is_dir():
            frozen_dir.mkdir(parents=True)
            result = _run(["tar", "xzf", str(archive), "-C", str(frozen_dir)])
            if result.returncode:
                raise RuntimeError(f"failed to extract frozen code: {result.stderr}")
        return archive, existing["sha256"], frozen_dir, existing["commit"]

    commit_result = _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    if commit_result.returncode:
        raise RuntimeError(commit_result.stderr)
    commit = commit_result.stdout.strip()
    CODE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=CODE_ROOT, suffix=".tgz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        result = _run(
            [
                "git",
                "-C",
                str(repo_root),
                "archive",
                "--format=tar.gz",
                f"--output={temporary}",
                "HEAD",
            ]
        )
        if result.returncode:
            raise RuntimeError(f"failed to archive code: {result.stderr}")
        digest = sha256(temporary)
        archive = CODE_ROOT / f"eval-code-{commit[:10]}-{digest[:12]}.tgz"
        if not archive.exists():
            temporary.replace(archive)
        else:
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)

    frozen_dir.mkdir(parents=True)
    result = _run(["tar", "xzf", str(archive), "-C", str(frozen_dir)])
    if result.returncode:
        raise RuntimeError(f"failed to extract frozen code: {result.stderr}")
    _write_json(code_path, {"archive": str(archive), "sha256": digest, "commit": commit})
    return archive, digest, frozen_dir, commit


def _export_value(value: Any) -> str:
    text = "" if value is None else str(value)
    if "," in text or "\n" in text:
        raise RuntimeError(f"unsafe value for sbatch --export: {text!r}")
    return text


def build_sbatch_command(
    *,
    config: SweepConfig,
    batch_id: str,
    batch_root: Path,
    runner: Path,
    archive: Path,
    archive_sha256: str,
    step: int | None,
    lane: str,
    dependency_job_id: str | None,
) -> list[str]:
    validate_safe_name(batch_id, "batch id")
    output = point_output(batch_root, config, step)
    point = "base" if step is None else point_name(step)
    requested_step: str | int = "base" if step is None else step
    qos = config.dev_qos if lane == "dev" else config.shared_qos
    job_tag = re.sub(r"[^A-Za-z0-9]+", "-", batch_id).strip("-")[-14:]
    job_name = f"bc-eval-{job_tag}-{point}"[:120]
    exports = {
        "EVAL_RUN_NAME": "shared-base" if step is None else config.run_name,
        "EVAL_POINT": point,
        "EVAL_REQUESTED_STEP": requested_step,
        "EVAL_OUTPUT_HOST": output,
        "EVAL_CODE_ARCHIVE": archive,
        "EVAL_CODE_ARCHIVE_SHA256": archive_sha256,
        "EVAL_N": config.samples_per_question,
        "EVAL_SEED": config.rollout_seed,
        "EVAL_EXPECTED_QUESTIONS": config.expected_questions,
        "BC_MODEL_SIZE": config.model_size,
        "BCPLUS_FIXED_SEARCH_TOPK": config.fixed_search_topk,
        "BCPLUS_DOC_WORDS_FULL": config.doc_words_full,
        "BCPLUS_SEARCH_CONCURRENCY": config.search_concurrency,
        "BCPLUS_JUDGE_CONCURRENCY": config.judge_concurrency,
    }
    export_arg = "--export=ALL," + ",".join(
        f"{name}={_export_value(value)}" for name, value in exports.items()
    )
    command = [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--gpus-per-node=8",
        "--ntasks-per-node=1",
        "--exclusive",
        "--cpus-per-task=64",
        "--mem=0",
        "--account=genai_interns",
        f"--qos={qos}",
        f"--time={config.walltime}",
        f"--job-name={job_name}",
        f"--output={output / 'slurm.log'}",
        export_arg,
    ]
    if dependency_job_id:
        command.append(f"--dependency=afterany:{dependency_job_id}")
    command.append(str(runner))
    return command


def slurm_state(job_id: str) -> str:
    result = _run(["squeue", "-h", "-j", job_id, "-o", "%T"])
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[0].split()[0].upper()
    result = _run(["sacct", "-n", "-X", "-j", job_id, "--format=State", "-P"])
    if result.returncode:
        return "STATUS_ERROR"
    states = [line.split("|", 1)[0].split("+", 1)[0].strip().upper() for line in result.stdout.splitlines()]
    return next((state for state in states if state), "UNKNOWN")


def update_statuses(state: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in state.get("jobs", {}).values():
        output = Path(record["output"])
        if (output / "_SUCCESS").is_file():
            status = "COMPLETE"
        else:
            status = slurm_state(str(record["job_id"]))
            if status == "COMPLETED":
                status = "COMPLETED_NO_SUCCESS"
        record["state"] = status
        record["last_checked_at"] = time.time()
        counts[status] = counts.get(status, 0) + 1
    return counts


def _outstanding_jobs(state: dict[str, Any]) -> int:
    terminal = FAILED_SLURM_STATES | {"COMPLETE", "COMPLETED_NO_SUCCESS"}
    return sum(record.get("state") not in terminal for record in state.get("jobs", {}).values())


def submit_point(
    *,
    config: SweepConfig,
    state: dict[str, Any],
    state_path: Path,
    batch_id: str,
    batch_root: Path,
    runner: Path,
    archive: Path,
    archive_sha256: str,
    step: int | None,
    lane: str,
) -> None:
    key = point_key(config, step)
    if key in state.setdefault("jobs", {}):
        return
    output = point_output(batch_root, config, step)
    output.mkdir(parents=True, exist_ok=True)
    lane_tails = state.setdefault("lane_tails", {})
    dependency = lane_tails.get(lane)
    command = build_sbatch_command(
        config=config,
        batch_id=batch_id,
        batch_root=batch_root,
        runner=runner,
        archive=archive,
        archive_sha256=archive_sha256,
        step=step,
        lane=lane,
        dependency_job_id=dependency,
    )
    result = _run(command)
    if result.returncode:
        raise RuntimeError(f"sbatch failed for {key}: {result.stderr.strip()}")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch returned an invalid job id for {key}: {result.stdout!r}")
    record = {
        "job_id": job_id,
        "lane": lane,
        "qos": config.dev_qos if lane == "dev" else config.shared_qos,
        "state": "SUBMITTED",
        "output": str(output),
        "submitted_at": time.time(),
        "dependency_job_id": dependency,
    }
    state["jobs"][key] = record
    lane_tails[lane] = job_id
    _write_json(state_path, state)
    print(f"[submit] {key} -> {job_id} ({lane}, dependency={dependency or 'none'})", flush=True)


def schedule_new_points(
    *,
    config: SweepConfig,
    state: dict[str, Any],
    state_path: Path,
    batch_id: str,
    batch_root: Path,
    runner: Path,
    archive: Path,
    archive_sha256: str,
    max_outstanding: int,
) -> None:
    capacity = max(0, max_outstanding - _outstanding_jobs(state))
    if capacity == 0:
        return
    jobs = state.setdefault("jobs", {})
    if "base" not in jobs:
        submit_point(
            config=config,
            state=state,
            state_path=state_path,
            batch_id=batch_id,
            batch_root=batch_root,
            runner=runner,
            archive=archive,
            archive_sha256=archive_sha256,
            step=None,
            lane="dev",
        )
        capacity -= 1
    missing = [step for step in config.steps if point_key(config, step) not in jobs]
    if capacity > 0 and missing:
        latest = max(missing)
        submit_point(
            config=config,
            state=state,
            state_path=state_path,
            batch_id=batch_id,
            batch_root=batch_root,
            runner=runner,
            archive=archive,
            archive_sha256=archive_sha256,
            step=latest,
            lane="dev",
        )
        missing.remove(latest)
        capacity -= 1
    shared_count = sum(record.get("lane", "").startswith("shared-") for record in jobs.values())
    for step in missing[:capacity]:
        lane = f"shared-{shared_count % config.shared_lanes}"
        submit_point(
            config=config,
            state=state,
            state_path=state_path,
            batch_id=batch_id,
            batch_root=batch_root,
            runner=runner,
            archive=archive,
            archive_sha256=archive_sha256,
            step=step,
            lane=lane,
        )
        shared_count += 1


def create_incremental_report(
    config: SweepConfig,
    state: dict[str, Any],
    state_path: Path,
    batch_root: Path,
    frozen_dir: Path,
) -> None:
    base = state.get("jobs", {}).get("base")
    complete_steps = [
        step
        for step in config.steps
        if state.get("jobs", {}).get(point_key(config, step), {}).get("state") == "COMPLETE"
    ]
    if not base or base.get("state") != "COMPLETE" or not complete_steps:
        return
    report_signature = ["base", *map(str, complete_steps)]
    if state.get("report_steps") == report_signature:
        return
    run_root = batch_root / "runs" / config.run_name
    pipeline = frozen_dir / "examples/supo_browsecomp/eval/eval_pipeline.py"
    result = _run(
        [
            "python3",
            str(pipeline),
            "report",
            "--run-root",
            str(run_root),
            "--base-root",
            str(batch_root / "base"),
            "--run-name",
            config.run_name,
            "--output-dir",
            str(run_root),
            "--allow-partial",
        ]
    )
    if result.returncode:
        raise RuntimeError(f"incremental report failed: {result.stderr}")
    state["report_steps"] = report_signature
    state["report_updated_at"] = time.time()
    _write_json(state_path, state)
    print(f"[report] {run_root / 'report.md'} ({len(complete_steps)} checkpoints)", flush=True)


def load_or_create_config(args: argparse.Namespace, batch_root: Path) -> SweepConfig:
    path = batch_root / "sweep_config.json"
    if path.is_file():
        config = SweepConfig.from_dict(json.loads(path.read_text()))
        if args.run_name and args.run_name != config.run_name:
            raise RuntimeError(f"batch is for {config.run_name}, not {args.run_name}")
        return config
    if not args.run_name:
        raise RuntimeError("a new batch requires --run")
    validate_safe_name(args.run_name, "run name")
    config = SweepConfig(
        run_name=args.run_name,
        model_size=args.model_size,
        expected_questions=args.expected_questions,
        samples_per_question=args.eval_n,
        rollout_seed=args.eval_seed,
        fixed_search_topk=args.fixed_search_topk,
        doc_words_full=args.doc_words_full,
        search_concurrency=args.search_concurrency,
        judge_concurrency=args.judge_concurrency,
        dev_qos=args.dev_qos,
        shared_qos=args.shared_qos,
        shared_lanes=args.shared_lanes,
        walltime=args.walltime,
    )
    _write_json(path, config.to_dict())
    return config


def orchestrate(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    batch_root = EVAL_ROOT / args.batch_id
    state_path = batch_root / "sweep_state.json"
    state = _read_json(
        state_path,
        {"batch_id": args.batch_id, "jobs": {}, "created_at": time.time()},
    )
    config = load_or_create_config(args, batch_root)
    archive, archive_sha256, frozen_dir, commit = stage_code(repo_root, batch_root)
    runner = frozen_dir / "examples/supo_browsecomp/aws/eval/run_eval_job.sh"
    if not runner.is_file():
        raise RuntimeError(f"frozen commit {commit} lacks {runner.relative_to(frozen_dir)}")

    while True:
        counts = update_statuses(state)
        tracker, stable_steps = discover_stable_steps(
            config,
            state,
            stability_seconds=args.stability_seconds,
        )
        additions = sorted(set(stable_steps) - set(config.steps))
        if additions:
            config = replace(config, steps=tuple(sorted(set(config.steps) | set(additions))))
            _write_json(batch_root / "sweep_config.json", config.to_dict())
            print(f"[discover] added stable checkpoints: {additions}", flush=True)
        schedule_new_points(
            config=config,
            state=state,
            state_path=state_path,
            batch_id=args.batch_id,
            batch_root=batch_root,
            runner=runner,
            archive=archive,
            archive_sha256=archive_sha256,
            max_outstanding=args.max_outstanding,
        )
        counts = update_statuses(state)
        create_incremental_report(config, state, state_path, batch_root, frozen_dir)
        state["last_poll_at"] = time.time()
        state["tracker"] = tracker
        _write_json(state_path, state)
        print(
            f"[monitor] tracker={tracker} stable={list(stable_steps)} jobs={counts}",
            flush=True,
        )

        target_done = (
            args.target_step is not None
            and tracker >= args.target_step
            and args.target_step in config.steps
        )
        all_submitted = all(point_key(config, step) in state["jobs"] for step in config.steps)
        all_terminal = all(
            record.get("state") in (FAILED_SLURM_STATES | {"COMPLETE", "COMPLETED_NO_SUCCESS"})
            for record in state["jobs"].values()
        )
        if not args.watch or (target_done and all_submitted and all_terminal):
            break
        time.sleep(args.poll_seconds)


def status_command(args: argparse.Namespace) -> None:
    batch_root = EVAL_ROOT / args.batch_id
    state_path = batch_root / "sweep_state.json"
    state = _read_json(state_path, None)
    if state is None:
        raise RuntimeError(f"unknown eval batch: {args.batch_id}")
    counts = update_statuses(state)
    _write_json(state_path, state)
    print(json.dumps({"batch_root": str(batch_root), "jobs": counts, "state": state}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("orchestrate", "status"))
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--run", dest="run_name")
    parser.add_argument("--model-size", choices=("4B", "9B"), default="4B")
    parser.add_argument("--expected-questions", type=positive_int, default=150)
    parser.add_argument("--eval-n", type=positive_int, default=4)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--fixed-search-topk", type=positive_int, default=5)
    parser.add_argument(
        "--model-controlled-topk",
        action="store_const",
        const=None,
        dest="fixed_search_topk",
    )
    parser.add_argument("--doc-words-full", type=positive_int, default=10000)
    parser.add_argument("--search-concurrency", type=positive_int, default=64)
    parser.add_argument("--judge-concurrency", type=positive_int, default=16)
    parser.add_argument("--dev-qos", default="a100_dev")
    parser.add_argument("--shared-qos", default="a100_genai_shared")
    parser.add_argument("--shared-lanes", type=positive_int, default=2)
    parser.add_argument("--walltime", default="12:00:00")
    parser.add_argument("--max-outstanding", type=positive_int, default=9)
    parser.add_argument("--stability-seconds", type=int, default=120)
    parser.add_argument("--poll-seconds", type=positive_int, default=120)
    parser.add_argument("--target-step", type=int)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--repo-root", default=str(Path(__file__).parents[4]))
    args = parser.parse_args()
    validate_safe_name(args.batch_id, "batch id")
    if args.command == "orchestrate" and args.stability_seconds < 0:
        parser.error("--stability-seconds must be non-negative")
    batch_root = EVAL_ROOT / args.batch_id
    with controller_lock(batch_root):
        if args.command == "orchestrate":
            orchestrate(args)
        else:
            status_command(args)


if __name__ == "__main__":
    main()
