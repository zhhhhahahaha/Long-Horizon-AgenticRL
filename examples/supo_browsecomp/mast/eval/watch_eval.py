#!/usr/bin/env python3
"""Watch a live MAST training run and extend its checkpoint eval batch."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import eval_sweep

TERMINAL_TRAINING_STATES = {"COMPLETE", "FAILED", "DEAD"}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}", flush=True)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def watcher_lock(batch_root: Path):
    lock_path = batch_root / ".watcher.lock"
    batch_root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise RuntimeError(f"eval watcher is already running: {owner}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def configured_steps(config_path: Path, run_name: str) -> tuple[int, ...]:
    if not config_path.is_file():
        return ()
    config = json.loads(config_path.read_text())
    matches = [run for run in config.get("runs", []) if run.get("name") == run_name]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one config entry for {run_name}, found {len(matches)}")
    return tuple(sorted(map(int, matches[0].get("steps", []))))


def controller_command(
    args: argparse.Namespace,
    command: str,
    *,
    initialize: bool = False,
    extend: bool = False,
) -> list[str]:
    result = [
        sys.executable,
        str(Path(__file__).with_name("eval_sweep.py")),
        command,
        "--batch-id",
        args.batch_id,
        "--repo-root",
        str(args.repo_root),
        "--poll-seconds",
        str(args.eval_poll_seconds),
    ]
    if initialize:
        result.extend(["--run", args.run, "--eval-config", str(args.eval_config)])
    if extend:
        result.append("--extend-checkpoints")
    if command == "orchestrate":
        result.append("--live-extend-checkpoints")
    return result


def run_controller(
    args: argparse.Namespace,
    command: str,
    *,
    initialize: bool = False,
    extend: bool = False,
) -> None:
    controller_args = controller_command(args, command, initialize=initialize, extend=extend)
    log(f"controller start: {' '.join(controller_args)}")
    result = subprocess.run(controller_args, check=False)
    if result.returncode:
        raise RuntimeError(f"eval controller failed with exit code {result.returncode}")
    log(f"controller complete: {command}")


def watcher_state(
    args: argparse.Namespace,
    *,
    status: str,
    training_state: str,
    observed_steps: tuple[int, ...],
    configured: tuple[int, ...],
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "status": status,
        "run": args.run,
        "training_job": args.training_job,
        "training_state": training_state,
        "batch_id": args.batch_id,
        "observed_steps": list(observed_steps),
        "configured_steps": list(configured),
        "report": str(args.batch_root / "runs" / args.run / "report.md"),
        "error": error,
        "updated_at": time.time(),
    }


def watch(args: argparse.Namespace) -> int:
    config_path = args.batch_root / "sweep_config.json"
    state_path = args.batch_root / "watch_state.json"
    with watcher_lock(args.batch_root):
        try:
            if not config_path.is_file():
                log(f"initializing batch={args.batch_id} run={args.run}")
                run_controller(args, "dry-run", initialize=True)

            observed = eval_sweep.discover_run(args.run, 1).steps
            configured = configured_steps(config_path, args.run)
            additions = sorted(set(observed) - set(configured))
            write_json(
                state_path,
                watcher_state(
                    args,
                    status="EVALUATING",
                    training_state=eval_sweep.mast_status(args.training_job),
                    observed_steps=observed,
                    configured=configured,
                ),
            )
            run_controller(args, "orchestrate", extend=bool(additions))
            configured = configured_steps(config_path, args.run)

            terminal_stable_polls = 0
            while True:
                training_state = eval_sweep.mast_status(args.training_job)
                try:
                    observed = eval_sweep.discover_run(args.run, 1).steps
                except RuntimeError as error:
                    log(f"checkpoint discovery is temporarily inconsistent: {error}")
                    write_json(
                        state_path,
                        watcher_state(
                            args,
                            status="WAITING_CHECKPOINT",
                            training_state=training_state,
                            observed_steps=(),
                            configured=configured,
                            error=str(error),
                        ),
                    )
                    terminal_stable_polls = 0
                    time.sleep(args.checkpoint_poll_seconds)
                    continue

                additions = sorted(set(observed) - set(configured))
                if additions:
                    log(f"new complete checkpoints: {additions}")
                    write_json(
                        state_path,
                        watcher_state(
                            args,
                            status="EVALUATING",
                            training_state=training_state,
                            observed_steps=observed,
                            configured=configured,
                        ),
                    )
                    run_controller(args, "orchestrate", extend=True)
                    configured = configured_steps(config_path, args.run)

                if training_state in TERMINAL_TRAINING_STATES and observed == configured:
                    terminal_stable_polls += 1
                else:
                    terminal_stable_polls = 0

                status = "TRAINING_TERMINAL" if training_state in TERMINAL_TRAINING_STATES else "WATCHING"
                write_json(
                    state_path,
                    watcher_state(
                        args,
                        status=status,
                        training_state=training_state,
                        observed_steps=observed,
                        configured=configured,
                    ),
                )
                log(
                    f"training={training_state} latest={observed[-1]} "
                    f"evaluated={configured[-1]} terminal_stable_polls={terminal_stable_polls}"
                )

                if terminal_stable_polls >= args.terminal_stable_polls:
                    final_status = "COMPLETE" if training_state == "COMPLETE" else "TRAINING_FAILED"
                    write_json(
                        state_path,
                        watcher_state(
                            args,
                            status=final_status,
                            training_state=training_state,
                            observed_steps=observed,
                            configured=configured,
                        ),
                    )
                    log(
                        f"watcher finished: status={final_status} report={args.batch_root / 'runs' / args.run / 'report.md'}"
                    )
                    return 0 if final_status == "COMPLETE" else 2

                time.sleep(args.checkpoint_poll_seconds)
        except Exception as error:
            configured = configured_steps(config_path, args.run) if config_path.is_file() else ()
            write_json(
                state_path,
                watcher_state(
                    args,
                    status="FAILED",
                    training_state=eval_sweep.mast_status(args.training_job),
                    observed_steps=(),
                    configured=configured,
                    error=str(error),
                ),
            )
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="checkpoint run directory name")
    parser.add_argument("--training-job", help="full MAST training job name; defaults to --run")
    parser.add_argument("--batch-id", required=True, help="persistent eval batch id")
    parser.add_argument("--eval-config", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[4])
    parser.add_argument("--checkpoint-poll-seconds", type=int, default=300)
    parser.add_argument("--eval-poll-seconds", type=int, default=120)
    parser.add_argument("--terminal-stable-polls", type=int, default=2)
    args = parser.parse_args()
    args.training_job = args.training_job or args.run
    args.repo_root = args.repo_root.resolve()
    args.eval_config = args.eval_config.resolve()
    args.batch_root = eval_sweep.DEV_STAGE / "evals" / args.batch_id
    eval_sweep.validate_safe_name(args.run, "run name")
    eval_sweep.validate_safe_name(args.training_job, "training job name")
    eval_sweep.validate_safe_name(args.batch_id, "batch id")
    if not args.eval_config.is_file():
        parser.error(f"eval config does not exist: {args.eval_config}")
    for name in ("checkpoint_poll_seconds", "eval_poll_seconds", "terminal_stable_polls"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(watch(parse_args()))
