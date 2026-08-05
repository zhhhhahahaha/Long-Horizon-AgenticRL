"""CPU tests for the live MAST checkpoint evaluation watcher."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sweep = _load("eval_sweep", ROOT / "examples/supo_browsecomp/mast/eval/eval_sweep.py")
watcher = _load("bcplus_eval_watch", ROOT / "examples/supo_browsecomp/mast/eval/watch_eval.py")


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run="training-run",
        training_job="training-job-suffix",
        batch_id="eval-batch",
        eval_config=tmp_path / "fixed.json",
        repo_root=ROOT,
        checkpoint_poll_seconds=1,
        eval_poll_seconds=7,
        terminal_stable_polls=2,
        batch_root=tmp_path / "evals/eval-batch",
    )


@pytest.mark.unit
def test_configured_steps_requires_one_matching_run(tmp_path):
    path = tmp_path / "sweep_config.json"
    path.write_text(json.dumps({"runs": [{"name": "training-run", "steps": [14, 4, 9]}]}))

    assert watcher.configured_steps(path, "training-run") == (4, 9, 14)
    with pytest.raises(RuntimeError, match="exactly one config entry"):
        watcher.configured_steps(path, "different-run")


@pytest.mark.unit
def test_controller_command_separates_initialization_and_extension(tmp_path):
    args = _args(tmp_path)

    initial = watcher.controller_command(args, "dry-run", initialize=True)
    extension = watcher.controller_command(args, "orchestrate", extend=True)

    assert initial[-4:] == ["--run", "training-run", "--eval-config", str(args.eval_config)]
    assert "--extend-checkpoints" not in initial
    assert "--live-extend-checkpoints" not in initial
    assert "--extend-checkpoints" in extension
    assert extension[-1] == "--live-extend-checkpoints"
    assert "--run" not in extension
    assert extension[extension.index("--poll-seconds") + 1] == "7"


@pytest.mark.unit
def test_watcher_initializes_resumes_and_waits_for_stable_training_completion(tmp_path, monkeypatch):
    args = _args(tmp_path)
    config_path = args.batch_root / "sweep_config.json"
    calls = []

    def run_controller(_args, command, *, initialize=False, extend=False):
        calls.append((command, initialize, extend))
        if command == "dry-run":
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps({"runs": [{"name": args.run, "steps": [4]}]}))

    monkeypatch.setattr(watcher, "run_controller", run_controller)
    monkeypatch.setattr(
        watcher.eval_sweep,
        "discover_run",
        lambda run, index: watcher.eval_sweep.RunConfig(name=run, alias="r01", steps=(4,)),
    )
    monkeypatch.setattr(watcher.eval_sweep, "mast_status", lambda job: "COMPLETE")
    monkeypatch.setattr(watcher.time, "sleep", lambda seconds: None)

    assert watcher.watch(args) == 0
    assert calls == [("dry-run", True, False), ("orchestrate", False, False)]
    state = json.loads((args.batch_root / "watch_state.json").read_text())
    assert state["status"] == "COMPLETE"
    assert state["configured_steps"] == [4]
    assert state["training_state"] == "COMPLETE"


@pytest.mark.unit
def test_watcher_lock_rejects_duplicate_process(tmp_path):
    batch_root = tmp_path / "batch"

    with watcher.watcher_lock(batch_root):
        with pytest.raises(RuntimeError, match="already running"):
            with watcher.watcher_lock(batch_root):
                pass


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
