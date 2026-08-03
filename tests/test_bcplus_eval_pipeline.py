"""CPU tests for the BC+ MAST eval analyzer and sweep contract."""

from __future__ import annotations

import importlib.util
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


pipeline = _load("bcplus_eval_pipeline", ROOT / "examples/supo_browsecomp/eval/eval_pipeline.py")
sweep = _load("bcplus_eval_sweep", ROOT / "examples/supo_browsecomp/mast/eval/eval_sweep.py")

EVAL_RUNNER = ROOT / "examples/supo_browsecomp/mast/eval/run_eval.sh"
EVAL_CONFIGS = ROOT / "examples/supo_browsecomp/mast/eval/configs"


def _sweep_config(*, local_report_root="/tmp/reports", base_source_batch=None):
    runs = tuple(
        sweep.RunConfig(name=f"run-{index}", alias=f"r{index:02d}", steps=(4, 9, 14, 19, 24, 29, 34, 39))
        for index in range(1, 4)
    )
    return sweep.SweepConfig(
        runs=runs,
        base_source_batch=base_source_batch,
        local_report_root=str(local_report_root),
    )


def _sample(
    *,
    rollout_id: int,
    query_id: str,
    sub_index: int,
    total: int,
    final: bool,
    score: float = 0.0,
    judge_failed: int = 0,
    outcome: str = "finished",
    source: str = "",
    summary_length=None,
):
    metadata = {
        "query_id": query_id,
        "query": f"question {query_id}",
        "_bcplus_sibling": {
            "sub_traj_index": sub_index,
            "total_sub_trajs": total,
            "is_final": final,
        },
        "_bcplus": {
            "outcome": outcome,
            "summary_source": source,
            "summary_content_len_tokens": summary_length,
            "finished": final and outcome == "finished",
            "finish_answer": "answer" if final else "",
            "n_turns_used": 2,
            "n_search": 1,
            "n_open": 1,
            "n_bad_tool_calls": 0,
            "n_search_server_error": 0,
            "response_len_tokens": 10,
        },
    }
    return {
        "rollout_id": rollout_id,
        "index": rollout_id,
        "metadata": metadata,
        "label": "answer",
        "reward": {"score": score, "judge_failed": judge_failed},
        "loss_mask": [1, 0, 1],
        "prompt": [{"role": "user", "content": f"question {query_id}"}],
    }


@pytest.mark.unit
def test_analyzer_reconstructs_siblings_and_uses_strict_full_credit():
    samples = [
        _sample(
            rollout_id=0,
            query_id="q0",
            sub_index=0,
            total=2,
            final=False,
            outcome="compressed",
            source="extracted",
            summary_length=12,
        ),
        _sample(rollout_id=0, query_id="q0", sub_index=1, total=2, final=True, score=1.0),
        _sample(rollout_id=1, query_id="q0", sub_index=0, total=1, final=True, score=0.5, judge_failed=1),
        _sample(
            rollout_id=2,
            query_id="q1",
            sub_index=0,
            total=2,
            final=False,
            outcome="compressed",
            source="fallback",
        ),
        _sample(rollout_id=2, query_id="q1", sub_index=1, total=2, final=True, score=1.0),
        _sample(rollout_id=3, query_id="q1", sub_index=0, total=1, final=True, score=1.0),
    ]

    metrics, rollouts, questions = pipeline.analyze_samples(samples, expected_questions=2, samples_per_question=2)

    assert metrics["pass@1"] == 0.75
    assert metrics["pass@n"] == 1.0
    assert metrics["all_correct_rate"] == 0.5
    assert metrics["judge_failed_count"] == 1
    assert metrics["summary_counts"] == {"extracted": 1, "fallback": 1, "empty": 0}
    assert metrics["extracted_summary_content_tokens"]["mean"] == 12.0
    assert rollouts[0]["n_turns"] == 4
    assert rollouts[1]["score"] == 0.5
    assert rollouts[1]["correct"] is False
    assert [row["successes"] for row in questions] == [1, 2]


@pytest.mark.unit
def test_analyzer_rejects_missing_final_sibling():
    samples = [
        _sample(rollout_id=0, query_id="q0", sub_index=0, total=1, final=False),
    ]
    with pytest.raises(ValueError, match="exactly one final sibling"):
        pipeline.analyze_samples(samples, expected_questions=1, samples_per_question=1)


@pytest.mark.unit
def test_load_log_verification_distinguishes_step_and_base():
    result = pipeline.verify_load_log("successfully loaded checkpoint at iteration 39", "39")
    assert result["actual_step"] == 39
    base = pipeline.verify_load_log("loading release distributed checkpoint from /models/base/release", "base")
    assert base["actual_step"] == "base"
    with pytest.raises(ValueError, match="does not confirm iteration 24"):
        pipeline.verify_load_log("successfully loaded checkpoint at iteration 39", "24")
    with pytest.raises(ValueError, match="release checkpoint load"):
        pipeline.verify_load_log(
            "loading release checkpoint from /models/base/release\n" "successfully loaded checkpoint at iteration 39",
            "39",
        )
    with pytest.raises(ValueError, match="does not confirm iteration 39"):
        pipeline.verify_load_log("training starts at iteration 39", "39")


@pytest.mark.unit
def test_curve_svg_records_metric_title(tmp_path):
    path = tmp_path / "pass_at_n_curve.svg"

    pipeline._write_curve_svg(path, [("base", 0.7), ("iter04", 0.8)], title="pass@4")

    svg = path.read_text()
    assert "pass@4" in svg
    assert "base" in svg
    assert "0.800" in svg


@pytest.mark.unit
def test_controller_lock_rejects_overlapping_process(tmp_path):
    batch_root = tmp_path / "evals/batch-1"

    with sweep.controller_lock(batch_root):
        with pytest.raises(RuntimeError, match="batch controller is already running"):
            with sweep.controller_lock(batch_root):
                pass

    with sweep.controller_lock(batch_root):
        assert (batch_root / ".controller.lock").is_file()


@pytest.mark.unit
def test_sweep_has_shared_base_and_24_unique_checkpoint_outputs(tmp_path):
    points = sweep.sweep_points(_sweep_config())
    assert len(points) == 25
    assert sum(point["point"] == "base" for point in points) == 1
    outputs = {sweep.output_dir(tmp_path, point) for point in points}
    assert len(outputs) == 25
    assert sweep.output_dir(tmp_path, points[0]) == tmp_path / "base"


@pytest.mark.unit
def test_mast_command_is_single_host_and_has_checkpoint_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "DEV_STAGE", tmp_path)
    monkeypatch.setattr(sweep, "MAST_STAGE", Path("/mnt/wsfuse/test"))
    archive = tmp_path / "eval-code" / "code.tgz"
    config = _sweep_config()
    point = next(
        point
        for point in sweep.sweep_points(config)
        if point["run_name"] == config.runs[0].name and point["step"] == 39
    )
    command = sweep.build_mast_command(
        config=config,
        batch_id="test-batch",
        batch_root=tmp_path / "evals/test-batch",
        archive=archive,
        archive_sha256="abc",
        point=point,
        dry_run=True,
    )
    joined = " ".join(command)
    assert "data_parallel_size=1" in joined
    assert "EVAL_REQUESTED_STEP=39" in joined
    assert "BCPLUS_FIXED_SEARCH_TOPK=5" in joined
    assert "BCPLUS_DOC_WORDS_FULL=10000" in joined
    assert "BCPLUS_SEARCH_CONCURRENCY=64" in joined
    assert "BCPLUS_JUDGE_CONCURRENCY=16" in joined
    assert "BC_MODEL_SIZE=4B" in joined
    assert "TORCH_NCCL_DUMP_ON_TIMEOUT=0" in joined
    assert "--job_priority=HIGH" in command
    assert "examples/supo_browsecomp/mast/eval/run_eval.sh" in joined
    assert "examples/supo_browsecomp/mast/run_eval.sh" in joined
    assert command[-1] == "--dryrun"

    model_9b = sweep.SweepConfig(runs=config.runs, model_size="9B")
    command_9b = sweep.build_mast_command(
        config=model_9b,
        batch_id="test-9b",
        batch_root=tmp_path / "evals/test-9b",
        archive=archive,
        archive_sha256="abc",
        point=point,
        dry_run=False,
    )
    assert "BC_MODEL_SIZE=9B" in " ".join(command_9b)


@pytest.mark.unit
@pytest.mark.parametrize("value", ("../run", "run/name", "run;false", ".", "..", ""))
def test_eval_sweep_rejects_unsafe_path_names(value):
    with pytest.raises(RuntimeError, match="invalid"):
        sweep.validate_safe_name(value, "test name")


@pytest.mark.unit
def test_eval_sweep_accepts_scheduler_names():
    assert sweep.validate_safe_name("supo_4b-run.v3", "test name") == "supo_4b-run.v3"


@pytest.mark.unit
def test_eval_runner_has_model_specific_sglang_concurrency_defaults():
    runner = EVAL_RUNNER.read_text()

    assert "DEFAULT_SGLANG_SERVER_CONCURRENCY=36" in runner
    assert "DEFAULT_SGLANG_SERVER_CONCURRENCY=32" in runner
    assert 'BCPLUS_SGLANG_SERVER_CONCURRENCY="${BCPLUS_SGLANG_SERVER_CONCURRENCY:-${DEFAULT_SGLANG_SERVER_CONCURRENCY}}"' in runner
    assert 'export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-0}"' in runner
    assert '\\"TORCH_NCCL_DUMP_ON_TIMEOUT\\": \\"${TORCH_NCCL_DUMP_ON_TIMEOUT}\\"' in runner


@pytest.mark.unit
def test_checkpoint_discovery_uses_all_complete_iters_and_latest_tracker(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "DEV_STAGE", tmp_path)
    root = tmp_path / "checkpoints/future-run"
    root.mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text("14\n")
    for step in (4, 9, 14):
        checkpoint = root / f"iter_{step:07d}"
        checkpoint.mkdir()
        (checkpoint / ".metadata").write_text(f"step={step}\n")
    incomplete = root / "iter_0000019"
    incomplete.mkdir()

    run = sweep.discover_run("future-run", 1)

    assert run == sweep.RunConfig(name="future-run", alias="r01", steps=(4, 9, 14))
    (root / "latest_checkpointed_iteration.txt").write_text("9\n")
    assert sweep.discover_run("future-run", 1).steps == (4, 9, 14)
    (root / "latest_checkpointed_iteration.txt").write_text("19\n")
    with pytest.raises(RuntimeError, match="latest complete checkpoint is 14"):
        sweep.discover_run("future-run", 1)


@pytest.mark.unit
def test_existing_batch_can_append_newly_completed_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "DEV_STAGE", tmp_path)
    root = tmp_path / "checkpoints/future-run"
    root.mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text("14\n")
    for step in (4, 9, 14):
        checkpoint = root / f"iter_{step:07d}"
        checkpoint.mkdir()
        (checkpoint / ".metadata").write_text(f"step={step}\n")

    batch_root = tmp_path / "evals/batch-1"
    original = sweep.SweepConfig(
        runs=(sweep.RunConfig(name="future-run", alias="r01", steps=(4, 9)),),
        search_concurrency=128,
    )
    sweep._write_json(batch_root / "sweep_config.json", original.to_dict())
    args = SimpleNamespace(
        command="orchestrate",
        run_names=None,
        reuse_base_from=None,
        extend_checkpoints=True,
    )

    extended = sweep.load_or_create_config(args, batch_root, {"jobs": {}})

    assert extended.runs == (sweep.RunConfig(name="future-run", alias="r01", steps=(4, 9, 14)),)
    assert extended.search_concurrency == 128
    assert sweep.SweepConfig.from_dict(sweep._read_json(batch_root / "sweep_config.json", {})) == extended


@pytest.mark.unit
def test_sweep_config_round_trip_and_local_report_sync(tmp_path):
    config = _sweep_config(local_report_root=tmp_path / "local")
    assert sweep.SweepConfig.from_dict(config.to_dict()) == config

    batch_root = tmp_path / "cloud/batch-1"
    sweep._write_json(batch_root / "sweep_config.json", config.to_dict())
    sweep._write_json(batch_root / "sweep_state.json", {"jobs": {}})
    for run in config.runs:
        run_root = batch_root / "runs" / run.name
        run_root.mkdir(parents=True)
        for artifact in sweep.REPORT_ARTIFACTS:
            (run_root / artifact).write_text(f"{run.name}/{artifact}\n")

    local_root = sweep.sync_local_reports(config, batch_root)

    assert local_root == tmp_path / "local/batch-1"
    for run in config.runs:
        for artifact in sweep.REPORT_ARTIFACTS:
            assert (local_root / run.name / artifact).read_text() == f"{run.name}/{artifact}\n"


@pytest.mark.unit
def test_legacy_sweep_config_preserves_original_tool_protocol():
    config = sweep.SweepConfig.from_dict(
        {
            "runs": [{"name": "old-run", "alias": "r01", "steps": [4]}],
            "evaluation": {},
        }
    )

    assert config.fixed_search_topk is None
    assert config.doc_words_full == 4096


@pytest.mark.unit
def test_legacy_manifest_preserves_original_tool_protocol(tmp_path):
    run = sweep.RunConfig(name="old-run", alias="r01", steps=(4,))
    config = sweep.SweepConfig(
        runs=(run,),
        base_source_batch="old-batch",
        fixed_search_topk=None,
        doc_words_full=4096,
    )
    batch_root = tmp_path / "evals/new-batch"
    base_root = tmp_path / "evals/old-batch/base"
    base_root.mkdir(parents=True)
    sampling = {
        "samples_per_question": 4,
        "rollout_seed": 42,
        "sampling_seeds": [42, 43, 44, 45],
        "deterministic": True,
        "temperature": 1.0,
    }
    manifest = {
        "model_name": "Qwen3.5-4B",
        "load_verification": {"actual_step": "base"},
        "dataset_sha256": "dataset-hash",
        "judge_model": "judge-model",
        "sampling": sampling,
    }
    sweep._write_json(base_root / "manifest.json", manifest)
    sweep._write_json(
        base_root / "point_metrics.json",
        {"n_questions": 150, "n_rollouts": 600, "samples_per_question": 4},
    )
    sweep._write_json(base_root / "_SUCCESS", {"status": "ok"})
    (base_root / "questions.jsonl").write_text("{}\n")

    sweep.validate_base_source(config, batch_root)
    with pytest.raises(RuntimeError, match="sampling"):
        sweep.validate_base_source(
            sweep.SweepConfig(runs=(run,), base_source_batch="old-batch"),
            batch_root,
        )

    point = sweep.sweep_points(config)[0]
    point_root = sweep.output_dir(batch_root, point)
    sweep._write_json(
        point_root / "manifest.json",
        {
            **manifest,
            "load_verification": {"actual_step": 4},
            "sampling": {**sampling, "fixed_search_topk": None, "doc_words_full": 4096},
        },
    )
    sweep.validate_base_against_point(config, batch_root, point)


@pytest.mark.unit
def test_custom_eval_config_applies_defaults_and_cli_overrides():
    args = SimpleNamespace(
        eval_config=EVAL_CONFIGS / "fixed_topk5_open10000.json",
        doc_words_full=12000,
    )

    sweep.apply_evaluation_config(args)

    assert args.fixed_search_topk == 5
    assert args.doc_words_full == 12000
    assert args.eval_n == 4
    assert args.search_concurrency == 64


@pytest.mark.unit
def test_model_controlled_eval_config_is_explicit():
    args = SimpleNamespace(eval_config=EVAL_CONFIGS / "model_topk_open4096.json")

    sweep.apply_evaluation_config(args)

    assert args.fixed_search_topk is None
    assert args.doc_words_full == 4096


@pytest.mark.unit
def test_custom_eval_config_rejects_unknown_or_invalid_settings(tmp_path):
    unknown = tmp_path / "unknown.json"
    unknown.write_text('{"evaluation":{"topk":5}}')
    with pytest.raises(RuntimeError, match="unknown evaluation settings"):
        sweep.load_evaluation_config(unknown)

    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"evaluation":{"fixed_search_topk":true}}')
    with pytest.raises(RuntimeError, match="positive integer or null"):
        sweep.load_evaluation_config(invalid)


@pytest.mark.unit
def test_existing_batch_accepts_matching_eval_config_and_rejects_changes(tmp_path):
    config = sweep.SweepConfig(runs=(sweep.RunConfig(name="run-1", alias="r01", steps=(4,)),))
    sweep._write_json(tmp_path / "sweep_config.json", config.to_dict())
    args = SimpleNamespace(
        command="report",
        run_names=None,
        reuse_base_from=None,
        eval_config=EVAL_CONFIGS / "fixed_topk5_open10000.json",
    )
    sweep.apply_evaluation_config(args)

    assert sweep.load_or_create_config(args, tmp_path, {"jobs": {}}) == config

    args = SimpleNamespace(
        command="report",
        run_names=None,
        reuse_base_from=None,
        eval_config=EVAL_CONFIGS / "model_topk_open4096.json",
    )
    sweep.apply_evaluation_config(args)
    with pytest.raises(RuntimeError, match="already frozen"):
        sweep.load_or_create_config(args, tmp_path, {"jobs": {}})

    args = SimpleNamespace(
        command="report",
        run_names=None,
        reuse_base_from=None,
        eval_config=None,
        doc_words_full=4096,
    )
    sweep.apply_evaluation_config(args)
    with pytest.raises(RuntimeError, match="already frozen"):
        sweep.load_or_create_config(args, tmp_path, {"jobs": {}})


@pytest.mark.unit
def test_report_loads_legacy_config_without_touching_training_checkpoints(tmp_path, monkeypatch):
    config = _sweep_config()
    legacy = config.to_dict()
    legacy["version"] = 2
    legacy.pop("model")
    legacy["smoke"] = {"run_name": config.runs[0].name, "step": 39}
    sweep._write_json(tmp_path / "sweep_config.json", legacy)
    monkeypatch.setattr(
        sweep,
        "validate_checkpoints",
        lambda config: (_ for _ in ()).throw(AssertionError("report touched checkpoints")),
    )

    loaded = sweep.load_or_create_config(
        SimpleNamespace(command="report", run_names=None, reuse_base_from=None),
        tmp_path,
        {"jobs": {}},
    )

    assert loaded == config


@pytest.mark.unit
def test_reused_base_omits_job_and_requires_matching_checkpoint_protocol(tmp_path):
    config = _sweep_config(base_source_batch="old-batch")
    batch_root = tmp_path / "evals/new-batch"
    base_root = tmp_path / "evals/old-batch/base"
    base_root.mkdir(parents=True)
    sampling = {
        "samples_per_question": 4,
        "rollout_seed": 42,
        "sampling_seeds": [42, 43, 44, 45],
        "deterministic": True,
        "temperature": 1.0,
        "fixed_search_topk": 5,
        "doc_words_full": 10000,
    }
    manifest = {
        "model_name": "Qwen3.5-4B",
        "load_verification": {"actual_step": "base"},
        "dataset_sha256": "dataset-hash",
        "judge_model": "judge-model",
        "sampling": sampling,
    }
    sweep._write_json(base_root / "manifest.json", manifest)
    sweep._write_json(
        base_root / "point_metrics.json",
        {"n_questions": 150, "n_rollouts": 600, "samples_per_question": 4},
    )
    sweep._write_json(base_root / "_SUCCESS", {"status": "ok"})
    (base_root / "questions.jsonl").write_text("{}\n")

    sweep.validate_base_source(config, batch_root)
    with pytest.raises(RuntimeError, match="expected Qwen3.5-9B"):
        sweep.validate_base_source(
            sweep.SweepConfig(
                runs=config.runs,
                model_size="9B",
                base_source_batch=config.base_source_batch,
            ),
            batch_root,
        )
    with pytest.raises(RuntimeError, match="sampling"):
        sweep.validate_base_source(
            sweep.SweepConfig(
                runs=config.runs,
                base_source_batch=config.base_source_batch,
                fixed_search_topk=None,
                doc_words_full=4096,
            ),
            batch_root,
        )
    assert len(sweep.sweep_points(config)) == 24
    assert all(point["point"] != "base" for point in sweep.sweep_points(config))
    assert sweep.base_output_dir(config, batch_root) == base_root

    point = next(point for point in sweep.sweep_points(config) if point["step"] == 39)
    point_root = sweep.output_dir(batch_root, point)
    sweep._write_json(point_root / "manifest.json", {**manifest, "load_verification": {"actual_step": 39}})
    sweep.validate_base_against_point(config, batch_root, point)

    sweep._write_json(
        point_root / "manifest.json",
        {**manifest, "dataset_sha256": "different-dataset", "load_verification": {"actual_step": 39}},
    )
    with pytest.raises(RuntimeError, match="incompatible with run-1/iter39"):
        sweep.validate_base_against_point(config, batch_root, point)


@pytest.mark.unit
def test_orchestrate_submits_every_point_before_waiting(tmp_path, monkeypatch):
    config = sweep.SweepConfig(
        runs=(sweep.RunConfig(name="run-1", alias="r01", steps=(4, 9)),),
        base_source_batch="old-batch",
    )
    batch_root = tmp_path / "evals/new-batch"
    state = {"batch_id": "new-batch", "jobs": {}}
    events = []
    monkeypatch.setattr(
        sweep,
        "prepare",
        lambda args: (tmp_path, batch_root, state, config, tmp_path / "code.tgz", "hash"),
    )
    monkeypatch.setattr(sweep, "health_gate", lambda **kwargs: events.append("health"))
    monkeypatch.setattr(
        sweep,
        "submit_point",
        lambda **kwargs: events.append(("submit", kwargs["point"]["key"], kwargs.get("dry_run", False))),
    )
    monkeypatch.setattr(
        sweep,
        "wait_for_keys",
        lambda state, state_path, keys, poll_seconds: events.append(("wait", keys)),
    )
    monkeypatch.setattr(sweep, "validate_base_against_point", lambda *args: events.append("base-check"))
    monkeypatch.setattr(sweep, "create_reports", lambda *args: events.append("report"))

    sweep.orchestrate(SimpleNamespace(poll_seconds=1))

    expected_keys = ["run-1/iter04", "run-1/iter09"]
    assert events == [
        "health",
        ("submit", expected_keys[0], False),
        ("submit", expected_keys[1], False),
        ("wait", expected_keys),
        "base-check",
        "report",
    ]


@pytest.mark.unit
def test_orchestrate_submits_only_steps_missing_from_existing_state(tmp_path, monkeypatch):
    config = sweep.SweepConfig(
        runs=(sweep.RunConfig(name="run-1", alias="r01", steps=(4, 9, 14)),),
        base_source_batch="old-batch",
    )
    batch_root = tmp_path / "evals/new-batch"
    state = {
        "batch_id": "new-batch",
        "jobs": {
            "run-1/iter04": {"state": "COMPLETE", "output": "unused"},
            "run-1/iter09": {"state": "COMPLETE", "output": "unused"},
        },
    }
    submitted = []
    monkeypatch.setattr(
        sweep,
        "prepare",
        lambda args: (tmp_path, batch_root, state, config, tmp_path / "code.tgz", "hash"),
    )
    monkeypatch.setattr(sweep, "health_gate", lambda **kwargs: None)
    monkeypatch.setattr(
        sweep,
        "submit_point",
        lambda **kwargs: submitted.append(kwargs["point"]["key"]),
    )
    monkeypatch.setattr(sweep, "wait_for_keys", lambda *args: None)
    monkeypatch.setattr(sweep, "validate_base_against_point", lambda *args: None)
    monkeypatch.setattr(sweep, "create_reports", lambda *args: None)

    sweep.orchestrate(SimpleNamespace(poll_seconds=1))

    assert submitted == ["run-1/iter14"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
