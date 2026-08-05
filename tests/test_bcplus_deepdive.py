"""CPU contract tests for the run-level BC+ deep-dive orchestrator."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.supo_browsecomp.eval.analysis import deepdive

NUM_GPUS = 0


def _config_value(tmp_path: Path, *, model_b: str = "model-b") -> dict:
    return {
        "schema_version": deepdive.CONFIG_SCHEMA_VERSION,
        "run_name": "test-run",
        "output_dir": "output",
        "points": [
            {"name": "base", "source_dir": "sources/base"},
            {"name": "iter04", "source_dir": "sources/iter04"},
        ],
        "judges": [
            {"name": "judge_a", "display_name": "Judge A", "model": "model-a"},
            {"name": "judge_b", "display_name": "Judge B", "model": model_b},
        ],
    }


def _write_config(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "deepdive.json"
    path.write_text(json.dumps(value))
    return path


def _fake_stage(point_dir: Path, output_dir: Path) -> dict:
    point = point_dir.name
    manifest = {
        "schema_version": deepdive.retention.SCHEMA_VERSION,
        "prefilter_version": deepdive.retention.PREFILTER_VERSION,
        "point": point,
        "source_point_dir": str(point_dir.resolve()),
        "counts": {"n_prefilter_candidates": 2, "n_semantic_match_tasks": 3},
    }
    deepdive._write_json(output_dir / "stage_manifest.json", manifest)
    deepdive._write_json(output_dir / "_STAGED", {"status": "ok"})
    (output_dir / "candidates.jsonl").write_text('{"candidate_id":"one"}\n')
    (output_dir / "failure_retrieval.jsonl").write_text('{"candidate_id":"one"}\n')
    return manifest


@pytest.mark.unit
def test_config_resolves_paths_and_defaults_to_all_judge_pairs(tmp_path):
    config = deepdive.load_config(_write_config(tmp_path, _config_value(tmp_path)))

    assert config.output_dir == (tmp_path / "output").resolve()
    assert config.points[0].source_dir == (tmp_path / "sources/base").resolve()
    assert config.judges[0].concurrency == 8
    assert config.judges[0].api_key_env == "LLAMA_API_KEY"
    assert config.comparisons == (
        deepdive.ComparisonConfig(name="judge_a_vs_judge_b", model_a="judge_a", model_b="judge_b"),
    )


@pytest.mark.unit
def test_config_rejects_unknown_fields_and_invalid_comparisons(tmp_path):
    value = _config_value(tmp_path)
    value["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        deepdive.load_config(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    value["comparisons"] = [{"name": "bad", "model_a": "judge_a", "model_b": "missing"}]
    with pytest.raises(ValueError, match="two different configured judges"):
        deepdive.load_config(_write_config(tmp_path, value))


@pytest.mark.unit
def test_stage_is_resumable_and_judges_share_static_artifacts(tmp_path, monkeypatch):
    config = deepdive.load_config(_write_config(tmp_path, _config_value(tmp_path)))
    calls = []

    def fake_stage(point_dir, output_dir):
        calls.append(point_dir)
        return _fake_stage(point_dir, output_dir)

    monkeypatch.setattr(deepdive.retention, "stage_point", fake_stage)
    first = deepdive.stage_all(config)
    second = deepdive.stage_all(config)

    assert len(calls) == 2
    assert all(not point["resumed"] for point in first["points"])
    assert all(point["resumed"] for point in second["points"])
    judge_stage = deepdive.materialize_judge_stage(config, config.judges[0], config.points[0])
    shared_candidates = config.output_dir / "stage/base/candidates.jsonl"
    assert (judge_stage / "candidates.jsonl").samefile(shared_candidates)


@pytest.mark.unit
def test_existing_output_rejects_a_changed_semantic_contract(tmp_path):
    config_path = _write_config(tmp_path, _config_value(tmp_path))
    deepdive.prepare_output(deepdive.load_config(config_path))
    config_path.write_text(json.dumps(_config_value(tmp_path, model_b="changed-model")))

    with pytest.raises(ValueError, match="contract changed"):
        deepdive.prepare_output(deepdive.load_config(config_path))


@pytest.mark.unit
def test_changed_comparisons_invalidate_run_success(tmp_path):
    config_path = _write_config(tmp_path, _config_value(tmp_path))
    config = deepdive.load_config(config_path)
    deepdive.prepare_output(config)
    deepdive._write_json(config.output_dir / "_SUCCESS", {"status": "ok"})

    value = _config_value(tmp_path)
    value["comparisons"] = []
    config_path.write_text(json.dumps(value))
    deepdive.prepare_output(deepdive.load_config(config_path))

    assert not (config.output_dir / "_SUCCESS").exists()


@pytest.mark.unit
def test_unknown_judge_is_rejected_before_staging(tmp_path, monkeypatch):
    config = deepdive.load_config(_write_config(tmp_path, _config_value(tmp_path)))
    stage_calls = []
    monkeypatch.setattr(
        deepdive.retention,
        "stage_point",
        lambda point_dir, output_dir: stage_calls.append((point_dir, output_dir)),
    )

    with pytest.raises(ValueError, match="unknown judge names"):
        asyncio.run(deepdive.judge_all(config, judge_names=["missing"]))

    assert stage_calls == []


@pytest.mark.unit
def test_judge_report_and_status_cover_the_whole_run(tmp_path, monkeypatch):
    config = deepdive.load_config(_write_config(tmp_path, _config_value(tmp_path)))
    monkeypatch.setattr(deepdive.retention, "stage_point", _fake_stage)
    judge_calls = []

    async def fake_judge(stage_dir, output_dir, **kwargs):
        judge_calls.append((stage_dir, kwargs["model"]))
        manifest = {
            "n_judgments": 2,
            "n_remaining": 0,
            "n_match_judgments": 3,
            "n_summary_judgments": 1,
            "complete": True,
        }
        deepdive._write_json(output_dir / "judge_manifest.json", manifest)
        deepdive._write_json(output_dir / "_JUDGED", {"status": "ok"})
        for name in ("match_judgments.jsonl", "summary_judgments.jsonl", "judgments.jsonl"):
            (output_dir / name).write_text("")
        return manifest

    def fake_report(analysis_dirs, output_dir):
        assert len(analysis_dirs) == 2
        deepdive._write_json(output_dir / "_SUMMARY_RETENTION_SUCCESS", {"status": "ok"})
        return {"points": [{"point": "base"}, {"point": "iter04"}]}

    def fake_comparison(model_a_dirs, model_b_dirs, output_dir, **kwargs):
        assert len(model_a_dirs) == len(model_b_dirs) == 2
        deepdive._write_json(output_dir / "_SUMMARY_RETENTION_COMPARISON_SUCCESS", {"status": "ok"})
        return {"points": [{"point": "base"}, {"point": "iter04"}]}

    monkeypatch.setattr(deepdive.retention, "judge_stage", fake_judge)
    monkeypatch.setattr(deepdive.retention, "build_report", fake_report)
    monkeypatch.setattr(deepdive.retention, "build_model_comparison", fake_comparison)

    asyncio.run(deepdive.judge_all(config))
    report = deepdive.report_all(config)
    state = deepdive.status(config)

    assert len(judge_calls) == 4
    assert len(report["reports"]) == 2
    assert len(report["comparisons"]) == 1
    assert state["complete"] is True
    assert all(judge["complete"] for judge in state["judges"])
    assert state["comparisons"] == [{"name": "judge_a_vs_judge_b", "complete": True}]

    deepdive.stage_all(config)
    asyncio.run(deepdive.judge_all(config))
    assert (config.output_dir / "_SUCCESS").is_file()

    (config.output_dir / "judges/judge_a/base/judgments.jsonl").unlink()
    assert deepdive.status(config)["complete"] is False


@pytest.mark.unit
def test_report_refuses_incomplete_judges(tmp_path, monkeypatch):
    config = deepdive.load_config(_write_config(tmp_path, _config_value(tmp_path)))
    monkeypatch.setattr(deepdive.retention, "stage_point", _fake_stage)
    deepdive.stage_all(config)
    deepdive._write_json(config.output_dir / "_SUCCESS", {"status": "stale"})

    with pytest.raises(ValueError, match="incomplete points"):
        deepdive.report_all(config)
    assert not (config.output_dir / "_SUCCESS").exists()

    deepdive._write_json(config.output_dir / "_SUCCESS", {"status": "stale"})
    assert deepdive.status(config)["complete"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
