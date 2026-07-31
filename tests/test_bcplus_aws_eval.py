"""CPU tests for the AWS BC+ evaluation scheduler."""

from __future__ import annotations

import importlib.util
import os
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


sweep = _load(
    "bcplus_aws_eval_sweep",
    ROOT / "examples/supo_browsecomp/aws/eval/eval_sweep.py",
)
pipeline = _load(
    "bcplus_eval_pipeline_aws_test",
    ROOT / "examples/supo_browsecomp/eval/eval_pipeline.py",
)


@pytest.mark.unit
def test_checkpoint_discovery_requires_tracker_metadata_and_stability(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "CHECKPOINT_ROOT", tmp_path / "checkpoints")
    config = sweep.SweepConfig(run_name="run-1")
    root = sweep.CHECKPOINT_ROOT / config.run_name
    root.mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text("4\n")
    checkpoint = root / "iter_0000004"
    checkpoint.mkdir()
    metadata = checkpoint / ".metadata"
    metadata.write_text("complete\n")
    os.utime(metadata, (100, 100))
    incomplete = root / "iter_0000009"
    incomplete.mkdir()

    state = {}
    tracker, steps = sweep.discover_stable_steps(
        config,
        state,
        stability_seconds=120,
        now=1000,
    )

    assert tracker == 4
    assert steps == (4,)
    (root / "latest_checkpointed_iteration.txt").write_text("9\n")
    (incomplete / ".metadata").write_text("still-writing\n")
    os.utime(incomplete / ".metadata", (1000, 1000))
    _, steps = sweep.discover_stable_steps(config, state, stability_seconds=120, now=1001)
    assert steps == (4,)
    _, steps = sweep.discover_stable_steps(config, state, stability_seconds=120, now=1121)
    assert steps == (4, 9)


@pytest.mark.unit
def test_sbatch_command_freezes_protocol_qos_and_lane_dependency(tmp_path):
    config = sweep.SweepConfig(run_name="fixed-top5-run")
    command = sweep.build_sbatch_command(
        config=config,
        batch_id="batch-1",
        batch_root=tmp_path / "evals/batch-1",
        runner=tmp_path / "frozen/run_eval_job.sh",
        archive=tmp_path / "code.tgz",
        archive_sha256="abc123",
        step=29,
        lane="shared-0",
        dependency_job_id="12345",
    )
    joined = " ".join(map(str, command))

    assert "--qos=a100_genai_shared" in command
    assert "--dependency=afterany:12345" in command
    assert "BCPLUS_FIXED_SEARCH_TOPK=5" in joined
    assert "BCPLUS_DOC_WORDS_FULL=10000" in joined
    assert "EVAL_CODE_ARCHIVE_SHA256=abc123" in joined
    assert "EVAL_REQUESTED_STEP=29" in joined


@pytest.mark.unit
def test_eval_manifest_records_v2_tool_protocol():
    args = SimpleNamespace(
        samples_per_question=4,
        rollout_seed=42,
        temperature=1.0,
        max_response_len=32768,
        max_context_len=65536,
        max_turns=64,
        max_sub_trajs=5,
        compression_threshold=0.85,
        fixed_search_topk=5,
        doc_words_full=10000,
    )

    sampling = pipeline.sampling_manifest(args)

    assert sampling["sampling_seeds"] == [42, 43, 44, 45]
    assert sampling["fixed_search_topk"] == 5
    assert sampling["doc_words_full"] == 10000
