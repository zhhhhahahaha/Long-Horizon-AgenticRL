"""CPU tests for the offline BC+ checkpoint slimming workflow."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


NUM_GPUS = 0
ROOT = Path(__file__).parents[1]


def _load_checkpoint_slim():
    path = ROOT / "examples/supo_browsecomp/mast/checkpoint_slim/checkpoint_slim.py"
    spec = importlib.util.spec_from_file_location("bcplus_checkpoint_slim", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checkpoint_slim = _load_checkpoint_slim()


@pytest.mark.unit
def test_checkpoint_slim_plan_protects_latest_checkpoint_at_any_iteration(tmp_path):
    root = tmp_path / "checkpoints/completed-run"
    root.mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text("57\n")
    for step in (4, 9, 57):
        checkpoint = root / f"iter_{step:07d}"
        checkpoint.mkdir()
        (checkpoint / ".metadata").write_text(f"step={step}\n")

    plan = checkpoint_slim.discover_plan(root)

    assert plan.protected_step == 57
    assert plan.steps == (4, 9)
    checkpoint_slim.validate_requested_steps(plan, (4, 9))
    with pytest.raises(RuntimeError, match="protected steps"):
        checkpoint_slim.validate_requested_steps(plan, (57,))
    (root / "latest_checkpointed_iteration.txt").write_text("56\n")
    with pytest.raises(RuntimeError, match="latest complete checkpoint=57"):
        checkpoint_slim.discover_plan(root)

    (root / "latest_checkpointed_iteration.txt").write_text("57\n")
    (root / "iter_0000061").mkdir()
    with pytest.raises(RuntimeError, match="newer incomplete checkpoint directories.*61"):
        checkpoint_slim.discover_plan(root)


@pytest.mark.unit
def test_checkpoint_slim_canary_promotion_rechecks_latest_checkpoint(tmp_path):
    source_root = tmp_path / "checkpoints/completed-run"
    staging_root = tmp_path / "staging/completed-run"
    state_dir = tmp_path / "state"
    source_root.mkdir(parents=True)
    staging_root.mkdir(parents=True)
    (source_root / "latest_checkpointed_iteration.txt").write_text("12\n")
    for step in (7, 12):
        checkpoint = source_root / f"iter_{step:07d}"
        checkpoint.mkdir()
        (checkpoint / ".metadata").write_text(f"source step={step}\n")

    staged = staging_root / "iter_0000012"
    staged.mkdir()
    (staged / ".metadata").write_text("staged step=12\n")

    with pytest.raises(RuntimeError, match="protected steps"):
        checkpoint_slim._promote_main(
            [
                "--source-root",
                str(source_root),
                "--staging-root",
                str(staging_root),
                "--state-dir",
                str(state_dir),
                "--step",
                "12",
            ]
        )

    assert (source_root / "iter_0000012/.metadata").read_text() == "source step=12\n"
    assert (staged / ".metadata").read_text() == "staged step=12\n"


@pytest.mark.unit
def test_checkpoint_slim_canary_promotion_rechecks_protected_metadata(tmp_path):
    source_root = tmp_path / "checkpoints/completed-run"
    staging_root = tmp_path / "staging/completed-run"
    state_dir = tmp_path / "state"
    source_root.mkdir(parents=True)
    staging_root.mkdir(parents=True)
    (source_root / "latest_checkpointed_iteration.txt").write_text("12\n")
    for step in (7, 12):
        checkpoint = source_root / f"iter_{step:07d}"
        checkpoint.mkdir()
        (checkpoint / ".metadata").write_text(f"source step={step}\n")

    staged = staging_root / "iter_0000007"
    staged.mkdir()
    (staged / ".metadata").write_text("staged step=7\n")
    manifest_path = state_dir / "manifests/completed-run/iter_0000007.json"
    checkpoint_slim.atomic_write_json(
        manifest_path,
        {
            "status": "staged_valid",
            "protected_step": 12,
            "protected_metadata_sha256": "stale-hash",
            "slim_metadata_sha256": checkpoint_slim.sha256_file(staged / ".metadata"),
        },
    )

    with pytest.raises(RuntimeError, match="protected checkpoint metadata changed"):
        checkpoint_slim._promote_main(
            [
                "--source-root",
                str(source_root),
                "--staging-root",
                str(staging_root),
                "--state-dir",
                str(state_dir),
                "--step",
                "7",
            ]
        )

    assert (source_root / "iter_0000007/.metadata").read_text() == "source step=7\n"
    assert (staged / ".metadata").read_text() == "staged step=7\n"

    manifest = {
        "status": "staged_valid",
        "protected_step": 12,
        "protected_metadata_sha256": checkpoint_slim.sha256_file(source_root / "iter_0000012/.metadata"),
        "slim_metadata_sha256": checkpoint_slim.sha256_file(staged / ".metadata"),
    }
    checkpoint_slim.atomic_write_json(manifest_path, manifest)
    assert (
        checkpoint_slim._promote_main(
            [
                "--source-root",
                str(source_root),
                "--staging-root",
                str(staging_root),
                "--state-dir",
                str(state_dir),
                "--step",
                "7",
            ]
        )
        == 0
    )
    assert (source_root / "iter_0000007/.metadata").read_text() == "staged step=7\n"
    assert not staged.exists()
    assert not (state_dir / "backups/completed-run/iter_0000007").exists()
    assert '"status": "promoted"' in manifest_path.read_text()


@pytest.mark.unit
def test_checkpoint_slim_promotion_deletes_backup_after_validation(tmp_path):
    source = tmp_path / "run/iter_0000004"
    staged = tmp_path / "staging/iter_0000004"
    backup = tmp_path / "state/backups/run/iter_0000004"
    source.mkdir(parents=True)
    staged.mkdir(parents=True)
    (source / ".metadata").write_text("full\n")
    (source / "payload").write_text("optimizer\n")
    (staged / ".metadata").write_text("slim\n")
    (staged / "payload").write_text("weights\n")

    checkpoint_slim.promote_checkpoint(
        source=source,
        staged=staged,
        backup=backup,
        validate=lambda path: (path / "payload").read_text() == "weights\n" or pytest.fail("wrong payload"),
        delete_backup=True,
    )

    assert (source / "payload").read_text() == "weights\n"
    assert not staged.exists()
    assert not backup.exists()


@pytest.mark.unit
def test_checkpoint_slim_promotion_rolls_back_on_validation_failure(tmp_path):
    source = tmp_path / "run/iter_0000004"
    staged = tmp_path / "staging/iter_0000004"
    backup = tmp_path / "state/backups/run/iter_0000004"
    source.mkdir(parents=True)
    staged.mkdir(parents=True)
    (source / ".metadata").write_text("full\n")
    (source / "payload").write_text("optimizer\n")
    (staged / ".metadata").write_text("slim\n")
    (staged / "payload").write_text("weights\n")

    def fail_validation(_path):
        raise RuntimeError("injected validation failure")

    with pytest.raises(RuntimeError, match="injected validation failure"):
        checkpoint_slim.promote_checkpoint(
            source=source,
            staged=staged,
            backup=backup,
            validate=fail_validation,
            delete_backup=True,
        )

    assert (source / "payload").read_text() == "optimizer\n"
    assert (staged / "payload").read_text() == "weights\n"
    assert not backup.exists()
