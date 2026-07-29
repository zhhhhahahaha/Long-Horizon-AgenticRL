from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from slime.utils import checkpoint_rotation
from slime.utils.checkpoint_rotation import (
    CheckpointInfo,
    RollingCheckpointManager,
    atomic_write_json,
    directory_size,
    sha256_file,
)


NUM_GPUS = 0


def write_checkpoint(path: Path, kind: str) -> CheckpointInfo:
    path.mkdir(parents=True)
    (path / ".metadata").write_text(kind)
    (path / "common.pt").write_text(kind)
    (path / "shard.distcp").write_bytes(b"checkpoint-shard")
    return inspect_checkpoint(path)


def inspect_checkpoint(path: Path) -> CheckpointInfo:
    kind = (path / ".metadata").read_text()
    return CheckpointInfo(
        metadata_sha256=sha256_file(path / ".metadata"),
        size_bytes=directory_size(path),
        has_optimizer=kind == "full",
        has_scheduler=kind == "full",
        has_rng=kind == "full",
    )


def write_tracker(root: Path, step: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest_checkpointed_iteration.txt").write_text(f"{step}\n")


def stage_and_register_full(manager: RollingCheckpointManager, step: int) -> None:
    slim = write_checkpoint(manager.staged(step), "slim")
    manager.record_staged(step, slim)
    full = write_checkpoint(manager.source(step), "full")
    write_tracker(manager.save_root, step)
    manager.record_full(step, full)


@pytest.mark.unit
def test_rolling_checkpoint_keeps_latest_full_and_rotates_predecessors(tmp_path):
    manager = RollingCheckpointManager(tmp_path / "checkpoints")

    stage_and_register_full(manager, 4)
    manager.reconcile(inspect_checkpoint)
    assert (manager.source(4) / ".metadata").read_text() == "full"
    assert (manager.staged(4) / ".metadata").read_text() == "slim"

    slim9 = write_checkpoint(manager.staged(9), "slim")
    manager.record_staged(9, slim9)
    full9 = write_checkpoint(manager.source(9), "full")
    write_tracker(manager.save_root, 9)
    manager.record_full(9, full9)
    manager.reconcile(inspect_checkpoint)

    assert (manager.source(4) / ".metadata").read_text() == "slim"
    assert (manager.source(9) / ".metadata").read_text() == "full"
    assert not manager.staged(4).exists()
    assert manager.read_manifest(4)["status"] == "promoted"
    assert manager.read_manifest(9)["status"] == "rotation_ready"

    write_checkpoint(manager.source(14), "full")
    write_tracker(manager.save_root, 14)
    manager.reconcile(inspect_checkpoint)
    manager.cleanup_workdirs()

    assert (manager.source(9) / ".metadata").read_text() == "slim"
    assert (manager.source(14) / ".metadata").read_text() == "full"
    assert manager.read_manifest(9)["status"] == "promoted"
    assert not manager.staging_root.exists()
    assert not manager.backup_root.exists()


@pytest.mark.unit
def test_rolling_checkpoint_recovers_crash_after_full_moved_to_backup(tmp_path):
    manager = RollingCheckpointManager(tmp_path / "checkpoints")
    stage_and_register_full(manager, 4)
    write_checkpoint(manager.source(9), "full")
    write_tracker(manager.save_root, 9)

    manifest = manager.read_manifest(4)
    manifest["status"] = "promoting"
    atomic_write_json(manager.manifest_path(4), manifest)
    manager.backup(4).parent.mkdir(parents=True)
    manager.source(4).replace(manager.backup(4))

    manager.reconcile(inspect_checkpoint)

    assert (manager.source(4) / ".metadata").read_text() == "slim"
    assert not manager.backup(4).exists()
    assert manager.read_manifest(4)["status"] == "promoted"


@pytest.mark.unit
def test_rolling_checkpoint_recovers_crash_after_slim_installed(tmp_path):
    manager = RollingCheckpointManager(tmp_path / "checkpoints")
    stage_and_register_full(manager, 4)
    write_checkpoint(manager.source(9), "full")
    write_tracker(manager.save_root, 9)

    manifest = manager.read_manifest(4)
    manifest["status"] = "promoting"
    atomic_write_json(manager.manifest_path(4), manifest)
    manager.backup(4).parent.mkdir(parents=True)
    manager.source(4).replace(manager.backup(4))
    manager.staged(4).replace(manager.source(4))

    manager.reconcile(inspect_checkpoint)

    assert (manager.source(4) / ".metadata").read_text() == "slim"
    assert not manager.backup(4).exists()
    assert manager.read_manifest(4)["status"] == "promoted"


@pytest.mark.unit
def test_rolling_checkpoint_recovers_crash_after_backup_deleted(tmp_path):
    manager = RollingCheckpointManager(tmp_path / "checkpoints")
    stage_and_register_full(manager, 4)
    write_checkpoint(manager.source(9), "full")
    write_tracker(manager.save_root, 9)

    manifest = manager.read_manifest(4)
    manifest["status"] = "promoting"
    atomic_write_json(manager.manifest_path(4), manifest)
    manager.backup(4).parent.mkdir(parents=True)
    manager.source(4).replace(manager.backup(4))
    manager.staged(4).replace(manager.source(4))
    manager.backup(4).rename(tmp_path / "deleted-backup-simulation")

    manager.reconcile(inspect_checkpoint)

    assert (manager.source(4) / ".metadata").read_text() == "slim"
    assert manager.read_manifest(4)["status"] == "promoted"


@pytest.mark.unit
def test_rolling_checkpoint_recovers_crash_before_full_moved_to_backup(tmp_path):
    manager = RollingCheckpointManager(tmp_path / "checkpoints")
    stage_and_register_full(manager, 4)
    write_checkpoint(manager.source(9), "full")
    write_tracker(manager.save_root, 9)

    manifest = manager.read_manifest(4)
    manifest["status"] = "promoting"
    atomic_write_json(manager.manifest_path(4), manifest)

    manager.reconcile(inspect_checkpoint)

    assert (manager.source(4) / ".metadata").read_text() == "slim"
    assert not manager.backup(4).exists()
    assert manager.read_manifest(4)["status"] == "promoted"


@pytest.mark.unit
def test_rolling_checkpoint_quarantines_uncommitted_newer_save(tmp_path):
    manager = RollingCheckpointManager(tmp_path / "checkpoints")
    stage_and_register_full(manager, 4)
    slim9 = write_checkpoint(manager.staged(9), "slim")
    manager.record_staged(9, slim9)
    write_checkpoint(manager.source(9), "full")

    manager.reconcile(inspect_checkpoint)

    assert not manager.source(9).exists()
    assert not manager.staged(9).exists()
    assert manager.read_manifest(9)["status"] == "discarded"
    assert len(list(manager.orphan_root.glob("iter_0000009-*"))) == 2
    assert (manager.source(4) / ".metadata").read_text() == "full"


@pytest.mark.unit
def test_rolling_checkpoint_retains_full_when_staging_metadata_changes(tmp_path, caplog):
    manager = RollingCheckpointManager(tmp_path / "checkpoints")
    stage_and_register_full(manager, 4)
    write_checkpoint(manager.source(9), "full")
    write_tracker(manager.save_root, 9)
    (manager.staged(4) / ".metadata").write_text("corrupt")

    manager.reconcile(inspect_checkpoint)

    assert (manager.source(4) / ".metadata").read_text() == "full"
    assert not manager.backup(4).exists()
    assert not manager.staged(4).exists()
    assert manager.read_manifest(4)["status"] == "retained_full"
    assert len(list(manager.orphan_root.glob("iter_0000004-failed-staging-*"))) == 1
    assert "retaining its full checkpoint and continuing training" in caplog.text


@pytest.mark.unit
def test_rolling_checkpoint_retains_full_when_staging_shard_is_truncated(tmp_path, caplog):
    manager = RollingCheckpointManager(tmp_path / "checkpoints")
    stage_and_register_full(manager, 4)
    write_checkpoint(manager.source(9), "full")
    write_tracker(manager.save_root, 9)
    (manager.staged(4) / "shard.distcp").write_bytes(b"x")

    manager.reconcile(inspect_checkpoint)

    assert (manager.source(4) / ".metadata").read_text() == "full"
    assert not manager.backup(4).exists()
    assert not manager.staged(4).exists()
    manifest = manager.read_manifest(4)
    assert manifest["status"] == "retained_full"
    assert "size does not match manifest" in manifest["promotion_error"]
    assert len(list(manager.orphan_root.glob("iter_0000004-failed-staging-*"))) == 1
    assert "retaining its full checkpoint and continuing training" in caplog.text


@pytest.mark.unit
def test_rolling_checkpoint_does_not_restat_installed_tree_after_rename(tmp_path):
    manager = RollingCheckpointManager(tmp_path / "checkpoints")
    stage_and_register_full(manager, 4)
    write_checkpoint(manager.source(9), "full")
    write_tracker(manager.save_root, 9)

    def inspect_with_stale_installed_size(path: Path) -> CheckpointInfo:
        info = inspect_checkpoint(path)
        if path == manager.source(4) and (path / ".metadata").read_text() == "slim":
            return replace(info, size_bytes=info.size_bytes + 1)
        return info

    manager.reconcile(inspect_with_stale_installed_size)

    assert (manager.source(4) / ".metadata").read_text() == "slim"
    assert not manager.backup(4).exists()
    assert manager.read_manifest(4)["status"] == "promoted"


@pytest.mark.unit
def test_rolling_checkpoint_rolls_back_when_installed_identity_check_fails(tmp_path, monkeypatch):
    manager = RollingCheckpointManager(tmp_path / "checkpoints")
    stage_and_register_full(manager, 4)
    write_checkpoint(manager.source(9), "full")
    write_tracker(manager.save_root, 9)

    def reject_installed_checkpoint(path: Path, manifest: dict) -> None:
        raise RuntimeError("injected installed identity mismatch")

    monkeypatch.setattr(manager, "_validate_installed_slim", reject_installed_checkpoint)
    manager.reconcile(inspect_checkpoint)

    assert (manager.source(4) / ".metadata").read_text() == "full"
    assert not manager.backup(4).exists()
    assert not manager.staged(4).exists()
    manifest = manager.read_manifest(4)
    assert manifest["status"] == "retained_full"
    assert "injected installed identity mismatch" in manifest["promotion_error"]
    assert len(list(manager.orphan_root.glob("iter_0000004-failed-staging-*"))) == 1


@pytest.mark.unit
def test_rolling_checkpoint_backup_cleanup_failure_is_nonfatal(tmp_path, monkeypatch, caplog):
    manager = RollingCheckpointManager(tmp_path / "checkpoints")
    stage_and_register_full(manager, 4)
    write_checkpoint(manager.source(9), "full")
    write_tracker(manager.save_root, 9)
    real_rmtree = shutil.rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        if Path(path) == manager.backup(4):
            raise OSError("injected backup cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(checkpoint_rotation.shutil, "rmtree", fail_backup_cleanup)

    manager.reconcile(inspect_checkpoint)

    assert (manager.source(4) / ".metadata").read_text() == "slim"
    assert (manager.backup(4) / ".metadata").read_text() == "full"
    manifest = manager.read_manifest(4)
    assert manifest["status"] == "promoted"
    assert manifest["backup_deleted"] is False
    assert "backup cleanup failed" in caplog.text


@pytest.mark.unit
def test_rolling_checkpoint_can_retain_pending_latest_as_final(tmp_path):
    manager = RollingCheckpointManager(tmp_path / "checkpoints")
    stage_and_register_full(manager, 4)

    manager.reconcile(inspect_checkpoint)
    manager.retain_full(4)
    manager.cleanup_workdirs()

    assert (manager.source(4) / ".metadata").read_text() == "full"
    assert manager.read_manifest(4)["status"] == "retained_full"
    assert not manager.staging_root.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
