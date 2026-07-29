"""Crash-safe filesystem state for rolling weights-only checkpoints.

This module intentionally has no torch or Megatron imports.  Distributed
checkpoint writers supply an inspection callback while the state machine owns
the destructive filesystem lifecycle and its recovery rules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_VERSION = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointInfo:
    metadata_sha256: str
    size_bytes: int
    has_optimizer: bool
    has_scheduler: bool
    has_rng: bool

    def require_full(self, path: Path) -> None:
        if not self.has_optimizer or not self.has_scheduler or not self.has_rng:
            raise RuntimeError(
                f"checkpoint is not a full resume checkpoint: {path} "
                f"(optimizer={self.has_optimizer}, scheduler={self.has_scheduler}, rng={self.has_rng})"
            )

    def require_slim(self, path: Path) -> None:
        if self.has_optimizer or self.has_scheduler or self.has_rng:
            raise RuntimeError(
                f"checkpoint is not weights-only: {path} "
                f"(optimizer={self.has_optimizer}, scheduler={self.has_scheduler}, rng={self.has_rng})"
            )


InspectCheckpoint = Callable[[Path], CheckpointInfo]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def checkpoint_dir(root: Path, step: int) -> Path:
    return root / f"iter_{step:07d}"


def checkpoint_step(path: Path) -> int | None:
    suffix = path.name.removeprefix("iter_")
    return int(suffix) if path.name.startswith("iter_") and suffix.isdigit() else None


def promote_checkpoint(
    *,
    source: Path,
    staged: Path,
    backup: Path,
    validate: Callable[[Path], None],
    delete_backup: bool,
) -> None:
    """Atomically install a staged checkpoint and restore the source on error."""
    if not source.is_dir() or not (source / ".metadata").is_file():
        raise RuntimeError(f"source checkpoint is incomplete: {source}")
    if not staged.is_dir() or not (staged / ".metadata").is_file():
        raise RuntimeError(f"staged checkpoint is incomplete: {staged}")
    if backup.exists():
        raise RuntimeError(f"backup path already exists; refusing promotion: {backup}")

    backup.parent.mkdir(parents=True, exist_ok=True)
    source.replace(backup)
    try:
        staged.replace(source)
        validate(source)
    except BaseException as error:
        if source.exists():
            if staged.exists():
                raise RuntimeError(f"cannot roll back because staging path was recreated: {staged}") from error
            source.replace(staged)
        backup.replace(source)
        raise

    if delete_backup:
        shutil.rmtree(backup)


def rollback_promotion(*, source: Path, staged: Path, backup: Path) -> None:
    if not backup.is_dir():
        raise RuntimeError(f"cannot roll back without backup: {backup}")
    if source.exists():
        if staged.exists():
            raise RuntimeError(f"cannot preserve failed slim checkpoint; staging exists: {staged}")
        source.replace(staged)
    backup.replace(source)


class RollingCheckpointManager:
    """Own the on-disk lifecycle for one Megatron checkpoint root."""

    def __init__(self, save_root: Path):
        self.save_root = save_root
        self.state_root = save_root / ".rolling_slim"
        self.staging_root = self.state_root / "staging"
        self.backup_root = self.state_root / "backups"
        self.manifest_root = self.state_root / "manifests"
        self.orphan_root = self.state_root / "orphans"

    def source(self, step: int) -> Path:
        return checkpoint_dir(self.save_root, step)

    def staged(self, step: int) -> Path:
        return checkpoint_dir(self.staging_root, step)

    def backup(self, step: int) -> Path:
        return checkpoint_dir(self.backup_root, step)

    def manifest_path(self, step: int) -> Path:
        return self.manifest_root / f"iter_{step:07d}.json"

    def read_tracker(self) -> int | None:
        path = self.save_root / "latest_checkpointed_iteration.txt"
        if not path.is_file():
            return None
        try:
            return int(path.read_text().strip())
        except ValueError as error:
            raise RuntimeError(f"invalid checkpoint tracker: {path}") from error

    def read_manifest(self, step: int) -> dict[str, Any] | None:
        path = self.manifest_path(step)
        return json.loads(path.read_text()) if path.is_file() else None

    def manifests(self) -> list[dict[str, Any]]:
        if not self.manifest_root.is_dir():
            return []
        manifests = [json.loads(path.read_text()) for path in self.manifest_root.glob("iter_*.json")]
        return sorted(manifests, key=lambda manifest: int(manifest["step"]))

    def record_staged(self, step: int, info: CheckpointInfo) -> None:
        staged = self.staged(step)
        info.require_slim(staged)
        existing = self.read_manifest(step)
        if existing is not None and existing.get("status") not in {"discarded", "retained_full", "staged_valid"}:
            raise RuntimeError(f"cannot stage iter {step}; manifest state is {existing.get('status')!r}")
        manifest = {
            "version": MANIFEST_VERSION,
            "step": step,
            "status": "staged_valid",
            "slim_metadata_sha256": info.metadata_sha256,
            "slim_bytes": info.size_bytes,
            "staged_at": time.time(),
            "backup_deleted": False,
        }
        atomic_write_json(self.manifest_path(step), manifest)

    def record_full(self, step: int, info: CheckpointInfo) -> None:
        source = self.source(step)
        info.require_full(source)
        manifest = self.read_manifest(step)
        if manifest is None or manifest.get("status") not in {"staged_valid", "rotation_ready"}:
            raise RuntimeError(f"iter {step} lacks a validated slim staging manifest")
        manifest.update(
            {
                "status": "rotation_ready",
                "source_metadata_sha256": info.metadata_sha256,
                "source_bytes": info.size_bytes,
                "ready_at": time.time(),
            }
        )
        atomic_write_json(self.manifest_path(step), manifest)

    def is_rotation_ready(self, step: int) -> bool:
        manifest = self.read_manifest(step)
        return manifest is not None and manifest.get("status") == "rotation_ready" and self.staged(step).is_dir()

    def reconcile(self, inspect: InspectCheckpoint) -> int | None:
        """Repair unambiguous interrupted states and rotate committed predecessors."""
        self._recover_backups()
        self._quarantine_unmanifested_staging()
        tracker = self.read_tracker()

        for manifest in self.manifests():
            step = int(manifest["step"])
            status = manifest.get("status")
            if status in {"promoted", "discarded", "retained_full"}:
                continue
            if status == "promoting":
                source = self.source(step)
                staged = self.staged(step)
                if source.is_dir() and not staged.exists():
                    self._validate_installed_slim(source, manifest)
                    self._mark_promoted(step, manifest)
                    continue
                elif source.is_dir() and staged.is_dir():
                    source_info = inspect(source)
                    source_info.require_full(source)
                    try:
                        self._validate_slim(staged, manifest, inspect, context="staged checkpoint")
                    except Exception as error:
                        self._retain_full_after_failed_promotion(step, manifest, error)
                        continue
                    manifest["status"] = "rotation_ready"
                    atomic_write_json(self.manifest_path(step), manifest)
                    status = "rotation_ready"
                if status != "rotation_ready":
                    raise RuntimeError(f"iter {step} is marked promoting without a recoverable backup")

            if status == "staged_valid":
                if tracker is None or step > tracker:
                    self._discard_uncommitted(step, manifest)
                    continue
                source = self.source(step)
                if (source / ".metadata").is_file() and sha256_file(source / ".metadata") == manifest[
                    "slim_metadata_sha256"
                ]:
                    self._validate_installed_slim(source, manifest)
                    self._mark_promoted(step, manifest)
                    continue
                source_info = inspect(source)
                self.record_full(step, source_info)
                manifest = self.read_manifest(step)
                status = "rotation_ready"

            if status != "rotation_ready":
                raise RuntimeError(f"unrecognized rolling checkpoint state for iter {step}: {status!r}")
            if tracker is None or step > tracker:
                raise RuntimeError(f"rotation-ready iter {step} is newer than tracker {tracker}")
            if step < tracker:
                self._promote(step, manifest, inspect)

        tracker = self.read_tracker()
        if tracker is not None:
            inspect(self.source(tracker)).require_full(self.source(tracker))
        return tracker

    def cleanup_workdirs(self) -> None:
        active = [
            manifest
            for manifest in self.manifests()
            if manifest.get("status") not in {"promoted", "discarded", "retained_full"}
        ]
        if active:
            logger.warning("leaving rolling checkpoint workdirs with active manifests: %s", active)
            return
        for root in (self.staging_root, self.backup_root):
            if not root.exists():
                continue
            iteration_dirs = [path for path in root.glob("iter_*") if path.is_dir()]
            if iteration_dirs:
                logger.warning("leaving non-empty rolling checkpoint workdir in place: %s", root)
                continue
            tracker = root / "latest_checkpointed_iteration.txt"
            if tracker.is_file():
                try:
                    tracker.unlink()
                except OSError as error:
                    logger.warning("could not clean rolling checkpoint tracker %s: %s", tracker, error)
                    continue
            try:
                root.rmdir()
            except OSError as error:
                logger.warning("could not clean rolling checkpoint workdir %s: %s", root, error)

    def retain_full(self, step: int) -> None:
        """Drop an unneeded slim copy when the current latest becomes final."""
        manifest = self.read_manifest(step)
        if manifest is None or manifest.get("status") in {"promoted", "discarded", "retained_full"}:
            return
        if manifest.get("status") not in {"staged_valid", "rotation_ready"}:
            raise RuntimeError(f"cannot retain full iter {step} from manifest state {manifest.get('status')!r}")
        staged = self.staged(step)
        if staged.is_dir():
            shutil.rmtree(staged)
        manifest.update({"status": "retained_full", "retained_at": time.time(), "backup_deleted": True})
        atomic_write_json(self.manifest_path(step), manifest)

    def _promote(self, step: int, manifest: dict[str, Any], inspect: InspectCheckpoint) -> None:
        source = self.source(step)
        staged = self.staged(step)
        backup = self.backup(step)
        source_info = inspect(source)
        source_info.require_full(source)
        if source_info.metadata_sha256 != manifest.get("source_metadata_sha256"):
            self._retain_full_after_failed_promotion(
                step,
                manifest,
                RuntimeError(f"full checkpoint metadata changed before rotation: {source}"),
            )
            return
        if source_info.size_bytes != manifest.get("source_bytes"):
            self._retain_full_after_failed_promotion(
                step,
                manifest,
                RuntimeError(f"full checkpoint size changed before rotation: {source}"),
            )
            return
        try:
            self._validate_slim(staged, manifest, inspect, context="staged checkpoint")
        except Exception as error:
            self._retain_full_after_failed_promotion(step, manifest, error)
            return

        manifest["status"] = "promoting"
        atomic_write_json(self.manifest_path(step), manifest)

        def validate(path: Path) -> None:
            self._validate_installed_slim(path, manifest)

        try:
            promote_checkpoint(
                source=source,
                staged=staged,
                backup=backup,
                validate=validate,
                delete_backup=False,
            )
        except Exception as error:
            self._retain_full_after_failed_promotion(step, manifest, error)
            return

        try:
            shutil.rmtree(backup)
        except OSError as error:
            logger.warning(
                "rolling checkpoint iter %d was slimmed, but backup cleanup failed; "
                "training will continue and leave %s in place: %s",
                step,
                backup,
                error,
            )
            self._mark_promoted(step, manifest, backup_deleted=False, cleanup_error=str(error))
        else:
            self._mark_promoted(step, manifest)

    def _recover_backups(self) -> None:
        if not self.backup_root.is_dir():
            return
        for backup in sorted(self.backup_root.glob("iter_*")):
            step = checkpoint_step(backup)
            if step is None or not backup.is_dir():
                continue
            manifest = self.read_manifest(step)
            if manifest is not None and manifest.get("status") == "promoted":
                try:
                    shutil.rmtree(backup)
                except OSError as error:
                    logger.warning("could not clean retained rolling checkpoint backup %s: %s", backup, error)
                else:
                    self._mark_promoted(step, manifest)
                continue
            if manifest is None or manifest.get("status") != "promoting":
                raise RuntimeError(f"untracked rolling checkpoint backup requires manual audit: {backup}")
            source = self.source(step)
            staged = self.staged(step)
            if source.is_dir():
                try:
                    self._validate_installed_slim(source, manifest)
                except Exception as error:
                    rollback_promotion(source=source, staged=staged, backup=backup)
                    self._retain_full_after_failed_promotion(step, manifest, error)
                    continue
                try:
                    shutil.rmtree(backup)
                except OSError as error:
                    logger.warning(
                        "recovered rolling checkpoint iter %d, but backup cleanup failed; "
                        "training will continue and leave %s in place: %s",
                        step,
                        backup,
                        error,
                    )
                    self._mark_promoted(step, manifest, backup_deleted=False, cleanup_error=str(error))
                else:
                    self._mark_promoted(step, manifest)
            elif staged.is_dir():
                backup.replace(source)
                manifest["status"] = "rotation_ready"
                atomic_write_json(self.manifest_path(step), manifest)
            else:
                raise RuntimeError(f"cannot recover rolling checkpoint promotion for iter {step}")

    def _discard_uncommitted(self, step: int, manifest: dict[str, Any]) -> None:
        moved = []
        for label, path in (("staging", self.staged(step)), ("source", self.source(step))):
            if path.exists():
                destination = self._orphan_path(step, label)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)
                moved.append(str(destination))
        manifest.update({"status": "discarded", "discarded_at": time.time(), "orphaned_paths": moved})
        atomic_write_json(self.manifest_path(step), manifest)

    def _quarantine_unmanifested_staging(self) -> None:
        if not self.staging_root.is_dir():
            return
        for staged in sorted(self.staging_root.glob("iter_*")):
            step = checkpoint_step(staged)
            if step is None or not staged.is_dir() or self.manifest_path(step).is_file():
                continue
            destination = self._orphan_path(step, "unmanifested-staging")
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(destination)

    def _orphan_path(self, step: int, label: str) -> Path:
        stamp = f"{time.time_ns()}-{os.getpid()}"
        return self.orphan_root / f"iter_{step:07d}-{label}-{stamp}"

    def _validate_slim(
        self,
        path: Path,
        manifest: dict[str, Any],
        inspect: InspectCheckpoint,
        *,
        context: str,
    ) -> None:
        info = inspect(path)
        info.require_slim(path)
        if info.metadata_sha256 != manifest.get("slim_metadata_sha256"):
            raise RuntimeError(f"{context} metadata does not match manifest: {path}")
        if info.size_bytes != manifest.get("slim_bytes"):
            raise RuntimeError(f"{context} size does not match manifest: {path}")

    def _validate_installed_slim(self, path: Path, manifest: dict[str, Any]) -> None:
        """Confirm that promotion installed the prevalidated slim checkpoint.

        A same-filesystem directory rename preserves the staged checkpoint's
        contents.  Avoid recursively restating every file immediately after
        rename because remote FUSE mounts can briefly return stale directory
        entries or attributes for the reused source path.
        """
        metadata = path / ".metadata"
        common = path / "common.pt"
        if not path.is_dir() or not metadata.is_file() or not common.is_file():
            raise RuntimeError(f"installed checkpoint is incomplete: {path}")
        if sha256_file(metadata) != manifest.get("slim_metadata_sha256"):
            raise RuntimeError(f"installed checkpoint metadata does not match manifest: {path}")

    def _retain_full_after_failed_promotion(
        self,
        step: int,
        manifest: dict[str, Any],
        error: Exception,
    ) -> None:
        staged = self.staged(step)
        orphaned_path = None
        if staged.exists():
            destination = self._orphan_path(step, "failed-staging")
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                staged.replace(destination)
                orphaned_path = str(destination)
            except OSError as quarantine_error:
                logger.warning(
                    "could not quarantine failed rolling checkpoint staging %s: %s",
                    staged,
                    quarantine_error,
                )
        manifest.update(
            {
                "status": "retained_full",
                "retained_at": time.time(),
                "backup_deleted": not self.backup(step).exists(),
                "promotion_error": f"{type(error).__name__}: {error}",
                "orphaned_staging": orphaned_path,
            }
        )
        try:
            atomic_write_json(self.manifest_path(step), manifest)
        except Exception as manifest_error:
            logger.warning(
                "could not record failed rolling checkpoint promotion for iter %d: %s",
                step,
                manifest_error,
            )
        logger.warning(
            "rolling checkpoint iter %d was not slimmed; retaining its full checkpoint and continuing training: %s",
            step,
            error,
        )

    def _mark_promoted(
        self,
        step: int,
        manifest: dict[str, Any],
        *,
        backup_deleted: bool = True,
        cleanup_error: str | None = None,
    ) -> None:
        manifest.update(
            {
                "status": "promoted",
                "promoted_at": time.time(),
                "backup_deleted": backup_deleted,
            }
        )
        if cleanup_error is not None:
            manifest["backup_cleanup_error"] = cleanup_error
        else:
            manifest.pop("backup_cleanup_error", None)
        atomic_write_json(self.manifest_path(step), manifest)
