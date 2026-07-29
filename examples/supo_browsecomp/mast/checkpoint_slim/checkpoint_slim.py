#!/usr/bin/env python3
"""Rewrite intermediate BC+ Megatron checkpoints without optimizer state.

The CPU-facing commands in this module intentionally avoid importing torch so
that checkpoint discovery, promotion, and rollback remain unit-testable.  The
``worker`` command is launched with torchrun inside the training image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Direct script execution sets sys.path to this nested directory rather than
# the repository root.  Keep the documented CPU-side CLI importable without an
# installed slime package.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.utils.checkpoint_rotation import (
    atomic_write_json,
    checkpoint_dir,
    directory_size,
    promote_checkpoint,
    rollback_promotion,
    sha256_file,
)


DEFAULT_CHECKPOINT_ROOT = Path("/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/checkpoints")


@dataclass(frozen=True)
class SlimPlan:
    run_name: str
    root: Path
    protected_step: int
    steps: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "root": str(self.root),
            "protected_step": self.protected_step,
            "steps": list(self.steps),
        }


def discover_plan(root: Path) -> SlimPlan:
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise RuntimeError(f"checkpoint root lacks tracker: {tracker}")
    try:
        tracked_step = int(tracker.read_text().strip())
    except ValueError as error:
        raise RuntimeError(f"invalid checkpoint tracker: {tracker}") from error
    checkpoint_paths = tuple(path for path in root.glob("iter_[0-9]*") if path.name.removeprefix("iter_").isdigit())
    steps = tuple(
        sorted(int(path.name.removeprefix("iter_")) for path in checkpoint_paths if (path / ".metadata").is_file())
    )
    if not steps:
        raise RuntimeError(f"{root} has no complete checkpoints")

    protected_step = steps[-1]
    newer_incomplete = sorted(
        int(path.name.removeprefix("iter_"))
        for path in checkpoint_paths
        if not (path / ".metadata").is_file() and int(path.name.removeprefix("iter_")) > protected_step
    )
    if newer_incomplete:
        raise RuntimeError(
            f"{root.name} has newer incomplete checkpoint directories {newer_incomplete}; "
            "refusing cleanup while checkpoint writing may still be active"
        )
    if tracked_step != protected_step:
        raise RuntimeError(
            f"{root.name} tracker={tracked_step}; latest complete checkpoint={protected_step}. "
            "Refusing cleanup until the tracker and checkpoint directories agree."
        )

    intermediate = steps[:-1]
    if not intermediate:
        raise RuntimeError(f"{root} has no intermediate checkpoints to slim")
    return SlimPlan(run_name=root.name, root=root, protected_step=protected_step, steps=intermediate)


def validate_requested_steps(plan: SlimPlan, requested_steps: tuple[int, ...]) -> None:
    if not requested_steps:
        raise RuntimeError("at least one --slim-step is required")
    if len(requested_steps) != len(set(requested_steps)):
        raise RuntimeError(f"duplicate slim steps: {requested_steps}")
    invalid = sorted(set(requested_steps) - set(plan.steps))
    if invalid:
        raise RuntimeError(f"refusing to slim unavailable or protected steps {invalid}; allowed={list(plan.steps)}")


def _parse_step_csv(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(part) for part in value.split(",") if part)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid step list: {value}") from error
    if not steps:
        raise argparse.ArgumentTypeError("step list must not be empty")
    return steps


def _plan_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Inspect BC+ checkpoints selected for optimizer-state removal")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--run", action="append", dest="runs", required=True)
    args = parser.parse_args(argv)
    if len(args.runs) != len(set(args.runs)):
        parser.error(f"duplicate --run values: {args.runs}")

    plans = [discover_plan(args.checkpoint_root / run_name) for run_name in args.runs]
    print(json.dumps({"runs": [plan.to_dict() for plan in plans]}, indent=2))
    return 0


def _promote_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Promote a canary-validated slim checkpoint")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args(argv)

    plan = discover_plan(args.source_root)
    validate_requested_steps(plan, (args.step,))
    source = checkpoint_dir(args.source_root, args.step)
    staged = checkpoint_dir(args.staging_root, args.step)
    backup = args.state_dir / "backups" / args.source_root.name / source.name
    manifest_path = args.state_dir / "manifests" / args.source_root.name / f"{source.name}.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing validated conversion manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "staged_valid":
        raise RuntimeError(f"checkpoint is not ready for promotion: {manifest_path}")
    if manifest.get("protected_step") != plan.protected_step:
        raise RuntimeError(
            f"protected checkpoint changed after staging: "
            f"manifest={manifest.get('protected_step')} current={plan.protected_step}"
        )
    protected = checkpoint_dir(args.source_root, plan.protected_step)
    protected_metadata = sha256_file(protected / ".metadata")
    if manifest.get("protected_metadata_sha256") != protected_metadata:
        raise RuntimeError(f"protected checkpoint metadata changed after staging: {protected}")

    expected_metadata = manifest["slim_metadata_sha256"]

    def validate(path: Path) -> None:
        actual = sha256_file(path / ".metadata")
        if actual != expected_metadata:
            raise RuntimeError(f"promoted metadata hash mismatch: {actual} != {expected_metadata}")

    promote_checkpoint(
        source=source,
        staged=staged,
        backup=backup,
        validate=validate,
        delete_backup=True,
    )
    manifest.update({"status": "promoted", "promoted_at": time.time(), "backup_deleted": True})
    atomic_write_json(manifest_path, manifest)
    return 0


def _add_worker_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hf-checkpoint", type=str, required=True)
    parser.add_argument("--custom-model-provider-path", type=str, default=None)
    parser.add_argument("--megatron-to-hf-mode", choices=["raw", "bridge"], default="raw")
    parser.add_argument("--allgather-cp", action="store_true", default=False)
    parser.add_argument("--slim-run-name", required=True)
    parser.add_argument("--slim-steps", type=_parse_step_csv, required=True)
    parser.add_argument("--slim-state-dir", type=Path, required=True)
    parser.add_argument("--slim-promote", action="store_true")
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except Exception:
        pass
    return parser


def _worker_main(argv: list[str]) -> int:  # noqa: C901 - lifecycle is intentionally explicit
    # Heavy imports stay inside the distributed worker so CPU safety tests can
    # import this module without a Megatron/PyTorch installation.
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dist_cp
    from megatron.core.enums import ModelType
    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.checkpointing import save_checkpoint
    from megatron.training.training import get_model

    import slime_plugins.mbridge  # noqa: F401
    from slime.backends.megatron_utils.arguments import set_default_megatron_args
    from slime.backends.megatron_utils.checkpoint import load_checkpoint
    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model_provider import get_model_provider_func
    from slime.utils.distributed_utils import get_gloo_group, init_gloo_group
    from slime.utils.logging_utils import configure_logger

    old_argv = sys.argv
    sys.argv = [old_argv[0], *argv]
    try:
        args = parse_args(_add_worker_args)
    finally:
        sys.argv = old_argv

    args = set_default_megatron_args(args)
    args.save_interval = 1
    args.micro_batch_size = 1
    args.global_batch_size = 1
    args.no_load_optim = True
    args.no_load_rng = True
    args.no_save_optim = True
    args.no_save_rng = True
    args.finetune = False
    args.ckpt_step = args.slim_steps[0]
    validate_args(args)
    configure_logger()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=args.distributed_timeout_minutes),
    )
    init_gloo_group()
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    init(args)

    source_root = Path(args.load)
    staging_root = Path(args.save)
    plan = discover_plan(source_root)
    validate_requested_steps(plan, args.slim_steps)
    if args.slim_run_name != source_root.name:
        raise RuntimeError(f"run name {args.slim_run_name} does not match source root {source_root.name}")

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    def model_digest() -> str:
        local = hashlib.sha256()
        for chunk_index, chunk in enumerate(model):
            tensors = list(chunk.named_parameters()) + list(chunk.named_buffers())
            for name, tensor in sorted(tensors, key=lambda item: item[0]):
                value = tensor.detach().contiguous()
                local.update(f"{chunk_index}:{name}:{value.dtype}:{tuple(value.shape)}".encode())
                local.update(value.view(torch.uint8).cpu().numpy().tobytes())
        rank_digests: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(rank_digests, local.hexdigest(), group=get_gloo_group())
        combined = hashlib.sha256()
        for rank, digest in enumerate(rank_digests):
            combined.update(f"{rank}:{digest}".encode())
        return combined.hexdigest()

    def load_from(root: Path, step: int) -> int:
        args.load = str(root)
        args.ckpt_step = step
        iteration, _ = load_checkpoint(model, None, None, {}, False)
        return int(iteration)

    def validate_slim_checkpoint(path: Path) -> tuple[str, int]:
        metadata = dist_cp.FileSystemReader(str(path)).read_metadata()
        disallowed = sorted(
            key for key in metadata.state_dict_metadata if key.startswith("optimizer") or key.startswith("rng_state")
        )
        if disallowed:
            raise RuntimeError(f"slim checkpoint still contains optimizer/RNG keys: {disallowed[:5]}")
        common = torch.load(path / "common.pt", map_location="cpu", weights_only=False)
        common_disallowed = sorted(set(common) & {"optimizer", "opt_param_scheduler", "rng_state"})
        if common_disallowed:
            raise RuntimeError(f"slim common state still contains: {common_disallowed}")
        return sha256_file(path / ".metadata"), directory_size(path)

    def validate_protected_checkpoint(path: Path) -> str:
        metadata = dist_cp.FileSystemReader(str(path)).read_metadata()
        if not any(key.startswith("optimizer") for key in metadata.state_dict_metadata):
            raise RuntimeError(f"protected latest checkpoint lacks optimizer state: {path}")
        return sha256_file(path / ".metadata")

    try:
        protected_metadata = None
        if args.rank == 0:
            protected_metadata = validate_protected_checkpoint(checkpoint_dir(source_root, plan.protected_step))
        dist.barrier()

        for step in args.slim_steps:
            manifest_path = args.slim_state_dir / "manifests" / args.slim_run_name / f"iter_{step:07d}.json"
            existing = None
            if manifest_path.is_file():
                existing = json.loads(manifest_path.read_text())
                if existing.get("status") == "promoted":
                    if args.rank == 0:
                        print(f"[slim] iter {step}: already promoted; skip", flush=True)
                    continue
                if existing.get("status") != "staged_valid":
                    raise RuntimeError(f"unrecognized manifest state: {manifest_path}")

            source = checkpoint_dir(source_root, step)
            staged = checkpoint_dir(staging_root, step)
            backup = args.slim_state_dir / "backups" / args.slim_run_name / source.name
            if backup.exists():
                raise RuntimeError(f"unresolved backup requires manual audit before retry: {backup}")

            if existing is not None:
                if not staged.is_dir():
                    raise RuntimeError(f"manifest says staged_valid but checkpoint is absent: {staged}")
                if args.rank == 0:
                    if existing.get("protected_step") != plan.protected_step:
                        raise RuntimeError(f"protected checkpoint changed after staging: {manifest_path}")
                    if existing.get("protected_metadata_sha256") != protected_metadata:
                        raise RuntimeError(f"protected checkpoint metadata changed after staging: {manifest_path}")
                    source_metadata = sha256_file(source / ".metadata")
                    slim_metadata, slim_size = validate_slim_checkpoint(staged)
                    if source_metadata != existing["source_metadata_sha256"]:
                        raise RuntimeError(f"source metadata changed after staging: {source}")
                    if slim_metadata != existing["slim_metadata_sha256"] or slim_size != existing["slim_bytes"]:
                        raise RuntimeError(f"staged checkpoint changed after validation: {staged}")

                loaded = load_from(source_root, step)
                if loaded != step:
                    raise RuntimeError(f"loaded iteration {loaded}, expected {step}")
                source_model_hash = model_digest()
                if source_model_hash != existing["model_sha256"]:
                    raise RuntimeError(f"source model changed after staging at iter {step}")

                loaded = load_from(staging_root, step)
                if loaded != step:
                    raise RuntimeError(f"reloaded slim iteration {loaded}, expected {step}")
                slim_model_hash = model_digest()
                if slim_model_hash != source_model_hash:
                    raise RuntimeError(f"model hash mismatch at iter {step}: {slim_model_hash} != {source_model_hash}")
                if args.rank == 0:
                    print(f"[slim] iter {step}: resumed validated staging checkpoint", flush=True)
                dist.barrier()
            else:
                source_metadata = sha256_file(source / ".metadata") if args.rank == 0 else None
                source_size = directory_size(source) if args.rank == 0 else None
                loaded = load_from(source_root, step)
                if loaded != step:
                    raise RuntimeError(f"loaded iteration {loaded}, expected {step}")
                source_model_hash = model_digest()

                if staged.exists():
                    raise RuntimeError(f"staging checkpoint already exists: {staged}")
                args.save = str(staging_root)
                save_checkpoint(step, model, None, None, 0)
                dist.barrier()

                slim_metadata = slim_size = None
                if args.rank == 0:
                    slim_metadata, slim_size = validate_slim_checkpoint(staged)
                loaded = load_from(staging_root, step)
                if loaded != step:
                    raise RuntimeError(f"reloaded slim iteration {loaded}, expected {step}")
                slim_model_hash = model_digest()
                if slim_model_hash != source_model_hash:
                    raise RuntimeError(f"model hash mismatch at iter {step}: {slim_model_hash} != {source_model_hash}")

                if args.rank == 0:
                    manifest = {
                        "version": 2,
                        "run_name": args.slim_run_name,
                        "step": step,
                        "protected_step": plan.protected_step,
                        "protected_metadata_sha256": protected_metadata,
                        "status": "staged_valid",
                        "source_metadata_sha256": source_metadata,
                        "slim_metadata_sha256": slim_metadata,
                        "model_sha256": source_model_hash,
                        "source_bytes": source_size,
                        "slim_bytes": slim_size,
                        "staged_at": time.time(),
                        "backup_deleted": False,
                    }
                    atomic_write_json(manifest_path, manifest)
                dist.barrier()

            if not args.slim_promote:
                continue

            if args.rank == 0:
                promote_checkpoint(
                    source=source,
                    staged=staged,
                    backup=backup,
                    validate=lambda path: validate_slim_checkpoint(path),
                    delete_backup=False,
                )
            dist.barrier()

            try:
                loaded = load_from(source_root, step)
                if loaded != step:
                    raise RuntimeError(f"final-path load returned iteration {loaded}, expected {step}")
                final_model_hash = model_digest()
                if final_model_hash != source_model_hash:
                    raise RuntimeError(
                        f"final-path model hash mismatch at iter {step}: " f"{final_model_hash} != {source_model_hash}"
                    )
            except BaseException:
                if args.rank == 0 and backup.exists():
                    rollback_promotion(source=source, staged=staged, backup=backup)
                raise

            if args.rank == 0:
                shutil.rmtree(backup)
                manifest = json.loads(manifest_path.read_text())
                manifest.update(
                    {
                        "status": "promoted",
                        "promoted_at": time.time(),
                        "backup_deleted": True,
                    }
                )
                atomic_write_json(manifest_path, manifest)
            dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("usage: checkpoint_slim.py {plan|promote|worker} ...")
        print("run 'checkpoint_slim.py <command> --help' for command-specific options")
        return 0
    command, argv = sys.argv[1], sys.argv[2:]
    if command == "plan":
        return _plan_main(argv)
    if command == "promote":
        return _promote_main(argv)
    if command == "worker":
        return _worker_main(argv)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
