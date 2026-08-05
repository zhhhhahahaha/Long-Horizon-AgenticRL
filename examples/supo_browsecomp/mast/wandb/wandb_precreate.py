#!/usr/bin/env python3
"""Pre-create grouped W&B runs before uploading active offline snapshots."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import wandb
from wandb.apis.public import Run
from wandb.errors import CommError
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def _run_id(path: str) -> str:
    name = Path(path).name
    match = re.fullmatch(r"offline-run-[^-]+-([A-Za-z0-9]+)", name)
    if match is None:
        raise ValueError(f"cannot extract W&B run id from {path!r}")
    return match.group(1)


def _run_metadata(path: str) -> dict[str, str]:
    wandb_files = sorted(Path(path).glob("run-*.wandb"))
    if len(wandb_files) != 1:
        raise ValueError(f"expected exactly one run-*.wandb file under {path!r}, found {len(wandb_files)}")

    store = DataStore()
    try:
        store.open_for_scan(str(wandb_files[0]))
        while (data := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") == "run":
                return {
                    "run_id": record.run.run_id,
                    "entity": record.run.entity,
                    "project": record.run.project,
                    "group": record.run.run_group,
                }
    finally:
        store.close()
    raise ValueError(f"W&B run metadata record not found in {wandb_files[0]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    api = wandb.Api(timeout=30)
    for path in args.paths:
        run_id = _run_id(path)
        metadata = _run_metadata(path)
        expected_metadata = {
            "run_id": run_id,
            "project": args.project,
            "group": args.group,
        }
        for field, expected in expected_metadata.items():
            actual = metadata[field]
            if actual != expected:
                raise RuntimeError(f"offline W&B run {path!r} has {field}={actual!r}; expected {expected!r}")

        offline_entity = metadata["entity"] or api.default_entity
        if not offline_entity:
            raise RuntimeError(f"cannot resolve the offline W&B entity for {path!r}")
        if offline_entity != args.entity:
            raise RuntimeError(f"offline W&B run {path!r} targets entity {offline_entity!r}; expected {args.entity!r}")

        full_path = f"{args.entity}/{args.project}/{run_id}"
        created = False
        try:
            run = api.run(full_path)
        except CommError:
            Run.create(
                api,
                run_id=run_id,
                project=args.project,
                entity=args.entity,
                state="running",
            )
            created = True
            for _ in range(30):
                try:
                    run = wandb.Api(timeout=30).run(full_path)
                    break
                except CommError:
                    time.sleep(1)
            else:
                raise RuntimeError(f"W&B run did not become visible after creation: {full_path}")

        if created:
            # The following offline sync supplies group, job_type, and display
            # name. Pre-creating the ID prevents active snapshot ingestion from
            # dropping those fields when it creates the remote run itself.
            print(f"W&B run {full_path} pre-created for grouped offline sync")
            continue
        if run.group in (None, ""):
            print(f"W&B run {full_path} exists without a group; " "continuing so offline sync can populate it")
            continue
        if run.group == args.group:
            print(f"W&B run {full_path} already grouped")
            continue
        raise RuntimeError(f"W&B run {full_path} already has unexpected group {run.group!r}")


if __name__ == "__main__":
    main()
