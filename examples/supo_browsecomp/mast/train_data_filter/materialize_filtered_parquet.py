#!/usr/bin/env python3
"""Materialize a BC+ parquet after excluding an audited query-id list."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_excluded_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"exclusion list is empty: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"exclusion list contains duplicate query ids: {path}")
    return ids


def query_ids(table: pa.Table) -> list[str]:
    if "extra_info" not in table.column_names:
        raise ValueError("source parquet has no extra_info column")

    ids = []
    for row_index, metadata in enumerate(table.column("extra_info").to_pylist()):
        if not isinstance(metadata, Mapping) or metadata.get("query_id") is None:
            raise ValueError(f"row {row_index} is missing extra_info.query_id")
        ids.append(str(metadata["query_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("source parquet contains duplicate extra_info.query_id values")
    return ids


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def materialize(
    source: Path,
    exclude_ids_path: Path,
    output: Path,
    manifest_path: Path,
    expected_source_sha256: str | None,
    expected_source_rows: int | None,
    expected_excluded: int | None,
) -> dict[str, Any]:
    source = source.resolve()
    exclude_ids_path = exclude_ids_path.resolve()
    output = output.resolve()
    manifest_path = manifest_path.resolve()

    if source == output:
        raise ValueError("output must not overwrite the source parquet")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")

    source_sha256 = sha256_file(source)
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise ValueError(
            f"source SHA-256 is {source_sha256}, expected {expected_source_sha256}"
        )

    excluded_ids = read_excluded_ids(exclude_ids_path)
    if expected_excluded is not None and len(excluded_ids) != expected_excluded:
        raise ValueError(
            f"exclusion list has {len(excluded_ids)} ids, expected {expected_excluded}"
        )

    source_table = pq.read_table(source)
    source_ids = query_ids(source_table)
    if expected_source_rows is not None and len(source_ids) != expected_source_rows:
        raise ValueError(
            f"source parquet has {len(source_ids)} rows, expected {expected_source_rows}"
        )

    excluded_set = set(excluded_ids)
    missing_ids = sorted(excluded_set - set(source_ids))
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} excluded query ids are absent from source: {missing_ids}")

    keep_mask = pa.array([query_id not in excluded_set for query_id in source_ids])
    output_table = source_table.filter(keep_mask)
    expected_output_ids = [query_id for query_id in source_ids if query_id not in excluded_set]
    if output_table.num_rows != len(source_ids) - len(excluded_ids):
        raise RuntimeError("filtered row count does not match source rows minus excluded ids")

    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.tmp-",
        suffix=".parquet",
    )
    os.close(file_descriptor)
    temporary_output = Path(temporary_name)
    try:
        pq.write_table(output_table, temporary_output, compression="snappy")
        persisted_table = pq.read_table(temporary_output)
        if persisted_table.schema != source_table.schema:
            raise RuntimeError("output parquet schema differs from source schema")
        if query_ids(persisted_table) != expected_output_ids:
            raise RuntimeError("output parquet query ids or row order differ from expectation")
        temporary_output.chmod(source.stat().st_mode & 0o777)
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)

    source_sha256_after = sha256_file(source)
    if source_sha256_after != source_sha256:
        output.unlink(missing_ok=True)
        raise RuntimeError("source parquet changed while the filtered output was being written")

    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filter": {
            "id_field": "extra_info.query_id",
            "excluded_ids_path": str(exclude_ids_path),
            "excluded_ids_sha256": sha256_file(exclude_ids_path),
            "excluded_count": len(excluded_ids),
            "preserved_source_row_order": True,
        },
        "source": {
            "path": str(source),
            "sha256": source_sha256,
            "rows": len(source_ids),
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "rows": output_table.num_rows,
            "schema_matches_source": True,
        },
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-source-rows", type=int)
    parser.add_argument("--expected-excluded", type=int)
    args = parser.parse_args()

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = materialize(
        source=args.source,
        exclude_ids_path=args.exclude_ids,
        output=args.output,
        manifest_path=manifest_path,
        expected_source_sha256=args.expected_source_sha256,
        expected_source_rows=args.expected_source_rows,
        expected_excluded=args.expected_excluded,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
