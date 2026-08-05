#!/usr/bin/env python3
"""Build a cross-iteration BC+ trajectory comparison from a training dump directory."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .bcplus_rollout_viewer import (
        TokenizerDecoder,
        build_view_data,
        load_selected_rows,
        render_html,
    )
except ImportError:
    from bcplus_rollout_viewer import (
        TokenizerDecoder,
        build_view_data,
        load_selected_rows,
        render_html,
    )


INDEX_VERSION = 1
PARQUET_NAME = re.compile(r"rollouts_iter_(\d+)_dp(\d+)\.parquet$")
INDEX_COLUMNS = (
    "iter_id",
    "query_id",
    "group_index",
    "rollout_id",
    "total_sub_trajs",
    "is_final",
    "outcome",
    "score_final",
)


def discover_complete_iterations(
    dump_dir: Path, expected_dp_shards: int | None = None
) -> tuple[dict[int, list[Path]], dict[int, list[int]], int]:
    if expected_dp_shards is not None and (
        isinstance(expected_dp_shards, bool) or expected_dp_shards < 1
    ):
        raise ValueError("expected_dp_shards must be a positive integer")
    files_by_iter: dict[int, dict[int, Path]] = defaultdict(dict)
    for path in dump_dir.glob("rollouts_iter_*_dp*.parquet"):
        match = PARQUET_NAME.fullmatch(path.name)
        if match:
            iter_id, dp_rank = map(int, match.groups())
            files_by_iter[iter_id][dp_rank] = path
    if not files_by_iter:
        raise FileNotFoundError(f"no rollout parquet files found under {dump_dir}")

    shard_count = expected_dp_shards or (max(max(ranks) for ranks in files_by_iter.values()) + 1)
    expected_ranks = set(range(shard_count))
    complete: dict[int, list[Path]] = {}
    incomplete: dict[int, list[int]] = {}
    for iter_id, ranked_paths in sorted(files_by_iter.items()):
        unexpected = sorted(set(ranked_paths) - expected_ranks)
        if unexpected:
            raise ValueError(f"iteration {iter_id} has unexpected DP ranks: {unexpected}")
        missing = sorted(expected_ranks - set(ranked_paths))
        if missing:
            incomplete[iter_id] = missing
        else:
            complete[iter_id] = [ranked_paths[rank] for rank in range(shard_count)]
    return complete, incomplete, shard_count


def _file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_final_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    missing = sorted(set(INDEX_COLUMNS) - set(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f"{path} is missing index columns: {', '.join(missing)}")
    rows = [
        row
        for row in parquet_file.read(columns=list(INDEX_COLUMNS), use_threads=False).to_pylist()
        if row["is_final"]
    ]
    _validate_file_index_rows(path, rows)
    return rows


def _validate_file_index_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    _validate_index_rows(rows)
    match = PARQUET_NAME.fullmatch(path.name)
    assert match is not None
    expected_iter = int(match.group(1))
    mismatched = sorted({int(row["iter_id"]) for row in rows if int(row["iter_id"]) != expected_iter})
    if mismatched:
        raise ValueError(f"{path} contains iter ids {mismatched}, expected {expected_iter}")


def _validate_index_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[tuple[int, int]] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"index row {row_index} is not an object")
        missing = sorted(set(INDEX_COLUMNS) - set(row))
        if missing:
            raise ValueError(f"index row {row_index} is missing columns: {', '.join(missing)}")
        query_id = row["query_id"]
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"index row {row_index} has an invalid query_id")
        key = (int(row["iter_id"]), int(row["rollout_id"]))
        if key in seen:
            raise ValueError(f"duplicate final rollout row for iter {key[0]} rollout {key[1]}")
        seen.add(key)


def build_rollout_index(
    dump_dir: Path,
    complete_iterations: dict[int, list[Path]],
    cache_path: Path,
) -> tuple[list[dict[str, Any]], int, int]:
    cached_files: dict[str, Any] = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text())
            files = cache.get("files")
            if (
                cache.get("version") == INDEX_VERSION
                and cache.get("dump_dir") == str(dump_dir)
                and isinstance(files, dict)
            ):
                cached_files = files
        except (OSError, ValueError, TypeError):
            cached_files = {}

    current: dict[str, dict[str, Any]] = {}
    to_read: list[Path] = []
    for paths in complete_iterations.values():
        for path in paths:
            signature = _file_signature(path)
            cached = cached_files.get(path.name)
            if isinstance(cached, dict) and cached.get("signature") == signature:
                cached_rows = cached.get("rows")
                if isinstance(cached_rows, list):
                    try:
                        _validate_file_index_rows(path, cached_rows)
                    except (TypeError, ValueError):
                        pass
                    else:
                        current[path.name] = cached
                        continue
            to_read.append(path)

    def read_entry(path: Path) -> tuple[str, dict[str, Any]]:
        return path.name, {
            "signature": _file_signature(path),
            "rows": _read_final_rows(path),
        }

    with ThreadPoolExecutor(max_workers=min(32, len(to_read) or 1)) as pool:
        for name, entry in pool.map(read_entry, to_read):
            current[name] = entry

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"version": INDEX_VERSION, "dump_dir": str(dump_dir), "files": current},
            separators=(",", ":"),
        )
    )
    temporary.replace(cache_path)

    rows = [row for entry in current.values() for row in entry["rows"]]
    _validate_index_rows(rows)
    return rows, len(current) - len(to_read), len(to_read)


def choose_query(rows: list[dict[str, Any]], requested: str | None) -> str:
    by_query: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_query[str(row["query_id"])][int(row["iter_id"])].append(row)
    if requested is not None:
        if requested not in by_query:
            raise ValueError(f"query_id {requested!r} was not found")
        return requested

    candidates = []
    for query_id, by_iter in by_query.items():
        paired = 0
        rates = []
        for iter_rows in by_iter.values():
            scores = [float(row["score_final"]) > 0 for row in iter_rows]
            paired += int(any(scores) and not all(scores))
            rates.append(sum(scores) / len(scores))
        spread = max(rates) - min(rates)
        candidates.append((paired, len(by_iter), spread, max(by_iter), query_id))
    if not candidates:
        raise ValueError("no query ids are available for comparison")
    return max(candidates)[-1]


def select_rollouts(rows: list[dict[str, Any]], query_id: str, max_iterations: int) -> list[dict[str, Any]]:
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    by_iter: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["query_id"]) == query_id:
            by_iter[int(row["iter_id"])].append(row)

    selected = []
    for iter_id, iter_rows in sorted(by_iter.items()):
        group_indices = {int(row["group_index"]) for row in iter_rows}
        if len(group_indices) != 1:
            raise ValueError(
                f"query_id {query_id!r} appears in multiple groups at iteration {iter_id}: "
                f"{sorted(group_indices)}"
            )
        passed = [row for row in iter_rows if float(row["score_final"]) > 0]
        failed = [row for row in iter_rows if float(row["score_final"]) <= 0]
        if not passed or not failed:
            continue
        success = min(passed, key=lambda row: (int(row["total_sub_trajs"]), int(row["rollout_id"])))
        failure = min(
            failed,
            key=lambda row: (
                str(row["outcome"]) != "finished",
                int(row["total_sub_trajs"]),
                int(row["rollout_id"]),
            ),
        )
        selected.append(
            {
                "iter_id": iter_id,
                "successes": len(passed),
                "total": len(iter_rows),
                "rollout_ids": [int(success["rollout_id"]), int(failure["rollout_id"])],
            }
        )
    if not selected:
        raise ValueError(f"query_id {query_id!r} has no iteration containing both success and failure")
    return selected[-max_iterations:]


def build_comparison_data(
    dump_dir: Path,
    query_id: str,
    selections: list[dict[str, Any]],
    decoder: TokenizerDecoder,
) -> dict[str, Any]:
    groups = []
    question = ""
    for selection in selections:
        iter_id = selection["iter_id"]
        rows = load_selected_rows(dump_dir, iter_id, selection["rollout_ids"])
        partial = build_view_data(rows, decoder, dump_dir, iter_id, selection["rollout_ids"])
        if len(partial["groups"]) != 1:
            raise ValueError(f"iteration {iter_id} selected rollouts from multiple prompt groups")
        group = partial["groups"][0]
        if not question:
            question = group["question"]
        elif group["question"] != question:
            raise ValueError(f"query_id {query_id!r} decodes to inconsistent questions")
        group["label"] = f"Iteration {iter_id} · {selection['successes']}/{selection['total']} correct"
        group["group_index"] = iter_id
        group["question"] = ""
        for trajectory in group["trajectories"]:
            trajectory["selection_label"] = "representative success" if trajectory["score_final"] > 0 else "representative failure"
        groups.append(group)

    trajectories = [trajectory for group in groups for trajectory in group["trajectories"]]
    sub_trajectory_count = sum(item["sub_trajectory_count"] for item in trajectories)
    return {
        "version": 3,
        "run_name": dump_dir.name,
        "title": f"Query {query_id} across training",
        "description": question,
        "group_filter_label": "All iterations",
        "iter_id": "comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rollout_count": len(trajectories),
        "group_count": len(groups),
        "pass_count": sum(item["score_final"] > 0 for item in trajectories),
        "sub_trajectory_count": sub_trajectory_count,
        "empty_summary_count": sum(item["empty_summary_count"] for item in trajectories),
        "metrics": [
            [len(groups), "iterations"],
            [len(trajectories), "selected rollouts"],
            [sub_trajectory_count, "sub-trajectories"],
            [sum(item["empty_summary_count"] for item in trajectories), "empty summaries"],
        ],
        "groups": groups,
    }


def resolve_tokenizer(dump_dir: Path, explicit: Path | None) -> Path:
    candidates = [explicit, dump_dir / "tokenizer.json"]
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError("tokenizer.json was not found; pass --tokenizer-json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--query-id")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--expected-dp-shards", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-cache", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dump_dir = args.dump_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.max_iterations <= 0:
        raise SystemExit("--max-iterations must be positive")
    if not dump_dir.is_dir():
        raise SystemExit(f"dump directory not found: {dump_dir}")

    complete, incomplete, shard_count = discover_complete_iterations(dump_dir, args.expected_dp_shards)
    cache_path = args.index_cache.expanduser().resolve() if args.index_cache else output.with_suffix(".index.json")
    rows, cached_files, scanned_files = build_rollout_index(dump_dir, complete, cache_path)
    query_id = choose_query(rows, args.query_id)
    selections = select_rollouts(rows, query_id, args.max_iterations)
    data = build_comparison_data(
        dump_dir,
        query_id,
        selections,
        TokenizerDecoder(resolve_tokenizer(dump_dir, args.tokenizer_json)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(data), encoding="utf-8")
    print(f"html: {output}")
    print(f"query_id: {query_id}")
    print(f"complete iterations: {len(complete)} (latest {max(complete)})")
    print(f"dp shards: {shard_count}; cached files: {cached_files}; scanned files: {scanned_files}")
    if incomplete:
        print(f"ignored incomplete iterations: {sorted(incomplete)}")


if __name__ == "__main__":
    main()
