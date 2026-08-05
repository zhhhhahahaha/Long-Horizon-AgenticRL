from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import bcplus_experiment_viewer as viewer


def _index_row(
    iter_id: int,
    query_id: str,
    rollout_id: int,
    score: float,
    *,
    group_index: int | None = None,
) -> dict:
    return {
        "iter_id": iter_id,
        "query_id": query_id,
        "group_index": iter_id if group_index is None else group_index,
        "rollout_id": rollout_id,
        "total_sub_trajs": 1,
        "is_final": True,
        "outcome": "finished",
        "score_final": score,
    }


@pytest.mark.unit
def test_discover_complete_iterations_ignores_partial_tail(tmp_path: Path):
    for name in (
        "rollouts_iter_00000_dp0.parquet",
        "rollouts_iter_00000_dp1.parquet",
        "rollouts_iter_00001_dp0.parquet",
    ):
        (tmp_path / name).touch()

    complete, incomplete, shard_count = viewer.discover_complete_iterations(tmp_path)

    assert shard_count == 2
    assert list(complete) == [0]
    assert incomplete == {1: [1]}

    with pytest.raises(ValueError, match="positive integer"):
        viewer.discover_complete_iterations(tmp_path, expected_dp_shards=0)

    (tmp_path / "rollouts_iter_00002_dp2.parquet").touch()
    with pytest.raises(ValueError, match="unexpected DP ranks"):
        viewer.discover_complete_iterations(tmp_path, expected_dp_shards=2)


@pytest.mark.unit
def test_index_cache_reuses_unchanged_parquet(tmp_path: Path, monkeypatch):
    path = tmp_path / "rollouts_iter_00000_dp0.parquet"
    pq.write_table(pa.Table.from_pylist([_index_row(0, "q1", 10, 1.0)]), path)
    cache_path = tmp_path / "index.json"

    rows, cached, scanned = viewer.build_rollout_index(tmp_path, {0: [path]}, cache_path)
    assert (len(rows), cached, scanned) == (1, 0, 1)

    monkeypatch.setattr(viewer, "_read_final_rows", lambda path: pytest.fail("cache miss"))
    rows, cached, scanned = viewer.build_rollout_index(tmp_path, {0: [path]}, cache_path)
    assert (len(rows), cached, scanned) == (1, 1, 0)


@pytest.mark.unit
def test_index_cache_rescans_invalid_structure_and_rejects_duplicate_rollouts(tmp_path: Path):
    first = tmp_path / "rollouts_iter_00000_dp0.parquet"
    second = tmp_path / "rollouts_iter_00000_dp1.parquet"
    row = _index_row(0, "q1", 10, 1.0)
    pq.write_table(pa.Table.from_pylist([row]), first)
    cache_path = tmp_path / "index.json"
    cache_path.write_text(
        json.dumps({"version": viewer.INDEX_VERSION, "dump_dir": str(tmp_path), "files": []})
    )

    rows, cached, scanned = viewer.build_rollout_index(tmp_path, {0: [first]}, cache_path)
    assert (len(rows), cached, scanned) == (1, 0, 1)

    cache = json.loads(cache_path.read_text())
    cache["files"][first.name]["rows"][0]["iter_id"] = 1
    cache_path.write_text(json.dumps(cache))
    rows, cached, scanned = viewer.build_rollout_index(tmp_path, {0: [first]}, cache_path)
    assert (len(rows), cached, scanned) == (1, 0, 1)

    pq.write_table(pa.Table.from_pylist([row]), second)
    with pytest.raises(ValueError, match="duplicate final rollout row"):
        viewer.build_rollout_index(tmp_path, {0: [first, second]}, cache_path)


@pytest.mark.unit
def test_choose_query_and_select_success_failure_pairs():
    rows = []
    for iter_id, successes in ((0, 1), (5, 2), (9, 3)):
        for sample in range(4):
            rows.append(
                _index_row(
                    iter_id,
                    "changing",
                    iter_id * 10 + sample,
                    float(sample < successes),
                )
            )
    for iter_id in (0, 5, 9):
        for sample in range(4):
            rows.append(_index_row(iter_id, "stable", 100 + iter_id * 10 + sample, 1.0))

    query_id = viewer.choose_query(rows, None)
    selections = viewer.select_rollouts(rows, query_id, max_iterations=2)

    assert query_id == "changing"
    assert [item["iter_id"] for item in selections] == [5, 9]
    assert all(len(item["rollout_ids"]) == 2 for item in selections)

    with pytest.raises(ValueError, match="max_iterations must be positive"):
        viewer.select_rollouts(rows, query_id, max_iterations=0)

    duplicated_query = [
        _index_row(1, "duplicate", 1, 1.0, group_index=10),
        _index_row(1, "duplicate", 2, 0.0, group_index=11),
    ]
    with pytest.raises(ValueError, match="multiple groups"):
        viewer.select_rollouts(duplicated_query, "duplicate", max_iterations=1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
