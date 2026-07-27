from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_viewer_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "bcplus_rollout_viewer.py"
    module_name = "test_bcplus_rollout_viewer_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeDecoder:
    def __init__(self, values: dict[int, str]) -> None:
        self._values = values

    def decode(self, token_ids) -> str:
        return self._values[token_ids[0]]


def _row(
    *,
    rollout_id: int,
    group_index: int,
    sub_traj_index: int,
    total_sub_trajs: int,
    is_final: bool,
    outcome: str,
    prompt_token: int,
    response_token: int,
    score: float,
) -> dict:
    return {
        "iter_id": 17,
        "group_index": group_index,
        "rollout_id": rollout_id,
        "sub_traj_index": sub_traj_index,
        "total_sub_trajs": total_sub_trajs,
        "is_final": is_final,
        "outcome": outcome,
        "summary_source": "" if is_final else "extracted",
        "score_raw": score,
        "score_final": score,
        "advantage": 0.5 if score else -0.5,
        "prompt_ids": [prompt_token],
        "response_ids": [response_token],
        "loss_mask": [1],
    }


@pytest.mark.unit
def test_parse_response_preserves_tool_and_summary_segments():
    viewer = _load_viewer_module()
    response = """research notes</think>
<tool_call>
<function=search>
<parameter=query>example query</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
[42] result text
</tool_response><|im_end|>
<|im_start|>assistant
<think>
compressing notes</think>
<summary>
handover text
</summary>"""

    segments = viewer.parse_response(response)

    assert [segment["type"] for segment in segments] == [
        "thinking",
        "tool_call",
        "tool_response",
        "thinking",
        "summary",
    ]
    assert segments[0]["text"] == "research notes"
    assert segments[1]["label"] == "search"
    assert segments[1]["parameters"] == [{"name": "query", "value": "example query"}]
    assert segments[2]["text"] == "[42] result text"
    assert segments[-1]["text"] == "handover text"


@pytest.mark.unit
def test_parse_response_marks_empty_summary_block():
    viewer = _load_viewer_module()

    segments = viewer.parse_response("notes</think><summary>\n</summary>")

    assert segments[-1] == {
        "type": "summary_empty",
        "label": "Handover summary (empty)",
        "text": "",
    }


@pytest.mark.unit
def test_build_view_data_groups_and_orders_selected_rollouts(tmp_path: Path):
    viewer = _load_viewer_module()
    prompt = "Question: Which answer?\n\nYour response should contain:\n..."
    decoder = _FakeDecoder(
        {
            1: prompt,
            2: "first thought</think><summary>continue</summary>",
            3: "second thought</think><tool_call><function=finish><parameter=exact_answer>A</parameter></function></tool_call>",
            4: "other thought</think><tool_call><function=finish><parameter=exact_answer>B</parameter></function></tool_call>",
        }
    )
    rows = [
        _row(
            rollout_id=20,
            group_index=7,
            sub_traj_index=0,
            total_sub_trajs=1,
            is_final=True,
            outcome="finished",
            prompt_token=1,
            response_token=4,
            score=0.0,
        ),
        _row(
            rollout_id=10,
            group_index=7,
            sub_traj_index=1,
            total_sub_trajs=2,
            is_final=True,
            outcome="finished",
            prompt_token=1,
            response_token=3,
            score=1.0,
        ),
        _row(
            rollout_id=10,
            group_index=7,
            sub_traj_index=0,
            total_sub_trajs=2,
            is_final=False,
            outcome="compressed",
            prompt_token=1,
            response_token=2,
            score=1.0,
        ),
    ]

    data = viewer.build_view_data(rows, decoder, tmp_path / "run-name", 17, [20, 10])

    assert data["rollout_count"] == 2
    assert data["sub_trajectory_count"] == 3
    assert data["pass_count"] == 1
    assert data["empty_summary_count"] == 0
    assert data["groups"][0]["question"] == "Which answer?"
    assert [item["rollout_id"] for item in data["groups"][0]["trajectories"]] == [
        10,
        20,
    ]
    assert [item["index"] for item in data["groups"][0]["trajectories"][0]["sub_trajectories"]] == [0, 1]


@pytest.mark.unit
def test_build_view_data_batched_merges_rollouts_from_the_same_group(tmp_path: Path, monkeypatch):
    viewer = _load_viewer_module()
    prompt = "Question: Which answer?\n\nYour response should contain:\n..."
    decoder = _FakeDecoder(
        {
            1: prompt,
            2: "thought</think><tool_call><function=finish><parameter=exact_answer>A</parameter></function></tool_call>",
            3: "thought</think><tool_call><function=finish><parameter=exact_answer>B</parameter></function></tool_call>",
        }
    )
    rows_by_rollout = {
        10: [
            _row(
                rollout_id=10,
                group_index=7,
                sub_traj_index=0,
                total_sub_trajs=1,
                is_final=True,
                outcome="finished",
                prompt_token=1,
                response_token=2,
                score=1.0,
            )
        ],
        20: [
            _row(
                rollout_id=20,
                group_index=7,
                sub_traj_index=0,
                total_sub_trajs=1,
                is_final=True,
                outcome="finished",
                prompt_token=1,
                response_token=3,
                score=0.0,
            )
        ],
    }
    calls = []

    def fake_load(dump_dir, iter_id, rollout_ids):
        calls.append(list(rollout_ids))
        return rows_by_rollout[rollout_ids[0]]

    monkeypatch.setattr(viewer, "load_selected_rows", fake_load)

    data = viewer.build_view_data_batched(tmp_path, 17, [20, 10], decoder)

    assert calls == [[20], [10]]
    assert data["group_count"] == 1
    assert data["rollout_count"] == 2
    assert [item["rollout_id"] for item in data["groups"][0]["trajectories"]] == [
        10,
        20,
    ]


@pytest.mark.unit
def test_render_html_escapes_embedded_script_content():
    viewer = _load_viewer_module()
    data = {
        "run_name": "run",
        "iter_id": 17,
        "generated_at": "now",
        "rollout_count": 0,
        "group_count": 0,
        "pass_count": 0,
        "sub_trajectory_count": 0,
        "groups": [{"question": "</script><script>alert(1)</script>"}],
    }

    output = viewer.render_html(data)

    assert "</script><script>alert(1)</script>" not in output
    assert r"\u003c/script\u003e\u003cscript\u003ealert(1)" in output
    assert "fetch(" not in output
    assert "http://" not in output
    assert "https://" not in output


@pytest.mark.unit
def test_validate_rollout_rows_rejects_non_contiguous_indices():
    viewer = _load_viewer_module()
    rows = [
        _row(
            rollout_id=10,
            group_index=7,
            sub_traj_index=0,
            total_sub_trajs=2,
            is_final=False,
            outcome="compressed",
            prompt_token=1,
            response_token=2,
            score=0.0,
        ),
        _row(
            rollout_id=10,
            group_index=7,
            sub_traj_index=2,
            total_sub_trajs=2,
            is_final=True,
            outcome="finished",
            prompt_token=1,
            response_token=3,
            score=0.0,
        ),
    ]

    with pytest.raises(ValueError, match="non-contiguous"):
        viewer._validate_rollout_rows(rows, 17, 10)


@pytest.mark.unit
def test_load_selected_rows_combines_dp_files(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    viewer = _load_viewer_module()

    rows = [
        _row(
            rollout_id=10,
            group_index=7,
            sub_traj_index=0,
            total_sub_trajs=2,
            is_final=False,
            outcome="compressed",
            prompt_token=1,
            response_token=2,
            score=1.0,
        ),
        _row(
            rollout_id=10,
            group_index=7,
            sub_traj_index=1,
            total_sub_trajs=2,
            is_final=True,
            outcome="finished",
            prompt_token=1,
            response_token=3,
            score=1.0,
        ),
    ]
    pq.write_table(pa.Table.from_pylist([rows[0]]), tmp_path / "rollouts_iter_00017_dp0.parquet")
    pq.write_table(pa.Table.from_pylist([rows[1]]), tmp_path / "rollouts_iter_00017_dp1.parquet")

    loaded = viewer.load_selected_rows(tmp_path, 17, [10])

    assert sorted(row["sub_traj_index"] for row in loaded) == [0, 1]
    assert {row["source_file"] for row in loaded} == {
        "rollouts_iter_00017_dp0.parquet",
        "rollouts_iter_00017_dp1.parquet",
    }
    with pytest.raises(ValueError, match="not found"):
        viewer.load_selected_rows(tmp_path, 17, [99])
