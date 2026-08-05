#!/usr/bin/env python3
"""Build a self-contained HTML viewer for BC+ rollout parquet dumps."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


REQUIRED_COLUMNS = (
    "iter_id",
    "group_index",
    "rollout_id",
    "sub_traj_index",
    "total_sub_trajs",
    "is_final",
    "outcome",
    "summary_source",
    "score_raw",
    "score_final",
    "advantage",
    "prompt_ids",
    "response_ids",
    "loss_mask",
)


class Decoder(Protocol):
    def decode(self, token_ids: Sequence[int]) -> str: ...


class TokenizerDecoder:
    def __init__(self, tokenizer_json: Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError("tokenizers is required; run this tool from the slime training environment") from error

        self._tokenizer = Tokenizer.from_file(str(tokenizer_json))

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)


def _unique_ids(values: Sequence[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate rollout id: {value}")
        seen.add(value)
        result.append(value)
    return result


def load_selected_rows(dump_dir: Path, iter_id: int, rollout_ids: Sequence[int]) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required; run this tool from the slime training environment") from error

    pattern = f"rollouts_iter_{iter_id:05d}_dp*.parquet"
    parquet_paths = sorted(dump_dir.glob(pattern))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet files matching {dump_dir / pattern}")

    requested = set(rollout_ids)
    rows: list[dict[str, Any]] = []
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        missing = sorted(set(REQUIRED_COLUMNS) - set(parquet_file.schema_arrow.names))
        if missing:
            raise ValueError(f"{parquet_path} is missing columns: {', '.join(missing)}")

        table = pq.read_table(
            parquet_path,
            columns=list(REQUIRED_COLUMNS),
            filters=[("rollout_id", "in", list(requested))],
            use_threads=False,
        )
        for row in table.to_pylist():
            row["source_file"] = parquet_path.name
            rows.append(row)

    found = {int(row["rollout_id"]) for row in rows}
    missing_ids = sorted(requested - found)
    if missing_ids:
        raise ValueError(f"rollout ids not found in iter {iter_id}: {missing_ids}")
    return rows


def _extract_question(prompt_text: str) -> str:
    match = re.search(
        r"Question:\s*(.*?)\n\nYour response should contain:",
        prompt_text,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else prompt_text.strip()


def _strip_role_end(text: str) -> str:
    return re.sub(r"<\|im_end\|>\s*$", "", text, count=1).strip()


def _parse_tool_call(block: str) -> dict[str, Any]:
    function_match = re.search(r"<function=([^>]+)>\s*(.*?)\s*</function>", block, flags=re.DOTALL)
    if not function_match:
        return {"type": "raw", "label": "Unparsed tool call", "text": block.strip()}

    function_name = function_match.group(1).strip()
    body = function_match.group(2)
    parameters = [
        {"name": match.group(1).strip(), "value": match.group(2).strip()}
        for match in re.finditer(
            r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
            body,
            flags=re.DOTALL,
        )
    ]
    return {
        "type": "finish" if function_name == "finish" else "tool_call",
        "label": function_name,
        "parameters": parameters,
        "text": block.strip(),
    }


def _append_raw_segment(segments: list[dict[str, Any]], text: str) -> None:
    value = text.strip()
    if value:
        segments.append({"type": "raw", "label": "Assistant output", "text": value})


def _parse_assistant_chunk(text: str) -> list[dict[str, Any]]:
    text = _strip_role_end(text)
    segments: list[dict[str, Any]] = []

    think_start = re.match(r"\s*<think>\s*", text)
    content_start = think_start.end() if think_start else 0
    think_end = text.find("</think>", content_start)
    if think_end >= 0:
        thinking = text[content_start:think_end].strip()
        if thinking:
            segments.append({"type": "thinking", "label": "Thinking", "text": thinking})
        remainder = text[think_end + len("</think>") :]
    elif text.strip():
        segments.append(
            {
                "type": "thinking",
                "label": "Thinking (unterminated)",
                "text": text[content_start:].strip(),
            }
        )
        return segments
    else:
        return segments

    tagged = re.compile(
        r"(<tool_call>.*?(?:</tool_call>|$)|<summary>.*?(?:</summary>|$))",
        flags=re.DOTALL,
    )
    cursor = 0
    for match in tagged.finditer(remainder):
        _append_raw_segment(segments, remainder[cursor : match.start()])
        block = match.group(1)
        if block.startswith("<tool_call>"):
            segments.append(_parse_tool_call(block))
        else:
            summary = re.sub(r"^<summary>|</summary>$", "", block.strip()).strip()
            segments.append(
                {
                    "type": "summary" if summary else "summary_empty",
                    "label": "Handover summary" if summary else "Handover summary (empty)",
                    "text": summary,
                }
            )
        cursor = match.end()
    _append_raw_segment(segments, remainder[cursor:])
    return segments


def parse_response(response_text: str) -> list[dict[str, Any]]:
    """Split a decoded response into assistant, tool, and compression segments."""
    role_marker = re.compile(r"<\|im_start\|>(assistant|user|system)\n")
    matches = list(role_marker.finditer(response_text))
    chunks: list[tuple[str, str]] = []

    if not matches:
        chunks.append(("assistant", response_text))
    else:
        prefix = response_text[: matches[0].start()]
        if prefix.strip():
            chunks.append(("assistant", prefix))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(response_text)
            chunks.append((match.group(1), response_text[match.end() : end]))

    segments: list[dict[str, Any]] = []
    for role, chunk in chunks:
        chunk = _strip_role_end(chunk)
        if not chunk:
            continue
        if role == "assistant":
            segments.extend(_parse_assistant_chunk(chunk))
            continue

        tool_response = re.search(r"<tool_response>\s*(.*?)\s*</tool_response>", chunk, flags=re.DOTALL)
        if tool_response:
            segments.append(
                {
                    "type": "tool_response",
                    "label": "Tool response",
                    "text": tool_response.group(1).strip(),
                }
            )
        elif "operational context is full" in chunk:
            segments.append(
                {
                    "type": "compression_request",
                    "label": "Compression request",
                    "text": chunk.strip(),
                }
            )
        else:
            segments.append({"type": role, "label": f"{role.title()} message", "text": chunk.strip()})
    return segments


def _validate_rollout_rows(rows: Sequence[dict[str, Any]], iter_id: int, rollout_id: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["sub_traj_index"]))
    indices = [int(row["sub_traj_index"]) for row in ordered]
    if len(indices) != len(set(indices)):
        raise ValueError(f"rollout {rollout_id} has duplicate sub-trajectory indices")
    if indices != list(range(len(indices))):
        raise ValueError(f"rollout {rollout_id} has non-contiguous sub-trajectory indices: {indices}")

    expected_counts = {int(row["total_sub_trajs"]) for row in ordered}
    if expected_counts != {len(ordered)}:
        raise ValueError(
            f"rollout {rollout_id} total_sub_trajs={sorted(expected_counts)} but found {len(ordered)} rows"
        )
    row_iters = {int(row["iter_id"]) for row in ordered}
    if row_iters != {iter_id}:
        raise ValueError(f"rollout {rollout_id} has iter ids {sorted(row_iters)}")
    final_rows = [row for row in ordered if bool(row["is_final"])]
    if len(final_rows) != 1 or final_rows[0] is not ordered[-1]:
        raise ValueError(f"rollout {rollout_id} must have exactly one final row at the end")
    return ordered


def build_view_data(
    rows: Sequence[dict[str, Any]],
    decoder: Decoder,
    dump_dir: Path,
    iter_id: int,
    rollout_ids: Sequence[int],
) -> dict[str, Any]:
    rows_by_rollout: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_rollout[int(row["rollout_id"])].append(row)

    groups: dict[int, dict[str, Any]] = {}
    for rollout_id in rollout_ids:
        ordered = _validate_rollout_rows(rows_by_rollout[rollout_id], iter_id, rollout_id)
        final_row = ordered[-1]
        group_index = int(final_row["group_index"])
        sub_trajectories: list[dict[str, Any]] = []
        question = ""
        total_prompt_tokens = 0
        total_response_tokens = 0
        total_trainable_tokens = 0

        for row in ordered:
            prompt_ids = [int(value) for value in row["prompt_ids"]]
            response_ids = [int(value) for value in row["response_ids"]]
            loss_mask = [int(value) for value in row["loss_mask"]]
            if len(loss_mask) != len(response_ids):
                raise ValueError(
                    f"rollout {rollout_id} sub-trajectory {row['sub_traj_index']} has "
                    f"loss_mask length {len(loss_mask)} != response length {len(response_ids)}"
                )

            prompt_text = decoder.decode(prompt_ids)
            response_text = decoder.decode(response_ids)
            if not question:
                question = _extract_question(prompt_text)
            trainable_tokens = sum(loss_mask)
            total_prompt_tokens += len(prompt_ids)
            total_response_tokens += len(response_ids)
            total_trainable_tokens += trainable_tokens
            segments = parse_response(response_text)
            has_empty_summary = any(segment["type"] == "summary_empty" for segment in segments)
            sub_trajectories.append(
                {
                    "index": int(row["sub_traj_index"]),
                    "outcome": str(row["outcome"]),
                    "is_final": bool(row["is_final"]),
                    "summary_source": str(row["summary_source"]),
                    "prompt_tokens": len(prompt_ids),
                    "response_tokens": len(response_ids),
                    "trainable_tokens": trainable_tokens,
                    "prompt": prompt_text,
                    "segments": segments,
                    "has_empty_summary": has_empty_summary,
                }
            )

        empty_summary_count = sum(sub["has_empty_summary"] for sub in sub_trajectories)
        trajectory = {
            "rollout_id": rollout_id,
            "score_raw": float(final_row["score_raw"]),
            "score_final": float(final_row["score_final"]),
            "advantage": float(final_row["advantage"]),
            "outcome": str(final_row["outcome"]),
            "sub_trajectory_count": len(ordered),
            "prompt_tokens": total_prompt_tokens,
            "response_tokens": total_response_tokens,
            "trainable_tokens": total_trainable_tokens,
            "empty_summary_count": empty_summary_count,
            "sub_trajectories": sub_trajectories,
        }
        group = groups.setdefault(
            group_index,
            {"group_index": group_index, "question": question, "trajectories": []},
        )
        if group["question"] != question:
            raise ValueError(f"group {group_index} contains inconsistent questions")
        group["trajectories"].append(trajectory)

    sorted_groups = sorted(groups.values(), key=lambda group: int(group["group_index"]))
    for group in sorted_groups:
        group["trajectories"].sort(key=lambda trajectory: (-trajectory["score_final"], trajectory["rollout_id"]))

    return _view_data_from_groups(sorted_groups, dump_dir, iter_id)


def _view_data_from_groups(sorted_groups: list[dict[str, Any]], dump_dir: Path, iter_id: int) -> dict[str, Any]:
    trajectories = [trajectory for group in sorted_groups for trajectory in group["trajectories"]]
    return {
        "version": 2,
        "run_name": dump_dir.name,
        "iter_id": iter_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rollout_count": len(trajectories),
        "group_count": len(sorted_groups),
        "pass_count": sum(trajectory["score_final"] > 0 for trajectory in trajectories),
        "sub_trajectory_count": sum(trajectory["sub_trajectory_count"] for trajectory in trajectories),
        "empty_summary_count": sum(trajectory["empty_summary_count"] for trajectory in trajectories),
        "groups": sorted_groups,
    }


def build_view_data_batched(
    dump_dir: Path,
    iter_id: int,
    rollout_ids: Sequence[int],
    decoder: Decoder,
) -> dict[str, Any]:
    """Load and decode one rollout at a time to limit peak memory usage."""
    groups: dict[int, dict[str, Any]] = {}
    for rollout_id in rollout_ids:
        rows = load_selected_rows(dump_dir, iter_id, [rollout_id])
        partial = build_view_data(rows, decoder, dump_dir, iter_id, [rollout_id])
        group = partial["groups"][0]
        group_index = int(group["group_index"])
        existing = groups.get(group_index)
        if existing is None:
            groups[group_index] = group
        else:
            if existing["question"] != group["question"]:
                raise ValueError(f"group {group_index} contains inconsistent questions")
            existing["trajectories"].extend(group["trajectories"])

    sorted_groups = sorted(groups.values(), key=lambda group: int(group["group_index"]))
    for group in sorted_groups:
        group["trajectories"].sort(key=lambda trajectory: (-trajectory["score_final"], trajectory["rollout_id"]))
    return _view_data_from_groups(sorted_groups, dump_dir, iter_id)


def _json_for_html(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BC+ rollout viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f7;
      --surface: #ffffff;
      --surface-muted: #f0f2f4;
      --line: #d7dce1;
      --text: #18202a;
      --muted: #596574;
      --accent: #16697a;
      --pass: #18794e;
      --fail: #b42318;
      --warn: #9a6700;
      --tool: #275dad;
      --summary: #6b4f9e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, select { font: inherit; letter-spacing: 0; }
    button { cursor: pointer; }
    header {
      background: var(--surface);
      border-bottom: 1px solid var(--line);
      padding: 24px clamp(16px, 4vw, 48px) 18px;
    }
    h1 { margin: 0; font-size: 24px; line-height: 1.25; }
    .subtitle { color: var(--muted); margin-top: 6px; overflow-wrap: anywhere; }
    .metrics { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 18px; }
    .metric strong { display: block; font-size: 20px; line-height: 1.15; }
    .metric span { color: var(--muted); font-size: 12px; }
    .data-warning {
      margin: 14px clamp(16px, 4vw, 48px) 0;
      border: 1px solid #e4c680;
      border-radius: 5px;
      background: #fff8e6;
      color: #6f4d00;
      padding: 10px 12px;
    }
    .comparison-intro {
      padding: 18px clamp(16px, 4vw, 48px) 0;
      max-width: 1180px;
      white-space: pre-wrap;
      font-size: 15px;
    }
    .toolbar {
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      padding: 10px clamp(16px, 4vw, 48px);
      background: rgba(245, 246, 247, 0.96);
      border-bottom: 1px solid var(--line);
    }
    .toolbar input { flex: 1 1 260px; min-width: 0; }
    .toolbar input, .toolbar select, .toolbar button {
      height: 34px;
      border: 1px solid #b9c1c9;
      border-radius: 5px;
      background: var(--surface);
      color: var(--text);
      padding: 0 10px;
    }
    .toolbar button:hover { border-color: var(--accent); color: var(--accent); }
    main { padding: 0 clamp(16px, 4vw, 48px) 48px; }
    .group { padding: 28px 0 34px; border-bottom: 1px solid var(--line); }
    .group:last-child { border-bottom: 0; }
    .group-heading { display: flex; gap: 12px; align-items: baseline; margin-bottom: 10px; }
    .group-heading h2 { margin: 0; font-size: 18px; }
    .group-heading span { color: var(--muted); }
    .question {
      margin: 0 0 18px;
      max-width: 1100px;
      font-size: 15px;
      white-space: pre-wrap;
    }
    .trajectory-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }
    .trajectory {
      min-width: 0;
      background: var(--surface);
      border: 1px solid var(--line);
      border-top: 3px solid var(--fail);
      border-radius: 6px;
      overflow: hidden;
    }
    .trajectory.pass { border-top-color: var(--pass); }
    .trajectory-head { padding: 14px 16px 12px; border-bottom: 1px solid var(--line); }
    .trajectory-title { display: flex; justify-content: space-between; gap: 10px; align-items: center; }
    .trajectory-title h3 { margin: 0; font-size: 16px; }
    .copy-button {
      border: 1px solid #b9c1c9;
      border-radius: 4px;
      background: white;
      padding: 4px 8px;
      color: var(--muted);
    }
    .badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 9px; }
    .badge {
      border: 1px solid var(--line);
      border-radius: 4px;
      background: var(--surface-muted);
      padding: 2px 6px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .badge.pass { border-color: #8dc7aa; background: #edf8f2; color: var(--pass); }
    .badge.fail { border-color: #efaaa5; background: #fff1f0; color: var(--fail); }
    .badge.warn { border-color: #e4c680; background: #fff8e6; color: var(--warn); }
    details { border-bottom: 1px solid var(--line); }
    details:last-child { border-bottom: 0; }
    summary {
      cursor: pointer;
      padding: 10px 16px;
      background: var(--surface-muted);
      color: var(--text);
      font-weight: 600;
      overflow-wrap: anywhere;
    }
    summary:hover { color: var(--accent); }
    .sub-meta { color: var(--muted); font-size: 12px; font-weight: 400; margin-left: 6px; }
    .segments { padding: 10px 14px 14px; }
    .segment { margin: 8px 0; border-left: 3px solid #aab3bd; padding-left: 10px; min-width: 0; }
    .segment.thinking { border-left-color: #7d8792; }
    .segment.tool_call, .segment.tool_response { border-left-color: var(--tool); }
    .segment.summary { border-left-color: var(--summary); }
    .segment.summary_empty { border-left-color: var(--fail); color: var(--fail); }
    .segment.finish { border-left-color: var(--pass); }
    .segment.compression_request { border-left-color: var(--warn); }
    .segment-label { font-weight: 700; margin-bottom: 4px; overflow-wrap: anywhere; }
    .segment details { border: 0; }
    .segment details summary { padding: 4px 0; background: transparent; font-weight: 700; }
    pre {
      margin: 0;
      max-height: 480px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      letter-spacing: 0;
    }
    .parameters { display: grid; gap: 7px; }
    .parameter-name { color: var(--muted); font: 600 11px/1.4 ui-monospace, monospace; }
    .empty { padding: 40px 0; color: var(--muted); }
    footer { padding: 20px clamp(16px, 4vw, 48px) 32px; color: var(--muted); font-size: 12px; }
    @media (max-width: 900px) {
      .trajectory-grid { grid-template-columns: minmax(0, 1fr); }
      .toolbar { position: static; }
    }
    @media (max-width: 520px) {
      header { padding-top: 18px; }
      h1 { font-size: 20px; }
      .metrics { gap: 12px 18px; }
      .toolbar input, .toolbar select, .toolbar button { flex: 1 1 140px; }
      .group { padding-top: 22px; }
      .trajectory-title { align-items: flex-start; }
    }
  </style>
</head>
<body>
  <header>
    <h1 id="title">BC+ rollout viewer</h1>
    <div id="subtitle" class="subtitle"></div>
    <div id="metrics" class="metrics"></div>
  </header>
  <div id="dataWarning" class="data-warning" hidden></div>
  <div id="comparisonIntro" class="comparison-intro" hidden></div>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Search trajectories">
    <select id="groupFilter" aria-label="Filter by group"></select>
    <select id="outcomeFilter" aria-label="Filter by outcome"></select>
    <select id="scoreFilter" aria-label="Filter by score">
      <option value="all">All scores</option>
      <option value="pass">Score 1</option>
      <option value="fail">Score 0</option>
    </select>
    <button id="expandAll" type="button">Expand all</button>
    <button id="collapseAll" type="button">Collapse all</button>
  </div>
  <main id="content"></main>
  <footer id="footer"></footer>
  <script id="rollout-data" type="application/json">__DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById('rollout-data').textContent);
    const content = document.getElementById('content');
    const search = document.getElementById('search');
    const groupFilter = document.getElementById('groupFilter');
    const outcomeFilter = document.getElementById('outcomeFilter');
    const scoreFilter = document.getElementById('scoreFilter');

    function el(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function metric(value, label) {
      const node = el('div', 'metric');
      node.append(el('strong', '', String(value)), el('span', '', label));
      return node;
    }

    function badge(text, kind) { return el('span', `badge ${kind || ''}`, text); }

    function appendText(container, text) {
      container.appendChild(el('pre', '', text || '(empty)'));
    }

    function appendLazyText(details, text) {
      const load = () => {
        if (!details.open || details.dataset.loaded === 'true') return;
        appendText(details, text);
        details.dataset.loaded = 'true';
        details.removeEventListener('toggle', load);
      };
      details.dataset.loaded = 'false';
      details.addEventListener('toggle', load);
      load();
    }

    function renderSegment(segment) {
      const node = el('div', `segment ${segment.type}`);
      const collapsible = ['thinking', 'tool_response', 'compression_request', 'raw', 'user', 'system'].includes(segment.type);
      if (collapsible) {
        const details = el('details');
        details.appendChild(el('summary', '', segment.label));
        appendLazyText(details, segment.text);
        node.appendChild(details);
        return node;
      }

      node.appendChild(el('div', 'segment-label', segment.label));
      if (segment.parameters && segment.parameters.length) {
        const parameters = el('div', 'parameters');
        for (const parameter of segment.parameters) {
          const item = el('div');
          item.appendChild(el('div', 'parameter-name', parameter.name));
          appendText(item, parameter.value);
          parameters.appendChild(item);
        }
        node.appendChild(parameters);
      } else {
        appendText(node, segment.text);
      }
      return node;
    }

    function trajectoryCopyText(group, trajectory) {
      const lines = [`Group ${group.group_index}`, group.question, '', `Rollout ${trajectory.rollout_id}`];
      for (const sub of trajectory.sub_trajectories) {
        lines.push('', `Sub-trajectory ${sub.index + 1}: ${sub.outcome}`, sub.prompt);
        for (const segment of sub.segments) {
          lines.push('', `[${segment.label}]`);
          if (segment.parameters) {
            for (const parameter of segment.parameters) lines.push(`${parameter.name}: ${parameter.value}`);
          } else {
            lines.push(segment.text || '');
          }
        }
      }
      return lines.join('\n');
    }

    function trajectorySearchText(group, trajectory) {
      const values = [group.question];
      for (const sub of trajectory.sub_trajectories) {
        for (const segment of sub.segments) {
          values.push(segment.label || '', segment.text || '');
          for (const parameter of segment.parameters || []) {
            values.push(parameter.name, parameter.value);
          }
        }
      }
      return values.join('\n').toLowerCase();
    }

    function renderTrajectory(group, trajectory) {
      const pass = trajectory.score_final > 0;
      const article = el('article', `trajectory ${pass ? 'pass' : ''}`);
      article.dataset.group = String(group.group_index);
      article.dataset.outcome = trajectory.outcome;
      article.dataset.score = pass ? 'pass' : 'fail';
      article.searchText = trajectorySearchText(group, trajectory);

      const head = el('div', 'trajectory-head');
      const title = el('div', 'trajectory-title');
      title.appendChild(el('h3', '', `Rollout ${trajectory.rollout_id}`));
      const copy = el('button', 'copy-button', 'Copy');
      copy.type = 'button';
      copy.addEventListener('click', async () => {
        await navigator.clipboard.writeText(trajectoryCopyText(group, trajectory));
        copy.textContent = 'Copied';
        setTimeout(() => { copy.textContent = 'Copy'; }, 1200);
      });
      title.appendChild(copy);
      head.appendChild(title);
      const badges = el('div', 'badges');
      badges.append(
        badge(`score ${trajectory.score_final}`, pass ? 'pass' : 'fail'),
        badge(trajectory.outcome, trajectory.outcome === 'finished' ? '' : 'warn'),
        badge(`${trajectory.sub_trajectory_count} sub-trajectories`),
        badge(`${trajectory.response_tokens.toLocaleString()} response tokens`),
        badge(`${trajectory.trainable_tokens.toLocaleString()} trainable`)
      );
      if (trajectory.selection_label) badges.appendChild(badge(trajectory.selection_label));
      if (trajectory.empty_summary_count) {
        badges.appendChild(badge(`${trajectory.empty_summary_count} empty summaries`, 'warn'));
      }
      head.appendChild(badges);
      article.appendChild(head);

      for (const sub of trajectory.sub_trajectories) {
        const details = el('details', 'sub-trajectory');
        details.open = true;
        const summary = el('summary');
        summary.appendChild(document.createTextNode(`Sub-trajectory ${sub.index + 1}: ${sub.outcome}`));
        summary.appendChild(el('span', 'sub-meta', `${sub.response_tokens.toLocaleString()} response / ${sub.trainable_tokens.toLocaleString()} trainable tokens`));
        if (sub.has_empty_summary) summary.appendChild(el('span', 'sub-meta', 'empty handover summary'));
        details.appendChild(summary);
        const segments = el('div', 'segments');

        const prompt = el('div', 'segment prompt');
        const promptDetails = el('details');
        promptDetails.appendChild(el('summary', '', 'Prompt context'));
        appendLazyText(promptDetails, sub.prompt);
        prompt.appendChild(promptDetails);
        segments.appendChild(prompt);
        for (const segment of sub.segments) segments.appendChild(renderSegment(segment));
        details.appendChild(segments);
        article.appendChild(details);
      }
      return article;
    }

    function renderAll() {
      content.replaceChildren();
      for (const group of data.groups) {
        const section = el('section', 'group');
        section.dataset.group = String(group.group_index);
        const heading = el('div', 'group-heading');
        heading.append(el('h2', '', group.label || `Group ${group.group_index}`), el('span', '', `${group.trajectories.length} selected rollouts`));
        section.appendChild(heading);
        if (group.question) section.appendChild(el('p', 'question', group.question));
        const grid = el('div', 'trajectory-grid');
        for (const trajectory of group.trajectories) {
          grid.appendChild(renderTrajectory(group, trajectory));
        }
        section.appendChild(grid);
        content.appendChild(section);
      }
    }

    function applyFilters() {
      const needle = search.value.trim().toLowerCase();
      let visible = 0;
      for (const section of content.querySelectorAll('.group')) {
        let groupVisible = 0;
        for (const node of section.querySelectorAll('.trajectory')) {
          const matches =
            (groupFilter.value === 'all' || node.dataset.group === groupFilter.value) &&
            (outcomeFilter.value === 'all' || node.dataset.outcome === outcomeFilter.value) &&
            (scoreFilter.value === 'all' || node.dataset.score === scoreFilter.value) &&
            (!needle || node.searchText.includes(needle));
          node.hidden = !matches;
          if (matches) {
            groupVisible += 1;
            visible += 1;
          }
        }
        section.hidden = groupVisible === 0;
      }
      let empty = content.querySelector('.empty');
      if (!visible && !empty) {
        empty = el('div', 'empty', 'No trajectories match the current filters.');
        content.appendChild(empty);
      } else if (visible && empty) {
        empty.remove();
      }
    }

    function addOption(select, value, label) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      select.appendChild(option);
    }

    document.getElementById('title').textContent = data.title || `Iter ${data.iter_id} rollout review`;
    document.getElementById('subtitle').textContent = data.run_name;
    const metrics = data.metrics || [
      [data.rollout_count, 'selected rollouts'],
      [data.group_count, 'prompt groups'],
      [data.pass_count, 'score 1'],
      [data.rollout_count - data.pass_count, 'score 0'],
      [data.sub_trajectory_count, 'sub-trajectories'],
      [data.empty_summary_count, 'empty summaries']
    ];
    document.getElementById('metrics').append(...metrics.map(item => metric(item[0], item[1])));
    if (data.description) {
      const intro = document.getElementById('comparisonIntro');
      intro.textContent = data.description;
      intro.hidden = false;
    }
    if (data.empty_summary_count) {
      const warning = document.getElementById('dataWarning');
      warning.textContent = `Data warning: ${data.empty_summary_count} <summary> blocks are empty even though the dump metadata labels them extracted. Their handover text cannot be reconstructed from this dump.`;
      warning.hidden = false;
    }
    document.getElementById('footer').textContent = `Generated ${data.generated_at}. Full model output is embedded in this internal report.`;

    addOption(groupFilter, 'all', data.group_filter_label || 'All groups');
    for (const group of data.groups) addOption(groupFilter, String(group.group_index), group.label || `Group ${group.group_index}`);
    const outcomes = [...new Set(data.groups.flatMap(group => group.trajectories.map(trajectory => trajectory.outcome)))].sort();
    addOption(outcomeFilter, 'all', 'All outcomes');
    for (const outcome of outcomes) addOption(outcomeFilter, outcome, outcome);
    for (const control of [search, groupFilter, outcomeFilter, scoreFilter]) control.addEventListener('input', applyFilters);
    document.getElementById('expandAll').addEventListener('click', () => document.querySelectorAll('details').forEach(node => { node.open = true; }));
    document.getElementById('collapseAll').addEventListener('click', () => document.querySelectorAll('details').forEach(node => { node.open = false; }));
    renderAll();
    applyFilters();
  </script>
</body>
</html>
"""


def render_html(data: dict[str, Any]) -> str:
    return HTML_TEMPLATE.replace("__DATA__", _json_for_html(data))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--iter", dest="iter_id", type=int, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--rollout-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_ids = _unique_ids(args.rollout_ids)
    dump_dir = args.dump_dir.expanduser().resolve()
    tokenizer_json = args.tokenizer_json.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not dump_dir.is_dir():
        raise SystemExit(f"dump directory not found: {dump_dir}")
    if not tokenizer_json.is_file():
        raise SystemExit(f"tokenizer json not found: {tokenizer_json}")

    data = build_view_data_batched(
        dump_dir,
        args.iter_id,
        rollout_ids,
        TokenizerDecoder(tokenizer_json),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(data), encoding="utf-8")
    print(f"html: {output}")
    print(
        f"groups: {data['group_count']} rollouts: {data['rollout_count']} "
        f"sub-trajectories: {data['sub_trajectory_count']}"
    )


if __name__ == "__main__":
    main()
