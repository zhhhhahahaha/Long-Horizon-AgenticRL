#!/usr/bin/env python3
"""Audit judge failures in saved BC+ training rollout dumps."""

from __future__ import annotations

import argparse
import gc
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def reward_value(sample: Any, key: str, default: float = 0.0) -> float:
    reward = value(sample, "reward", {})
    raw = value(reward, key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def metadata(sample: Any) -> Mapping[str, Any]:
    raw = value(sample, "metadata", {})
    return raw if isinstance(raw, Mapping) else {}


def is_final(sample: Any) -> bool:
    sibling = metadata(sample).get("_bcplus_sibling", {})
    return isinstance(sibling, Mapping) and sibling.get("is_final") is True


def analyze(path: Path) -> dict[str, Any]:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    samples = value(blob, "samples", [])
    groups: dict[Any, list[Any]] = defaultdict(list)
    for index, sample in enumerate(samples):
        rollout_id = value(sample, "rollout_id", value(sample, "index", index))
        groups[rollout_id].append(sample)

    finals = []
    malformed_groups = 0
    for siblings in groups.values():
        selected = [sample for sample in siblings if is_final(sample)]
        if len(selected) == 1:
            finals.append(selected[0])
        else:
            malformed_groups += 1
            if len(siblings) == 1:
                finals.append(siblings[0])

    failed = sum(reward_value(sample, "judge_failed") > 0 for sample in finals)
    finished = sum(bool(metadata(sample).get("_bcplus", {}).get("finished")) for sample in finals)
    correct = sum(reward_value(sample, "score") == 1.0 for sample in finals)
    result = {
        "file": path.name,
        "step": int(re.search(r"(\d+)$", path.stem).group(1)),
        "n_samples": len(samples),
        "n_rollouts": len(groups),
        "n_finals": len(finals),
        "malformed_groups": malformed_groups,
        "finished": finished,
        "judge_failed": failed,
        "judge_failed_rate_all": failed / len(finals) if finals else 0.0,
        "judge_failed_rate_finished": failed / finished if finished else 0.0,
        "correct": correct,
        "observed_pass1": correct / len(finals) if finals else 0.0,
    }
    del blob, samples, groups, finals
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = sorted(
        args.rollout_dir.glob("rollout_*.pt"),
        key=lambda path: int(re.search(r"(\d+)$", path.stem).group(1)),
    )
    rows = []
    for path in paths:
        row = analyze(path)
        rows.append(row)
        print(
            f"step={row['step']:02d} rollouts={row['n_rollouts']} finished={row['finished']} "
            f"judge_failed={row['judge_failed']} "
            f"failed/all={row['judge_failed_rate_all']:.4%} "
            f"failed/finished={row['judge_failed_rate_finished']:.4%}",
            flush=True,
        )

    contaminated = [row["step"] for row in rows if row["judge_failed"]]
    first_contaminated = min(contaminated) if contaminated else None
    checkpoints = []
    if args.checkpoint_root:
        checkpoints = sorted(
            int(path.name.removeprefix("iter_"))
            for path in args.checkpoint_root.glob("iter_[0-9]*")
            if (path / ".metadata").is_file()
        )
    last_clean_checkpoint = None
    if first_contaminated is not None:
        last_clean_checkpoint = max(
            (step for step in checkpoints if step < first_contaminated),
            default=None,
        )
    elif checkpoints:
        last_clean_checkpoint = checkpoints[-1]

    report = {
        "rollout_dir": str(args.rollout_dir),
        "checkpoint_root": str(args.checkpoint_root) if args.checkpoint_root else None,
        "first_contaminated_rollout": first_contaminated,
        "last_clean_checkpoint": last_clean_checkpoint,
        "checkpoint_steps": checkpoints,
        "rows": rows,
    }
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
