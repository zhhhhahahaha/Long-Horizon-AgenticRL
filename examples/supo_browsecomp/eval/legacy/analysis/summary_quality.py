#!/usr/bin/env python3
"""Per-checkpoint summary-writing quality across the sweep.

Answers: did the model learn to emit a valid <summary>...</summary> block during
context compression as training progressed? Every compression sub-trajectory
records metadata._bcplus.summary_source in {extracted, fallback, empty}:
  extracted = parsed a real <summary> block   (learned the pattern)
  fallback  = no parseable block, used raw text (wrong format)
  empty     = produced nothing

Scans ALL sub-trajs in each point's eval_0.pt (not just the final one).

Usage (in the slime container, has torch):
    python summary_quality.py --eval-root /genai_hh/evals/<run>-sweep
"""
from __future__ import annotations
import argparse, json, os, re
from collections import defaultdict
import torch


def _pt_key(name):
    if name == "base":
        return (0, 0)
    m = re.match(r"iter0*(\d+)", name)
    return (1, int(m.group(1))) if m else (2, name)


def _bc(s):
    m = s.get("metadata") or {}
    b = m.get("_bcplus") if isinstance(m, dict) else None
    return b if isinstance(b, dict) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    args = ap.parse_args()
    root = args.eval_root

    points = sorted(
        [d for d in os.listdir(root)
         if os.path.isfile(os.path.join(root, d, "rollout_data", "eval_0.pt"))],
        key=_pt_key)

    rows = []
    print(f"{'point':8} {'rollouts':>8} {'rollouts_w_compress':>18} "
          f"{'compress_turns':>14} {'extracted':>10} {'fallback':>9} {'empty':>7} {'%extracted':>11}")
    for pt in points:
        blob = torch.load(os.path.join(root, pt, "rollout_data", "eval_0.pt"), weights_only=False)
        samples = blob.get("samples", [])
        by_roll = defaultdict(list)
        for i, s in enumerate(samples):
            rid = s.get("rollout_id")
            if rid is None:
                rid = s.get("index", i)
            by_roll[rid].append(s)

        cat = defaultdict(int)          # summary_source -> count (compression turns)
        rolls_with_compress = 0
        for rid, sibs in by_roll.items():
            saw = False
            for s in sibs:
                src = _bc(s).get("summary_source", "") or ""
                if src != "":
                    cat[src] += 1
                    saw = True
            if saw:
                rolls_with_compress += 1

        turns = cat["extracted"] + cat["fallback"] + cat["empty"]
        pct = (100.0 * cat["extracted"] / turns) if turns else 0.0
        rows.append({
            "point": pt, "n_rollouts": len(by_roll),
            "rollouts_with_compression": rolls_with_compress,
            "compression_turns": turns,
            "extracted": cat["extracted"], "fallback": cat["fallback"], "empty": cat["empty"],
            "pct_extracted": round(pct, 1),
        })
        print(f"{pt:8} {len(by_roll):>8} {rolls_with_compress:>18} "
              f"{turns:>14} {cat['extracted']:>10} {cat['fallback']:>9} {cat['empty']:>7} {pct:>10.1f}%")

    out = os.path.join(root, "summary_quality.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
