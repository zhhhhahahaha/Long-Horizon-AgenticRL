#!/usr/bin/env python3
"""Whole-trajectory (rollout-level) tool/turn stats — corrects build_eval_report.py,
which took these counts from the FINAL sub-trajectory only. Here we SUM across all
sibling sub-trajs (same rollout_id) so a compressed rollout's search/turn/open counts
reflect the entire rollout, not just its last leg.

Usage (in container): python traj_stats.py --eval-root /genai_hh/evals/<run>-sweep
"""
from __future__ import annotations
import argparse, json, os, re
from collections import defaultdict
import torch

def _pt_key(name):
    if name == "base": return (0, 0)
    m = re.match(r"iter0*(\d+)", name); return (1, int(m.group(1))) if m else (2, name)

def _bc(s):
    m = s.get("metadata") or {}; b = m.get("_bcplus") if isinstance(m, dict) else None
    return b if isinstance(b, dict) else {}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--eval-root", required=True)
    root = ap.parse_args().eval_root
    points = sorted([d for d in os.listdir(root)
                     if os.path.isfile(os.path.join(root, d, "rollout_data", "eval_0.pt"))], key=_pt_key)
    rows = []
    hdr = f"{'point':8}{'rolls':>7}{'subtraj':>9}{'turns_WHOLE':>12}{'turns_final':>12}{'search_WHOLE':>13}{'search_final':>13}{'open_WHOLE':>11}{'bad_WHOLE':>10}"
    print(hdr)
    for pt in points:
        blob = torch.load(os.path.join(root, pt, "rollout_data", "eval_0.pt"), weights_only=False)
        g = defaultdict(list)
        for i, s in enumerate(blob.get("samples", [])):
            rid = s.get("rollout_id"); rid = s.get("index", i) if rid is None else rid
            g[rid].append(s)
        n = len(g)
        agg = defaultdict(float)
        for rid, sibs in g.items():
            def _sti(s): return _bc(s).get("sub_traj_index", 0) or 0
            sibs = sorted(sibs, key=_sti)
            final = ([s for s in sibs if _bc(s).get("finished")] or sibs)[-1]
            # WHOLE = sum across sub-trajs; final = last sub-traj only (the old buggy way)
            agg["subtraj"] += len(sibs)
            agg["turns_whole"] += sum(int(_bc(s).get("n_turns_used", 0) or 0) for s in sibs)
            agg["turns_final"] += int(_bc(final).get("n_turns_used", 0) or 0)
            agg["search_whole"] += sum(int(_bc(s).get("n_search", 0) or 0) for s in sibs)
            agg["search_final"] += int(_bc(final).get("n_search", 0) or 0)
            agg["open_whole"] += sum(int(_bc(s).get("n_open", 0) or 0) for s in sibs)
            agg["bad_whole"] += sum(int(_bc(s).get("n_bad_tool_calls", 0) or 0) for s in sibs)
        r = {k: round(v / n, 2) for k, v in agg.items()}
        r["point"] = pt; r["n"] = n; rows.append(r)
        print(f"{pt:8}{n:>7}{r['subtraj']:>9}{r['turns_whole']:>12}{r['turns_final']:>12}"
              f"{r['search_whole']:>13}{r['search_final']:>13}{r['open_whole']:>11}{r['bad_whole']:>10}")
    json.dump(rows, open(os.path.join(root, "traj_stats.json"), "w"), indent=2)
    print("\nwrote", os.path.join(root, "traj_stats.json"))

if __name__ == "__main__":
    main()
