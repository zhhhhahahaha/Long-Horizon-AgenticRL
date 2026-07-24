#!/usr/bin/env python3
"""Empirical breakdown of WHY tool calls were 'bad' across the eval sweep.

A turn is counted into `n_bad_tool_calls` (generate_with_bcplus.py) when it
produced zero effective tool calls. Each such turn injects a DISTINCTIVE
observation string back into the trajectory. We grep the decoded `response`
of every sub-trajectory for those markers to attribute each bad turn to a cause.

Causes (exact templates from _run_action / the rollout loop):
  parse_fail     : "No function call was detected in the model response."   (no <tool_call> parsed)
  search_no_query: 'The "search" function requires a "query" argument.'
  open_no_target : 'The "open_page" function requires either a "docid" or a "url".'
  empty_finish   : "Fail to parse answer. Please resubmit"                  (finish w/ empty answer)
  unknown_fn     : '" is not supported.'                                    (unknown function name)

Server-side failures ("[Search server error]") are NOT bad tool calls; counted separately for context.

Run via run_report.sh with REPORT_SCRIPT=examples/supo_browsecomp/eval/bad_tool_breakdown.py
Output: <eval-root>/bad_tool_breakdown.json  + a table to stdout.
"""
import argparse, json, os
from collections import defaultdict
import torch

MARKERS = {
    "parse_fail":      "No function call was detected in the model response.",
    "search_no_query": 'The "search" function requires a "query" argument.',
    "open_no_target":  'The "open_page" function requires either a "docid" or a "url".',
    "empty_finish":    "Fail to parse answer. Please resubmit",
    "unknown_fn":      '" is not supported.',
}
SERVER_ERR = "[Search server error]"


def _bc(meta):
    if not isinstance(meta, dict):
        return {}
    return meta.get("_bcplus", {}) or {}


def _score_of(reward):
    if isinstance(reward, dict):
        for k in ("score", "reward", "value"):
            if k in reward:
                try:
                    return float(reward[k])
                except (TypeError, ValueError):
                    pass
        return 0.0
    try:
        return float(reward)
    except (TypeError, ValueError):
        return 0.0


def _point_sort_key(name):
    if name == "base":
        return (0, 0)
    if name.startswith("iter"):
        try:
            return (1, int(name[4:]))
        except ValueError:
            return (2, name)
    return (3, name)


def analyze_point(pt_dir):
    dump = os.path.join(pt_dir, "rollout_data", "eval_0.pt")
    if not os.path.isfile(dump):
        return None
    blob = torch.load(dump, weights_only=False)
    samples = blob.get("samples", [])

    groups = defaultdict(list)
    for i, s in enumerate(samples):
        rid = s.get("rollout_id")
        if rid is None:
            rid = s.get("index", i)
        groups[rid].append(s)

    counts = defaultdict(int)                     # cause -> total marker hits (all rollouts)
    counts_by_correct = {True: defaultdict(int), False: defaultdict(int)}
    server_errs = 0
    meta_bad_sum = 0                              # sum of metadata n_bad_tool_calls (cross-check)
    n_rollouts = 0
    rollouts_with_any_bad = 0

    for rid, sibs in groups.items():
        n_rollouts += 1
        score = max(_score_of(s.get("reward")) for s in sibs)
        correct = score >= 0.5
        roll_bad = 0
        for s in sibs:
            resp = s.get("response", "") or ""
            meta_bad_sum += int(_bc(s.get("metadata")).get("n_bad_tool_calls", 0) or 0)
            server_errs += resp.count(SERVER_ERR)
            for cause, mk in MARKERS.items():
                c = resp.count(mk)
                if c:
                    counts[cause] += c
                    counts_by_correct[correct][cause] += c
                    roll_bad += c
        if roll_bad:
            rollouts_with_any_bad += 1

    total_markers = sum(counts.values())
    return {
        "n_rollouts": n_rollouts,
        "total_bad_markers": total_markers,
        "meta_n_bad_sum": meta_bad_sum,           # should ~= total_markers
        "avg_bad_per_rollout": round(total_markers / n_rollouts, 4) if n_rollouts else 0,
        "rollouts_with_any_bad": rollouts_with_any_bad,
        "frac_rollouts_with_bad": round(rollouts_with_any_bad / n_rollouts, 4) if n_rollouts else 0,
        "server_errors": server_errs,
        "by_cause": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "by_cause_correct": dict(counts_by_correct[True]),
        "by_cause_wrong": dict(counts_by_correct[False]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    args = ap.parse_args()
    root = args.eval_root

    points = []
    for name in sorted(os.listdir(root), key=_point_sort_key):
        pt = os.path.join(root, name)
        if os.path.isdir(pt) and os.path.isfile(os.path.join(pt, "rollout_data", "eval_0.pt")):
            points.append(name)

    out = {}
    causes = list(MARKERS.keys())
    hdr = "%-7s %8s %8s | " % ("point", "bad/roll", "%roll") + " ".join("%-15s" % c for c in causes) + " | srv_err"
    print(hdr)
    for name in points:
        r = analyze_point(os.path.join(root, name))
        if r is None:
            continue
        out[name] = r
        row = "%-7s %8.3f %7.1f%% | " % (name, r["avg_bad_per_rollout"], 100 * r["frac_rollouts_with_bad"])
        row += " ".join("%-15d" % r["by_cause"].get(c, 0) for c in causes)
        row += " | %d" % r["server_errors"]
        print(row)
        if r["meta_n_bad_sum"] != r["total_bad_markers"]:
            print("   [note] meta_n_bad_sum=%d vs marker_total=%d (mismatch = some bad turns had no "
                  "distinct marker, e.g. multi-error turns counted once in metadata)"
                  % (r["meta_n_bad_sum"], r["total_bad_markers"]))

    dst = os.path.join(root, "bad_tool_breakdown.json")
    with open(dst, "w") as f:
        json.dump({"points": points, "metrics": out, "markers": MARKERS}, f, indent=2)
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
