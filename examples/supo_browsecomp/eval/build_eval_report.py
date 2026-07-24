#!/usr/bin/env python3
"""Stage-A eval report builder for the BC+ 4B checkpoint sweep.

Parses the per-checkpoint eval dumps written by slime's `--dump-details`
(`<point>/rollout_data/eval_0.pt`, each a dict with `samples=[Sample.to_dict()]`)
and produces:

  <EVAL_ROOT>/eval_summary.json        machine-readable per-point metrics + flips
  <EVAL_ROOT>/summary_table.md         accuracy trend + metric table (base->iterNN)
  <EVAL_ROOT>/failures/<point>.jsonl   one line per FAILED rollout (score<1) with
                                       question / gold / predicted / full decoded
                                       trajectory + token ids  (input for Stage B)
  <EVAL_ROOT>/trajectories/<point>.jsonl  one line per rollout (all), lighter

Grouping: all sub-trajectories of one generate() call share rollout_id (==
sample.index); each (question, sample j) call is one "rollout". Per rollout we
take the FINAL/finished sub-traj's judge score. Per-question accuracy averages the
n samples; pass@n = any sample correct.

Usage (inside the slime container, which has torch):
    python build_eval_report.py --eval-root /genai_hh/evals/<run>-sweep
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import torch


def _point_sort_key(name: str):
    """base first, then iterNN ascending."""
    if name == "base":
        return (0, 0)
    m = re.match(r"iter0*(\d+)", name)
    return (1, int(m.group(1))) if m else (2, name)


def _score_of(reward):
    if reward is None:
        return 0.0
    if isinstance(reward, dict):
        return float(reward.get("score", 0.0) or 0.0)
    try:
        return float(reward)
    except (TypeError, ValueError):
        return 0.0


def _bc(meta):
    if isinstance(meta, dict):
        bc = meta.get("_bcplus")
        if isinstance(bc, dict):
            return bc
    return {}


def _question_of(sample):
    meta = sample.get("metadata") or {}
    if isinstance(meta, dict):
        q = meta.get("query") or meta.get("problem_statement")
        if q:
            return str(q)
    # Fall back to the prompt string (list-of-messages or str).
    p = sample.get("prompt")
    if isinstance(p, list):
        for m in reversed(p):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))[:4000]
    return str(p)[:4000]


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())[:500]


def load_point(pt_dir):
    """Load one point's eval dump -> list of per-rollout dicts."""
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

    rollouts = []
    for rid, sibs in groups.items():
        # Order sub-trajs; pick the final/finished one for the score + answer.
        def _sti(s):
            return _bc(s.get("metadata")).get("sub_traj_index", 0) or 0
        sibs_sorted = sorted(sibs, key=_sti)
        finished = [s for s in sibs_sorted if _bc(s.get("metadata")).get("finished")]
        final = finished[-1] if finished else sibs_sorted[-1]
        bc = _bc(final.get("metadata"))

        # Score: max across siblings (only the finished one carries the judge
        # score; reward_post_process may or may not broadcast in eval).
        score = max(_score_of(s.get("reward")) for s in sibs_sorted)

        tokens = final.get("tokens") or []
        resp_len = final.get("response_length") or 0
        lm = final.get("loss_mask") or []
        gen_tokens = int(sum(1 for x in lm if x)) if lm else 0
        rollouts.append({
            "rollout_id": rid,
            "question": _question_of(final),
            "gold": final.get("label"),
            "finish_answer": bc.get("finish_answer", ""),
            "score": score,
            "correct": score >= 0.5,
            "finished": bool(bc.get("finished", False)),
            # "truncated" = ran out of turn/context budget without finishing.
            "truncated": not bool(bc.get("finished", False)),
            "outcome": bc.get("outcome", ""),
            "final_stop_reason": bc.get("final_stop_reason", ""),
            # WHOLE-ROLLOUT totals: sum across all sub-trajectories of the rollout
            # (a rollout compresses into ~2.5-3 sub-trajs; taking only `final`
            # undercounts search/turns ~3x). finish/outcome/answer stay from `final`.
            "n_turns_used": sum(int(_bc(s.get("metadata")).get("n_turns_used", 0) or 0) for s in sibs_sorted),
            "n_search": sum(int(_bc(s.get("metadata")).get("n_search", 0) or 0) for s in sibs_sorted),
            "n_open": sum(int(_bc(s.get("metadata")).get("n_open", 0) or 0) for s in sibs_sorted),
            "n_bad_tool_calls": sum(int(_bc(s.get("metadata")).get("n_bad_tool_calls", 0) or 0) for s in sibs_sorted),
            "n_search_server_error": sum(int(_bc(s.get("metadata")).get("n_search_server_error", 0) or 0) for s in sibs_sorted),
            "n_sub_trajs": len(sibs_sorted),
            "response_length": resp_len,
            "traj_tokens": len(tokens),   # full trajectory (prompt+resp incl. tool obs)
            "gen_tokens": gen_tokens,     # trainable/model-generated tokens (sum loss_mask)
            # full decoded trajectory + token ids kept for Stage B / inspection
            "trajectory": final.get("response", ""),
            "prompt": final.get("prompt", ""),
            "tokens": tokens,
        })
    return rollouts


def summarize(rollouts):
    n = len(rollouts)
    if n == 0:
        return {"n_rollouts": 0}
    acc = sum(r["score"] for r in rollouts) / n
    finish_rate = sum(r["finished"] for r in rollouts) / n
    trunc_rate = sum(r["truncated"] for r in rollouts) / n
    outcome_dist = defaultdict(int)
    for r in rollouts:
        outcome_dist[r["outcome"] or "?"] += 1

    # per-question: pass@1 (mean) and pass@n (any correct)
    byq = defaultdict(list)
    for r in rollouts:
        byq[_norm_q(r["question"])].append(r)
    q_acc = [sum(x["score"] for x in v) / len(v) for v in byq.values()]
    q_passn = [1.0 if any(x["correct"] for x in v) else 0.0 for v in byq.values()]

    def _avg(key):
        return sum(r[key] for r in rollouts) / n

    return {
        "n_rollouts": n,
        "n_questions": len(byq),
        "accuracy": round(acc, 4),
        "pass@1_meanq": round(sum(q_acc) / len(q_acc), 4),
        "pass@n": round(sum(q_passn) / len(q_passn), 4),
        "finish_rate": round(finish_rate, 4),
        "truncation_rate": round(trunc_rate, 4),
        "avg_turns": round(_avg("n_turns_used"), 2),
        "avg_search": round(_avg("n_search"), 2),
        "avg_open": round(_avg("n_open"), 2),
        "avg_traj_tokens": round(_avg("traj_tokens"), 1),
        "avg_gen_tokens": round(_avg("gen_tokens"), 1),
        "avg_bad_tool_calls": round(_avg("n_bad_tool_calls"), 3),
        "avg_search_errors": round(_avg("n_search_server_error"), 3),
        "outcome_dist": dict(sorted(outcome_dist.items(), key=lambda kv: -kv[1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--max-failures-per-point", type=int, default=100000,
                    help="cap failure lines written per point (default: all)")
    args = ap.parse_args()
    root = args.eval_root

    points = []
    for name in sorted(os.listdir(root), key=_point_sort_key):
        pt_dir = os.path.join(root, name)
        if os.path.isdir(pt_dir) and os.path.isfile(
            os.path.join(pt_dir, "rollout_data", "eval_0.pt")
        ):
            points.append(name)
    if not points:
        raise SystemExit(f"No eval_0.pt dumps found under {root}")

    os.makedirs(os.path.join(root, "failures"), exist_ok=True)
    os.makedirs(os.path.join(root, "trajectories"), exist_ok=True)

    summary = {}
    per_point_rollouts = {}
    for pt in points:
        rollouts = load_point(os.path.join(root, pt))
        per_point_rollouts[pt] = rollouts
        summary[pt] = summarize(rollouts)
        print(f"[{pt}] {summary[pt]}")

        # failures.jsonl (Stage B input) — drop full token ids to keep it light,
        # but keep the decoded trajectory + prompt so agents can read it.
        fpath = os.path.join(root, "failures", f"{pt}.jsonl")
        written = 0
        with open(fpath, "w") as f:
            for r in rollouts:
                if r["correct"]:
                    continue
                if written >= args.max_failures_per_point:
                    break
                rec = {k: v for k, v in r.items() if k != "tokens"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
        # trajectories.jsonl (all rollouts, light: no tokens, trajectory trimmed)
        tpath = os.path.join(root, "trajectories", f"{pt}.jsonl")
        with open(tpath, "w") as f:
            for r in rollouts:
                rec = {k: v for k, v in r.items() if k not in ("tokens",)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- question-level flips across points (correct/incorrect transitions) ----
    # Map question -> {point: best_correct}
    q_by_point = {}
    all_qs = set()
    for pt in points:
        m = {}
        byq = defaultdict(list)
        for r in per_point_rollouts[pt]:
            byq[_norm_q(r["question"])].append(r)
        for q, v in byq.items():
            m[q] = 1 if any(x["correct"] for x in v) else 0
            all_qs.add(q)
        q_by_point[pt] = m

    flips = {"gained": [], "lost": [], "always_wrong": [], "always_right": []}
    first, last = points[0], points[-1]
    # pick a readable question label from the last point's rollouts
    qlabel = {}
    for pt in (last, first):
        for r in per_point_rollouts[pt]:
            qlabel.setdefault(_norm_q(r["question"]), r["question"][:200])
    for q in all_qs:
        a = q_by_point.get(first, {}).get(q, 0)
        b = q_by_point.get(last, {}).get(q, 0)
        lbl = qlabel.get(q, q[:200])
        if a == 0 and b == 1:
            flips["gained"].append(lbl)
        elif a == 1 and b == 0:
            flips["lost"].append(lbl)
        elif a == 0 and b == 0:
            flips["always_wrong"].append(lbl)
        else:
            flips["always_right"].append(lbl)

    out = {
        "points": points,
        "first_point": first,
        "last_point": last,
        "metrics": summary,
        "flips_first_to_last": {
            "n_gained": len(flips["gained"]),
            "n_lost": len(flips["lost"]),
            "n_always_wrong": len(flips["always_wrong"]),
            "n_always_right": len(flips["always_right"]),
            "gained_examples": flips["gained"][:40],
            "lost_examples": flips["lost"][:40],
        },
    }
    with open(os.path.join(root, "eval_summary.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- markdown summary table ----
    cols = ["accuracy", "pass@n", "finish_rate", "truncation_rate",
            "avg_turns", "avg_search", "avg_open", "avg_traj_tokens", "n_rollouts"]
    lines = ["# BC+ 4B checkpoint sweep — eval summary", "",
             f"Run points: {', '.join(points)}", "",
             "| point | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for pt in points:
        s = summary[pt]
        row = [pt] + [str(s.get(c, "")) for c in cols]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["",
              f"**Flips {first}→{last}**: gained {len(flips['gained'])}, "
              f"lost {len(flips['lost'])}, always-wrong {len(flips['always_wrong'])}, "
              f"always-right {len(flips['always_right'])}.", ""]
    for pt in points:
        s = summary[pt]
        lines.append(f"- **{pt}** outcomes: {s.get('outcome_dist', {})}")
    with open(os.path.join(root, "summary_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\nWrote:")
    print(" ", os.path.join(root, "eval_summary.json"))
    print(" ", os.path.join(root, "summary_table.md"))
    print(" ", os.path.join(root, "failures/<point>.jsonl"))
    print(" ", os.path.join(root, "trajectories/<point>.jsonl"))


if __name__ == "__main__":
    main()
