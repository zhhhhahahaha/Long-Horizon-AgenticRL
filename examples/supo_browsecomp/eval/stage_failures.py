#!/usr/bin/env python3
"""Stage failure records for the Stage-B per-trajectory agent deep-dive.

The eval dumps + failures/*.jsonl live on /genai (visible to the login pod but
NOT to the sandbox where Stage-B agents run). The sandbox DOES share /home with
the login pod. So this script (run on the login pod) reads the failures JSONL
from the /genai sweep dir and writes one JSON file per failing trajectory into a
/home staging dir that sandbox agents can Read.

Runs with plain python3 (no torch) — failures/*.jsonl is plain JSON written by
build_eval_report.py.

Usage (on the login pod, via the bridge):
    python3 stage_failures.py \
        --sweep /genai/fsx-project/hhzhang01/evals/<run>-sweep \
        --out   /home/hhzhang01/eval_stage/<run> \
        --points iter44 base \
        --max-per-point 0            # 0 = no cap
"""
import argparse
import json
import os
import re


def _norm_q(q):
    return re.sub(r"\s+", " ", (q or "").strip().lower())[:500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="sweep dir on /genai (has failures/<point>.jsonl)")
    ap.add_argument("--out", required=True, help="staging dir on /home (sandbox-visible)")
    ap.add_argument("--points", nargs="+", default=["iter44", "base"],
                    help="which points to stage")
    ap.add_argument("--max-per-point", type=int, default=0, help="cap files per point (0=all)")
    ap.add_argument("--dedup-by-question", action="store_true",
                    help="keep only the first failing rollout per distinct question")
    ap.add_argument("--which", choices=["failure", "correct"], default="failure",
                    help="stage failing rollouts (from failures/<pt>.jsonl) or CORRECT ones "
                         "(from trajectories/<pt>.jsonl, correct==True) for a success-trajectory review")
    args = ap.parse_args()

    index = {"sweep": args.sweep, "points": {}, "which": args.which}
    for pt in args.points:
        src = os.path.join(args.sweep,
                           "failures" if args.which == "failure" else "trajectories",
                           f"{pt}.jsonl")
        if not os.path.isfile(src):
            print(f"[stage] WARN: {src} missing, skipping {pt}")
            continue
        out_pt = os.path.join(args.out, pt)
        os.makedirs(out_pt, exist_ok=True)
        paths = []
        seen_q = set()
        i = 0
        with open(src) as f:
            for line in f:
                rec = json.loads(line)
                if args.which == "correct" and not rec.get("correct"):
                    continue  # only keep correctly-answered rollouts
                if args.dedup_by_question:
                    qk = _norm_q(rec.get("question"))
                    if qk in seen_q:
                        continue
                    seen_q.add(qk)
                if args.max_per_point and i >= args.max_per_point:
                    break
                # keep the fields an agent needs to diagnose one failure
                slim = {
                    "point": pt,
                    "rollout_id": rec.get("rollout_id"),
                    "question": rec.get("question"),
                    "gold_answer": rec.get("gold"),
                    "model_answer": rec.get("finish_answer"),
                    "score": rec.get("score"),
                    "finished": rec.get("finished"),
                    "outcome": rec.get("outcome"),
                    "n_turns_used": rec.get("n_turns_used"),
                    "n_search": rec.get("n_search"),
                    "n_open": rec.get("n_open"),
                    "trajectory": rec.get("trajectory"),
                    "prompt": rec.get("prompt"),
                }
                p = os.path.join(out_pt, f"traj_{i:04d}.json")
                with open(p, "w") as g:
                    json.dump(slim, g, ensure_ascii=False, indent=2)
                paths.append(p)
                i += 1
        index["points"][pt] = {"n": len(paths), "dir": out_pt, "files": paths}
        print(f"[stage] {pt}: staged {len(paths)} failure trajectories -> {out_pt}")

    idx_path = os.path.join(args.out, "index.json")
    os.makedirs(args.out, exist_ok=True)
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"[stage] wrote index: {idx_path}")


if __name__ == "__main__":
    main()
