#!/usr/bin/env python3
"""Estimate the ACCURACY COST of bad tool calls (parse_fail in particular):
how often did the model write a CORRECT answer in free-form prose but score 0
because it never submitted a `finish` tool call?

For each point:
  - group sub-trajs by rollout_id; score = max judge score across sibs; correct = score>=0.5
  - finished = final sub-traj's _bcplus.finished
  - "lost" rollout = wrong (score<0.5)
  - scan ALL sub-traj responses for a prose answer line ("Exact Answer: X" /
    "Answer: X"); take the last one found
  - does the gold label appear in that prose answer (normalized substring)?

Reports per point:
  wrong_unfinished           : wrong AND never finished  (the direct parse_fail victims)
  wu_prose_hits_gold         : of those, prose answer matched gold  -> "knew but didn't submit"
  wrong_finished_prose_hits  : wrong but DID finish, yet an earlier prose answer matched gold
                               (submitted wrong despite having written the right answer)
Both expressed as a fraction of all 600 rollouts = recoverable accuracy ceiling.
"""
import argparse, json, os, re
from collections import defaultdict
import torch

ANS_RE = re.compile(r"(?:Exact Answer|Final Answer|Answer)\s*[:：]\s*(.+)", re.IGNORECASE)


def _bc(meta):
    return (meta or {}).get("_bcplus", {}) if isinstance(meta, dict) else {}


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


def _norm(s):
    if s is None:
        return ""
    if isinstance(s, dict):
        s = s.get("answer") or s.get("target") or s.get("gold") or json.dumps(s)
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def _prose_answer(sibs):
    """Last prose 'Exact Answer: X' across all sub-trajs (first line only)."""
    found = None
    for s in sibs:
        for m in ANS_RE.finditer(s.get("response", "") or ""):
            cand = m.group(1).strip().splitlines()[0].strip()
            cand = cand.strip("*` ").rstrip(".")
            if cand:
                found = cand
    return found


def _match(gold, prose):
    g, p = _norm(gold), _norm(prose)
    if not g or not p:
        return False
    return g in p or (len(p) >= 3 and p in g)


def _point_sort_key(name):
    if name == "base":
        return (0, 0)
    if name.startswith("iter"):
        try:
            return (1, int(name[4:]))
        except ValueError:
            return (2, name)
    return (3, name)


def analyze(pt_dir):
    dump = os.path.join(pt_dir, "rollout_data", "eval_0.pt")
    if not os.path.isfile(dump):
        return None, []
    blob = torch.load(dump, weights_only=False)
    groups = defaultdict(list)
    for i, s in enumerate(blob.get("samples", [])):
        rid = s.get("rollout_id") or s.get("index", i)
        groups[rid].append(s)

    n = 0
    wrong = wrong_unfinished = wu_prose = wu_hit = wf_hit = 0
    examples = []
    for rid, sibs in groups.items():
        n += 1
        score = max(_score_of(s.get("reward")) for s in sibs)
        correct = score >= 0.5
        finished = bool(_bc(sibs[-1].get("metadata")).get("finished")) or any(
            _bc(s.get("metadata")).get("finished") for s in sibs)
        gold = None
        for s in sibs:
            gold = s.get("label") or gold
        if correct:
            continue
        wrong += 1
        prose = _prose_answer(sibs)
        hit = _match(gold, prose)
        if not finished:
            wrong_unfinished += 1
            if prose:
                wu_prose += 1
            if hit:
                wu_hit += 1
                if len(examples) < 6:
                    examples.append({"rid": rid, "gold": str(gold)[:80], "prose": str(prose)[:80],
                                     "finished": finished})
        else:
            if hit:
                wf_hit += 1
    return {
        "n_rollouts": n,
        "wrong": wrong,
        "wrong_unfinished": wrong_unfinished,
        "wu_with_prose_answer": wu_prose,
        "wu_prose_hits_gold": wu_hit,
        "wf_prose_hits_gold": wf_hit,
        "recoverable_ceiling_pct": round(100 * wu_hit / n, 2) if n else 0,
    }, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    args = ap.parse_args()
    root = args.eval_root
    points = [d for d in sorted(os.listdir(root), key=_point_sort_key)
              if os.path.isfile(os.path.join(root, d, "rollout_data", "eval_0.pt"))]

    out = {}
    print("%-7s %5s %6s %10s %9s %9s %9s | recover%%" % (
        "point", "n", "wrong", "wrong_unfin", "wu_prose", "wu_HIT", "wf_HIT"))
    all_ex = {}
    for name in points:
        r, ex = analyze(os.path.join(root, name))
        if r is None:
            continue
        out[name] = r
        all_ex[name] = ex
        print("%-7s %5d %6d %10d %9d %9d %9d | %.2f%%" % (
            name, r["n_rollouts"], r["wrong"], r["wrong_unfinished"],
            r["wu_with_prose_answer"], r["wu_prose_hits_gold"], r["wf_prose_hits_gold"],
            r["recoverable_ceiling_pct"]))

    print("\n=== examples of 'wrote correct answer in prose but never finished' ===")
    for name in points:
        for e in all_ex.get(name, [])[:3]:
            print("[%s] gold=%r  prose=%r" % (name, e["gold"], e["prose"]))

    dst = os.path.join(root, "bad_tool_cost.json")
    with open(dst, "w") as f:
        json.dump({"points": points, "metrics": out}, f, indent=2)
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
