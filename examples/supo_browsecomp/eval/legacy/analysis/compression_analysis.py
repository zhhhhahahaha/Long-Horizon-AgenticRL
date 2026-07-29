#!/usr/bin/env python3
"""Compression-focused analysis of the BC+ 4B eval sweep (torch, in-container).

The research question for this project is *compression in long-horizon agents*:
does SUPO's context compression (write a `<summary>`, drop all prior context, start
a fresh sub-trajectory from the summary alone) itself CAUSE failures? Two hypotheses:

  (H1) compression info-loss — an EARLY sub-traj already retrieved the answer (or a
       strong lead), but the `<summary>` handover dropped it, so the later sub-traj
       never recovers.
  (H2) top_k overwhelm — the model can set `topk` up to 20 (default 10); big result
       sets bloat context, drown the signal, and/or trigger premature compression.

This script reconstructs each rollout's FULL sub-traj chain from the per-checkpoint
`eval_0.pt` dumps (each Sample = one sub-traj; `metadata._bcplus.summary` = the
handover it wrote; `metadata._bcplus_sibling.{sub_traj_index,total_sub_trajs,is_final}`;
`response` = that sub-traj's searches+observations+reasoning; `prompt` = what it
started from, incl. the inherited summary for sub-traj>0). It then computes, per
checkpoint, an OBJECTIVE lower-bound on compression info-loss:

  dropped_by_compression = gold string present in an early sub-traj's text
                           AND not surviving into the final sub-traj's context
                           AND the rollout failed.

plus whether the summary specifically dropped it, the topk histogram (failure vs
correct), and observation bloat. Substring match (word-boundary, normalized) is a
LOWER bound — it misses paraphrase and over-counts very short golds; the agent pass
(stage_b_compression_workflow.js) confirms the flagged chains and catches semantic
cases. Writes:

  <root>/chains/<point>.jsonl        one line per rollout: full sub-traj chain + flags
  <root>/compression_stats.json      per-checkpoint objective aggregates

Usage (in the slime container):
  python compression_analysis.py --eval-root /genai_hh/evals/<run>-sweep \
      --points base iter04 iter24 iter44
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import torch

SEARCH_TOPK_DEFAULT = 10  # BCPLUS_CONFIGS["search_topk_default"]


def _point_sort_key(name: str):
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


def _sib(meta):
    if isinstance(meta, dict):
        s = meta.get("_bcplus_sibling")
        if isinstance(s, dict):
            return s
    return {}


def _question_of(sample):
    meta = sample.get("metadata") or {}
    if isinstance(meta, dict):
        q = meta.get("query") or meta.get("problem_statement")
        if q:
            return str(q)
    p = sample.get("prompt")
    if isinstance(p, list):
        for m in reversed(p):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))[:4000]
    return str(p)[:4000]


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _contains(haystack_norm: str, needle_norm: str) -> bool:
    """Word-boundary substring match on already-normalized strings.

    Word-boundary reduces false positives for short golds (e.g. 'rain' won't match
    'training'); still a lower bound (misses paraphrase / alternate surface forms).
    """
    if not needle_norm or len(needle_norm) < 2:
        return False
    return re.search(r"\b" + re.escape(needle_norm) + r"\b", haystack_norm) is not None


def _topk_list(resp_text: str):
    """Parse the topk each search call used (default 10 when the param is absent)."""
    out = []
    for block in re.findall(r"<function=search>(.*?)</function>", resp_text, re.DOTALL):
        m = re.search(r"<parameter=topk>\s*(\d+)", block)
        out.append(int(m.group(1)) if m else SEARCH_TOPK_DEFAULT)
    return out


def _n_doc_markers(resp_text: str) -> int:
    """Rough count of retrieved-doc blocks in the observations (search + open_page)."""
    return len(re.findall(r"--- #\d+:", resp_text)) + resp_text.count("[Opened Page Content]")


def load_chains(pt_dir):
    """Return list of per-rollout chain dicts (all sub-trajs, ordered)."""
    dump = os.path.join(pt_dir, "rollout_data", "eval_0.pt")
    if not os.path.isfile(dump):
        return None
    blob = torch.load(dump, weights_only=False)
    samples = blob.get("samples", [])

    groups = defaultdict(list)
    order = {}
    for i, s in enumerate(samples):
        rid = s.get("rollout_id")
        if rid is None:
            rid = s.get("index", i)
        order.setdefault(rid, i)
        groups[rid].append((i, s))

    chains = []
    for rid, items in groups.items():
        # order sub-trajs by sibling index, falling back to insertion order
        def _key(it):
            idx, s = it
            si = _sib(s.get("metadata")).get("sub_traj_index")
            return si if si is not None else idx
        items_sorted = sorted(items, key=_key)
        sibs = [s for _, s in items_sorted]
        total = len(sibs)
        final = sibs[-1]
        fbc = _bc(final.get("metadata"))

        score = max(_score_of(s.get("reward")) for s in sibs)
        correct = score >= 0.5
        gold_norm = _norm(final.get("label"))

        sub = []
        topk_all = []
        n_search_total = 0
        gold_present = []      # gold in sub-traj i's own response text
        gold_in_prompt = []    # gold in sub-traj i's inherited prompt (summary handover)
        gold_in_summary = []   # gold in the summary this sub-traj wrote
        for i, s in enumerate(sibs):
            bc = _bc(s.get("metadata"))
            resp = s.get("response", "") or ""
            prm = s.get("prompt", "")
            prm = prm if isinstance(prm, str) else json.dumps(prm)[:6000]
            summ = bc.get("summary")
            tks = _topk_list(resp)
            topk_all.extend(tks)
            n_search_total += int(bc.get("n_search", 0) or 0)
            rn, pn, sn = _norm(resp), _norm(prm), _norm(summ)
            gold_present.append(_contains(rn, gold_norm))
            gold_in_prompt.append(_contains(pn, gold_norm))
            gold_in_summary.append(_contains(sn, gold_norm))
            sub.append({
                "i": i,
                "outcome": bc.get("outcome", ""),
                "summary_source": bc.get("summary_source", ""),
                "finished": bool(bc.get("finished", False)),
                "finish_answer": bc.get("finish_answer", ""),
                "n_search": int(bc.get("n_search", 0) or 0),
                "n_open": int(bc.get("n_open", 0) or 0),
                "topk_used": tks,
                "n_doc_markers": _n_doc_markers(resp),
                "resp_chars": len(resp),
                "summary": summ,
                "prompt": prm,
                "response": resp,
            })

        # --- objective compression info-loss signal (only meaningful w/ >=2 sub-trajs) ---
        had_in_early = any(gold_present[:-1]) or any(gold_in_summary[:-1])
        # did gold survive into the FINAL sub-traj's context (inherited summary or its own retrieval)?
        gold_in_final_ctx = gold_in_prompt[-1] or gold_present[-1]
        gold_in_any_summary = any(gold_in_summary)
        dropped_by_compression = (total >= 2) and had_in_early and (not gold_in_final_ctx)
        # narrower: gold was retrieved early but the SUMMARY text omitted it
        summary_dropped_gold = (total >= 2) and any(gold_present[:-1]) and (not gold_in_any_summary)

        chains.append({
            "point": os.path.basename(pt_dir),
            "rollout_id": rid,
            "question": _question_of(final),
            "gold": final.get("label"),
            "final_answer": fbc.get("finish_answer", ""),
            "score": score,
            "correct": correct,
            "n_sub_trajs": total,
            "compressed": total >= 2,
            "outcomes": [x["outcome"] for x in sub],
            "summary_sources": [x["summary_source"] for x in sub],
            "n_searches_total": n_search_total,
            "topk_used": topk_all,
            "max_topk": max(topk_all) if topk_all else 0,
            "n_topk20": sum(1 for t in topk_all if t >= 20),
            # objective flags
            "gold_present_by_subtraj": gold_present,
            "gold_in_summary_by_subtraj": gold_in_summary,
            "gold_in_final_ctx": gold_in_final_ctx,
            "had_gold_early": had_in_early,
            "dropped_by_compression": dropped_by_compression,
            "summary_dropped_gold": summary_dropped_gold,
            "sub_trajs": sub,
        })
    return chains


def _hist(vals):
    h = defaultdict(int)
    for v in vals:
        h[v] += 1
    return dict(sorted(h.items()))


def summarize_point(chains):
    fails = [c for c in chains if not c["correct"]]
    corr = [c for c in chains if c["correct"]]
    fails_multi = [c for c in fails if c["n_sub_trajs"] >= 2]
    corr_multi = [c for c in corr if c["n_sub_trajs"] >= 2]

    def rate(num, den):
        return round(num / den, 4) if den else 0.0

    n_dropped = sum(c["dropped_by_compression"] for c in fails)
    n_sumdrop = sum(c["summary_dropped_gold"] for c in fails)
    n_fail_had_early = sum(c["had_gold_early"] for c in fails_multi)

    all_topk = [t for c in chains for t in c["topk_used"]]
    fail_topk = [t for c in fails for t in c["topk_used"]]
    corr_topk = [t for c in corr for t in c["topk_used"]]

    return {
        "n_rollouts": len(chains),
        "n_correct": len(corr),
        "n_fail": len(fails),
        "n_fail_multi_subtraj": len(fails_multi),
        "n_correct_multi_subtraj": len(corr_multi),
        "avg_sub_trajs": round(sum(c["n_sub_trajs"] for c in chains) / max(1, len(chains)), 3),
        # --- H1: compression info-loss (objective lower bound) ---
        "dropped_by_compression": n_dropped,
        "dropped_rate_over_fail": rate(n_dropped, len(fails)),
        "dropped_rate_over_fail_multi": rate(n_dropped, len(fails_multi)),
        "summary_dropped_gold": n_sumdrop,
        "fail_multi_had_gold_early": n_fail_had_early,
        "fail_multi_had_gold_early_rate": rate(n_fail_had_early, len(fails_multi)),
        # control: correct rollouts that compressed and still carried gold through
        "correct_multi_gold_survived": sum(
            c["gold_in_final_ctx"] for c in corr_multi if c["had_gold_early"]),
        "correct_multi_had_gold_early": sum(c["had_gold_early"] for c in corr_multi),
        # --- H2: top_k ---
        "topk_hist_all": _hist(all_topk),
        "topk_hist_fail": _hist(fail_topk),
        "topk_hist_correct": _hist(corr_topk),
        "frac_searches_topk20_fail": rate(sum(1 for t in fail_topk if t >= 20), len(fail_topk)),
        "frac_searches_topk20_correct": rate(sum(1 for t in corr_topk if t >= 20), len(corr_topk)),
        "avg_topk_fail": round(sum(fail_topk) / max(1, len(fail_topk)), 2),
        "avg_topk_correct": round(sum(corr_topk) / max(1, len(corr_topk)), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--points", nargs="+", default=None,
                    help="which points to analyze (default: all with a dump)")
    args = ap.parse_args()
    root = args.eval_root

    avail = []
    for name in sorted(os.listdir(root), key=_point_sort_key):
        if os.path.isfile(os.path.join(root, name, "rollout_data", "eval_0.pt")):
            avail.append(name)
    points = [p for p in (args.points or avail) if p in avail]
    if not points:
        raise SystemExit(f"No matching eval_0.pt dumps under {root} (avail={avail})")

    os.makedirs(os.path.join(root, "chains"), exist_ok=True)
    stats = {}
    for pt in points:
        chains = load_chains(os.path.join(root, pt))
        stats[pt] = summarize_point(chains)
        print(f"[{pt}] {json.dumps(stats[pt])}")
        cpath = os.path.join(root, "chains", f"{pt}.jsonl")
        with open(cpath, "w") as f:
            for c in chains:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(os.path.join(root, "compression_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("\nWrote:")
    print(" ", os.path.join(root, "compression_stats.json"))
    print(" ", os.path.join(root, "chains/<point>.jsonl"))


if __name__ == "__main__":
    main()
