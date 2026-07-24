#!/usr/bin/env python3
"""Stage compression-failure chains to /home for the Stage-B agent deep-dive.

Reads the full per-rollout sub-traj chains written by compression_analysis.py
(`<sweep>/chains/<point>.jsonl`, on /genai) and writes one JSON file per selected
rollout into a /home staging dir the sandbox agents can Read. Plain python (no
torch) — runs on the login pod.

Buckets (so the agent pass gets both PRECISION and RECALL on compression-loss):
  dropped        objective dropped_by_compression=True  -> CONFIRM it's a real loss
  summary_lossy  summary_dropped_gold=True but not dropped -> lossy summary, recovered
  hi_topk        failure with max_topk>=20                -> H2 (overwhelm) probe
  control        random compression-failure with NO flag  -> catch losses substring missed

For each sub-traj we keep: outcome, summary_source, n_search, topk_used, the FULL
summary text (the handover), a truncated response, and `gold_windows` = raw-text
snippets around each gold-answer occurrence (so the agent can verify the objective
flag isn't a short-string false positive and see exactly what compression dropped).

Usage (login pod, via bridge):
  python3 stage_compression_home.py \
      --sweep /genai/fsx-project/hhzhang01/evals/<run>-sweep \
      --out   /home/hhzhang01/eval_stage/4b-compression \
      --points base iter04 iter24 iter44 \
      --control-per-point 15 --hitopk-per-point 20
"""
import argparse
import json
import os
import re


def _norm(s):
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _gold_windows(text, gold, radius=260, maxw=3):
    """Raw-text snippets around each word-boundary occurrence of gold (case-insensitive)."""
    g = _norm(gold)
    if not g or len(g) < 2:
        return []
    out = []
    for m in re.finditer(r"\b" + re.escape(g) + r"\b", text, flags=re.IGNORECASE):
        a = max(0, m.start() - radius)
        b = min(len(text), m.end() + radius)
        out.append(("..." if a > 0 else "") + text[a:b] + ("..." if b < len(text) else ""))
        if len(out) >= maxw:
            break
    return out


def _trunc(text, head=6000, tail=3000):
    if len(text) <= head + tail + 40:
        return text
    return f"{text[:head]}\n\n...[TRUNCATED {len(text) - head - tail} chars]...\n\n{text[-tail:]}"


def _slim_subtraj(st, gold):
    resp = st.get("response", "") or ""
    return {
        "i": st.get("i"),
        "outcome": st.get("outcome"),
        "summary_source": st.get("summary_source"),
        "finished": st.get("finished"),
        "finish_answer": st.get("finish_answer"),
        "n_search": st.get("n_search"),
        "n_open": st.get("n_open"),
        "topk_used": st.get("topk_used"),
        "n_doc_markers": st.get("n_doc_markers"),
        "resp_chars": st.get("resp_chars"),
        # gold occurrences in THIS sub-traj's own work (observations/reasoning)
        "gold_windows": _gold_windows(resp, gold),
        # the handover this sub-traj wrote to the next (FULL text — the key artifact)
        "summary": st.get("summary"),
        "response": _trunc(resp),
    }


def _bucket(c):
    if c.get("dropped_by_compression"):
        return "dropped"
    if c.get("summary_dropped_gold"):
        return "summary_lossy"
    if (c.get("max_topk") or 0) >= 20:
        return "hi_topk"
    return "control"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", nargs="+", default=["base", "iter04", "iter24", "iter44"])
    ap.add_argument("--control-per-point", type=int, default=15)
    ap.add_argument("--hitopk-per-point", type=int, default=20)
    args = ap.parse_args()

    index = {"sweep": args.sweep, "points": {}}
    for pt in args.points:
        src = os.path.join(args.sweep, "chains", f"{pt}.jsonl")
        if not os.path.isfile(src):
            print(f"[stage] WARN missing {src}")
            continue
        fails = []
        with open(src) as f:
            for line in f:
                c = json.loads(line)
                if c.get("correct"):
                    continue
                if c.get("n_sub_trajs", 1) < 2:
                    continue  # compression study => only rollouts that compressed
                fails.append(c)

        # deterministic bucket selection (no RNG): control/hi_topk capped by first-N order
        picked, n_ctrl, n_hitopk = [], 0, 0
        for c in fails:
            b = _bucket(c)
            if b == "control":
                if n_ctrl >= args.control_per_point:
                    continue
                n_ctrl += 1
            elif b == "hi_topk":
                if n_hitopk >= args.hitopk_per_point:
                    continue
                n_hitopk += 1
            c["_bucket"] = b
            picked.append(c)

        out_pt = os.path.join(args.out, pt)
        os.makedirs(out_pt, exist_ok=True)
        paths, bcounts = [], {}
        for i, c in enumerate(picked):
            rec = {
                "point": pt,
                "rollout_id": c.get("rollout_id"),
                "bucket": c["_bucket"],
                "question": c.get("question"),
                "gold_answer": c.get("gold"),
                "final_answer": c.get("final_answer"),
                "score": c.get("score"),
                "n_sub_trajs": c.get("n_sub_trajs"),
                "outcomes": c.get("outcomes"),
                "summary_sources": c.get("summary_sources"),
                "max_topk": c.get("max_topk"),
                "n_topk20": c.get("n_topk20"),
                # objective flags (agent should CONFIRM/REFUTE these)
                "flag_dropped_by_compression": c.get("dropped_by_compression"),
                "flag_summary_dropped_gold": c.get("summary_dropped_gold"),
                "flag_had_gold_early": c.get("had_gold_early"),
                "flag_gold_in_final_ctx": c.get("gold_in_final_ctx"),
                "gold_present_by_subtraj": c.get("gold_present_by_subtraj"),
                "gold_in_summary_by_subtraj": c.get("gold_in_summary_by_subtraj"),
                "sub_trajs": [_slim_subtraj(st, c.get("gold")) for st in c.get("sub_trajs", [])],
            }
            p = os.path.join(out_pt, f"chain_{i:04d}.json")
            with open(p, "w") as g:
                json.dump(rec, g, ensure_ascii=False, indent=2)
            paths.append(p)
            bcounts[c["_bucket"]] = bcounts.get(c["_bucket"], 0) + 1
        index["points"][pt] = {"n": len(paths), "dir": out_pt, "files": paths, "buckets": bcounts}
        print(f"[stage] {pt}: {len(paths)} chains {bcounts} -> {out_pt}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"[stage] wrote {os.path.join(args.out, 'index.json')}")


if __name__ == "__main__":
    main()
