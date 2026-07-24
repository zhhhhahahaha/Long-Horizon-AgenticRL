#!/usr/bin/env python3
"""Retrieval-grounding vs parametric-knowledge check for the BC+ 4B eval.

Answers two validity questions about "correct" rollouts, independent of whether the
model CITED a docid in its finish answer:

  (A) citation false-negative: did the model actually RETRIEVE the answer (it appears
      in a <tool_response> observation) even if it didn't cite it? -> grounding is
      undercounted by the citation metric.
  (B) data leakage / parametric win: did the model answer correctly WITHOUT the answer
      ever appearing in any retrieved observation? -> it likely used prior/parametric
      knowledge (the question/answer may be in pretraining).

Method: for each rollout, concatenate the text inside every <tool_response>...</tool_response>
block across ALL sub-trajs = exactly what retrieval returned (search snippets + opened
pages), EXCLUDING the model's own reasoning/finish text. Then word-boundary substring
match the gold answer against that retrieved text.

  answer_in_retrieval = gold answer string appears in some observation
  correct & answer_in_retrieval      -> grounded by retrieval (cited or not)
  correct & NOT answer_in_retrieval  -> parametric / leakage candidate

Caveats (two-sided noise): substring misses paraphrase / alternate surface forms
(=> OVER-counts parametric); very short golds can match coincidentally (=> UNDER-counts
parametric). So parametric_rate is an estimate, best read as a bracket. A stronger
version joins the parquet gold DOCIDs (see grounding_check.py) to also catch paraphrase.

Reads the chains/*.jsonl written by compression_analysis.py (plain JSON, full sub-traj
responses) — runs with plain python3 on the login pod, no torch.

Usage:
  python3 retrieval_grounding.py --sweep /genai/.../<run>-sweep --points base iter04 iter24 iter44
"""
import argparse
import json
import os
import re

OBS_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)


def _norm(s):
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _contains(hay, needle):
    if not needle or len(needle) < 2:
        return False
    return re.search(r"\b" + re.escape(needle) + r"\b", hay) is not None


def _parts(gold_norm):
    """Split a compound gold label ('Band, Album' / 'X; Y' / 'A and B') into parts.

    BC+ gold answers are often multi-fact; the concatenated surface form never appears
    verbatim in one observation even when every part WAS retrieved. Requiring all parts
    present (each >=3 chars) is a much better 'was the evidence retrieved' test."""
    raw = re.split(r"[;,]| and ", gold_norm)
    return [p.strip() for p in raw if len(p.strip()) >= 3]


def _all_parts_in(hay, gold_norm):
    parts = _parts(gold_norm)
    if not parts:
        return _contains(hay, gold_norm)
    return all(_contains(hay, p) for p in parts)


def _obs_text(chain):
    parts = []
    for st in chain.get("sub_trajs", []):
        parts.extend(OBS_RE.findall(st.get("response", "") or ""))
    return _norm(" ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--points", nargs="+", default=["base", "iter04", "iter24", "iter44"])
    args = ap.parse_args()

    stats = {}
    for pt in args.points:
        src = os.path.join(args.sweep, "chains", f"{pt}.jsonl")
        if not os.path.isfile(src):
            print(f"[grnd] WARN missing {src}")
            continue
        n_corr = n_corr_ret = n_corr_ret_parts = 0
        n_fail = n_fail_ret = 0
        parametric_examples = []
        with open(src) as f:
            for line in f:
                c = json.loads(line)
                gold = _norm(c.get("gold"))
                obs = _obs_text(c)
                in_ret = _contains(obs, gold)          # strict: whole gold string
                in_ret_parts = _all_parts_in(obs, gold)  # lenient: all compound parts
                if c.get("correct"):
                    n_corr += 1
                    if in_ret:
                        n_corr_ret += 1
                    if in_ret_parts:
                        n_corr_ret_parts += 1
                    elif len(parametric_examples) < 12:
                        # truly nothing (not even parts) in retrieval -> best leakage candidate
                        parametric_examples.append({
                            "rollout_id": c.get("rollout_id"),
                            "gold": c.get("gold"),
                            "final_answer": c.get("final_answer"),
                            "n_sub_trajs": c.get("n_sub_trajs"),
                            "n_searches_total": c.get("n_searches_total"),
                        })
                else:
                    n_fail += 1
                    if in_ret:
                        n_fail_ret += 1

        def r(a, b):
            return round(a / b, 4) if b else 0.0

        stats[pt] = {
            "n_correct": n_corr,
            "correct_answer_in_retrieval_strict": n_corr_ret,
            "correct_grounded_rate_strict": r(n_corr_ret, n_corr),
            "correct_answer_parts_in_retrieval": n_corr_ret_parts,
            "correct_grounded_rate_parts": r(n_corr_ret_parts, n_corr),
            # parametric = correct but NOT EVEN all compound parts retrieved (best leakage estimate)
            "parametric_candidates_parts": n_corr - n_corr_ret_parts,
            "parametric_rate_parts": r(n_corr - n_corr_ret_parts, n_corr),
            # loose upper bound (strict whole-string miss; inflated by compound golds)
            "parametric_rate_strict_upperbound": r(n_corr - n_corr_ret, n_corr),
            "n_fail": n_fail,
            "fail_answer_in_retrieval": n_fail_ret,
            "fail_had_answer_in_retrieval_rate": r(n_fail_ret, n_fail),
            "parametric_examples": parametric_examples,
        }
        s = stats[pt]
        print(f"[{pt}] correct={n_corr} | grounded(strict)={s['correct_grounded_rate_strict']:.1%} "
              f"grounded(parts)={s['correct_grounded_rate_parts']:.1%} | "
              f"parametric(parts)={s['parametric_rate_parts']:.1%} "
              f"(strict-upperbnd {s['parametric_rate_strict_upperbound']:.1%}) | "
              f"fail_had_answer_in_retrieval={s['fail_had_answer_in_retrieval_rate']:.1%}")

    out = os.path.join(args.sweep, "retrieval_grounding.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
