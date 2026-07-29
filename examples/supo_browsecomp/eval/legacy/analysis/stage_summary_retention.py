#!/usr/bin/env python3
"""Stage summary-retention candidates for the agent pass (part-wise pre-filter).

Goal: measure whether the model's FINAL handover summary retained an answer it had
already retrieved earlier — a clean "summary content ability" metric. We only agent-
review FAILURES where the answer was plausibly retrieved (a LENIENT part-wise match on
the observation text), because failures where gold was never retrieved cannot be a
summary loss. The agent then judges semantically (handles paraphrase). Only failures,
only the 4 report checkpoints.

Candidate = failure AND n_sub_trajs>=2 AND gold's parts all appear in the OBSERVATIONS
(<tool_response> blocks) of at least one NON-final sub-traj.

Reads chains/<pt>.jsonl (on /genai), writes one JSON per candidate to /home for agents.
Plain python (login pod).
"""
import argparse, json, os, re

OBS_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)

def _norm(s): return re.sub(r"\s+", " ", str(s or "").strip().lower())
def _contains(hay, needle):
    return bool(needle) and len(needle) >= 2 and re.search(r"\b"+re.escape(needle)+r"\b", hay) is not None
def _parts(g):
    return [p.strip() for p in re.split(r"[;,]| and ", g) if len(p.strip()) >= 3]
def _all_parts_in(hay, g):
    ps = _parts(g)
    return all(_contains(hay, p) for p in ps) if ps else _contains(hay, g)

def _obs(resp): return _norm(" ".join(OBS_RE.findall(resp or "")))

def _obs_windows(resp, gold, radius=260, maxw=2):
    """Raw obs snippets around gold parts (so the agent sees the retrieval evidence)."""
    text = " ".join(OBS_RE.findall(resp or ""))
    out = []
    for part in (_parts(_norm(gold)) or [_norm(gold)]):
        for m in re.finditer(r"\b"+re.escape(part)+r"\b", text, flags=re.IGNORECASE):
            a, b = max(0, m.start()-radius), min(len(text), m.end()+radius)
            out.append(("..." if a>0 else "")+text[a:b]+("..." if b<len(text) else ""))
            break
        if len(out) >= maxw: break
    return out

def _trunc(t, head=4000, tail=2000):
    return t if len(t) <= head+tail+40 else f"{t[:head]}\n...[TRUNC {len(t)-head-tail}]...\n{t[-tail:]}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", nargs="+", default=["base","iter04","iter24","iter44"])
    args = ap.parse_args()

    index = {"sweep": args.sweep, "points": {}}
    for pt in args.points:
        src = os.path.join(args.sweep, "chains", f"{pt}.jsonl")
        if not os.path.isfile(src):
            print(f"[stage] WARN missing {src}"); continue
        n_fail = n_cand = 0
        out_pt = os.path.join(args.out, pt); os.makedirs(out_pt, exist_ok=True)
        paths = []
        for line in open(src):
            c = json.loads(line)
            if c.get("correct"): continue
            n_fail += 1
            subs = c.get("sub_trajs", [])
            if len(subs) < 2: continue
            gold = _norm(c.get("gold"))
            # part-wise gold in observations of a NON-final sub-traj
            hit = any(_all_parts_in(_obs(s.get("response","")), gold) for s in subs[:-1])
            if not hit: continue
            rec = {
                "point": pt, "rollout_id": c.get("rollout_id"),
                "question": c.get("question"), "gold_answer": c.get("gold"),
                "final_answer": c.get("final_answer"), "n_sub_trajs": c.get("n_sub_trajs"),
                "outcomes": c.get("outcomes"), "summary_sources": c.get("summary_sources"),
                "sub_trajs": [{
                    "i": s.get("i"), "outcome": s.get("outcome"),
                    "is_final": (s.get("i") == len(subs)-1),
                    "obs_gold_windows": ([] if s.get("i")==len(subs)-1 else _obs_windows(s.get("response",""), c.get("gold"))),
                    "summary": s.get("summary"),
                    "response": _trunc(s.get("response","")),
                } for s in subs],
            }
            p = os.path.join(out_pt, f"cand_{n_cand:04d}.json")
            json.dump(rec, open(p,"w"), ensure_ascii=False, indent=2)
            paths.append(p); n_cand += 1
        index["points"][pt] = {"n": n_cand, "dir": out_pt, "files": paths}
        print(f"[stage] {pt}: failures={n_fail} candidates={n_cand} -> {out_pt}")
    os.makedirs(args.out, exist_ok=True)
    json.dump(index, open(os.path.join(args.out,"index.json"),"w"), indent=2, ensure_ascii=False)
    print(f"[stage] total candidates = {sum(v['n'] for v in index['points'].values())}")

if __name__ == "__main__":
    main()
