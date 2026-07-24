#!/usr/bin/env python3
"""Grounding check: does the model's FINAL finish answer actually cite the
gold / evidence docids for that question?

Join: eval sample metadata.query_id  ->  bc_test.parquet extra_info
  gold_docs     : the essential doc(s)      (small; strict)
  evidence_docs : broader supporting set    (lenient)
Cited docids: bracketed ints inside the LAST <function=finish>...</function>
block of the rollout's final sub-trajectory response.

Reports per checkpoint, split correct vs wrong:
  cites_any   : % answers citing >=1 docid at all
  hit_evid    : % citing >=1 evidence_doc
  hit_gold    : % citing the gold_doc(s)
  evid_recall : mean |cited ∩ evidence| / |evidence|
  precision   : mean |cited ∩ evidence| / |cited|   (over answers that cite)
"""
import argparse, json, os, re
from collections import defaultdict
import torch
import pandas as pd

PARQUET = "/genai_hh/datasets/BC+/bc_test.parquet"
FINISH_RE = re.compile(r"<function=finish>(.*?)</function>", re.DOTALL)
BRACKET_RE = re.compile(r"\[([\d,\s]+)\]")


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


def _docset(arr):
    out = set()
    if arr is None:
        return out
    for d in arr:
        if isinstance(d, dict) and d.get("docid") is not None:
            out.add(str(d["docid"]))
    return out


def load_gold():
    df = pd.read_parquet(PARQUET)
    gold, evid = {}, {}
    for _, r in df.iterrows():
        ei = r["extra_info"]
        qid = str(ei.get("query_id"))
        gold[qid] = _docset(ei.get("gold_docs"))
        evid[qid] = _docset(ei.get("evidence_docs"))
    return gold, evid


def cited_docids(final_sample):
    resp = final_sample.get("response", "") or ""
    blocks = FINISH_RE.findall(resp)
    if not blocks:
        return set()
    text = blocks[-1]  # last finish block = the actual submission
    out = set()
    for m in BRACKET_RE.findall(text):
        for tok in m.split(","):
            tok = tok.strip()
            if tok.isdigit():
                out.add(tok)
    return out


def _point_key(name):
    if name == "base":
        return (0, 0)
    if name.startswith("iter"):
        try:
            return (1, int(name[4:]))
        except ValueError:
            return (2, name)
    return (3, name)


def analyze(pt_dir, gold, evid):
    dump = os.path.join(pt_dir, "rollout_data", "eval_0.pt")
    if not os.path.isfile(dump):
        return None
    blob = torch.load(dump, weights_only=False)
    groups = defaultdict(list)
    for i, s in enumerate(blob.get("samples", [])):
        rid = s.get("rollout_id") or s.get("index", i)
        groups[rid].append(s)

    agg = {True: defaultdict(float), False: defaultdict(float)}
    cnt = {True: 0, False: 0}
    for rid, sibs in groups.items():
        # final sub-traj: is_final flag else max sub_traj_index
        def _fin(s):
            sib = (s.get("metadata") or {}).get("_bcplus_sibling", {}) if isinstance(s.get("metadata"), dict) else {}
            return sib.get("is_final", False), sib.get("sub_traj_index", 0)
        final = sorted(sibs, key=lambda s: (_fin(s)[0], _fin(s)[1]))[-1]
        score = max(_score_of(s.get("reward")) for s in sibs)
        correct = score >= 0.5
        qid = None
        md = final.get("metadata")
        if isinstance(md, dict):
            qid = str(md.get("query_id"))
        g, e = gold.get(qid, set()), evid.get(qid, set())
        cited = cited_docids(final)
        c = correct
        cnt[c] += 1
        a = agg[c]
        a["cites_any"] += 1 if cited else 0
        a["hit_evid"] += 1 if (cited & e) else 0
        a["hit_gold"] += 1 if (cited & g) else 0
        if e:
            a["evid_recall"] += len(cited & e) / len(e)
        if cited:
            a["prec_num"] += len(cited & e) / len(cited)
            a["prec_den"] += 1
        a["n_cited_docs"] += len(cited)
    return agg, cnt


def _row(label, a, n):
    if n == 0:
        return "%-16s n=0" % label
    prec = (a["prec_num"] / a["prec_den"]) if a["prec_den"] else 0.0
    return ("%-16s n=%3d | cites_any=%4.0f%% hit_evid=%4.0f%% hit_gold=%4.0f%% "
            "| evid_recall=%.2f prec=%.2f avg_cited=%.1f" % (
        label, n, 100*a["cites_any"]/n, 100*a["hit_evid"]/n, 100*a["hit_gold"]/n,
        a["evid_recall"]/n, prec, a["n_cited_docs"]/n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    args = ap.parse_args()
    root = args.eval_root
    gold, evid = load_gold()
    ng = sum(1 for v in gold.values() if v); ne = sum(1 for v in evid.values() if v)
    print("gold table: %d qs, %d with gold_docs, %d with evidence_docs" % (len(gold), ng, ne))
    points = [d for d in sorted(os.listdir(root), key=_point_key)
              if os.path.isfile(os.path.join(root, d, "rollout_data", "eval_0.pt"))]
    out = {}
    for name in points:
        res = analyze(os.path.join(root, name), gold, evid)
        if not res:
            continue
        agg, cnt = res
        print("\n== %s ==" % name)
        print("  " + _row("CORRECT", agg[True], cnt[True]))
        print("  " + _row("WRONG", agg[False], cnt[False]))
        out[name] = {
            "correct": {k: agg[True][k] for k in agg[True]}, "n_correct": cnt[True],
            "wrong": {k: agg[False][k] for k in agg[False]}, "n_wrong": cnt[False],
        }
    with open(os.path.join(root, "grounding_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", os.path.join(root, "grounding_check.json"))


if __name__ == "__main__":
    main()
