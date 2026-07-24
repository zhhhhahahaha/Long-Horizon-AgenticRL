#!/usr/bin/env python3
"""Extract concrete parse_fail examples: the assistant text that FAILED to parse
into a <tool_call>, i.e. the turn right before a
"No function call was detected in the model response." observation.

Run via run_report.sh with REPORT_SCRIPT=.../parse_fail_examples.py
EXTRA_ARGS="--point iter24 --n 8"
"""
import argparse, os, re
import torch

MARK = "No function call was detected in the model response."
# assistant turn boundary in the decoded trajectory
ASSIST = "<|im_start|>assistant"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--point", default="iter24")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tail", type=int, default=900, help="chars of assistant tail to show")
    args = ap.parse_args()

    dump = os.path.join(args.eval_root, args.point, "rollout_data", "eval_0.pt")
    blob = torch.load(dump, weights_only=False)
    samples = blob.get("samples", [])

    shown = 0
    for s in samples:
        resp = s.get("response", "") or ""
        if MARK not in resp:
            continue
        # split trajectory into assistant turns; for each occurrence of the
        # marker, grab the assistant chunk that immediately precedes it.
        idx = 0
        while True:
            m = resp.find(MARK, idx)
            if m < 0:
                break
            idx = m + len(MARK)
            # assistant turn preceding this observation
            a_start = resp.rfind(ASSIST, 0, m)
            chunk = resp[a_start:m] if a_start >= 0 else resp[max(0, m - 1500):m]
            # strip the trailing tool_response opener if present
            chunk = re.split(r"<\|im_start\|>user", chunk)[0]
            tail = chunk[-args.tail:]
            shown += 1
            print("\n" + "=" * 90)
            print("EXAMPLE %d  (point=%s)  assistant turn tail [%d chars]:" % (shown, args.point, len(tail)))
            print("-" * 90)
            print(tail)
            if shown >= args.n:
                print("\n[shown %d examples]" % shown)
                return
    print("\n[shown %d examples total]" % shown)


if __name__ == "__main__":
    main()
