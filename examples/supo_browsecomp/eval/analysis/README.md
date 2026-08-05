# Summary-retention evaluation

`summary_retention.py` is the supported version of the legacy
`stage_summary_retention.py` plus `stage_b_summret_workflow.js` study. It asks a
narrow question about failed, compressed rollouts:

> When an earlier tool observation genuinely contained the gold answer, did the
> final handover summary carry that answer or an unambiguous identifying fact
> into the final sub-trajectory?

This is a semantic research metric. It does not replace strict answer accuracy
and is not included in `pass@1`.

## Protocol

The analysis has three separately auditable stages.

### 1. Stage candidates

```bash
POINT=/path/to/evals/<batch>/runs/<run>/iter04
ANALYSIS="$POINT/summary_retention"

python examples/supo_browsecomp/eval/analysis/summary_retention.py stage \
  --point-dir "$POINT" \
  --output-dir "$ANALYSIS"
```

The point must already satisfy the active `_SUCCESS` contract and retain
`rollout_data/eval_0.pt`. At startup the analysis checks that
`manifest.point` agrees with `load_verification.actual_step`, then trusts and
directly reads that completed point. The pre-filter selects only:

1. strict model failures (`correct == false` and `judge_failed == false`),
2. rollouts with at least two sub-trajectories, and
3. rollouts where every normalized gold-answer part occurs somewhere across the
   tool observations of non-final sub-trajectories.

Gold parts are derived from `<qN>` answer fields when present, otherwise from a
lenient comma/semicolon/`and` split. The match is deliberately high recall. It
is only a candidate generator and must not be reported as summary loss.

Outputs:

- `candidates.jsonl`: question, answer, deduplicated matching tool responses,
  and final handover.
- `failure_retrieval.jsonl`: one lightweight docid-retrieval record per model
  failure, used for mutually exclusive failure attribution.
- `stage_manifest.json`: source point/checkpoint identity, protocol version, and
  all filter counts.
- `_STAGED`: completion marker written last.

### 2. Judge matches, then summaries

```bash
export LLAMA_API_KEY=...

python examples/supo_browsecomp/eval/analysis/summary_retention.py judge \
  --stage-dir "$ANALYSIS" \
  --model gpt-5-4-genai-dss4 \
  --concurrency 8
```

For a judge sanity check, limit only the number of new candidates processed by this
invocation:

```bash
python examples/supo_browsecomp/eval/analysis/summary_retention.py judge \
  --stage-dir "$ANALYSIS" \
  --max-new-candidates 12 \
  --keep-raw-responses
```

This leaves the analysis incomplete without `_JUDGED`. A later invocation resumes
at the individual match or summary boundary; remove the limit to finish the point.

Search responses are split into individual matching document blocks; matching
opened pages retain broader context. Only byte-identical response units for the
same tool/document are deduplicated, while all occurrence locations and search
queries are preserved. Every response has a stable `evidence_id`. Every lexical
`(gold_part, evidence_id)` pair has a stable `match_id`, including multiple parts
matched by the same response. The response's `content` is the complete tool-returned
unit; there is no second length cap.

Judging uses two separate prompts and API calls:

1. Each `match_id` is judged independently as `yes`, `no`, or `unclear`. The
   judge receives one gold part and one full matching response. It decides only
   whether that occurrence has the part's intended semantic identity/value/role.
2. Only after early retrieval is confirmed, a separate judge sees only the gold
   answer and final handover and labels it `carried`, `dropped`, `distorted`, or
   `unclear`.

`Dropped` means the gold answer itself is absent, even if the summary asserts a
different wrong answer or retains related clues. `Distorted` is reserved for a
recognizable gold answer that remains present but has been materially corrupted,
partially lost, or explicitly rejected. A clearly named usable gold candidate is
`carried` even if the summary prefers another candidate.

The code derives candidate-level `early_retrieval` deterministically across all
non-final sub-trajectories. Every required gold part must have at least one semantic
`yes`, but those confirmations may come from different sub-trajectories. If every
part has `yes|unclear` and at least one lacks `yes`, the aggregate is `unclear`;
otherwise it is `no`.

The match judge does not decide whether one response proves the whole question or
whether retrieval caused the final error. The gold is authoritative; the question
only identifies the part's semantic role. Search queries are retrieval metadata,
not factual evidence. The summary judge receives no retrieval responses, so it
cannot revise the early-retrieval decision or condition retention on evidence quality.

The command validates the exact schema and cross-field constraints, retries invalid
responses, and atomically checkpoints completed pair and summary judgments. Rerunning
with the same staged data, judge model, and protocol resumes missing work. A changed
judge contract is rejected. Raw API responses are omitted by default; pass
`--keep-raw-responses` only when debugging.

Outputs:

- `match_judgments.jsonl`: one resumable semantic verdict per `match_id`.
- `summary_judgments.jsonl`: one resumable retention verdict per confirmed candidate.
- `judgments.jsonl`: compact candidate verdict deterministically derived from the two files above.
- `judge_manifest.json`: model, protocol version, and candidate/judgment counts.
- `_JUDGED`: completion marker written only after one-to-one coverage.

### 3. Build checkpoint metrics

```bash
python examples/supo_browsecomp/eval/analysis/summary_retention.py report \
  --analysis-dir /path/to/base/summary_retention \
  --analysis-dir /path/to/iter04/summary_retention \
  --analysis-dir /path/to/iter24/summary_retention \
  --output-dir /path/to/run/summary_retention_report
```

All points must use the same pre-filter version, judge protocol, and judge model.
Candidate, match, summary, and derived-judgment coverage must match exactly.

The primary metric is:

```text
drop_rate = dropped / (carried + dropped + distorted)
```

The denominator contains only candidates with `early_retrieval=yes` and a
resolved retention verdict. `retention_coverage` reports what fraction of
confirmed early retrievals received a resolved verdict. Reports also expose
pre-filter false-positive rate, retrieval uncertainty, and each component
count/rate. The broader
`summary_loss_rate=(dropped+distorted)/(carried+dropped+distorted)` is retained
as a secondary diagnostic; `distorted` remains separate because that category
has a less reliable boundary than strict `dropped`.

The same report deterministically measures retrieval coverage over every model
failure, including failures that did not enter the lexical candidate set:

- `failure_no_gold_doc_rate` uses the benchmark's strict `gold_docs` docids.
- `failure_no_evidence_doc_rate` uses the broader supporting `evidence_docs`
  docids and is the stronger proxy for never retrieving useful evidence.
- Opened-document rates distinguish a search-result hit from a page the agent
  explicitly opened.
- `drop_share_of_model_failures` is the observed summary-drop failure mode as a
  fraction of all model failures.
- `optimistic_drop_uplift_all_rollouts` divides observed drops by all rollouts.
  It is only an upper bound: the analysis does not establish that retaining the
  answer would necessarily make the final answer correct.

Trajectories without the relevant docid annotations are excluded from that
retrieval rate's denominator and reported through their annotation count.

The human-readable report separates these overlapping retrieval diagnostics
from a mutually exclusive failure-mode table. Each model failure enters exactly
one category, in this priority order:

1. A semantically confirmed answer is classified by its summary outcome:
   `summary_dropped`, `summary_distorted`, or
   `summary_carried_final_wrong`.
2. Remaining resolved cases are classified as
   `no_evidence_doc_retrieved` or
   `evidence_doc_retrieved_answer_not_confirmed`.
3. Missing annotations and unclear semantic/retention judgments are
   `unresolved`.

The categories sum to all strict model failures. This is a deterministic,
hierarchical failure-mode assignment, not proof that changing only the assigned
component would make the rollout correct.

`evidence_doc_retrieved_answer_not_confirmed` specifically means that the
summary-retention pipeline did not confirm a correct answer before the final
handover. It is a residual bucket, not proof that the answer never appeared. It
can include open-page truncation, answers first seen in an uncompressed or final
sub-trajectory, and answer-surface normalization misses.

Outputs are `summary_retention_report.md`, `summary_retention_metrics.csv`,
`summary_retention_metrics.json`, and `_SUMMARY_RETENTION_SUCCESS`.

### 4. Compare two judge models

After both model-specific reports complete, the `compare` command validates that
they used identical staged candidates and match IDs, then reports pair-level,
candidate-level, and summary-label agreement alongside both metric tables:

```bash
python examples/supo_browsecomp/eval/analysis/summary_retention.py compare \
  --model-a-name GPT-5.4 \
  --model-b-name Claude-4.8-Opus \
  --model-a-analysis-dir /path/to/gpt/base \
  --model-a-analysis-dir /path/to/gpt/iter04 \
  --model-b-analysis-dir /path/to/opus/base \
  --model-b-analysis-dir /path/to/opus/iter04 \
  --output-dir /path/to/comparison
```

Repeat each analysis-dir option for every checkpoint. Outputs are
`summary_retention_model_comparison.md`, its machine-readable JSON equivalent,
and `_SUMMARY_RETENTION_COMPARISON_SUCCESS`. Summary-label agreement is computed
only on candidates both judges confirmed as early retrieval; the report does not
silently merge disagreements into a third label.

## Storage

The analysis never copies `eval_0.pt` or full trajectories. It writes each matching
search-document/open-page response once in `candidates.jsonl`, plus the final handover,
short pair/summary verdicts, manifests, and reports. Pair-level checkpoints do not
duplicate tool-response content. Non-matching responses and model reasoning are not
copied, and raw API responses are off by default. The existing raw rollout dump remains
the dominant storage cost.

## Interpretation limits

- The metric conditions on failed rollouts and cannot estimate summary quality
  over all rollouts.
- It measures whether the final handover retained any genuinely retrieved
  answer. With more than two sub-trajectories, it does not by itself identify
  the first handoff where information was lost.
- The lexical pre-filter can miss paraphrases and alternate answer forms, so
  the resulting loss rate applies to the staged candidate population rather
  than every possible semantic retrieval.
- Judge-model or rubric changes define a new protocol and must not be mixed in
  one checkpoint trend.
