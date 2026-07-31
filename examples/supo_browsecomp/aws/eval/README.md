# BrowseComp-Plus evaluation on AWS Slurm

This directory runs the active evaluator in `examples/supo_browsecomp/eval/eval_pipeline.py` on AWS. It does not use the legacy Slurm report code under `eval/legacy`.

## Evaluation contract

- One point uses one exclusive 8-GPU node.
- The test set has 150 questions and four deterministic samples per question (seeds 42-45).
- V2 defaults are fixed search top-k 5 and full-page open capped at 10,000 words.
- Base and trained checkpoints are evaluated with the same tool protocol. The protocol is recorded in `manifest.json` and compared before report generation.
- A checkpoint is eligible only after its `iter_*/.metadata` exists, the tracker covers the step, and its file signature has remained stable for the configured interval.
- `_SUCCESS` is the sole completion marker. A Slurm `COMPLETED` state without `_SUCCESS` is treated as a failure.

## Scheduling

The controller uses three serial lanes by default:

- `dev`: base, then the newest discovered checkpoint, on `a100_dev`.
- `shared-0` and `shared-1`: older checkpoints on `a100_genai_shared`.

Jobs in each lane use `afterany` dependencies. This bounds active evaluation concurrency to three nodes while allowing queued work to survive the controller. The controller caps outstanding submissions at nine, below the cluster's per-user submission limit of ten.

## Start or resume a sweep

Run the controller from the `v2_dev` tmux bridge because `/genai` and Slurm are visible there:

```bash
tmux -S /home/hhzhang01/.tmux-forclaude attach -t v2_dev
cd /storage/home/hhzhang01/slime-top5

python3 examples/supo_browsecomp/aws/eval/eval_sweep.py orchestrate \
  --batch-id fixedtopk5-open10000-100step-v1 \
  --run supo-bcplus-dynamic-fixedtopk5-open10000w-qwen3p5-4b-100step-20260730-0841 \
  --fixed-search-topk 5 \
  --doc-words-full 10000 \
  --target-step 99 \
  --watch
```

The command is idempotent for a batch id. It preserves submission state in `sweep_state.json`, discovers new stable checkpoints, and updates the report whenever another point completes.

To evaluate the original model-controlled search protocol instead, start a separate batch with `--model-controlled-topk --doc-words-full 4096`.

## Outputs

The default root is:

```text
/genai/fsx-llm/interns/hhzhang01/evals/<batch-id>/
```

Important files:

```text
sweep_config.json
sweep_state.json
code.json
base/{manifest.json,point_metrics.json,questions.jsonl,_SUCCESS}
runs/<run>/iterNN/{manifest.json,point_metrics.json,questions.jsonl,_SUCCESS}
runs/<run>/{report.md,checkpoint_metrics.csv,question_changes.csv,metrics.json}
```

Each point also keeps `rollout_data/eval_0.pt`, `eval.log`, and `slurm.log` for debugging. The frozen source archive is stored under `/genai/fsx-llm/interns/hhzhang01/eval-code/` and its SHA-256 is recorded in every manifest.

## Status

```bash
python3 examples/supo_browsecomp/aws/eval/eval_sweep.py status \
  --batch-id fixedtopk5-open10000-100step-v1
```

If a job exits without `_SUCCESS`, inspect that point's `slurm.log` and `eval.log`. The controller intentionally does not overwrite a partial `.pt` or automatically retry a deterministic failure.
