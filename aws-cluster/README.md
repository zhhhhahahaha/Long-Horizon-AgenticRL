# AWS cluster (Meta HPC) helpers for slime

Anything in this directory is **specific to running slime on the Meta shared AWS
cluster** (login pod = EKS, compute nodes = A100-80GB via Slurm, FSx home,
Lustre `/genai/fsx-project/`, `enroot` for containers). It is not upstream
slime code — it's here so this directory is a single grep-able answer to
"what do I need to run slime on this cluster."

## Files

- **`wandb-args.sh`** — sourced by every slime run script *inside* the
  enroot container. Defines the `WANDB_ARGS` bash array with `--wandb-mode
  offline` and `--wandb-dir /data/wandb`. Requires `RUN_NAME` env var to be
  set by the caller.
- **`wandb-sync.sh`** — after a training job finishes (or every 5 min mid-run
  via `launch_all.sh`), run this on the **login pod** to upload the offline
  wandb runs to wandb.ai. Reads the API key from `/home/hhzhang01/.wandb-key`
  (chmod 600, kept outside the repo). Uses `python3 -m wandb sync --append`.
  `--append` is required because the script re-syncs the same *still-growing*
  offline runs every 5 min: plain `wandb sync` closes the cloud run after the
  first upload, freezing a live/resumed run at its first-synced step; `--append`
  resumes each run and pushes only the new steps. Also the on-demand command for
  "sync now so I can see the curves": `bash wandb-sync.sh <RUN_NAME>`.
- **`wandb-sync-merged.py`** — **DEPRECATED**, kept for reference only.
  Was an earlier attempt to fix the multi-process wandb offline sync bug
  (see below); introduced its own file_stream offset collision bug. Will be
  deleted once the multi-run+group approach has proven stable.

## Multi-run + group wandb layout (offline mode gotcha)

wandb's SDK isn't built for the scenario where multiple offline processes
all attach to one shared `run_id` and later get sync'd together — their
per-process `_step` counters collide and either the config `metrics` array
gets clobbered (stock `wandb sync`, per-dir SendManager) or history rows
get overlaid at the wrong offsets (any shared-SendManager workaround).
Verified during run `qwen3p5-4b-cp2-tp4-20260707-2318` on 2026-07-08 —
cloud showed `bcplus`'s `rollout/step=[0,1,1,1,1,2,6,3,3,3,4,11,...]`
instead of the correct `0..19`.

Fix: **each Ray actor gets its own wandb run, joined by shared
`group=RUN_NAME`**. wandb UI auto-groups runs with matching group values
(https://docs.wandb.ai/guides/track/log/distributed-training/), so opening
the group shows all constituent runs together with overlay curves — no
manual grouping needed. Implementation in
`slime/utils/wandb_utils.py:init_wandb_secondary`; each actor picks a
`job_type` (`"rollout"` for RolloutManager, `"train"` for Megatron actor,
`"critic"` for critic actor). Driver process's `init_wandb_primary` is a
no-op — it doesn't call `wandb.log()`, so we don't create an empty run for it.

**In wandb UI:** filter by `group=<RUN_NAME>`, you'll see 2 runs
(`<RUN_NAME>-rollout`, `<RUN_NAME>-train`) auto-grouped. To view combined
plots, use the default group view.

## Why offline mode is mandatory

Compute nodes on this cluster have **no external network egress**. Slime's
`wandb.log()` runs inside training actors on compute nodes, so real-time
upload is impossible. The login pod does have an X2P proxy — that's where
`wandb sync` runs after the srun job returns.

See also memory `[[wandb-setup]]` and `[[slime-enroot-import]]`.

## Every new slime run script on this cluster needs

1. **Outer launcher (login-pod side)**:
   - Defines `RUN_NAME="<something>-$(date +%Y%m%d-%H%M)"`, `export RUN_NAME`
   - `srun ... bash -c "... ENROOT_MOUNT_HOME=false enroot start ... --mount /home/hhzhang01/slime/aws-cluster:/aws-cluster --mount /genai/fsx-project/hhzhang01/wandb:/data/wandb --env RUN_NAME='${RUN_NAME}' ..."`
   - After srun exits: `bash /home/hhzhang01/slime/aws-cluster/wandb-sync.sh "${RUN_NAME}"`
2. **Inner script (in-container)**:
   - After `set -ex`: `: "${RUN_NAME:?...}"` and `source /aws-cluster/wandb-args.sh` (note the in-container path — `/home/hhzhang01/` is not visible inside the container because we disable `ENROOT_MOUNT_HOME`)
   - Passes `"${WANDB_ARGS[@]}"` into `python3 train.py` (or `ray job submit ... -- python3 train.py`)

Miss any one and wandb logging silently breaks (no group, or no upload).
