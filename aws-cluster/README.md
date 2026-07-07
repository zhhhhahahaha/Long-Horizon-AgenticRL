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
- **`wandb-sync.sh`** — after a training job finishes, run this on the
  **login pod** to upload the offline wandb runs to wandb.ai. Reads the API
  key from `/home/hhzhang01/.wandb-key` (chmod 600, kept outside the repo).

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
