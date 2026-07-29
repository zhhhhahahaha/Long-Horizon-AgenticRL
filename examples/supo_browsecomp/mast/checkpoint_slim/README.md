# Megatron checkpoint slimming

This directory contains the complete workflow for replacing intermediate BC+
Megatron checkpoints with weights-only checkpoints. The purpose is to retain
every checkpoint needed by evaluation while keeping optimizer state only in
the latest checkpoint, which remains suitable for exact training resume.

## Safety contract

Read these rules before running anything:

- Stop training before cleanup. Do not slim a run that may still write
  checkpoints.
- A complete checkpoint is an `iter_XXXXXXX` directory containing `.metadata`.
- The tool automatically protects the numerically largest complete checkpoint.
  There is no fixed final iteration.
- `latest_checkpointed_iteration.txt` must equal that protected iteration. A
  mismatch aborts the operation instead of guessing which checkpoint is final.
- A numerically newer `iter_*` directory without `.metadata` also aborts the
  operation because checkpoint writing may still be active.
- Requested slim steps must be intermediate checkpoints. The protected latest
  checkpoint is rejected even if it is supplied explicitly.
- Before converting anything, the distributed worker verifies that the
  protected checkpoint metadata contains optimizer state. Its iteration and
  metadata SHA-256 are recorded in every conversion manifest and rechecked
  before canary promotion.
- Intermediate weights-only checkpoints support evaluation but not exact
  training resume. Only the protected checkpoint retains optimizer, scheduler,
  and RNG state.
- Never delete individual `.distcp` files. Model and optimizer records may be
  mixed in the same files, so checkpoints must be rewritten through Megatron.
- Never run two slimming jobs against the same training run concurrently.

The current launcher is configured for Qwen3.5-4B checkpoints on one 8-GPU
`zionex_80g` host using TP4/CP2. Supporting another model size requires matching
the model configuration and parallel layout used to save that checkpoint.

## Training-time rolling slimming

New BC+ training runs should set the following in their MAST experiment config:

```bash
BC_SLIM_INTERMEDIATE_CHECKPOINTS=1
```

The runner maps this to slime's opt-in `--slim-intermediate-checkpoints`. At
every non-final save N, training first writes a weights-only copy under
`<save>/.rolling_slim/staging/iter_N`, then writes the normal full `iter_N`.
Only after full N and its tracker are validated does it atomically replace the
previous full checkpoint with the already prepared weights-only copy. The
normal checkpoint names never change, so eval discovery remains unchanged.

For save steps 4, 9, and final 14, the stable states are:

```text
after 4:   normal 4=full                 staging 4=slim
after 9:   normal 4=slim, 9=full         staging 9=slim
after 14:  normal 4=slim, 9=slim, 14=full
```

The temporary backup used during replacement is a same-filesystem rename, not
another copy. Before the rename, the staged checkpoint must pass metadata hash,
total-size, and weights-only content validation. After the rename, slime checks
the installed `.metadata` identity and `common.pt` presence before deleting the
backup. It intentionally does not recursively restat the renamed directory:
remote FUSE mounts can briefly expose stale directory entries or file sizes for
the reused source path. Interrupted unambiguous states are repaired on startup;
conflicts retain the old full checkpoint instead of interrupting training.

Slimming old checkpoints is best-effort and must not interrupt training. Before
promotion, slime rechecks the staged metadata hash, total byte size, and absence
of optimizer, scheduler, and RNG state. If a staged checkpoint changed or is
incomplete, it is moved under `.rolling_slim/orphans`, the old full checkpoint
is marked `retained_full`, and training continues with a warning. Failure to
delete a post-promotion backup is also only a warning. Only an invalid latest
full checkpoint or a tracker/load mismatch is fatal.

When enabling the option while resuming a checkpoint produced by older code,
slime loads the existing latest full checkpoint normally and creates its slim
staging copy before the next rollout. Older historical full checkpoints are
left untouched; run the offline workflow below before resume if those should
also be slimmed. A newly submitted MAST job receives a new scheduler suffix, so
its resume config must set `BC_RUN_NAME` to the exact existing checkpoint
directory basename. The runner refuses an explicit resume name without a valid
`latest_checkpointed_iteration.txt` instead of silently starting a new run.

Rolling slimming is synchronous in its first version and rejects
`--async-save`, `--release-train`, `--no-save-optim`, `--no-save-rng`, and
Megatron checkpoint retention. It records separate slim-save, full-save, and
promotion durations in the actor log.

## Files

- `checkpoint_slim.py`: CPU-safe planning and promotion commands plus the
  distributed Megatron conversion worker.
- `run_checkpoint_slim.sh`: one-host conversion entrypoint. It can stage only
  or stage, validate, and promote checkpoints.
- `run_checkpoint_slim_canary_eval.sh`: runs the normal eval path against one
  staged checkpoint and promotes it only after eval succeeds.

## Storage layout

On the development host, the source checkpoints normally live under:

```text
/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/checkpoints/<run>/
```

Inside MAST they are mounted at:

```text
/mnt/wsfuse/hhzhang01/supo-slime/checkpoints/<run>/
```

For `SLIM_BATCH_ID=<batch>`, working state is stored at:

```text
/mnt/wsfuse/hhzhang01/supo-slime/checkpoint-slim/<batch>/
├── staging/<run>/iter_XXXXXXX/
├── backups/<run>/
├── manifests/<run>/iter_XXXXXXX.json
└── logs/
```

Use a new, descriptive batch ID for each cleanup campaign. Keep the manifests
and logs after completion; they are small and provide provenance.

## End-to-end workflow

The commands in steps 3-5 are job entrypoints inside the MAST container after
the repository archive has been extracted to `/slime-src`. Use one 8-GPU MAST
task per command. The mounted checkpoint storage must be available at
`/mnt/wsfuse/hhzhang01/supo-slime`.

### 1. Inspect the plan on the development host

From the repository root:

```bash
RUN="replace-with-completed-training-run-name"

python examples/supo_browsecomp/mast/checkpoint_slim/checkpoint_slim.py plan \
  --run "${RUN}"
```

For a non-default checkpoint parent, add:

```bash
--checkpoint-root /path/to/checkpoints
```

The output contains:

- `protected_step`: the latest complete checkpoint that will not be modified.
- `steps`: all intermediate checkpoints eligible for slimming.

Stop if the selected run or steps are not exactly what you expect. A tracker
that disagrees with the latest complete checkpoint causes this command to fail.
Planning is intentionally metadata-only: an intermediate checkpoint that was
already slimmed still appears in `steps`. Check existing cleanup manifests and
checkpoint sizes before selecting `SLIM_STEPS`; do not blindly reprocess every
planned step.

### 2. Archive the exact code used by MAST

The MAST container does not read the development checkout directly. Archive
the current repository, including uncommitted files that are part of this
workflow, into storage visible to MAST:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
ARCHIVE_DIR=/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/slim-code
ARCHIVE="${ARCHIVE_DIR}/checkpoint-slim-$(date +%Y%m%d-%H%M%S).tgz"

mkdir -p "${ARCHIVE_DIR}"
tar -czf "${ARCHIVE}" \
  --exclude=.git \
  --exclude=.pytest_cache \
  --exclude=__pycache__ \
  --exclude='*.pyc' \
  -C "${REPO_ROOT}" .
sha256sum "${ARCHIVE}"
```

Record both the archive path and SHA-256. In MAST, the corresponding path is
under `/mnt/wsfuse/hhzhang01/supo-slime/slim-code/`. Every conversion and
canary job in one cleanup batch must use this same archive.

### 3. Stage one canary checkpoint

Choose the first intermediate step and a unique batch ID. In a single-host,
8-GPU MAST job, run:

```bash
export SLIM_RUN_NAME="replace-with-completed-training-run-name"
export SLIM_STEPS="4"  # Replace with one step reported by plan.
export SLIM_BATCH_ID="checkpoint-slim-YYYYMMDD-v1"
export SLIM_PROMOTE=0

bash /slime-src/examples/supo_browsecomp/mast/checkpoint_slim/run_checkpoint_slim.sh
```

This loads the original checkpoint without optimizer/RNG state, saves a native
weights-only `torch_dist` checkpoint, reloads it, and compares a distributed
SHA-256 over all model parameters and buffers. It writes a `staged_valid`
  manifest but does not replace the source checkpoint. The manifest also
  records the protected checkpoint iteration and metadata hash.

Do not proceed unless the MAST job completes and the canary manifest has
`"status": "staged_valid"`.

### 4. Run the normal eval path and promote the canary

Submit a fresh single-host, 8-GPU MAST job from the same code archive:

```bash
export SLIM_RUN_NAME="replace-with-completed-training-run-name"
export SLIM_STEP="4"  # Must equal the staged canary step.
export SLIM_BATCH_ID="checkpoint-slim-YYYYMMDD-v1"
export SLIM_EVAL_ALIAS="__checkpoint_slim_canary_RUN_iter4_YYYYMMDD"
export EVAL_OUTPUT_DIR="/mnt/wsfuse/hhzhang01/supo-slime/evals/checkpoint-slim-canary-RUN-iter4"
export EVAL_CODE_ARCHIVE_SHA256="replace-with-recorded-archive-sha256"

bash /slime-src/examples/supo_browsecomp/mast/checkpoint_slim/run_checkpoint_slim_canary_eval.sh
```

The wrapper exposes only the staged checkpoint through a temporary symlink,
runs the unchanged `mast/eval/run_eval.sh` with `EVAL_N=1`, removes the symlink, and
promotes the staged checkpoint only after eval succeeds. The eval environment
must have the normal search service and judge credentials used by BC+ eval.

Verify that the MAST job completed, `EVAL_OUTPUT_DIR/_SUCCESS` exists, and the
manifest status changed to `promoted` before continuing.

### 5. Convert the remaining intermediate checkpoints

Run the remaining steps in one or more single-host, 8-GPU MAST jobs. Steps in a
comma-separated list are processed sequentially:

```bash
export SLIM_RUN_NAME="replace-with-completed-training-run-name"
export SLIM_STEPS="9,14,19"  # Replace with the remaining planned steps.
export SLIM_BATCH_ID="checkpoint-slim-YYYYMMDD-v1"
export SLIM_PROMOTE=1

bash /slime-src/examples/supo_browsecomp/mast/checkpoint_slim/run_checkpoint_slim.sh
```

`SLIM_PROMOTE=1` performs all of the following for every step:

1. Load and hash the original model.
2. Save and validate a weights-only checkpoint in staging.
3. Reload staging and compare its model hash with the original.
4. Atomically move the original to backup and staging to the original path.
5. Reload from the final path and compare the model hash again.
6. Delete the backup only after final-path validation succeeds.

The latest protected checkpoint is never accepted in `SLIM_STEPS`.

### 6. Perform final read-only checks

Replace the placeholders below with the development-host paths:

```bash
RUN="replace-with-completed-training-run-name"
BATCH="checkpoint-slim-YYYYMMDD-v1"
ROOT=/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime

python examples/supo_browsecomp/mast/checkpoint_slim/checkpoint_slim.py plan \
  --checkpoint-root "${ROOT}/checkpoints" \
  --run "${RUN}"

jq -s 'map({step,protected_step,protected_metadata_sha256,status,backup_deleted,source_bytes,slim_bytes})' \
  "${ROOT}/checkpoint-slim/${BATCH}/manifests/${RUN}"/*.json

find "${ROOT}/checkpoint-slim/${BATCH}/backups/${RUN}" \
  -mindepth 1 -print
```

Confirm that:

- every requested step has `status=promoted` and `backup_deleted=true`;
- the backup search prints nothing;
- the tracker still names `protected_step`;
- the protected checkpoint metadata hash and directory size have not changed;
- every evaluation checkpoint still contains `.metadata`.

Record the protected checkpoint metadata hash and size before the first
promotion if you need byte-for-byte evidence that it was untouched.

## Environment variables

| Variable | Required by | Meaning |
| --- | --- | --- |
| `SLIM_RUN_NAME` | both wrappers | Source directory name under `checkpoints/` |
| `SLIM_STEPS` | conversion | Comma-separated intermediate iterations |
| `SLIM_BATCH_ID` | both wrappers | Unique state/manifests/logs namespace |
| `SLIM_PROMOTE` | conversion | `0` stages only; `1` installs after validation |
| `SLIM_STEP` | canary eval | Single staged iteration to evaluate |
| `SLIM_EVAL_ALIAS` | canary eval | Unique name beginning with `__checkpoint_slim_canary_` |
| `EVAL_OUTPUT_DIR` | canary eval | New output directory with no stale partial eval |
| `EVAL_CODE_ARCHIVE_SHA256` | canary eval | Submitted repository archive SHA-256 |

## Resume and failure behavior

- A `promoted` manifest is skipped on retry.
- A `staged_valid` manifest is resumable. The worker rechecks source metadata,
  staged metadata, staged size, and both model hashes before continuing.
- A failed staging or reload does not touch the source checkpoint.
- During promotion the source is first moved to `backups/`. If final-path
  validation fails, the code restores the original checkpoint.
- A pre-existing unresolved backup aborts the run and requires manual audit.
  Do not delete it merely to make a retry proceed.
- The canary alias cleanup removes only a symlink with the required safe name;
  it refuses to remove a real directory.

## MAST submission contract

The shell files above are container entrypoints, not host-side MAST submitters.
When constructing a MAST job, follow the existing one-host eval submission
pattern in `mast/eval/eval_sweep.py`:

- extract the exact repository archive to `/slime-src`;
- run one `trainer_0` task on `zionex_80g` with eight GPUs;
- set data parallelism and context parallelism to one at the MAST task layer;
- export the variables for exactly one of the commands above;
- run the appropriate wrapper as `docker_custom_cmd`;
- dry-run the MAST submission before the real submission;
- retain the archive SHA-256 and MAST job name with the cleanup records.

Do not reuse an archive after editing these scripts: create a new archive and
record its new SHA-256.
