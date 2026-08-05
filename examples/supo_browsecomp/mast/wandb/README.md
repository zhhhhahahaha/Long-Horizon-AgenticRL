# MAST W&B Online Logging

This branch adds an opt-in online mode for formal SUPO training without giving
up the durable OILFS snapshots used by the existing offline workflow. The
default remains `offline`; each immutable experiment config must explicitly
select `online` before submission.

## Modes

Set the mode in the experiment config:

```bash
BC_WANDB_MODE=online
BC_WANDB_ENTITY=hhzhang01
BC_WANDB_PROJECT=supo-bcplus-mast
BC_WANDB_HOST=https://meta-3.wandb.io
```

`BC_WANDB_MODE=offline` preserves the previous behavior. In both modes W&B
writes its transaction log to node-local `/tmp`, never directly to OILFS.

### Online data flow

1. The submitter stages `~/.wandb-key` to a mode-`0600` OILFS file.
2. Each MAST task reads that file once before starting Ray and writes the key to
   the same node-local `/tmp` path.
3. The few Ray processes that own W&B runs read the node-local key file. The
   secret is not placed in MAST arguments, Ray runtime JSON, or W&B config.
4. W&B uploads metrics live to Meta W&B and simultaneously writes a local
   `run-*` transaction log.
5. The existing snapshot process packages the local W&B directory to OILFS at
   the configured interval and once more during normal shutdown.
6. No devserver sync watcher runs while the online job is healthy.

The current training topology normally creates one rollout run and one train
run, grouped under `BC_RUN_NAME` or the full MAST job name. It does not create a
connection per GPU or per compute node.

### Offline data flow

Offline mode writes `offline-run-*` directories locally, snapshots them to
OILFS, and starts the existing devserver watcher. The watcher extracts completed
snapshots to devserver-local disk and runs `wandb sync`.

## Network routing

Training needs two different HTTPS paths:

- Judge requests keep using the node-local relay at `http://127.0.0.1:9080`.
- W&B alone uses `WANDB_HTTPS_PROXY=http://fwdproxy:8080`.

`slime.utils.wandb_utils` passes the W&B-specific proxy through
`wandb.Settings`. Do not replace the global `https_proxy`, because that would
change the judge path.

## Choosing an instance

`meta.wandb.io`, `meta-2.wandb.io` and `meta-3.wandb.io` are **three independent
deployments**, not shards of one cluster: separate Terraform stacks, separate SSO
groups, and separate identity stores — an API key issued by one is rejected by the
others with `invalid api key`. Runs do not replicate between them, so switching
instances leaves your history behind.

Because they are independent, one can be degraded while the others are fine. During
T283503334 (2026-08-03..05) `meta.wandb.io` history ingest ran hours behind while
`meta-3.wandb.io` returned rows in under 30 seconds. Online mode is only useful
against a healthy instance — a backlogged one accepts the filestream POST with
`200 OK` and shows nothing for hours.

Before pointing a long run at an instance, check it:

```bash
# writes 10 rows and reports HEALTHY vs DROPPING WRITES
bash examples/supo_browsecomp/mast/wandb/wandb_instance_check.sh \
  https://meta-3.wandb.io ~/.wandb-key-meta3 <entity>
```

Per-instance access lives in separate AMP groups
(`weights_and_biases_llama` / `_2` / `_3`); get the matching key from
`https://<host>/authorize`. `BC_WANDB_HOST` and `WANDB_KEY_FILE` are the only
settings that need to change.

Confirmed on 2026-08-05: the MAST data project's fwdproxy allowlist already permits
`meta-3.wandb.io` — the online smoke reported `endpoint_status: 200` from inside a
compute container (`supo-wandb-online-smoke-meta3-hrw0z236`, run `1r5weg0g`).

## Submit

Run the normal dry-run and submission commands. The API key is required only
for a real online submission; a dry-run contains the OILFS key path but never
the key value.

```bash
bash examples/supo_browsecomp/mast/submit_experiment.sh --dry-run \
  examples/supo_browsecomp/mast/configs/<config>.sh

WANDB_KEY_FILE="${HOME}/.wandb-key" \
  bash examples/supo_browsecomp/mast/submit_experiment.sh \
  examples/supo_browsecomp/mast/configs/<config>.sh
```

The default persistent key paths are:

```text
devserver: /data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime/.wandb-online-key
container: /mnt/wsfuse/hhzhang01/supo-slime/.wandb-online-key
```

The staged file is shared only for bootstrap. Actors do not repeatedly read it
from OILFS.

## Durability and failure behavior

Online mode is not treated as the only copy of the metrics. W&B continues to
write a local transaction log, and `wandb_snapshot.sh` retains completed tar
snapshots under:

```text
/mnt/wsfuse/hhzhang01/supo-slime/wandb-snapshots/<run-name>/
```

By default snapshots are published every 60 seconds and the newest three per
task are retained. A hard node loss can therefore lose data written after the
latest completed snapshot. Normal shutdown calls `wandb.finish()` before the
trainer publishes its final snapshot. Completed archives use mode `0644` so the
devserver user can extract files created by the root-owned compute container.

If online initialization fails, the W&B process falls back to an offline run in
the same local directory. Training continues and the fallback run is included
in the snapshots. Errors after initialization are buffered by the W&B SDK; the
local snapshot remains the recovery source if the node disappears before the
upload completes.

## Recover an online run

Do not continuously sync online snapshots. That would compete with the live
uploader. After the MAST job reaches a terminal state, use recovery only when a
run is missing or incomplete:

```bash
WANDB_KEY_FILE="${HOME}/.wandb-key" \
  bash examples/supo_browsecomp/mast/wandb/wandb_sync.sh \
  recover-online <full-mast-job-name>
```

For a resumed checkpoint namespace, snapshots use the logical run name:

```bash
MAST_WANDB_RUN_NAME=<logical-run-name> \
WANDB_KEY_FILE="${HOME}/.wandb-key" \
  bash examples/supo_browsecomp/mast/wandb/wandb_sync.sh \
  recover-online <full-mast-job-name>
```

Recovery extracts only completed snapshot archives and invokes `wandb sync`
with `--include-online --include-synced --append`. It also discovers an
`offline-run-*` created by initialization fallback.

## Validation sequence

Before enabling online mode for a long run:

1. Run `examples/supo_browsecomp/mast/wandb/smoke/submit_wandb_online_smoke.sh`
   to verify Meta W&B authentication and the
   TTLS fwdproxy from one compute container.
2. Run `configs/4b_1node_wandb_online_smoke.sh` and confirm both the rollout and
   train runs receive metrics. This config disables checkpoint saving and
   rollout dumps and performs one training iteration.
3. Verify OILFS contains `run-*` in the latest snapshot and test
   `recover-online` against a disposable project or run.
4. Scale to the intended node count only after the formal smoke completes.

Keep `wandb_snapshot.sh`, `wandb_sync.sh`, and the offline mode until the online
path has completed representative long-running jobs. Merging this branch does
not require changing existing configs because the default mode is still
offline.

## Validation record

The one-node formal smoke completed on 2026-08-03 with no task restart:

- MAST job:
  [`supo-4b-wandb-online-training-smoke-wwl4kvsx`](https://www.internalfb.com/mlhub/pipelines/runs/mast/supo-4b-wandb-online-training-smoke-wwl4kvsx)
- Rollout run:
  [`zc40rlgi`](https://meta.wandb.io/hhzhang01/supo-bcplus-mast/runs/zc40rlgi)
- Train run:
  [`lycs4wkn`](https://meta.wandb.io/hhzhang01/supo-bcplus-mast/runs/lycs4wkn)

Both W&B runs reached `finished`. The rollout run uploaded `rollout/*`,
`bcplus_*`, and `perf/*` metrics; the train run uploaded `train/*`, `rollout/*`,
`multi_turn/*`, and `perf/*` metrics. Neither actor used offline fallback.

The smoke exposed that the previous publisher preserved `mktemp` mode `0600`
on OILFS, making root-owned snapshots unreadable from the devserver. This branch
publishes completed archives as `0644` and includes a regression test for that
permission. The live job proved periodic and final archive creation; the
permission and `recover-online` paths are covered by local contract tests.
