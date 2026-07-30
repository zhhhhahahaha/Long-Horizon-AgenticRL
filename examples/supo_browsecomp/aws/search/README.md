# BrowseComp-Plus search server on AWS

This directory owns the long-lived retrieval service used by BrowseComp-Plus
rollouts. The service runs on one A100 in Slurm and serves the shared corpus on
port 8000. Fixed search top-k and `open_page` word limits are rollout-client
settings; they do not require separate baseline and V2 search servers.

## Start or reuse the service

Run the launcher from a persistent login-node tmux session and from the
checkout or worktree whose search code should be mounted:

```bash
cd <repo-or-worktree>
bash examples/supo_browsecomp/aws/search/launch_search_server.sh
```

The launcher derives `SLIME_HOST_DIR` from its own path, so a launcher in a Git
worktree mounts that worktree at `/slime`. Set `SLIME_HOST_DIR` explicitly only
when a different checkout is intentional.

The operation is idempotent. It looks for the current user's job named
`supo-search-server` and:

- reuses a healthy running server with at least 48 hours remaining;
- waits for an existing pending job instead of submitting a duplicate;
- replaces a running server with insufficient time remaining after a 10-second
  warning window;
- otherwise submits a new job and waits for `/health`.

The default allocation is one GPU, 128 GB RAM, 8 CPUs, seven days, account
`genai_interns`, and QOS `a100_dev`. Override `MIN_HOURS_REMAINING`, `QOS`,
`SERVER_PORT`, or `SEARCH_JOB_NAME` through environment variables when needed.
`SEARCH_GPUS` controls both the Slurm GPU request and the number of embedding
workers; `SEARCH_CPUS` defaults to eight CPUs per GPU. `SEARCH_MEM` accepts a
Slurm memory value such as `128G` or `0` for all node memory.

## Blue-green full-node replacement

Do not stop a search server that an active trainer is using. Start and validate
the replacement under a different job name, address file, and log file first:

```bash
SEARCH_JOB_NAME=supo-search-server-8gpu \
SEARCH_GPUS=8 SEARCH_CPUS=64 SEARCH_MEM=0 \
SEARCH_HOST_FILE=/genai/fsx-project/hhzhang01/logs/search-server-8gpu.hostname \
SEARCH_LOG_FILE=/genai/fsx-project/hhzhang01/logs/search-server-8gpu.log \
bash examples/supo_browsecomp/aws/search/launch_search_server.sh
```

The two jobs may use the same port because they run on different hosts. Before
switching a consumer, require `/health` to report eight workers and send a real
`/search` request. Promote the new address to `search-server.hostname` only at
a checkpoint boundary. Existing trainers pin `LOCAL_SEARCH_URL` at startup and
must be resumed to change servers; pending eval jobs should resolve the address
file when they start.

## Monitor and connect

```bash
squeue -u "$USER" -n supo-search-server -o '%.18i %.12T %.12L %.20N'
tail -f /genai/fsx-project/hhzhang01/logs/search-server.log
cat /genai/fsx-project/hhzhang01/logs/search-server.hostname
curl -sf "http://$(cat /genai/fsx-project/hhzhang01/logs/search-server.hostname)/health"
```

On a new server, seeing `Loading corpus dataset ...` for several minutes is
normal while the corpus, embeddings, and Qwen3-Embedding-8B model load. FastAPI
`on_event` deprecation warnings are non-fatal. The launcher waits up to 10
minutes for health after the Slurm job starts. A cold start on this cluster was
observed to become healthy in about 180 seconds on 2026-07-30.

Do not edit `launch_search_server.sh` while that same file is running. Bash may
continue reading a long-lived script after startup, so changing its byte offsets
during the health wait can make the launcher fail after the server is already
healthy. Finish or stop the launcher before editing it, then rerun it; the
idempotent path will reuse the healthy job.

After a successful launch it writes `<hostname>:<port>` to
`/genai/fsx-project/hhzhang01/logs/search-server.hostname` and prints:

```bash
export LOCAL_SEARCH_URL=http://<hostname>:8000
```

The 4B training launcher reads the hostname file automatically. To pin a
specific service and skip its ensure/replacement logic, export
`LOCAL_SEARCH_URL` before starting training.

## Stop the service

The server is shared by training and evaluation jobs. Check for consumers
before cancellation, then stop it by job ID:

```bash
squeue -u "$USER" -n supo-search-server
scancel <job-id>
```

## Enroot invariants

The launcher intentionally uses the shared imported `slime-test` rootfs and
sets both `ENROOT_TEMP_PATH=/dev/shm` and `ENROOT_MOUNT_HOME=false`. It also
pins `ENROOT_DATA_PATH` to `/storage/home/hhzhang01/.local/share/enroot`.
Changing these can cause overlay whiteout failures, hide image contents under
the host home mount, or make the rootfs unavailable on the compute node.
