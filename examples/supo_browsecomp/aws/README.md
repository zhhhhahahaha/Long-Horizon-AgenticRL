# BrowseComp on AWS (Slurm + enroot)

This directory contains the AWS-cluster launchers for the SUPO / BrowseComp-Plus
pipeline. Everything here is scheduler-specific: two-part (login pod →
in-container) scripts that submit `srun`/`sbatch` jobs and run enroot on
`/genai/fsx-project/...`. The model rollout, reward, and eval-metric
implementation stays scheduler-independent in the parent directory. The MAST
counterparts live under [`../mast/`](../mast/README.md).

## Workflows

| Workflow | Entry point |
|---|---|
| Training (8-node colocate) | `run_qwen3p5_4B_colocate.sh` (canonical), `run_qwen3p5_9B_colocate.sh` |
| Search service | `search/launch_search_server.sh` |
| HF → mcore conversion (one-shot) | `convert_qwen3p5_9B.sh` |

See [`../README.md`](../README.md) for prerequisites (search server, MetaGen
judge, checkpoints, parquet) and the full run walkthrough.

## Directory ownership

- Training launchers live at this root because both 4B and 9B share the same
  colocate machinery (per-node `/dev/shm` rootfs staging, ray head, weight sync,
  wandb-sync poll). `run_qwen3p5_4B_colocate.sh` is the live canonical script;
  `run_qwen3p5_9B_colocate.sh` is its 9B sibling.
- `search/` owns the long-lived retrieval-server orchestrator. It is idempotent:
  reuses a running `supo-search-server` job with enough walltime left, otherwise
  scancels + resubmits. Both colocate scripts call it automatically.
- `convert_qwen3p5_9B.sh` is a one-shot preflight that turns the Qwen3.5-9B HF
  checkpoint into the Megatron `torch_dist` format. It writes to FSx
  (`/genai/fsx-project/.../models/Qwen3.5-9B_torch_dist`). MAST consumes the
  same artifact from its own path and does not run this script.

The 1-node debug wrappers (`../debug_scripts/`) are kept out of git
(personal tooling); they re-use these launchers' in-container half via
`SLIME_INNER=1`.

Do not put rollout, reward, or metric logic here. That belongs in the parent
directory's shared Python (`generate_with_bcplus.py`, `dynamic_sampling.py`,
`summary_advantage.py`, `eval/eval_pipeline.py`) so MAST can reuse it.
