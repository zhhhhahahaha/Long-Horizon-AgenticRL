# slime rollout data model & the `rollout_id` name collision

*A mental model for anyone touching slime's rollout / loss / logging code — especially before adding compression, async training, or per-rollout metric logging.*

## The 4 conceptual levels

Use these names (not slime's — slime's identifiers overload each other). Example numbers use `rollout-batch-size=4, n-samples-per-prompt=4`.

| # | Name | What it is | Count per rollout |
|---|------|------------|-----------------|
| 0 | **rollout batch** | one call to `RolloutManager.generate(rollout_id)`; corresponds to one training step in sync mode | 1 (per driver step) |
| 1 | **prompt** | one question from the training parquet | `rollout-batch-size` = 4 |
| 2 | **trajectory** | one full sampled response for a prompt (temperature-seeded independently). GRPO groups the `n` trajectories of a prompt together for advantage normalization | `rollout-batch-size × n-samples-per-prompt` = 16 |
| 3 | **sub-trajectory** | a trajectory optionally split into shorter chunks for training (compression / very long trajectories). Without compression, sub-traj == trajectory | ≥ 16 |
| 4 | **training sample** | one `Sample` object handed to the loss reducer. Equals one sub-trajectory | ≥ 16 |

**With compression** (planned): one Level-2 trajectory of e.g. 30k tokens is split into 3 sub-trajectories of ~10k each; each becomes an independent training sample. **Without compression** (current BC+): each trajectory is one sample; Levels 2/3/4 collapse.

## slime's identifiers, mapped to the levels

| slime field | Level it identifies | Semantics |
|---|---|---|
| `sample.index` | Level 4 (training sample = sub-trajectory) | Global monotonic counter, set by `data_source.get_samples` at line `sample.index += 1`. Survives across rollouts. **Not** a per-rollout position. |
| `sample.group_index` | Level 1 (prompt) | Global monotonic prompt counter. All Level-2 siblings of one prompt share the same group_index. GRPO group normalization reshapes on this. |
| `sample.rollout_id` | Level 2 (trajectory) | **Not** the driver's step id. It's the trajectory-execution grouping ID used by the **loss reducer**. See "The name collision" below. |
| driver's `rollout_id` (function parameter) | Level 0 (rollout batch) | The training step counter. Equals wandb `rollout/step`. Present in `RolloutManager.generate(rollout_id)`, `_log_rollout_data(rollout_id, …)`, and every custom hook that receives it. |

### The name collision (this is the trap)

**Two different concepts share the name `rollout_id` in slime**:

- **Driver's `rollout_id`** (parameter) — Level 0 — "which training step is this rollout batch". Always well-defined, monotonic from 0. This is what you want on the x-axis of your wandb curves.
- **`sample.rollout_id`** (field) — Level 2 — "which trajectory execution did this training sample come from". Its purpose is to tell the loss reducer that N sub-trajectories from one trajectory should be counted as one rollout unit, not N.

They are **not** the same. Confusing them (e.g., stamping `sample.rollout_id = driver's rollout_id`) is silently wrong under async and compression.

## Loss reducer's use of `sample.rollout_id`

`slime/ray/rollout.py:_convert_samples_to_train_data` (lines 703-711) computes `rollout_mask_sums`:

```python
rollout_id_list = train_data["rollout_ids"]
mask_sums_per_sample = [sum(m) for m in loss_masks]
rollout_total_mask: dict[int, int] = {}
for rid, ms in zip(rollout_id_list, mask_sums_per_sample, strict=True):
    rollout_total_mask[rid] = rollout_total_mask.get(rid, 0) + ms
train_data["rollout_mask_sums"] = [rollout_total_mask[rid] for rid in rollout_id_list]
```

Loss for sample `i` is then divided by `rollout_mask_sums[i]` (per-rollout-unit total), not by that sample's own `loss_mask.sum()`.

**Without compression** (all `sample.rollout_id` distinct): each sample's denominator is its own mask sum → per-sample loss. Correct because each sample is its own trajectory.

**With compression** (siblings share `rollout_id`): all N sub-trajectory siblings use their combined mask sum as denominator → per-trajectory loss. Correct because the trajectory should contribute 1 gradient's worth, not N.

**If compression forgets to share `rollout_id`**: 3 sub-trajectories of 100 tokens each with distinct rollout_ids give `(100/100) + (100/100) + (100/100) = 3` rollout losses instead of `300/300 = 1`. **3× over-counting**.

## What each code path does with `sample.rollout_id`

| Path | Sets `sample.rollout_id`? | Why |
|------|--------------------------|-----|
| **Sync default** (`sglang_rollout.generate_rollout`) | No — stays `None`. `_convert_samples_to_train_data` assigns unique `tmp_id` per sample as fallback. | No compression; each sample is its own trajectory unit; per-sample loss is desired. |
| **Async** (`fully_async_rollout.generate_rollout_fully_async`) | No — stays `None`. Same fallback. | Same reason. **Crucially**, the driver's `rollout_id` param is only used for the async worker's own logs — it is NOT written to samples. Async in-flight pool can span multiple driver steps, so stamping driver's step id would be a lie about which trajectory execution the sample came from. |
| **Compression / subagent** (`agent/trajectory.py:251`, `examples/multi_agent`) | **Yes** — explicitly set to `base_sample.rollout_id or base_sample.index`, shared across all siblings. Slime validates this at `rollout.py:880 _validate_rollout_id_annotated`. | To make the loss reducer count each trajectory once. |

Rule: **`sample.rollout_id` lives at the trajectory execution layer (Level 2)**. The framework and downstream code (loss reducer, dp_schedule grouping) treat it that way. If you write custom rollout code that fans out a trajectory into sub-trajectories, you must share `rollout_id` across siblings.

## Where to get the driver's `rollout_id` (Level 0) for your own hooks

**Not from `sample.rollout_id`.** Get it from slime's hooks that receive it as a parameter:

- `--custom-rollout-log-function-path` — signature `(rollout_id, args, samples, extra_metrics, rollout_time) -> bool`. **Called for every training step in both sync and async modes.** This is the correct hook for wandb `rollout/step`-aligned logging. Returning `False` (or `None`) lets slime's default log run afterwards.
- `--custom-eval-rollout-log-function-path` — analogous, called for eval.
- `--custom-generate-function-path` receives `(args, sample, sampling_params)` — **does NOT receive `rollout_id`**. Don't try to log per-rollout metrics from here.
- `--custom-rm-path` receives `(args, sample)` (per sample) — same. Not the right place for per-rollout logging.
- `--custom-reward-post-process-path` receives `(args, samples)` — **also does NOT receive `rollout_id`**. Historically we mis-used this for wandb logging; the resulting `sample.rollout_id` fallback (`None`) meant the bcplus curves never got the correct step binding and only "worked" by accident when wandb's internal `_step` counter happened to align with the true step id.

## Rules of thumb

1. **Wandb x-axis / per-rollout-batch metrics** → use `--custom-rollout-log-function-path`. The `rollout_id` param is the driver's step counter. Works for sync/async/compression.
2. **Per-sample rewards / filters** → use `--custom-rm-path` / `--custom-generate-function-path`. Don't touch `sample.rollout_id`.
3. **Writing a compression / subagent rollout function** → you MUST set `sample.rollout_id = <shared id>` on every sub-trajectory sibling. Otherwise the loss reducer over-counts.
4. **Never** conflate the driver's `rollout_id` with `sample.rollout_id`. They live at different levels and mean different things. Stamping driver's `rollout_id` onto samples is silently wrong under async and under compression.

## Key files to read when this comes up again

- `slime/utils/types.py:90-110` — `Sample` dataclass with docstrings for the 3 ID fields
- `slime/rollout/data_source.py:108-116` — where `index` and `group_index` are assigned
- `slime/ray/rollout.py:534-547` — `RolloutManager.generate(rollout_id)` entry point (sync and async both call this)
- `slime/ray/rollout.py:703-711` — sample.rollout_id fallback logic (`tmp_id` counter)
- `slime/ray/rollout.py:751-756` — `rollout_mask_sums` computation from rollout_id groups
- `slime/ray/rollout.py:880-909` — `_validate_rollout_id_annotated` (enforces compression contract)
- `slime/ray/rollout.py:1273-1280` — `_log_rollout_data` dispatch (where `custom_rollout_log_function_path` is invoked with the driver's rollout_id)
- `slime/agent/trajectory.py:251` — example of correct sample.rollout_id assignment in a compact/subagent rollout
- `slime/rollout/fully_async_rollout.py:194-256` — async rollout function; note that its `rollout_id` param is only used for logging, not written to samples
