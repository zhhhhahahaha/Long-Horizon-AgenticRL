import json
import os
from argparse import ArgumentParser
from pathlib import Path
from shlex import quote

import torch
import torch.distributed.checkpoint as dist_cp

import slime.utils.external_utils.command_utils as U


ENABLE_EVAL = bool(int(os.environ.get("SLIME_TEST_ENABLE_EVAL", "1")))

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 8


parser = ArgumentParser()
parser.add_argument("--async-save", action="store_true", help="Whether to test async save/load.")
parser.add_argument("--save-optimizer", choices=["cpu", "gpu"], default="cpu", help="Optimizer placement for save.")
parser.add_argument("--load-optimizer", choices=["cpu", "gpu"], default="cpu", help="Optimizer placement for load.")
parser.add_argument("--checkpoint-dir", default=None, help="Directory used for the save/load checkpoint roundtrip.")
parser.add_argument(
    "--rolling-slim",
    action="store_true",
    help="Save intermediate checkpoints weights-only while retaining a full final checkpoint.",
)


def default_checkpoint_dir(args):
    save_mode = "async" if args.async_save else "sync"
    if args.rolling_slim:
        save_mode += "_rolling_slim"
    return f"/root/models/{MODEL_NAME}_slime_{save_mode}_{args.save_optimizer}_save_{args.load_optimizer}_load"


def prepare(checkpoint_dir: str):
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command(f"rm -rf {quote(checkpoint_dir)}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")

    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
    )


def optimizer_args(optimizer: str):
    args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--use-precision-aware-optimizer "
    )
    if optimizer == "cpu":
        args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d "
    return args


def execute(mode: str = "", optimizer: str = "cpu", checkpoint_dir: str = "", rolling_slim: bool = False):
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
    checkpoint_dir_arg = quote(checkpoint_dir)
    if mode == "save":
        ckpt_args += f"--save {checkpoint_dir_arg} "
        ckpt_args += "--save-interval 1 " if rolling_slim else "--save-interval 2 "
        if rolling_slim:
            ckpt_args += "--slim-intermediate-checkpoints "
    elif mode == "async_save":
        ckpt_args += f"--save {checkpoint_dir_arg} "
        ckpt_args += "--save-interval 2 "
        ckpt_args += "--async-save "
    elif mode == "load":
        ckpt_args += f"--load {checkpoint_dir_arg} "
        ckpt_args += f"--ckpt-step {3 if rolling_slim else 1} "

    num_rollout = 4 if rolling_slim else 2
    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {num_rollout} "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 16 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 2 --sglang-mem-fraction-static 0.8 --sglang-cuda-graph-max-bs 16 "

    ci_args = "--ci-test "

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args(optimizer)} "
        f"{ppo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


def checkpoint_contents(path: Path) -> tuple[bool, bool, bool]:
    metadata = dist_cp.FileSystemReader(str(path)).read_metadata()
    distributed_keys = set(metadata.state_dict_metadata)
    common_keys = set(torch.load(path / "common.pt", map_location="cpu", weights_only=False))
    has_optimizer = "optimizer" in common_keys or any(
        key == "optimizer" or key.startswith("optimizer.") for key in distributed_keys
    )
    has_scheduler = "opt_param_scheduler" in common_keys or any(
        key.startswith("opt_param_scheduler") for key in distributed_keys
    )
    has_rng = "rng_state" in common_keys or any(key.startswith("rng_state") for key in distributed_keys)
    return has_optimizer, has_scheduler, has_rng


def assert_rolling_checkpoint_layout(checkpoint_dir: str) -> int:
    root = Path(checkpoint_dir)
    steps = sorted(
        int(path.name.removeprefix("iter_"))
        for path in root.glob("iter_*")
        if path.name.removeprefix("iter_").isdigit() and (path / ".metadata").is_file()
    )
    assert len(steps) >= 2, f"rolling checkpoint test expected intermediate and final checkpoints, got {steps}"
    final_step = steps[-1]
    assert int((root / "latest_checkpointed_iteration.txt").read_text().strip()) == final_step
    for step in steps[:-1]:
        assert checkpoint_contents(root / f"iter_{step:07d}") == (False, False, False)
    assert checkpoint_contents(root / f"iter_{final_step:07d}") == (True, True, True)

    state_root = root / ".rolling_slim"
    for workdir in (state_root / "staging", state_root / "backups"):
        assert not workdir.exists(), f"rolling checkpoint workdir was not cleaned: {workdir}"
    manifests = [json.loads(path.read_text()) for path in (state_root / "manifests").glob("iter_*.json")]
    assert manifests
    assert all(manifest["status"] == "promoted" for manifest in manifests)
    return final_step


if __name__ == "__main__":
    args = parser.parse_args()
    assert not (args.async_save and args.rolling_slim), "--rolling-slim is intentionally synchronous"
    # TODO also use typer
    checkpoint_dir = args.checkpoint_dir or default_checkpoint_dir(args)
    prepare(checkpoint_dir)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(
        "save" if not args.async_save else "async_save",
        optimizer=args.save_optimizer,
        checkpoint_dir=checkpoint_dir,
        rolling_slim=args.rolling_slim,
    )
    if args.rolling_slim:
        assert_rolling_checkpoint_layout(checkpoint_dir)
    execute("load", optimizer=args.load_optimizer, checkpoint_dir=checkpoint_dir, rolling_slim=args.rolling_slim)
