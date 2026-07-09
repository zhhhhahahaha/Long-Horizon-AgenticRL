import logging
import os
from copy import deepcopy

import wandb

logger = logging.getLogger(__name__)


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def init_wandb_primary(args):
    """Driver-side wandb bootstrap. No-op body — the training driver process
    itself does not call wandb.log(). All logging happens from the
    RolloutManager and Megatron actors, each of which creates its OWN wandb
    run in init_wandb_secondary(), all joined by shared group=args.wandb_group.

    This avoids the wandb SDK design pitfall where multiple offline processes
    sharing one run_id have their _step counters collide during `wandb sync`
    (see aws-cluster/README.md for the full story). wandb officially
    recommends the multi-run + same-group pattern for distributed logging:
    https://docs.wandb.ai/guides/track/log/distributed-training/

    We still set args.wandb_run_id = None so downstream code that reads it
    (currently only init_wandb_secondary) doesn't NameError.
    """
    args.wandb_run_id = None

    if not args.use_wandb:
        return

    # Set W&B mode env var if specified (so secondary actors inherit it via
    # the WANDB_MODE env var if their args.wandb_mode is unset for any reason).
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")


def _args_to_config_dict(args):
    return deepcopy(args.__dict__)


def _prefix_config_keys(config, prefix):
    return {f"{prefix}/{key}": value for key, value in config.items()}


def _compute_secondary_config_for_logging(args, role=None):
    config = _args_to_config_dict(args)
    if role == "critic":
        return _prefix_config_keys(config, "critic")
    return config


# Map slime's role string -> wandb job_type (shown in the group view).
# role=None: RolloutManager (see slime/ray/rollout.py:457 init_tracking(args, primary=False))
# role="actor": Megatron train actor (slime/backends/megatron_utils/actor.py:65)
# role="critic": Megatron critic actor
_ROLE_TO_JOB_TYPE = {
    None: "rollout",
    "actor": "train",
    "critic": "critic",
}


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-independently
def init_wandb_secondary(args, role=None):
    """Each Ray actor calls this to create its OWN wandb run, joined to the
    shared group=args.wandb_group. wandb UI auto-groups runs with matching
    group names — no manual grouping needed.

    We deliberately do NOT pass id=<shared>/resume="allow" here. See
    init_wandb_primary docstring and aws-cluster/README.md for why.
    """
    if not args.use_wandb:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    settings_kwargs = dict(mode="offline") if offline else dict(mode="online")

    job_type = _ROLE_TO_JOB_TYPE.get(role, role or "rollout")
    run_name = f"{args.wandb_group}-{job_type}"

    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": args.wandb_group,
        "job_type": job_type,
        "name": run_name,
        "config": _compute_secondary_config_for_logging(args, role=role),
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
