#!/usr/bin/env python3
"""Verify online W&B logging from a MAST compute container."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from types import SimpleNamespace


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is empty: {name}")
    return value


def _configure_environment() -> dict[str, str]:
    base_url = os.environ.get("WANDB_BASE_URL", "https://meta.wandb.io").rstrip("/")
    proxy_url = (
        os.environ.get("MAST_WANDB_HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTPS_PROXY")
        or "http://fwdproxy:8080"
    )
    entity = _required_env("WANDB_ENTITY")
    project = _required_env("WANDB_PROJECT")
    group = _required_env("WANDB_RUN_GROUP")

    os.environ["WANDB_BASE_URL"] = base_url
    os.environ["WANDB_MODE"] = "online"
    os.environ["https_proxy"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["no_proxy"] = "127.0.0.1,localhost,::1"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
    return {
        "base_url": base_url,
        "proxy_url": proxy_url,
        "entity": entity,
        "project": project,
        "group": group,
    }


def _check_endpoint(base_url: str, proxy_url: str) -> int:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": proxy_url}))
    request = urllib.request.Request(base_url, method="GET", headers={"User-Agent": "mast-wandb-online-smoke"})
    with opener.open(request, timeout=20) as response:
        return response.status


def _write_result(result_path: str, payload: dict[str, object]) -> None:
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    temporary.replace(path)


def run_smoke() -> dict[str, object]:
    config = _configure_environment()
    endpoint_status = _check_endpoint(config["base_url"], config["proxy_url"])
    print(
        f"[wandb-online-smoke] endpoint reachable: status={endpoint_status} "
        f"url={config['base_url']} proxy={config['proxy_url']}"
    )

    # Authenticate through WANDB_API_KEY in the environment. Passing the key in
    # args.wandb_key would copy it into slime's W&B config dictionary.
    _required_env("WANDB_API_KEY")
    args = SimpleNamespace(
        use_wandb=True,
        use_tensorboard=False,
        wandb_mode="online",
        wandb_key=None,
        wandb_host=config["base_url"],
        wandb_team=config["entity"],
        wandb_project=config["project"],
        wandb_group=config["group"],
        wandb_dir=os.environ.get("WANDB_DIR", "/tmp/mast-wandb-online-smoke"),
        wandb_explicit_teardown=True,
    )

    import wandb
    from slime.utils.logging_utils import finish_tracking, init_tracking, log

    try:
        init_tracking(args, primary=False, role="actor")
        if wandb.run is None:
            raise RuntimeError("wandb.init returned without an active run")
        run_id = wandb.run.id
        run_url = wandb.run.url
        for step in range(3):
            log(
                args,
                {
                    "train/step": step,
                    "train/wandb_online_smoke": float(step + 1),
                },
                step_key="train/step",
            )
    finally:
        finish_tracking(args)

    payload = {
        "status": "ok",
        "endpoint_status": endpoint_status,
        "entity": config["entity"],
        "project": config["project"],
        "group": config["group"],
        "run_id": run_id,
        "run_url": run_url,
    }
    result_path = os.environ.get("MAST_WANDB_RESULT_PATH")
    if result_path:
        _write_result(result_path, payload)
        print(f"[wandb-online-smoke] result={result_path}")
    print(f"[wandb-online-smoke] PASS run={run_url}")
    return payload


if __name__ == "__main__":
    try:
        run_smoke()
    except Exception as error:
        result_path = os.environ.get("MAST_WANDB_RESULT_PATH")
        if result_path:
            _write_result(
                result_path,
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise
