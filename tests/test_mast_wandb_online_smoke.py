"""CPU contract tests for the MAST direct W&B smoke job."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).parents[1]
SMOKE = ROOT / "examples/supo_browsecomp/mast/wandb_online_smoke.py"
SUBMIT = ROOT / "examples/supo_browsecomp/mast/submit_wandb_online_smoke.sh"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("test_mast_wandb_online_smoke_module", SMOKE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_smoke_uses_environment_auth_and_real_slime_tracking_contract(tmp_path, monkeypatch):
    smoke = _load_smoke_module()
    monkeypatch.setenv("WANDB_API_KEY", "fake-secret-that-must-not-enter-config")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    monkeypatch.setenv("WANDB_RUN_GROUP", "test-group")
    monkeypatch.setenv("https_proxy", "http://fwdproxy:8080")
    monkeypatch.setenv("MAST_WANDB_RESULT_PATH", str(tmp_path / "result.json"))
    monkeypatch.setattr(smoke, "_check_endpoint", lambda base_url, proxy_url: 200)

    calls = []
    fake_wandb = ModuleType("wandb")
    fake_wandb.run = SimpleNamespace(id="run-123", url="https://meta.wandb.io/test/run-123")
    fake_logging = ModuleType("slime.utils.logging_utils")

    def init_tracking(args, primary, role):
        calls.append(("init", args, primary, role))

    def log(args, metrics, step_key):
        calls.append(("log", args, metrics, step_key))

    def finish_tracking(args):
        calls.append(("finish", args))

    fake_logging.init_tracking = init_tracking
    fake_logging.log = log
    fake_logging.finish_tracking = finish_tracking
    fake_slime = ModuleType("slime")
    fake_slime.__path__ = []
    fake_utils = ModuleType("slime.utils")
    fake_utils.__path__ = []
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setitem(sys.modules, "slime", fake_slime)
    monkeypatch.setitem(sys.modules, "slime.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "slime.utils.logging_utils", fake_logging)

    payload = smoke.run_smoke()

    init_call = calls[0]
    assert init_call[0] == "init"
    args = init_call[1]
    assert init_call[2:] == (False, "actor")
    assert args.wandb_mode == "online"
    assert args.wandb_key is None
    assert args.wandb_host == "https://meta.wandb.io"
    assert args.wandb_explicit_teardown is True
    assert os.environ["HTTPS_PROXY"] == "http://fwdproxy:8080"
    assert [call[2]["train/step"] for call in calls if call[0] == "log"] == [0, 1, 2]
    assert calls[-1][0] == "finish"
    assert payload["run_id"] == "run-123"
    result_text = (tmp_path / "result.json").read_text()
    assert json.loads(result_text) == payload
    assert "fake-secret" not in result_text


def _write_fake_cli(path: Path) -> None:
    path.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\\0' "$@" >> "${FAKE_CLI_ARGS}"
for argument in "$@"; do
  if [[ "${argument}" == "--dryrun" ]]; then
    cat <<'JSON'
{"status":"ok","dryrun":true,"spec":{"hpc_job_definition":{"hpcTaskGroups":[{"name":"trainer_0","taskCount":1}]},"app_def":{"roles":[{"name":"trainer_0","env":{"ROLE_ASSIGNMENT_MAP":"trainer_0=8"}}]}}}
JSON
    exit 0
  fi
done
cat <<'JSON'
{"status":"ok","dryrun":false,"job":{"job_name":"supo-wandb-online-smoke-abcd1234","mast_url":"https://mlhub/supo-wandb-online-smoke-abcd1234"}}
JSON
"""
    )
    path.chmod(0o755)


def _submit_env(tmp_path: Path) -> dict[str, str]:
    fake_cli = tmp_path / "fake-mast-cli"
    _write_fake_cli(fake_cli)
    key = tmp_path / "wandb-key"
    key.write_text("fake-wandb-api-key\n")
    archive = tmp_path / "code.tgz"
    archive.write_bytes(b"test archive")
    return {
        **os.environ,
        "MAST_RL_CLI": str(fake_cli),
        "MAST_JQ_BIN": shutil.which("jq") or "jq",
        "FAKE_CLI_ARGS": str(tmp_path / "cli-args"),
        "WANDB_KEY_FILE": str(key),
        "MAST_WANDB_KEY_HOST_PATH": str(tmp_path / "staged-wandb-key"),
        "MAST_WANDB_KEY_CONTAINER_PATH": "/mnt/test/.wandb-key",
        "MAST_CODE_ARCHIVE_HOST_PATH": str(archive),
        "MAST_CODE_ARCHIVE_CONTAINER_PATH": "/mnt/test/code.tgz",
        "MAST_WANDB_RESULT_PATH": "/mnt/test/result.json",
        "MAST_WANDB_STATE_ROOT": str(tmp_path / "state"),
        "MAST_BUILD_CODE_ARCHIVE": "0",
    }


@pytest.mark.unit
def test_submitter_keeps_key_out_of_mast_spec_and_bypasses_sync_wrapper(tmp_path):
    if shutil.which("jq") is None:
        pytest.skip("jq is required by the MAST submission wrapper")
    env = _submit_env(tmp_path)

    result = subprocess.run(["bash", str(SUBMIT)], capture_output=True, text=True, env=env, check=False)

    assert result.returncode == 0, result.stderr
    raw_args = Path(env["FAKE_CLI_ARGS"]).read_bytes()
    assert raw_args.split(b"\0").count(b"mast") == 2
    assert b"fake-wandb-api-key" not in raw_args
    assert b"submit_with_wandb" not in raw_args
    custom_commands = [arg for arg in raw_args.split(b"\0") if arg.startswith(b"--docker_custom_cmd=")]
    assert len(custom_commands) == 2
    assert all(b"WANDB_API_KEY=$(tr" in command for command in custom_commands)
    assert all(b"WANDB_BASE_URL=https://meta.wandb.io" in command for command in custom_commands)
    assert all(b"wandb_online_smoke.py" in command for command in custom_commands)
    assert b"--docker_host_cmd=" not in raw_args
    assert raw_args.split(b"\0").count(b"--retries=0") == 2
    staged_key = Path(env["MAST_WANDB_KEY_HOST_PATH"])
    assert staged_key.read_text() == "fake-wandb-api-key"
    assert staged_key.stat().st_mode & 0o777 == 0o600
    assert "job=supo-wandb-online-smoke-abcd1234" in result.stdout


@pytest.mark.unit
def test_submitter_rejects_missing_key_before_calling_mast(tmp_path):
    env = _submit_env(tmp_path)
    env["WANDB_KEY_FILE"] = str(tmp_path / "missing-key")

    result = subprocess.run(
        ["bash", str(SUBMIT), "--dry-run"], capture_output=True, text=True, env=env, check=False
    )

    assert result.returncode != 0
    assert "W&B API key is missing or empty" in result.stderr
    assert not Path(env["FAKE_CLI_ARGS"]).exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
