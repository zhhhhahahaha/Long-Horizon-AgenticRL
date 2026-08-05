"""CPU contract tests for BC+ tool protocol propagation through MAST training."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).parents[1]
SUBMIT = ROOT / "examples/supo_browsecomp/mast/submit_experiment.sh"
TRAINER = ROOT / "examples/supo_browsecomp/mast/run_trainer.sh"


def _config() -> str:
    return """\
MAST_JOB_NAME=bcplus-tool-protocol-test
MAST_NUM_NODES=1
MAST_GPUS_PER_NODE=8
MAST_DATA_PARALLEL_SIZE=1
MAST_CONTEXT_PARALLEL_SIZE=8
BC_MODEL_SIZE=4B
BC_NUM_ROLLOUT=1
BC_ROLLOUT_BATCH_SIZE=1
BC_N_SAMPLES=1
BC_GLOBAL_BATCH_SIZE=1
SEARCH_ADDR_FILE=/mnt/wsfuse/hhzhang01/supo-slime/search-servers/test-search.addr
BC_WANDB_MODE=online
BC_WANDB_ENTITY=test-entity
BCPLUS_FIXED_SEARCH_TOPK=5
BCPLUS_DOC_WORDS_FULL=10000
"""


@pytest.mark.unit
def test_mast_training_dry_run_propagates_tool_protocol(tmp_path):
    jq = shutil.which("jq")
    if jq is None:
        pytest.skip("jq is required by the MAST submission wrapper")

    args_path = tmp_path / "args"
    fake_cli = tmp_path / "fake-mast-cli"
    fake_cli.write_text("""#!/bin/bash
printf '%s\\0' "$@" > "${FAKE_CLI_ARGS}"
cat <<'JSON'
{"status":"ok","dryrun":true,"spec":{"hpc_job_definition":{"hpcTaskGroups":[{"name":"trainer_0","taskCount":1}]},"app_def":{"roles":[{"name":"trainer_0","env":{"ROLE_ASSIGNMENT_MAP":"trainer_0=8"}}]}}}
JSON
""")
    fake_cli.chmod(0o755)
    config = tmp_path / "config.sh"
    config.write_text(_config())
    env = {
        **os.environ,
        "FAKE_CLI_ARGS": str(args_path),
        "MAST_RL_CLI": str(fake_cli),
        "MAST_JQ_BIN": jq,
        "MAST_DRYRUN_ROOT": str(tmp_path / "dryruns"),
    }

    result = subprocess.run(
        ["bash", str(SUBMIT), "--dry-run", str(config)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    cli_args = args_path.read_bytes().split(b"\0")
    custom_command = next(arg for arg in cli_args if arg.startswith(b"--docker_custom_cmd="))
    assert b"BCPLUS_FIXED_SEARCH_TOPK=5" in custom_command
    assert b"BCPLUS_DOC_WORDS_FULL=10000" in custom_command
    assert b"SEARCH_ADDR_FILE=/mnt/wsfuse/hhzhang01/supo-slime/search-servers/test-search.addr" in custom_command
    assert b"BC_WANDB_MODE=online" in custom_command
    assert b"BC_WANDB_ENTITY=test-entity" in custom_command
    assert b"MAST_WANDB_KEY_FILE=/mnt/wsfuse/hhzhang01/supo-slime/.wandb-online-key" in custom_command
    assert b"WANDB_API_KEY" not in custom_command
    assert "search_topk=5 open_words=10000" in result.stdout
    assert "search discovery: /mnt/wsfuse/hhzhang01/supo-slime/search-servers/test-search.addr" in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ("BCPLUS_FIXED_SEARCH_TOPK=0", "BCPLUS_FIXED_SEARCH_TOPK must be a positive integer"),
        ("BCPLUS_DOC_WORDS_FULL=invalid", "BCPLUS_DOC_WORDS_FULL must be a positive integer"),
    ],
)
def test_mast_training_rejects_invalid_tool_protocol(tmp_path, setting, message):
    config = tmp_path / "config.sh"
    lines = [line for line in _config().splitlines() if not line.startswith(setting.split("=", 1)[0] + "=")]
    config.write_text("\n".join([*lines, setting, ""]))

    result = subprocess.run(
        ["bash", str(SUBMIT), "--dry-run", str(config)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.unit
def test_mast_trainer_passes_tool_protocol_to_ray_actors():
    trainer = TRAINER.read_text()

    assert '\\"BCPLUS_FIXED_SEARCH_TOPK\\": \\"${BCPLUS_FIXED_SEARCH_TOPK}\\"' in trainer
    assert '\\"BCPLUS_DOC_WORDS_FULL\\": \\"${BCPLUS_DOC_WORDS_FULL}\\"' in trainer
    assert '--wandb-key-file "${WANDB_LOCAL_KEY_FILE}"' in trainer
    assert '\\"WANDB_HTTPS_PROXY\\": \\"${WANDB_HTTPS_PROXY}\\"' in trainer


@pytest.mark.unit
def test_mast_training_rejects_relative_search_addr_file(tmp_path):
    config = tmp_path / "config.sh"
    config.write_text(_config().replace(
        "SEARCH_ADDR_FILE=/mnt/wsfuse/hhzhang01/supo-slime/search-servers/test-search.addr",
        "SEARCH_ADDR_FILE=search-server.addr",
    ))

    result = subprocess.run(
        ["bash", str(SUBMIT), "--dry-run", str(config)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "SEARCH_ADDR_FILE must be an absolute container path" in result.stderr


@pytest.mark.unit
def test_mast_training_requires_search_addr_file(tmp_path):
    config = tmp_path / "config.sh"
    config.write_text("\n".join(
        line for line in _config().splitlines() if not line.startswith("SEARCH_ADDR_FILE=")
    ))

    result = subprocess.run(
        ["bash", str(SUBMIT), "--dry-run", str(config)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "experiment config must set SEARCH_ADDR_FILE" in result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
