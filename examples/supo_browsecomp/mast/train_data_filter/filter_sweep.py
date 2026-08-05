#!/usr/bin/env python3
"""Freeze, validate, and submit the BC+ training-data filter sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


RUN_NAME = "supo_4b_n8_8n_40iter_dump_groupfix-mj1d0qw1"
STEPS = (4, 9)
EXPECTED_QUESTIONS = 680
SAMPLES_PER_QUESTION = 8
ROLLOUT_SEED = 42
SGLANG_SERVER_CONCURRENCY = 36
SEARCH_CONCURRENCY = 64
JUDGE_CONCURRENCY = 16
DOC_WORDS_FULL = 4096
TRAIN_DATA_SHA256 = "44ce3691039433c28ab50086bf60e1fb380e06b0d268ea5b9ae701d2247a79f4"

DEV_STAGE = Path("/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-slime")
MAST_STAGE = Path("/mnt/wsfuse/hhzhang01/supo-slime")
CHECKPOINT_STAGE = DEV_STAGE / "checkpoints"
CLI = Path("/data/users/hhzhang01/fbsource/genai/msl/rl/cli.sh")
IMAGE = "588845226011.dkr.ecr.us-east-2.amazonaws.com/msl_infra/slime:hhz-20260629a"
WSF_SRC = "ws://ws.ai.eag0genai/genai_fair_llm"
TENANT = "rhea_assistant_lens"
REGION = "eag"
HOST = "grandteton_80g_roce"
PRIORITY = "CRITICAL"
FAILED_STATES = {"FAILED", "DEAD"}
SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def validate_safe_name(value: str, label: str) -> str:
    if SAFE_NAME.fullmatch(value) is None or value in {".", ".."}:
        raise RuntimeError(f"invalid {label}: {value!r}")
    return value


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CGROUPED"] = "1"
    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        start_new_session=True,
        env=env,
    )


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mast_path(path: Path) -> str:
    return str(MAST_STAGE / path.relative_to(DEV_STAGE))


def points(
    steps: tuple[int, ...] = STEPS,
    *,
    base_only: bool = False,
    checkpoint_only: bool = False,
) -> list[dict[str, Any]]:
    if base_only and checkpoint_only:
        raise RuntimeError("base_only and checkpoint_only are mutually exclusive")
    configured = [] if checkpoint_only else [{"name": "base", "step": "base"}]
    if not base_only:
        configured.extend({"name": f"iter{step:02d}", "step": step} for step in steps)
    return configured


def frozen_config(batch_id: str, args: argparse.Namespace) -> dict[str, Any]:
    run_name = validate_safe_name(args.run_name, "run name")
    tenant_path = args.tenant_path or (
        "gen_ai/msl/tbd_research/rhea/msl_tbd_rhea_friends_data/"
        f"rhea_assistant/{args.tenant}"
    )
    comparison = None
    if args.compare_query_ids:
        reference_path = Path(args.compare_query_ids).resolve()
        if not reference_path.is_file():
            raise RuntimeError(f"comparison query-id list does not exist: {reference_path}")
        comparison = {
            "name": args.comparison_name,
            "query_ids_path": str(reference_path),
            "query_ids_sha256": sha256(reference_path),
            "query_id_count": len(set(reference_path.read_text().split())),
        }
    return {
        "version": 1,
        "batch_id": batch_id,
        "run_name": run_name,
        "points": points(
            tuple(args.steps),
            base_only=args.base_only,
            checkpoint_only=args.checkpoint_only,
        ),
        "dataset": {
            "name": "bcplus_train",
            "dev_path": "/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-data/BC+/bc_train.parquet",
            "mast_path": "/mnt/wsfuse/hhzhang01/supo-data/BC+/bc_train.parquet",
            "sha256": TRAIN_DATA_SHA256,
        },
        "evaluation": {
            "expected_questions": EXPECTED_QUESTIONS,
            "samples_per_question": SAMPLES_PER_QUESTION,
            "rollout_seed": ROLLOUT_SEED,
            "sampling_seeds": list(range(ROLLOUT_SEED, ROLLOUT_SEED + SAMPLES_PER_QUESTION)),
            "deterministic": True,
            "sglang_engines": 8,
            "sglang_tensor_parallel_size": 1,
            "sglang_server_concurrency_per_engine": SGLANG_SERVER_CONCURRENCY,
            "search_concurrency": SEARCH_CONCURRENCY,
            "judge_concurrency": JUDGE_CONCURRENCY,
            "fixed_search_topk": args.fixed_search_topk,
            "doc_words_full": args.doc_words_full,
        },
        "mast": {
            "tenant": args.tenant,
            "tenant_path": tenant_path,
            "region": REGION,
            "host": HOST,
            "priority": args.priority,
            "nodes_per_point": 1,
            "gpus_per_node": 8,
            "hardware": "T20_GRAND_TETON_HBM3_ROCE (8x H100 80GB)",
            "search_server_tenant": args.search_server_tenant,
            "search_addr_file": args.search_addr_file,
        },
        "comparison": comparison,
    }


def validate_inputs(config: dict[str, Any]) -> None:
    run_name = validate_safe_name(config["run_name"], "run name")
    root = CHECKPOINT_STAGE / run_name
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise RuntimeError(f"missing checkpoint tracker: {tracker}")
    tracked_step = int(tracker.read_text().strip())
    for point in config["points"]:
        if point["step"] == "base":
            metadata = Path(
                "/data/users/hhzhang01/wsfuse_mnt/hhzhang01/supo-data/Qwen3.5-4B_torch_dist/release/.metadata"
            )
        else:
            metadata = root / f"iter_{int(point['step']):07d}" / ".metadata"
            if int(point["step"]) > tracked_step:
                raise RuntimeError(f"requested step {point['step']} is newer than tracker {tracked_step}")
        if not metadata.is_file():
            raise RuntimeError(f"missing checkpoint metadata: {metadata}")
    dataset = Path(config["dataset"]["dev_path"])
    if not dataset.is_file():
        raise RuntimeError(f"missing training dataset: {dataset}")
    actual_dataset_sha256 = sha256(dataset)
    if actual_dataset_sha256 != config["dataset"]["sha256"]:
        raise RuntimeError(
            f"training dataset changed: expected {config['dataset']['sha256']}, "
            f"found {actual_dataset_sha256}"
        )
    fixed_search_topk = config["evaluation"].get("fixed_search_topk")
    if fixed_search_topk is not None and int(fixed_search_topk) < 1:
        raise RuntimeError("fixed_search_topk must be positive or null")
    if int(config["evaluation"].get("doc_words_full", DOC_WORDS_FULL)) < 1:
        raise RuntimeError("doc_words_full must be positive")
    comparison = config.get("comparison")
    if comparison:
        reference_path = Path(comparison["query_ids_path"])
        if not reference_path.is_file():
            raise RuntimeError(f"comparison query-id list is missing: {reference_path}")
        if sha256(reference_path) != comparison["query_ids_sha256"]:
            raise RuntimeError(f"comparison query-id list changed: {reference_path}")


def stage_code(repo_root: Path, batch_root: Path) -> tuple[Path, str]:
    archive_dir = DEV_STAGE / "train-data-filter-code"
    archive_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_dir:
        candidate = Path(temporary_dir) / "train-data-filter-code.tgz"
        result = run(
            [
                "tar",
                "-czf",
                str(candidate),
                "--exclude=.git",
                "--exclude=.pytest_cache",
                "--exclude=__pycache__",
                "--exclude=*.pyc",
                "-C",
                str(repo_root),
                ".",
            ]
        )
        if result.returncode:
            raise RuntimeError(f"failed to build code archive: {result.stderr}")
        digest = sha256(candidate)
        destination = archive_dir / f"train-data-filter-code-{digest[:16]}.tgz"
        if not destination.exists():
            temporary = destination.with_suffix(".tmp")
            shutil.copyfile(candidate, temporary)
            temporary.replace(destination)
    write_json(batch_root / "code.json", {"archive": str(destination), "sha256": digest})
    return destination, digest


def build_mast_command(
    config: dict[str, Any],
    batch_root: Path,
    archive: Path,
    archive_sha256: str,
    point: dict[str, Any],
    *,
    dry_run: bool,
) -> list[str]:
    destination = batch_root / "points" / point["name"]
    batch_tag = re.sub(r"[^A-Za-z0-9]+", "-", config["batch_id"]).strip("-")[-18:]
    job_name = f"bcplus-train-filter-{batch_tag}-{point['name']}"[:120]
    evaluation = config["evaluation"]
    environment_values = [
        ("FILTER_RUN_NAME", config["run_name"]),
        ("FILTER_POINT", point["name"]),
        ("FILTER_REQUESTED_STEP", point["step"]),
        ("FILTER_OUTPUT_DIR", mast_path(destination)),
        ("FILTER_CODE_ARCHIVE_SHA256", archive_sha256),
        ("FILTER_N", evaluation["samples_per_question"]),
        ("FILTER_SEED", evaluation["rollout_seed"]),
        ("FILTER_EXPECTED_QUESTIONS", evaluation["expected_questions"]),
        (
            "SEARCH_ADDR_FILE",
            config["mast"].get("search_addr_file", f"{MAST_STAGE}/search-server.addr"),
        ),
        ("BCPLUS_DOC_WORDS_FULL", evaluation.get("doc_words_full", DOC_WORDS_FULL)),
        ("BCPLUS_SGLANG_SERVER_CONCURRENCY", evaluation["sglang_server_concurrency_per_engine"]),
        ("BCPLUS_SEARCH_CONCURRENCY", evaluation["search_concurrency"]),
        ("BCPLUS_JUDGE_CONCURRENCY", evaluation["judge_concurrency"]),
        ("TORCH_NCCL_DUMP_ON_TIMEOUT", 0),
    ]
    if evaluation.get("fixed_search_topk") is not None:
        environment_values.append(("BCPLUS_FIXED_SEARCH_TOPK", evaluation["fixed_search_topk"]))
    custom_command = " && ".join(
        [
            "mkdir -p /slime-src",
            f"tar xzf {shlex.quote(mast_path(archive))} -C /slime-src",
            "cd /slime-src",
            "export "
            + " ".join(
                f"{name}={shlex.quote(str(value))}"
                for name, value in environment_values
            ),
            "bash /slime-src/examples/supo_browsecomp/mast/train_data_filter/run_filter_eval.sh",
        ]
    )
    command = [
        str(CLI),
        "mast",
        "--json",
        f"--tenant={config['mast']['tenant']}",
        f"--region={config['mast']['region']}",
        f"--job_priority={config['mast'].get('priority', 'HIGH')}",
        "--workspace=None",
        "--main_package=xlformers_pretrain1:latest",
        "program",
        "avocado.rev1.rl.debug_80m",
        "--roles=trainer_0",
        f"--job_name={job_name}",
        "--enable_ttls=True",
        "--retries=3",
        "--use_conda_docker=True",
        f"--conda_docker_image={IMAGE}",
        "--docker_host_cmd=sh -c 'nohup python3 /mnt/wsfuse/hhzhang01/slime-sanity/connect_proxy.py 9080 >/tmp/relay.log 2>&1 &'",
        f"--docker_custom_cmd={custom_command}",
        f"--host={config['mast']['host']}",
        f"--wsf_src={WSF_SRC}",
        "--overrides=cluster_config.trainer_parallelism.data_parallel_size=1,cluster_config.trainer_parallelism.context_parallel_size=1",
    ]
    if dry_run:
        command.append("--dryrun")
    return command


def parse_response(stdout: str) -> dict[str, Any]:
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"MAST did not return JSON: {stdout[-2000:]}") from error
    if response.get("status") != "ok":
        raise RuntimeError(f"MAST request failed: {response}")
    return response


def validate_dry_run(
    response: dict[str, Any],
    expected_sglang_concurrency: int,
    expected_fixed_search_topk: int | None,
    expected_doc_words_full: int,
    expected_search_addr_file: str,
    expected_tenant: str,
    expected_priority: str,
) -> None:
    if response.get("dryrun") is not True:
        raise RuntimeError("MAST response is not a dry-run")
    task_groups = response["spec"]["hpc_job_definition"]["hpcTaskGroups"]
    role = next(group for group in task_groups if group["name"] == "trainer_0")
    if role["taskCount"] != 1:
        raise RuntimeError(f"dry-run requested {role['taskCount']} hosts, expected 1")
    if role["spec"]["resourceLimit"]["compute"]["gpu"] != 8:
        raise RuntimeError("dry-run did not request 8 GPUs")
    subtype = role["spec"]["machineConstraints"]["types"]["serverSubTypes"]
    if subtype != [200007]:
        raise RuntimeError(f"dry-run did not select Grand Teton H100: {subtype}")
    app_role = next(role for role in response["spec"]["app_def"]["roles"] if role["name"] == "trainer_0")
    if app_role["env"].get("ROLE_ASSIGNMENT_MAP") != "trainer_0=1":
        raise RuntimeError(f"unexpected role assignment: {app_role['env'].get('ROLE_ASSIGNMENT_MAP')}")
    entrypoint = app_role["entrypoint"]
    actual_tenant = response["spec"]["args"]["cfg"]["rmAttribution"]
    if actual_tenant != expected_tenant:
        raise RuntimeError(f"dry-run uses unexpected tenant: {actual_tenant}")
    actual_priority = response["spec"]["args"]["cfg"]["jobPriority"]
    if actual_priority != expected_priority:
        raise RuntimeError(f"dry-run uses unexpected priority: {actual_priority}")
    for expected in (
        "run_filter_eval.sh",
        "FILTER_N=8",
        f"BCPLUS_DOC_WORDS_FULL={expected_doc_words_full}",
        f"SEARCH_ADDR_FILE={expected_search_addr_file}",
        f"BCPLUS_SGLANG_SERVER_CONCURRENCY={expected_sglang_concurrency}",
    ):
        if expected not in entrypoint:
            raise RuntimeError(f"dry-run entrypoint is missing {expected}")
    if expected_fixed_search_topk is not None:
        expected = f"BCPLUS_FIXED_SEARCH_TOPK={expected_fixed_search_topk}"
        if expected not in entrypoint:
            raise RuntimeError(f"dry-run entrypoint is missing {expected}")


def search_stats(config: dict[str, Any]) -> dict[str, Any]:
    configured_path = Path(
        config["mast"].get("search_addr_file", f"{MAST_STAGE}/search-server.addr")
    )
    try:
        address_path = DEV_STAGE / configured_path.relative_to(MAST_STAGE)
    except ValueError:
        address_path = configured_path
    address = address_path.read_text().strip()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{address}/stats", timeout=5) as response:
        return json.load(response)


def mast_status(job_name: str) -> str:
    result = run(["with-proxy", "mast", "--output", "json", "get-status", job_name])
    if result.returncode:
        return "STATUS_ERROR"
    try:
        return str(json.loads(result.stdout)["data"]["state"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return "STATUS_ERROR"


def prepare(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any], dict[str, Any], Path, str]:
    batch_id = validate_safe_name(args.batch_id, "batch id")
    repo_root = Path(args.repo_root).resolve()
    batch_root = DEV_STAGE / "train-data-filter" / batch_id
    batch_root.mkdir(parents=True, exist_ok=True)
    config_path = batch_root / "config.json"
    if config_path.is_file():
        config = read_json(config_path)
        if config.get("batch_id") != batch_id:
            raise RuntimeError(f"frozen config has the wrong batch id: {config_path}")
        configured_points = config.get("points")
        if not isinstance(configured_points, list) or not configured_points:
            raise RuntimeError(f"frozen config targets unexpected model points: {config_path}")
        for point in configured_points:
            step = point.get("step")
            expected_name = "base" if step == "base" else f"iter{int(step):02d}"
            if point.get("name") != expected_name:
                raise RuntimeError(f"frozen config has invalid point: {point}")
        if config.get("dataset", {}).get("sha256") != TRAIN_DATA_SHA256:
            raise RuntimeError(f"frozen config targets an unexpected dataset: {config_path}")
    else:
        config = frozen_config(batch_id, args)
        validate_inputs(config)
        write_json(config_path, config)
    validate_inputs(config)
    state_path = batch_root / "state.json"
    state = read_json(state_path, {"batch_id": batch_id, "jobs": {}, "created_at": time.time()})
    code = read_json(batch_root / "code.json")
    if code:
        archive, digest = Path(code["archive"]), str(code["sha256"])
        if not archive.is_file() or sha256(archive) != digest:
            raise RuntimeError(f"frozen code archive is missing or changed: {archive}")
    else:
        archive, digest = stage_code(repo_root, batch_root)
    return repo_root, batch_root, config, state, archive, digest


def dry_run_command(args: argparse.Namespace) -> None:
    _, batch_root, config, _, archive, digest = prepare(args)
    command = build_mast_command(config, batch_root, archive, digest, config["points"][0], dry_run=True)
    result = run(command)
    if result.returncode:
        raise RuntimeError(f"MAST dry-run failed rc={result.returncode}: {result.stderr[-3000:]}")
    response = parse_response(result.stdout)
    validate_dry_run(
        response,
        config["evaluation"]["sglang_server_concurrency_per_engine"],
        config["evaluation"].get("fixed_search_topk"),
        config["evaluation"].get("doc_words_full", DOC_WORDS_FULL),
        config["mast"].get("search_addr_file", f"{MAST_STAGE}/search-server.addr"),
        config["mast"]["tenant"],
        config["mast"].get("priority", "HIGH"),
    )
    write_json(batch_root / "dry_run.json", response)
    print(
        json.dumps(
            {
                "status": "validated",
                "tenant": config["mast"]["tenant"],
                "host": config["mast"]["host"],
                "task_count": 1,
                "gpus": 8,
                "batch_root": str(batch_root),
            },
            indent=2,
        )
    )


def submit_command(args: argparse.Namespace) -> None:
    _, batch_root, config, state, archive, digest = prepare(args)
    state_path = batch_root / "state.json"
    stats = search_stats(config)
    if stats.get("status") not in (None, "healthy") or int(stats.get("pending", 0)) > 1500:
        raise RuntimeError(f"search server is not ready: {stats}")

    if not (batch_root / "dry_run.json").is_file():
        command = build_mast_command(config, batch_root, archive, digest, config["points"][0], dry_run=True)
        result = run(command)
        if result.returncode:
            raise RuntimeError(f"MAST dry-run failed rc={result.returncode}: {result.stderr[-3000:]}")
        response = parse_response(result.stdout)
        validate_dry_run(
            response,
            config["evaluation"]["sglang_server_concurrency_per_engine"],
            config["evaluation"].get("fixed_search_topk"),
            config["evaluation"].get("doc_words_full", DOC_WORDS_FULL),
            config["mast"].get("search_addr_file", f"{MAST_STAGE}/search-server.addr"),
            config["mast"]["tenant"],
            config["mast"].get("priority", "HIGH"),
        )
        write_json(batch_root / "dry_run.json", response)

    for point in config["points"]:
        point_name = point["name"]
        output = batch_root / "points" / point_name
        if (output / "_SUCCESS").is_file() or point_name in state["jobs"]:
            print(f"[skip] {point_name} is complete or already recorded", flush=True)
            continue
        result = run(build_mast_command(config, batch_root, archive, digest, point, dry_run=False))
        if result.returncode:
            raise RuntimeError(f"submission failed for {point_name}: {result.stderr[-3000:]}")
        response = parse_response(result.stdout)
        job = response.get("job") or {}
        if not job.get("job_name"):
            raise RuntimeError(f"submission response lacks job name: {response}")
        state["jobs"][point_name] = {
            "job_name": job["job_name"],
            "mast_url": job.get("mast_url"),
            "state": "SUBMITTED",
            "output": str(output),
            "submitted_at": time.time(),
        }
        write_json(state_path, state)
        print(f"[submit] {point_name} -> {job['job_name']}", flush=True)
    print(json.dumps({"batch_root": str(batch_root), "jobs": state["jobs"]}, indent=2))


def status_command(args: argparse.Namespace) -> None:
    _, batch_root, config, state, _, _ = prepare(args)
    for point in config["points"]:
        record = state["jobs"].get(point["name"])
        if record is None:
            continue
        if (Path(record["output"]) / "_SUCCESS").is_file():
            record["state"] = "COMPLETE"
        else:
            record["state"] = mast_status(record["job_name"])
        record["last_checked_at"] = time.time()
    write_json(batch_root / "state.json", state)
    print(
        json.dumps(
            {"batch_root": str(batch_root), "jobs": state["jobs"], "search": search_stats(config)},
            indent=2,
        )
    )


def finalize_command(args: argparse.Namespace) -> None:
    repo_root, batch_root, config, state, _, _ = prepare(args)
    missing = [
        point["name"]
        for point in config["points"]
        if not (batch_root / "points" / point["name"] / "_SUCCESS").is_file()
    ]
    if missing:
        states = {name: state["jobs"].get(name, {}).get("state", "NOT_SUBMITTED") for name in missing}
        raise RuntimeError(f"points are not complete: {states}")
    aggregate = repo_root / "examples/supo_browsecomp/mast/train_data_filter/aggregate.py"
    result = run(["python3", str(aggregate), "--batch-root", str(batch_root)])
    if result.returncode:
        raise RuntimeError(f"aggregation failed: {result.stderr}")
    print(result.stdout, end="")
    comparison = config.get("comparison")
    if comparison:
        current_ids = set((batch_root / "filter_candidate_query_ids.txt").read_text().split())
        reference_ids = set(Path(comparison["query_ids_path"]).read_text().split())
        intersection = current_ids & reference_ids
        union = current_ids | reference_ids
        report = {
            "reference_name": comparison["name"],
            "reference_count": len(reference_ids),
            "current_count": len(current_ids),
            "intersection_count": len(intersection),
            "reference_only_count": len(reference_ids - current_ids),
            "current_only_count": len(current_ids - reference_ids),
            "reference_retention": round(len(intersection) / len(reference_ids), 6)
            if reference_ids
            else None,
            "jaccard": round(len(intersection) / len(union), 6) if union else 1.0,
            "intersection_query_ids": sorted(intersection),
            "reference_only_query_ids": sorted(reference_ids - current_ids),
            "current_only_query_ids": sorted(current_ids - reference_ids),
        }
        write_json(batch_root / "comparison_with_reference.json", report)
        print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("dry-run", "submit", "status", "finalize"))
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).parents[4]))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--base-only", action="store_true")
    mode.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--steps", type=positive_int, nargs="+", default=list(STEPS))
    parser.add_argument("--fixed-search-topk", type=positive_int)
    parser.add_argument("--doc-words-full", type=positive_int, default=DOC_WORDS_FULL)
    parser.add_argument("--tenant", default=TENANT)
    parser.add_argument("--tenant-path")
    parser.add_argument("--priority", type=str.upper, default=PRIORITY)
    parser.add_argument(
        "--search-addr-file",
        default=f"{MAST_STAGE}/search-server.addr",
        help="MAST-container path to the search server address file",
    )
    parser.add_argument("--search-server-tenant", default="rhea_assistant_interns")
    parser.add_argument("--compare-query-ids")
    parser.add_argument("--comparison-name", default="reference")
    args = parser.parse_args()
    if args.command == "dry-run":
        dry_run_command(args)
    elif args.command == "submit":
        submit_command(args)
    elif args.command == "status":
        status_command(args)
    else:
        finalize_command(args)


if __name__ == "__main__":
    main()
