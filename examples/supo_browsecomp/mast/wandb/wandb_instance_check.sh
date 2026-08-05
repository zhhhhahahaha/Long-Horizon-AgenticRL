#!/bin/bash
# Is a W&B instance actually storing run history, or silently queueing it?
#
#   wandb_instance_check.sh <base-url> <key-file> <entity>
#
# Writes a 10-row run and polls for ~2 minutes, then reports HEALTHY or
# DROPPING WRITES. A degraded instance answers the filestream POST with 200 OK
# and still shows nothing, so "the client said it uploaded" proves nothing --
# only reading the rows back does. Run this before pointing a long training job
# at an instance in online mode. See README.md "Choosing an instance".
set -euo pipefail

BASE="${1:?usage: $0 <base-url> <key-file> <entity>}"
KEYF="${2:?usage: $0 <base-url> <key-file> <entity>}"
ENTITY="${3:?usage: $0 <base-url> <key-file> <entity>}"
PY="${WANDB_PYTHON_BIN:-/data/users/hhzhang01/slime-sanity/wandb-0.27.2-venv/bin/python}"
WITH_PROXY_BIN="${WITH_PROXY_BIN:-with-proxy}"

[[ -s "${KEYF}" ]] || { echo "$0: key file is missing or empty: ${KEYF}" >&2; exit 1; }
[[ -x "${PY}" ]] || { echo "$0: python is not executable: ${PY}" >&2; exit 1; }

export WANDB_BASE_URL="${BASE%/}"
WANDB_API_KEY="$(tr -d ' \t\r\n' < "${KEYF}")"
export WANDB_API_KEY

"${WITH_PROXY_BIN}" "${PY}" - "${ENTITY}" <<'PYEOF'
import base64, json, os, sys, time, uuid
import requests, wandb

BASE = os.environ["WANDB_BASE_URL"]
KEY = os.environ["WANDB_API_KEY"]
ENTITY = sys.argv[1]
PROJECT = os.environ.get("WANDB_INSTANCE_CHECK_PROJECT", "wandb-instance-check")
RID = os.environ.get("WANDB_INSTANCE_CHECK_RUN_ID")
if not RID:
    RID = f"instancecheck-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
ROWS = 10
POLL_DELAYS = tuple(
    int(delay) for delay in os.environ.get("WANDB_INSTANCE_CHECK_POLL_DELAYS", "5,25,30,60").split(",")
)
H = {
    "Content-Type": "application/json",
    "Authorization": "Basic " + base64.b64encode(f":{KEY}".encode()).decode(),
}

print(f"instance : {BASE}")
print(f"client   : wandb {wandb.__version__}")
print(f"run id   : {RID}")

who = requests.post(
    f"{BASE}/graphql", json={"query": "query{viewer{username}}"}, headers=H, timeout=60
).json()
if "errors" in who or who.get("error"):
    sys.exit(f"!! key rejected by {BASE} -- keys are per-instance, get one from {BASE}/authorize")
print("viewer   :", who["data"]["viewer"]["username"])

run = wandb.init(
    entity=ENTITY,
    project=PROJECT,
    id=RID,
    name="instance-check",
    settings=wandb.Settings(console="off", disable_code=True),
)
for step in range(ROWS):
    run.log({"loss": 1.0 / (step + 1), "acc": step / 10.0}, step=step)
run.finish()

PH = (
    'query PH($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){run(name:$r){'
    'historyLineCount parquetHistory(liveKeys:["_step"]){liveData parquetUrls}}}}'
)
healthy = False
for wait in POLL_DELAYS:
    time.sleep(wait)
    response = requests.post(
        f"{BASE}/graphql",
        json={"query": PH, "variables": {"e": ENTITY, "p": PROJECT, "r": RID}},
        headers=H,
        timeout=120,
    ).json()
    node = (((response.get("data") or {}).get("project") or {}).get("run"))
    if node is None:
        print("  run is not visible yet")
        continue
    parquet = len(node["parquetHistory"]["parquetUrls"])
    live = len(node["parquetHistory"]["liveData"])
    print(f"  lines={node['historyLineCount']}  parquet={parquet}  live={live}")
    if parquet or live:
        healthy = True
        break

readable = 0
if healthy:
    try:
        readable = len(list(wandb.Api(timeout=60).run(f"{ENTITY}/{PROJECT}/{RID}").scan_history()))
    except Exception as exc:
        print(f"history readback failed: {exc}")
print(f"\nscan_history rows: {readable} / {ROWS}")
print(f"VERDICT: {BASE} is {'HEALTHY' if healthy and readable == ROWS else 'DROPPING WRITES'}")
sys.exit(0 if healthy and readable == ROWS else 1)
PYEOF
