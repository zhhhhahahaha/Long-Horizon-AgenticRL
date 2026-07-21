#!/bin/bash
# BrowseComp-Plus retrieval server — docker-on-MAST (Path B) version.
#
# Runs as its OWN 1-node MAST job (8×H100). The trainer job dials it over the
# RoCE/backend net via HTTP on the node's routable IPv6 (cross-job HTTP was
# verified 2026-07-20). This script runs INSIDE the slime container; the launcher
# extracts slime-code.tgz to /slime-src first and invokes this file.
#
# Discovery: writes its "[<ipv6>]:<port>" to an OILFS file the trainer polls.
# No egress needed (server is inbound-only); all model+corpus come from OILFS.
#
# Launched via (see submit_search.sh / the memory runbook):
#   --docker_custom_cmd='mkdir -p /slime-src && tar xzf \
#       /mnt/wsfuse/hhzhang01/supo-slime/slime-code.tgz -C /slime-src && \
#       bash /slime-src/examples/supo_browsecomp/mast/run_search_server.sh'
set -ex
export PYTHONUNBUFFERED=1
# Local model + local HF-format corpus/embeds → force offline so no code path
# tries to reach the hub (MAST has no egress here anyway).
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# The container's baked http_proxy (fwdproxy:8080) is dead on MAST and the
# --enable_ttls proxy env would hijack even localhost — the server needs no
# egress, so clear all proxy and mark everything direct.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="*" NO_PROXY="*"
# MAST env sets TRITON_CACHE_MANAGER=msl_tools.* (unimportable under our
# PYTHONPATH). transformers/flash-attn may touch triton → use the local cache.
unset TRITON_CACHE_MANAGER
export TRITON_CACHE_DIR=/tmp/triton_cache_slime
# The bind-mounted host numactl is broken in-container; nothing here uses it,
# but disable the sglang numa wrapper defensively.
export SGLANG_NUMA_BIND_V2=0

D=/mnt/wsfuse/hhzhang01/supo-data
SLIME=/slime-src
PORT="${SEARCH_PORT:-8000}"
ADDR_FILE="${SEARCH_ADDR_FILE:-/mnt/wsfuse/hhzhang01/supo-slime/search-server.addr}"

# Routable fb IPv6 (first non-loopback, non-link-local). Capturing this INSIDE
# a staged script (not the inline docker_custom_cmd) avoids the quoting mangle
# that wrote an empty ":8000" during the cross-job test.
MYIP=$(hostname -i | tr ' ' '\n' | grep ':' | grep -vE '^(::1|fe80)' | head -1)
if [[ -z "${MYIP}" ]]; then
  echo "ERROR: could not determine routable IPv6 from 'hostname -i'" >&2
  hostname -i >&2
  exit 1
fi
mkdir -p "$(dirname "${ADDR_FILE}")"
echo "[${MYIP}]:${PORT}" > "${ADDR_FILE}"
echo "SEARCH_ADDR=[${MYIP}]:${PORT} -> ${ADDR_FILE}"

cd "${SLIME}/examples/supo_browsecomp"
export PYTHONPATH=/root/Megatron-LM:${SLIME}
# Belt-and-suspenders: spawn workers re-import the module and reset MODEL_NAME to
# its HF-repo-id default; this env var lets them recover the local path offline.
export SEARCH_SERVER_MODEL="${D}/Qwen3-Embedding-8B"

# Bind IPv6 (::) so cross-node trainer can reach [<ipv6>]:PORT. Auto-uses all 8
# GPUs (search_server.py: NUM_GPUS defaults to torch.cuda.device_count()) → 8
# embedding workers for rollout-time search throughput.
exec python -u search_server.py \
  --model "${D}/Qwen3-Embedding-8B" \
  --corpus "${D}/browsecomp-plus-corpus" \
  --corpus-embedding-dataset "${D}/browsecomp-plus-embeds" \
  --host :: --port "${PORT}"
