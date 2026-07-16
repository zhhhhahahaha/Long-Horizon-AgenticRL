#!/bin/bash
# Smoke test for the enroot multi-node rootfs staging fix in
# run_qwen3p5_9B_colocate.sh. Submits a 2-node srun with tiny walltime that
# only exercises the outer-part `cp -a` + `enroot start` per-node dance.
# Inside the container, prints "SMOKE OK from node ${SLURM_NODEID} @ ${HOSTNAME}"
# and exits. No ray, no training, no ckpt load. Just verifies the fix works.
#
# Expected output (2 lines, order not guaranteed):
#   SMOKE OK from node 0 @ a100-XXX-XXX
#   SMOKE OK from node 1 @ a100-YYY-YYY
#
# If we see [ERROR] Could not acquire rootfs lock → fix didn't work.
# If we see 2 SMOKE OK lines → fix worked; safe to submit the full 8-node run.
#
# Usage:
#   bash examples/supo_browsecomp/smoke_enroot_multinode.sh

set -euo pipefail

if [[ "${SLIME_INNER:-0}" != "1" ]]; then
    SLIME_HOST_DIR=/home/hhzhang01/slime
    ENROOT_ROOTFS="${ENROOT_ROOTFS:-slime-test}"
    SLURM_ACCOUNT="${SLURM_ACCOUNT:-genai_interns}"
    # shared has ample GRES headroom; 2-node backfill lands in seconds.
    # (interns_high is QOSGrpGRES-capped, dev per-user cap is 2 nodes but
    # our search server on dev already uses 1 node.)
    QOS="${QOS:-a100_genai_shared}"
    WALLTIME="${WALLTIME:-0:15:00}"
    NUM_NODES="${NUM_NODES:-2}"
    JOB_NAME="smoke-enroot-multinode-$(date +%H%M%S)"
    LOG_PATH=/genai/fsx-project/hhzhang01/logs/${JOB_NAME}.log
    mkdir -p "$(dirname "${LOG_PATH}")"
    echo "log path: ${LOG_PATH}"

    exec srun \
        --nodes=${NUM_NODES} --gpus-per-node=8 --ntasks-per-node=1 --exclusive \
        --cpus-per-task=16 --mem=0 \
        --account="${SLURM_ACCOUNT}" --qos="${QOS}" \
        --time="${WALLTIME}" \
        --mpi=none \
        --job-name="${JOB_NAME}" \
        --output="${LOG_PATH}" \
        bash -c "
            # Same per-node staging logic as run_qwen3p5_9B_colocate.sh
            LOCAL_ENROOT_DATA=/dev/shm/enroot-\${USER}-\${SLURM_JOB_ID}
            LOCAL_ROOTFS=\${LOCAL_ENROOT_DATA}/${ENROOT_ROOTFS}
            if [[ ! -d \${LOCAL_ROOTFS} ]]; then
                mkdir -p \${LOCAL_ENROOT_DATA}
                echo \"[node \${SLURM_NODEID:-0}] copying rootfs FSx -> \${LOCAL_ROOTFS} ...\"
                time cp -a /storage/home/hhzhang01/.local/share/enroot/${ENROOT_ROOTFS} \${LOCAL_ENROOT_DATA}/
                echo \"[node \${SLURM_NODEID:-0}] rootfs staged, size=\$(du -sh \${LOCAL_ROOTFS} 2>/dev/null | cut -f1)\"
            else
                echo \"[node \${SLURM_NODEID:-0}] rootfs already staged (reused)\"
            fi

            ENROOT_TEMP_PATH=/dev/shm \
            ENROOT_DATA_PATH=\${LOCAL_ENROOT_DATA} \
            ENROOT_MOUNT_HOME=false \
            enroot start \
                --env SLURM_NODEID=\${SLURM_NODEID:-0} \
                ${ENROOT_ROOTFS} \
                bash -c 'echo \"SMOKE OK from node \${SLURM_NODEID} @ \$(hostname) @ \$(date -Iseconds)\"'
        "
fi
