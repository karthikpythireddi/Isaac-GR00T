#!/bin/bash
# Download full 1000-episode demo data for all 4 GR1 tabletop tasks
# from nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim
#
# Uses snapshot_download for efficiency — downloads entire task directory
# in one call instead of file-by-file, which avoids rate limit issues.
#
# Storage estimate: ~8GB per task × 4 tasks = ~32GB total
# Runtime: ~30-60 min depending on network
#
# Run after: huggingface-cli login
# Usage: bash scripts/download_demo_tasks.sh

set -e
PYTHON=${PYTHON:-python}
BASE=examples/robocasa-gr1-tabletop-tasks/gr1_finetune_data
REPO=nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim

download_task() {
    local TASK=$1
    local LOCAL_DIR="${BASE}/${TASK}"

    if [ -d "${LOCAL_DIR}/meta" ] && [ -d "${LOCAL_DIR}/data/chunk-000" ]; then
        echo "[skip] ${TASK} already exists at ${LOCAL_DIR}"
        return
    fi

    echo "[download] ${TASK} (all 1000 episodes) ..."
    mkdir -p "${BASE}/_dl_tmp"

    # Use huggingface-cli which handles large dataset downloads reliably
    huggingface-cli download "$REPO" \
        --repo-type dataset \
        --include "LeRobot/${TASK}/**" \
        --local-dir "${BASE}/_dl_tmp"

    # Move from nested LeRobot/<TASK> into expected flat location
    if [ -d "${BASE}/_dl_tmp/LeRobot/${TASK}" ]; then
        mv "${BASE}/_dl_tmp/LeRobot/${TASK}" "${LOCAL_DIR}"
        rm -rf "${BASE}/_dl_tmp"
        echo "[done] ${TASK} -> ${LOCAL_DIR}"
    else
        echo "[error] Expected path not found: ${BASE}/_dl_tmp/LeRobot/${TASK}"
        ls "${BASE}/_dl_tmp/" 2>/dev/null || true
        exit 1
    fi
}

echo "========================================"
echo "Downloading full demo data (1000 eps each)"
echo "for all 4 GR1 tabletop evaluation tasks"
echo "Estimated: ~32GB total, ~30-60 min"
echo "========================================"

download_task "gr1_arms_waist.CuttingboardToBasket"
download_task "gr1_unified.PnPBottleToCabinetClose"
download_task "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA"
download_task "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA"

echo ""
echo "[done] All 4 tasks downloaded!"
echo "Demo data at: ${BASE}/"
ls -d ${BASE}/gr1_* 2>/dev/null
