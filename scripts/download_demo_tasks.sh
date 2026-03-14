#!/bin/bash
# Download full 1000-episode demo data for all 4 GR1 tabletop tasks
# from nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim
#
# Run after: huggingface-cli login
# Usage: bash scripts/download_demo_tasks.sh

set -e
PYTHON=${PYTHON:-python}
BASE=examples/robocasa-gr1-tabletop-tasks/gr1_finetune_data
REPO=nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim

# Exact task names as they appear in the repo under LeRobot/
TASKS=(
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA"
    "gr1_unified.PnPBottleToCabinetClose"
    "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA"
    "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA"
)

download_task() {
    local TASK=$1
    local LOCAL_DIR="${BASE}/${TASK}"

    if [ -d "${LOCAL_DIR}/meta" ] && [ -d "${LOCAL_DIR}/data/chunk-000" ]; then
        echo "[skip] ${TASK} already exists"
        return
    fi

    echo "[download] ${TASK} ..."
    mkdir -p "${BASE}/_dl_tmp"

    $PYTHON -c "
import os, shutil
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='${REPO}',
    repo_type='dataset',
    allow_patterns=['LeRobot/${TASK}/*', 'LeRobot/${TASK}/**/*'],
    local_dir='${BASE}/_dl_tmp',
)

src = os.path.join('${BASE}/_dl_tmp', 'LeRobot', '${TASK}')
if os.path.isdir(src):
    shutil.move(src, '${LOCAL_DIR}')
    shutil.rmtree('${BASE}/_dl_tmp', ignore_errors=True)
    print('[done] ${TASK}')
else:
    import subprocess
    subprocess.run(['find', '${BASE}/_dl_tmp', '-maxdepth', '4', '-type', 'd'])
    raise SystemExit('[error] Task dir not found after download')
"
}

echo "========================================"
echo "Downloading full demo data (1000 eps each)"
echo "for all 4 GR1 tabletop evaluation tasks"
echo "========================================"

for TASK in "${TASKS[@]}"; do
    download_task "$TASK"
done

echo ""
echo "[done] All 4 tasks downloaded to: ${BASE}/"
ls -d ${BASE}/gr1_unified.* 2>/dev/null
