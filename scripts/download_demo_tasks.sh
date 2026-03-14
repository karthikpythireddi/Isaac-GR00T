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
    $PYTHON -c "
from huggingface_hub import snapshot_download
import shutil, os

REPO = '${REPO}'
TASK = '${TASK}'
LOCAL_DIR = '${LOCAL_DIR}'

# Download entire task directory at once — much faster than file-by-file
# and avoids HF rate limits
tmp = snapshot_download(
    repo_id=REPO,
    repo_type='dataset',
    allow_patterns=[f'LeRobot/{TASK}/**'],
    local_dir='${BASE}/_snapshot_tmp',
)

# Move from nested LeRobot/<TASK> to expected flat location
src = os.path.join('${BASE}/_snapshot_tmp', 'LeRobot', TASK)
if os.path.isdir(src):
    shutil.move(src, LOCAL_DIR)
    shutil.rmtree('${BASE}/_snapshot_tmp', ignore_errors=True)
    print(f'[done] {TASK} -> {LOCAL_DIR}')
else:
    print(f'[error] Expected path not found: {src}')
    raise SystemExit(1)
"
    echo "[done] ${TASK}"
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
