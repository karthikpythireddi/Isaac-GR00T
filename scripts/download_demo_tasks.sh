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
import os, time
from huggingface_hub import hf_hub_download, list_repo_tree

REPO = '${REPO}'
TASK = '${TASK}'
LOCAL_DIR = '${LOCAL_DIR}'
DELAY = 0.35   # seconds between requests — keeps us under 1000 req/5min limit

# List all files under LeRobot/TASK
print(f'Listing files for {TASK}...')
items = list(list_repo_tree(REPO, repo_type='dataset',
                             path_in_repo=f'LeRobot/{TASK}', recursive=True))
files = [getattr(i, 'path', None) for i in items if getattr(i, 'size', None) is not None]
print(f'Found {len(files)} files. Downloading sequentially (~{len(files)*DELAY/60:.0f} min)...')

for i, fpath in enumerate(files):
    dest = os.path.join(LOCAL_DIR, os.path.relpath(fpath, f'LeRobot/{TASK}'))
    if os.path.exists(dest):
        continue  # already downloaded — resume safely
    for attempt in range(5):
        try:
            hf_hub_download(
                repo_id=REPO, repo_type='dataset',
                filename=fpath, local_dir=LOCAL_DIR,
            )
            # Move from nested LeRobot/TASK/... to LOCAL_DIR/...
            src = os.path.join(LOCAL_DIR, 'LeRobot', TASK,
                               os.path.relpath(fpath, f'LeRobot/{TASK}'))
            if os.path.exists(src) and not os.path.exists(dest):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                os.rename(src, dest)
            break
        except Exception as e:
            if '429' in str(e) and attempt < 4:
                wait = 300  # wait 5 min on rate limit
                print(f'  [rate limit] waiting {wait}s before retry...')
                time.sleep(wait)
            else:
                print(f'  [warn] failed {fpath}: {e}')
                break
    time.sleep(DELAY)
    if (i + 1) % 100 == 0:
        print(f'  {i+1}/{len(files)} files downloaded')

# Clean up empty LeRobot/ dir if present
import shutil
lerobot_dir = os.path.join(LOCAL_DIR, 'LeRobot')
if os.path.isdir(lerobot_dir):
    shutil.rmtree(lerobot_dir, ignore_errors=True)
print(f'[done] {TASK}')
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
