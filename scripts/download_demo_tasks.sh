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

    $PYTHON - <<PYEOF
import os, time, shutil
from huggingface_hub import hf_hub_download

REPO = '${REPO}'
TASK = '${TASK}'
LOCAL_DIR = '${LOCAL_DIR}'
N = 1000
DELAY = 0.35  # ~170 files/min, safely under 1000 req/5min

# Build file list from known structure — no list_repo_tree API calls needed
META_FILES = [
    'meta/info.json', 'meta/tasks.jsonl', 'meta/episodes.jsonl',
    'meta/stats.json', 'meta/modality.json', 'meta/relative_stats.json',
]
DATA_FILES = [f'data/chunk-000/episode_{i:06d}.parquet' for i in range(N)]
VIDEO_FILES = [f'videos/chunk-000/observation.images.ego_view/episode_{i:06d}.mp4' for i in range(N)]
ALL_FILES = META_FILES + DATA_FILES + VIDEO_FILES

print(f'Downloading {len(ALL_FILES)} files for {TASK}...')
skipped = 0
for idx, rel_path in enumerate(ALL_FILES):
    dest = os.path.join(LOCAL_DIR, rel_path)
    if os.path.exists(dest):
        skipped += 1
        continue
    hf_path = f'LeRobot/{TASK}/{rel_path}'
    for attempt in range(5):
        try:
            result = hf_hub_download(
                repo_id=REPO, repo_type='dataset',
                filename=hf_path, local_dir=LOCAL_DIR,
            )
            # hf_hub_download saves to LOCAL_DIR/LeRobot/TASK/rel_path
            # move to LOCAL_DIR/rel_path
            src = os.path.join(LOCAL_DIR, 'LeRobot', TASK, rel_path)
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(src, dest)
            break
        except Exception as e:
            if '429' in str(e) and attempt < 4:
                print(f'  [rate limit] waiting 300s...')
                time.sleep(300)
            else:
                print(f'  [warn] {hf_path}: {e}')
                break
    time.sleep(DELAY)
    done = idx + 1 - skipped
    if done % 200 == 0:
        print(f'  {idx+1}/{len(ALL_FILES)} ({skipped} skipped)')

# Clean up empty nested dirs
lerobot_dir = os.path.join(LOCAL_DIR, 'LeRobot')
if os.path.isdir(lerobot_dir):
    shutil.rmtree(lerobot_dir, ignore_errors=True)
print(f'[done] {TASK} ({skipped} already existed)')
PYEOF
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
