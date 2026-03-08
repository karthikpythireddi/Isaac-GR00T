#!/bin/bash
# Download demo data for 3 remaining tasks (first 50 episodes only to avoid rate limits)
# Run after: huggingface-cli login
#
# Usage: conda activate groot && bash scripts/download_demo_tasks.sh

set -e
PYTHON=${PYTHON:-/home/karthik/miniconda3/envs/groot/bin/python}
BASE=examples/robocasa-gr1-tabletop-tasks/gr1_finetune_data
REPO=nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim
N_EPISODES=50  # Only download first 50 episodes (not all 1000)

download_task() {
    local TASK=$1
    local LOCAL_DIR="${BASE}/${TASK}"

    if [ -d "${LOCAL_DIR}/meta" ] && [ -d "${LOCAL_DIR}/data/chunk-000" ]; then
        echo "[skip] ${TASK} already exists"
        return
    fi

    echo "[download] ${TASK} (first ${N_EPISODES} episodes) ..."
    local TEMP_DIR="${BASE}/_dl_${TASK}"
    $PYTHON -c "
import time
from huggingface_hub import hf_hub_download, list_repo_tree
import os

REPO = '${REPO}'
TASK = '${TASK}'
TEMP_DIR = '${TEMP_DIR}'
N = ${N_EPISODES}

# Build list of specific files to download (meta + first N episodes)
files_to_download = []

# Meta files (small, always needed)
meta_items = list(list_repo_tree(REPO, repo_type='dataset', path_in_repo=f'LeRobot/{TASK}/meta'))
for item in meta_items:
    if hasattr(item, 'size'):  # is a file, not a dir
        files_to_download.append(item.path)

# First N parquet files
for i in range(N):
    files_to_download.append(f'LeRobot/{TASK}/data/chunk-000/episode_{i:06d}.parquet')

# First N video files
for i in range(N):
    files_to_download.append(f'LeRobot/{TASK}/videos/chunk-000/observation.images.ego_view/episode_{i:06d}.mp4')

print(f'Downloading {len(files_to_download)} files...')

# Download files one batch at a time with small delays
for i, fpath in enumerate(files_to_download):
    try:
        hf_hub_download(
            repo_id=REPO,
            repo_type='dataset',
            filename=fpath,
            local_dir=TEMP_DIR,
        )
    except Exception as e:
        if '429' in str(e):
            print(f'  Rate limited at file {i}/{len(files_to_download)}, waiting 60s...')
            time.sleep(60)
            hf_hub_download(
                repo_id=REPO,
                repo_type='dataset',
                filename=fpath,
                local_dir=TEMP_DIR,
            )
        else:
            print(f'  [warn] Failed: {fpath}: {e}')

    if (i + 1) % 50 == 0:
        print(f'  Downloaded {i+1}/{len(files_to_download)} files')
        time.sleep(5)  # Brief pause every 50 files

print('Download complete')
"
    # Move from nested LeRobot/ path to expected location
    if [ -d "${TEMP_DIR}/LeRobot/${TASK}" ]; then
        mv "${TEMP_DIR}/LeRobot/${TASK}" "${LOCAL_DIR}"
        rm -rf "${TEMP_DIR}"
        echo "[done] ${TASK} -> ${LOCAL_DIR}"
    elif [ -d "${TEMP_DIR}/meta" ]; then
        mv "${TEMP_DIR}" "${LOCAL_DIR}"
        echo "[done] ${TASK} -> ${LOCAL_DIR}"
    else
        echo "[error] Unexpected download structure in ${TEMP_DIR}"
        ls -R "${TEMP_DIR}" | head -20
    fi
    echo "  Waiting 60s before next task to avoid rate limits..."
    sleep 60
}

echo "========================================"
echo "Downloading demo data for 3 tasks"
echo "(first ${N_EPISODES} episodes each)"
echo "========================================"

download_task "gr1_unified.PnPBottleToCabinetClose"
download_task "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA"
download_task "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA"

echo ""
echo "[done] All downloads complete!"
echo "Demo data at: ${BASE}/"
ls -d ${BASE}/gr1_unified.* 2>/dev/null
