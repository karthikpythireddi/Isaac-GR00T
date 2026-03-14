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

probe_repo_structure() {
    echo "[probe] Checking repo structure..."
    $PYTHON -c "
from huggingface_hub import list_repo_tree
# Top level
items = list(list_repo_tree('${REPO}', repo_type='dataset', recursive=False))
print('Top-level:')
for item in items:
    print(' ', getattr(item, 'path', item))
# Inside LeRobot/
print('LeRobot/ contents:')
items2 = list(list_repo_tree('${REPO}', repo_type='dataset', path_in_repo='LeRobot', recursive=False))
for item in items2[:30]:
    print(' ', getattr(item, 'path', item))
"
}

download_task() {
    local TASK=$1
    local LOCAL_DIR="${BASE}/${TASK}"

    if [ -d "${LOCAL_DIR}/meta" ] && [ -d "${LOCAL_DIR}/data/chunk-000" ]; then
        echo "[skip] ${TASK} already exists at ${LOCAL_DIR}"
        return
    fi

    echo "[download] ${TASK} (all 1000 episodes) ..."
    mkdir -p "${BASE}/_dl_tmp"

    $PYTHON -c "
import os, shutil
from huggingface_hub import snapshot_download, list_repo_tree

REPO = '${REPO}'
TASK = '${TASK}'
LOCAL_DIR = '${LOCAL_DIR}'
TMP_DIR = '${BASE}/_dl_tmp'

# Find the correct prefix by listing repo top-level
top = list(list_repo_tree(REPO, repo_type='dataset', recursive=False))
top_paths = [getattr(i, 'path', str(i)) for i in top]
print('Top-level:', top_paths[:10])

# Try different prefix patterns
candidates = [
    f'LeRobot/{TASK}',
    f'{TASK}',
    f'data/{TASK}',
]
prefix = None
for c in candidates:
    # Check if any top-level entry starts with this
    if any(p == c or p.startswith(c + '/') or p.startswith(c.split('/')[0]) for p in top_paths):
        prefix = c.split('/')[0]  # use top-level dir
        break

if prefix is None:
    prefix = top_paths[0] if top_paths else 'LeRobot'
    print(f'[warn] Could not detect prefix, using: {prefix}')
else:
    print(f'[info] Detected prefix: {prefix}')

# Download using detected prefix
snapshot_download(
    repo_id=REPO,
    repo_type='dataset',
    allow_patterns=[f'{prefix}/{TASK}/**', f'{prefix}/{TASK}/*'],
    local_dir=TMP_DIR,
)

# Find the downloaded task dir
for root, dirs, files in os.walk(TMP_DIR):
    if os.path.basename(root) == TASK:
        shutil.move(root, LOCAL_DIR)
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        print(f'[done] {TASK} -> {LOCAL_DIR}')
        raise SystemExit(0)

print(f'[error] Task dir not found after download. Contents:')
os.system(f'find {TMP_DIR} -maxdepth 3 -type d')
raise SystemExit(1)
"
}

echo "========================================"
echo "Downloading full demo data (1000 eps each)"
echo "for all 4 GR1 tabletop evaluation tasks"
echo "Estimated: ~32GB total, ~30-60 min"
echo "========================================"

probe_repo_structure
download_task "gr1_arms_waist.CuttingboardToBasket"
download_task "gr1_unified.PnPBottleToCabinetClose"
download_task "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA"
download_task "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA"

echo ""
echo "[done] All 4 tasks downloaded!"
echo "Demo data at: ${BASE}/"
ls -d ${BASE}/gr1_* 2>/dev/null
