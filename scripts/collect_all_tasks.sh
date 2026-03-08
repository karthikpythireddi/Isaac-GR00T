#!/bin/bash
# Collect 200 preference pairs across 4 tasks (50 per task).
# Runs sequentially since only one GR00T server is available.
# PnPCounterToPan already has 20 pairs (seeds 0-39), so collect 30 more (seed_offset=40).
# All other tasks collect 50 pairs fresh (seed_offset=0).
#
# Usage (from Isaac-GR00T root, with GR00T server already running):
#   bash scripts/collect_all_tasks.sh 2>&1 | tee collect_all_tasks.log

set -e
PYTHON=/home/karthik/miniconda3/envs/gr1_sim/bin/python
SCRIPT=scripts/collect_preferences_groot.py
OUTDIR=preference_data/gr1
COMMON="--noise_injection --noise_scale 0.05"

echo "========================================"
echo "Starting multi-task preference collection"
echo "Target: 200 pairs across 4 tasks"
echo "========================================"

# Task 1: PnPCounterToPan — 30 more pairs (we already have 20, seeds 0-39)
echo ""
echo "[1/4] PnPCounterToPan — 30 pairs (seed_offset=40)"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PnPCounterToPan_GR1ArmsAndWaistFourierHands_Env" \
    --n_pairs 30 \
    --seed_offset 40 \
    --output_dir "${OUTDIR}/PnPCounterToPan_batch2" \
    $COMMON

# Task 2: PnPCounterToBowl — 50 pairs fresh
echo ""
echo "[2/4] PnPCounterToBowl — 50 pairs"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PnPCounterToBowl_GR1ArmsAndWaistFourierHands_Env" \
    --n_pairs 50 \
    --seed_offset 0 \
    --output_dir "${OUTDIR}/PnPCounterToBowl" \
    $COMMON

# Task 3: PnPCounterToPlate — 50 pairs fresh
echo ""
echo "[3/4] PnPCounterToPlate — 50 pairs"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PnPCounterToPlate_GR1ArmsAndWaistFourierHands_Env" \
    --n_pairs 50 \
    --seed_offset 0 \
    --output_dir "${OUTDIR}/PnPCounterToPlate" \
    $COMMON

# Task 4: PnPOnionToBowl — 50 pairs fresh
echo ""
echo "[4/4] PnPOnionToBowl — 50 pairs"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PnPOnionToBowl_GR1ArmsAndWaistFourierHands_Env" \
    --n_pairs 50 \
    --seed_offset 0 \
    --output_dir "${OUTDIR}/PnPOnionToBowl" \
    $COMMON

echo ""
echo "========================================"
echo "All tasks done! Now merging into one file..."
echo "========================================"

$PYTHON scripts/merge_preference_hdf5.py \
    ${OUTDIR}/PnPCounterToPan_GR1ArmsAndWaistFourierHands_Env_preferences.hdf5 \
    ${OUTDIR}/PnPCounterToPan_batch2/PnPCounterToPan_GR1ArmsAndWaistFourierHands_Env_preferences.hdf5 \
    ${OUTDIR}/PnPCounterToBowl/PnPCounterToBowl_GR1ArmsAndWaistFourierHands_Env_preferences.hdf5 \
    ${OUTDIR}/PnPCounterToPlate/PnPCounterToPlate_GR1ArmsAndWaistFourierHands_Env_preferences.hdf5 \
    ${OUTDIR}/PnPOnionToBowl/PnPOnionToBowl_GR1ArmsAndWaistFourierHands_Env_preferences.hdf5 \
    --output ${OUTDIR}/all_tasks_200pairs.hdf5

echo ""
echo "[done] Merged dataset: ${OUTDIR}/all_tasks_200pairs.hdf5"
