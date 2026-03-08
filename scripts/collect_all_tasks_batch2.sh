#!/bin/bash
# Collect 200 more preference pairs across 4 tasks (50 per task, seed_offset=100).
# Adds to existing 200 pairs; final merge produces 400-pair dataset.
#
# Usage:
#   bash scripts/collect_all_tasks_batch2.sh 2>&1 | tee collect_all_tasks_batch2.log

set -e
PYTHON=/home/karthik/miniconda3/envs/gr1_sim/bin/python
SCRIPT=scripts/collect_preferences_groot.py
OUTDIR=preference_data/gr1
COMMON="--noise_injection --noise_scale 0.05 --seed_offset 100"

echo "========================================"
echo "Batch 2: 200 more pairs across 4 tasks"
echo "seed_offset=100 for all tasks"
echo "========================================"

echo ""
echo "[1/4] PnPCounterToPan — 50 pairs (seed_offset=100)"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PnPCounterToPan_GR1ArmsAndWaistFourierHands_Env" \
    --n_pairs 50 --output_dir "${OUTDIR}/PnPCounterToPan_batch3" $COMMON

echo ""
echo "[2/4] PnPCounterToBowl — 50 pairs (seed_offset=100)"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PnPCounterToBowl_GR1ArmsAndWaistFourierHands_Env" \
    --n_pairs 50 --output_dir "${OUTDIR}/PnPCounterToBowl_batch2" $COMMON

echo ""
echo "[3/4] PnPCounterToPlate — 50 pairs (seed_offset=100)"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PnPCounterToPlate_GR1ArmsAndWaistFourierHands_Env" \
    --n_pairs 50 --output_dir "${OUTDIR}/PnPCounterToPlate_batch2" $COMMON

echo ""
echo "[4/4] PnPOnionToBowl — 50 pairs (seed_offset=100)"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PnPOnionToBowl_GR1ArmsAndWaistFourierHands_Env" \
    --n_pairs 50 --output_dir "${OUTDIR}/PnPOnionToBowl_batch2" $COMMON

echo ""
echo "========================================"
echo "Merging all 400 pairs into one file..."
echo "========================================"

$PYTHON scripts/merge_preference_hdf5.py \
    ${OUTDIR}/all_tasks_200pairs.hdf5 \
    ${OUTDIR}/PnPCounterToPan_batch3/PnPCounterToPan_GR1ArmsAndWaistFourierHands_Env_preferences.hdf5 \
    ${OUTDIR}/PnPCounterToBowl_batch2/PnPCounterToBowl_GR1ArmsAndWaistFourierHands_Env_preferences.hdf5 \
    ${OUTDIR}/PnPCounterToPlate_batch2/PnPCounterToPlate_GR1ArmsAndWaistFourierHands_Env_preferences.hdf5 \
    ${OUTDIR}/PnPOnionToBowl_batch2/PnPOnionToBowl_GR1ArmsAndWaistFourierHands_Env_preferences.hdf5 \
    --output ${OUTDIR}/all_tasks_400pairs.hdf5

echo ""
echo "[done] Merged dataset: ${OUTDIR}/all_tasks_400pairs.hdf5"
