#!/bin/bash
# Collect preference pairs from the SFT fine-tuned GR00T policy.
# These replace the base-model pairs collected previously.
#
# Model: karthikpythireddi93/gr00t-n16-gr1-tabletop-sft (trained on H100)
# Output: preference_data/gr1_sft/all_tasks_preferences.hdf5
#
# Requires GR00T inference server already running in 'groot' conda env:
#   conda activate groot
#   cd /home/karthik/CS234/IssacLab/Isaac-GR00T
#   python gr00t/eval/run_gr00t_server.py \
#       --model-path karthikpythireddi93/gr00t-n16-gr1-tabletop-sft \
#       --embodiment-tag new_embodiment \
#       --use-sim-policy-wrapper \
#       --denoising-steps 4 \
#       --port 5555
#
# Then in a second terminal (gr1_sim env):
#   conda activate gr1_sim
#   cd /home/karthik/CS234/IssacLab/Isaac-GR00T
#   bash scripts/collect_sft_rollouts.sh 2>&1 | tee collect_sft_rollouts.log

set -e
PYTHON=/home/karthik/miniconda3/envs/gr1_sim/bin/python
SCRIPT=scripts/collect_preferences_groot.py
OUTDIR=preference_data/gr1_sft
N_PAIRS=20
COMMON="--noise_injection --noise_scale 0.05 --host localhost --port 5555"

mkdir -p "$OUTDIR"

echo "========================================"
echo " SFT Preference Data Collection"
echo " Model : karthikpythireddi93/gr00t-n16-gr1-tabletop-sft"
echo " Output: $OUTDIR"
echo " Pairs : $N_PAIRS per task x 4 tasks = 80 total"
echo " (matches LIBERO: 20 pairs/task)"
echo "========================================"

echo ""
echo "[1/4] PnPCounterToPan — $N_PAIRS pairs"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PnPCounterToPan_GR1ArmsAndWaistFourierHands_Env" \
    --n_pairs $N_PAIRS \
    --seed_offset 0 \
    --output_dir "$OUTDIR" \
    $COMMON

echo ""
echo "[2/4] PnPCounterToBowl — $N_PAIRS pairs"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PnPCounterToBowl_GR1ArmsAndWaistFourierHands_Env" \
    --n_pairs $N_PAIRS \
    --seed_offset 0 \
    --output_dir "$OUTDIR" \
    $COMMON

echo ""
echo "[3/4] PnPCounterToPlate — $N_PAIRS pairs"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PnPCounterToPlate_GR1ArmsAndWaistFourierHands_Env" \
    --n_pairs $N_PAIRS \
    --seed_offset 0 \
    --output_dir "$OUTDIR" \
    $COMMON

echo ""
echo "[4/4] PnPOnionToBowl — $N_PAIRS pairs"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PnPOnionToBowl_GR1ArmsAndWaistFourierHands_Env" \
    --n_pairs $N_PAIRS \
    --seed_offset 0 \
    --output_dir "$OUTDIR" \
    $COMMON

echo ""
echo "========================================"
echo "All tasks done! Merging into one file..."
echo "========================================"

$PYTHON scripts/merge_preference_hdf5.py \
    "$OUTDIR"/PnPCounterToPan_GR1ArmsAndWaistFourierHands_Env_preferences.hdf5 \
    "$OUTDIR"/PnPCounterToBowl_GR1ArmsAndWaistFourierHands_Env_preferences.hdf5 \
    "$OUTDIR"/PnPCounterToPlate_GR1ArmsAndWaistFourierHands_Env_preferences.hdf5 \
    "$OUTDIR"/PnPOnionToBowl_GR1ArmsAndWaistFourierHands_Env_preferences.hdf5 \
    --output "$OUTDIR/all_tasks_preferences.hdf5"

echo ""
echo "[done] Merged dataset: $OUTDIR/all_tasks_preferences.hdf5"
echo "[done] Ready for: bash launch_groot_h100.sh dpo / rwr / ppo (on H100)"
