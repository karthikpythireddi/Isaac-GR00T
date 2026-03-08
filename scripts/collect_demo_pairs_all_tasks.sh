#!/bin/bash
# =============================================================================
# Collect demo-vs-rollout preference pairs for 4 RoboCasa GR1 tasks
#
# Uses expert demos as WINNERS and failed base GR00T N1.6 rollouts as LOSERS.
# Produces clean "demo_vs_failure" preference pairs for DPO/RWR/PPO training.
#
# Prerequisites:
#   1. Demo data downloaded:
#        conda activate groot
#        python scripts/download_robocasa_demos.py --all_recommended
#
#   2. GR00T server running (in a separate terminal):
#        conda activate groot
#        python gr00t/eval/run_gr00t_server.py \
#            --model-path nvidia/GR00T-N1.6-3B \
#            --embodiment-tag GR1 \
#            --use-sim-policy-wrapper --port 5555
#
#   3. Run this script (in gr1_sim env):
#        conda activate gr1_sim
#        bash scripts/collect_demo_pairs_all_tasks.sh 2>&1 | tee collect_demo_pairs.log
# =============================================================================
set -e

PYTHON=${PYTHON:-python}
SCRIPT=scripts/build_demo_preference_pairs.py
OUTDIR=preference_data/gr1_demo_pairs
DEMO_BASE=examples/robocasa-gr1-tabletop-tasks/gr1_finetune_data
N_PAIRS=${1:-50}

echo "========================================"
echo "Demo-vs-Rollout Preference Collection"
echo "n_pairs per task: $N_PAIRS"
echo "Output dir: $OUTDIR"
echo "========================================"

# Task 1: CuttingboardToBasket (58.0% base success → ~42% failures)
echo ""
echo "[1/4] PosttrainPnPNovelFromCuttingboardToBasketSplitA — $N_PAIRS pairs"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env" \
    --demo_data_dir "${DEMO_BASE}/gr1_arms_waist.CuttingboardToBasket" \
    --n_pairs $N_PAIRS --output_dir "$OUTDIR" --seed_offset 0

# Task 2: BottleToCabinetClose (51.5% → ~49% failures)
echo ""
echo "[2/4] PnPBottleToCabinetClose — $N_PAIRS pairs"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env" \
    --demo_data_dir "${DEMO_BASE}/gr1_unified.PnPBottleToCabinetClose" \
    --n_pairs $N_PAIRS --output_dir "$OUTDIR" --seed_offset 0

# Task 3: PlateToBowl (57.0% → ~43% failures)
echo ""
echo "[3/4] PosttrainPnPNovelFromPlateToBowlSplitA — $N_PAIRS pairs"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env" \
    --demo_data_dir "${DEMO_BASE}/gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA" \
    --n_pairs $N_PAIRS --output_dir "$OUTDIR" --seed_offset 0

# Task 4: TrayToPot (64.5% → ~35% failures)
echo ""
echo "[4/4] PosttrainPnPNovelFromTrayToPotSplitA — $N_PAIRS pairs"
$PYTHON $SCRIPT \
    --env_name "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env" \
    --demo_data_dir "${DEMO_BASE}/gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA" \
    --n_pairs $N_PAIRS --output_dir "$OUTDIR" --seed_offset 0

echo ""
echo "========================================"
echo "Merging all pairs into one file..."
echo "========================================"

$PYTHON scripts/merge_preference_hdf5.py \
    ${OUTDIR}/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env_demo_preferences.hdf5 \
    ${OUTDIR}/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env_demo_preferences.hdf5 \
    ${OUTDIR}/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env_demo_preferences.hdf5 \
    ${OUTDIR}/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env_demo_preferences.hdf5 \
    --output ${OUTDIR}/all_4tasks_demo_preferences.hdf5

echo ""
echo "[done] Merged dataset: ${OUTDIR}/all_4tasks_demo_preferences.hdf5"
echo "       Total pairs: $((N_PAIRS * 4))"
echo ""
echo "Next steps:"
echo "  1. Train DPO:  python gr00t_rlhf/algos/dpo.py --model_path nvidia/GR00T-N1.6-3B --hdf5_path ${OUTDIR}/all_4tasks_demo_preferences.hdf5"
echo "  2. Train RWR:  python gr00t_rlhf/algos/rwr.py --model_path nvidia/GR00T-N1.6-3B --hdf5_path ${OUTDIR}/all_4tasks_demo_preferences.hdf5"
