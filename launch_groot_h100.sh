#!/bin/bash
# =============================================================================
# GR00T N1.6 RLHF Pipeline — RoboCasa GR1 Tabletop Tasks (H100)
#
# Uses expert demos as winners + base policy failures as losers.
# Chain: Base GR00T N1.6 → DPO / RWR / PPO
#
# Evaluation compares all checkpoints side-by-side.
#
# Usage: bash launch_groot_h100.sh <step>
#
# Steps:
#   install    install dependencies
#   rollouts   collect demo-vs-rollout preference pairs (4 tasks × 50 pairs)
#   dpo        DPO fine-tune from base model
#   rwr        RWR fine-tune from base model
#   ppo        PPO fine-tune from base model
#   eval       evaluate all checkpoints: base, dpo, rwr, ppo
#   all        install + rollouts + dpo + rwr + ppo + eval
# =============================================================================
set -e

STEP=${1:-"all"}

# ---- Config ------------------------------------------------------------------
BASE_MODEL="nvidia/GR00T-N1.6-3B"

PREFERENCE_DIR="preference_data/gr1_demo_pairs"
PREFERENCE_HDF5="$PREFERENCE_DIR/all_4tasks_demo_preferences.hdf5"
DEMO_BASE="examples/robocasa-gr1-tabletop-tasks/gr1_finetune_data"

DPO_OUTPUT="outputs/dpo"
RWR_OUTPUT="outputs/rwr"
PPO_OUTPUT="outputs/ppo"
EVAL_OUTPUT="outputs/eval"

SERVER_PORT=5555

# 4 tasks with moderate base policy success (good mix of failures for losers)
EVAL_ENVS=(
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
)

# Demo data directories (task_name -> demo_dir)
declare -A DEMO_DIRS
DEMO_DIRS["PosttrainPnPNovelFromCuttingboardToBasketSplitA"]="gr1_arms_waist.CuttingboardToBasket"
DEMO_DIRS["PnPBottleToCabinetClose"]="gr1_unified.PnPBottleToCabinetClose"
DEMO_DIRS["PosttrainPnPNovelFromPlateToBowlSplitA"]="gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA"
DEMO_DIRS["PosttrainPnPNovelFromTrayToPotSplitA"]="gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA"
# ------------------------------------------------------------------------------

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

echo "============================================================"
echo " GR00T RLHF Pipeline (H100)"
echo " Step    : $STEP"
echo " Model   : $BASE_MODEL"
echo " Chain   : Base → DPO / RWR / PPO"
echo "============================================================"

# ---- Install -----------------------------------------------------------------
step_install() {
    echo "[install] Installing dependencies..."
    pip install -e ".[train]" --ignore-requires-python --quiet
    pip install -e external_dependencies/robocasa-gr1-tabletop-tasks --quiet
    pip install gymnasium h5py wandb pyarrow opencv-python pytz --quiet
    echo "[install] Done."
}

# ---- Download demo data ------------------------------------------------------
step_download() {
    echo "[download] Downloading demo data for 4 tasks..."
    bash scripts/download_demo_tasks.sh
    echo "[download] Done."
}

# ---- Rollout collection (base policy, paired with demos) ---------------------
step_rollouts() {
    echo "[rollouts] Starting GR00T inference server ($BASE_MODEL)..."
    python gr00t/eval/run_gr00t_server.py \
        --model-path "$BASE_MODEL" \
        --embodiment-tag GR1 \
        --use-sim-policy-wrapper \
        --port $SERVER_PORT &
    SERVER_PID=$!
    echo "[rollouts] Server PID: $SERVER_PID — waiting 60s for warmup..."
    sleep 60

    mkdir -p "$PREFERENCE_DIR"

    for ENV in "${EVAL_ENVS[@]}"; do
        TASK=$(echo "$ENV" | cut -d'/' -f2 | sed 's/_GR1ArmsAndWaistFourierHands_Env//')
        DEMO_DIR="${DEMO_BASE}/${DEMO_DIRS[$TASK]}"
        OUT="$PREFERENCE_DIR/${TASK}_GR1ArmsAndWaistFourierHands_Env_demo_preferences.hdf5"

        if [ -f "$OUT" ]; then
            echo "[rollouts] $TASK already collected, skipping."
            continue
        fi

        echo "[rollouts] Collecting 50 pairs for $TASK ..."
        python scripts/build_demo_preference_pairs.py \
            --env_name "$ENV" \
            --demo_data_dir "$DEMO_DIR" \
            --host localhost \
            --port $SERVER_PORT \
            --n_pairs 50 \
            --output_dir "$PREFERENCE_DIR"
    done

    kill $SERVER_PID 2>/dev/null || true
    echo "[rollouts] Server stopped."

    echo "[rollouts] Merging preference files..."
    python scripts/merge_preference_hdf5.py \
        "$PREFERENCE_DIR"/*_demo_preferences.hdf5 \
        --output "$PREFERENCE_HDF5"
    echo "[rollouts] Done. Preference data: $PREFERENCE_HDF5"
}

# ---- DPO (Base -> DPO) ------------------------------------------------------
step_dpo() {
    echo "[dpo] DPO fine-tuning from base model..."
    python gr00t_rlhf/algos/dpo.py \
        --model_path "$BASE_MODEL" \
        --hdf5_path  "$PREFERENCE_HDF5" \
        --output_dir "$DPO_OUTPUT" \
        --beta 0.1 \
        --n_epochs 3 \
        --batch_size 8 \
        --lr 1e-5 \
        --n_windows_per_pair 5 \
        --use_wandb \
        --wandb_project gr00t-rlhf
    echo "[dpo] Done. Checkpoint: $DPO_OUTPUT"
}

# ---- RWR (Base -> RWR) ------------------------------------------------------
step_rwr() {
    echo "[rwr] RWR fine-tuning from base model..."
    python gr00t_rlhf/algos/rwr.py \
        --model_path "$BASE_MODEL" \
        --hdf5_path  "$PREFERENCE_HDF5" \
        --output_dir "$RWR_OUTPUT" \
        --temperature 1.0 \
        --n_epochs 3 \
        --batch_size 8 \
        --lr 1e-5 \
        --n_windows_per_pair 5 \
        --use_wandb \
        --wandb_project gr00t-rlhf
    echo "[rwr] Done. Checkpoint: $RWR_OUTPUT"
}

# ---- PPO (Base -> PPO) ------------------------------------------------------
step_ppo() {
    echo "[ppo] PPO fine-tuning from base model..."
    python gr00t_rlhf/algos/ppo.py \
        --model_path "$BASE_MODEL" \
        --hdf5_path  "$PREFERENCE_HDF5" \
        --output_dir "$PPO_OUTPUT" \
        --ppo_iters 30 \
        --batch_size 4 \
        --lr 1e-5 \
        --clip_eps 0.2 \
        --kl_coeff 0.1 \
        --n_windows_per_pair 5 \
        --use_wandb \
        --wandb_project gr00t-rlhf
    echo "[ppo] Done. Checkpoint: $PPO_OUTPUT"
}

# ---- Evaluation: all checkpoints side-by-side --------------------------------
step_eval() {
    echo "[eval] Evaluating all checkpoints..."
    echo "[eval] Chain: Base → DPO / RWR / PPO"

    declare -A MODELS
    MODELS["base"]="$BASE_MODEL"
    MODELS["dpo"]="$DPO_OUTPUT"
    MODELS["rwr"]="$RWR_OUTPUT"
    MODELS["ppo"]="$PPO_OUTPUT"

    for LABEL in base dpo rwr ppo; do
        MODEL_DIR="${MODELS[$LABEL]}"
        # Skip missing RLHF checkpoints
        if [[ "$LABEL" != "base" ]] && [ ! -d "$MODEL_DIR" ]; then
            echo "[eval] $LABEL checkpoint not found at $MODEL_DIR, skipping."
            continue
        fi

        echo "[eval] --- $LABEL ---"
        python gr00t/eval/run_gr00t_server.py \
            --model-path "$MODEL_DIR" \
            --embodiment-tag GR1 \
            --use-sim-policy-wrapper \
            --port $((SERVER_PORT + 10)) &
        EVAL_SERVER_PID=$!
        sleep 60

        for ENV in "${EVAL_ENVS[@]}"; do
            TASK=$(echo "$ENV" | cut -d'/' -f2)
            echo "[eval] $LABEL / $TASK"
            python scripts/eval_policy.py \
                --env_name "$ENV" \
                --host localhost \
                --port $((SERVER_PORT + 10)) \
                --n_episodes 20 \
                --output_dir "$EVAL_OUTPUT/$LABEL/$TASK" || \
            echo "[eval] eval failed for $LABEL/$TASK"
        done

        kill $EVAL_SERVER_PID 2>/dev/null || true
        sleep 5
    done

    echo ""
    echo "[eval] =================================================="
    echo "[eval]  Results summary"
    echo "[eval] =================================================="
    python scripts/print_eval_summary.py \
        --eval_dir "$EVAL_OUTPUT" \
        --model_order base dpo rwr ppo 2>/dev/null || \
    echo "[eval] Run scripts/print_eval_summary.py manually to see table."
    echo "[eval] Full results in: $EVAL_OUTPUT"
}

# ---- Dispatcher --------------------------------------------------------------
case $STEP in
    all)       step_install; step_download; step_rollouts; step_dpo; step_rwr; step_ppo; step_eval ;;
    install)   step_install  ;;
    download)  step_download ;;
    rollouts)  step_rollouts ;;
    dpo)       step_dpo      ;;
    rwr)       step_rwr      ;;
    ppo)       step_ppo      ;;
    eval)      step_eval     ;;
    *)
        echo "Usage: bash launch_groot_h100.sh <step>"
        echo ""
        echo "  install    install dependencies"
        echo "  download   download demo data for 4 tasks"
        echo "  rollouts   collect demo-vs-rollout preference pairs"
        echo "  dpo        DPO fine-tune  → outputs/dpo"
        echo "  rwr        RWR fine-tune  → outputs/rwr"
        echo "  ppo        PPO fine-tune  → outputs/ppo"
        echo "  eval       evaluate all checkpoints (base, dpo, rwr, ppo)"
        echo "  all        run full pipeline"
        exit 1
        ;;
esac

echo "============================================================"
echo " Pipeline step '$STEP' complete!"
echo "============================================================"
