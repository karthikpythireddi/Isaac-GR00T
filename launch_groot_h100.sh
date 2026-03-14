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

# Trap to clean up server on unexpected exit
cleanup() {
    echo "[cleanup] Killing any lingering GR00T servers..."
    pkill -f run_gr00t_server 2>/dev/null || true
}
trap cleanup EXIT

STEP=${1:-"all"}

# ---- Config ------------------------------------------------------------------
BASE_MODEL="nvidia/GR00T-N1.6-3B"
SFT_CHECKPOINT="outputs/gr1_tabletop_sft_h100/checkpoint-30000"

# Use SFT checkpoint for DPO/RWR if it exists, otherwise fall back to base
_policy_base() {
    if [ -d "$SFT_CHECKPOINT" ]; then
        echo "$SFT_CHECKPOINT"
    else
        echo "[warn] SFT checkpoint not found at $SFT_CHECKPOINT, using zero-shot base model." >&2
        echo "$BASE_MODEL"
    fi
}
POLICY_BASE=$(_policy_base)

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

    # FlashAttention2 — required by GR00T backbone
    # Download wheel directly to /tmp to avoid cross-device link errors
    # Wheel targets: torch 2.7, cu12, cp312, cxx11abiTRUE
    FLASH_WHL="flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${FLASH_WHL}"
    echo "[install] Downloading flash-attn wheel..."
    wget -q "$FLASH_URL" -O "/tmp/${FLASH_WHL}" && \
        pip install "/tmp/${FLASH_WHL}" --quiet && \
        rm "/tmp/${FLASH_WHL}" || \
        echo "[warn] flash_attn install failed — model will fall back to eager attention"

    # robosuite — required for rollout collection environments
    pip install robosuite --quiet

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

# ---- SFT fine-tuning (optional but recommended) ------------------------------
step_sft() {
    echo "[sft] Running SFT fine-tuning on H100..."
    bash scripts/finetune_gr1_h100.sh 2>&1 | tee outputs/finetune_gr1_h100.log
    # Refresh POLICY_BASE after SFT
    POLICY_BASE=$(_policy_base)
    echo "[sft] Done. POLICY_BASE=$POLICY_BASE"
}

# ---- DPO (SFT -> DPO) -------------------------------------------------------
step_dpo() {
    echo "[dpo] DPO fine-tuning from: $POLICY_BASE"
    python gr00t_rlhf/algos/dpo.py \
        --model_path "$POLICY_BASE" \
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

# ---- RWR (SFT -> RWR) -------------------------------------------------------
step_rwr() {
    echo "[rwr] RWR fine-tuning from: $POLICY_BASE"
    python gr00t_rlhf/algos/rwr.py \
        --model_path "$POLICY_BASE" \
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
    # Disable set -e inside eval so one task failure doesn't kill the whole run
    set +e

    echo "[eval] Evaluating all checkpoints..."
    echo "[eval] Chain: Base → DPO / RWR / PPO"

    declare -A MODELS
    MODELS["base"]="$BASE_MODEL"
    MODELS["dpo"]="$DPO_OUTPUT"
    MODELS["rwr"]="$RWR_OUTPUT"
    MODELS["ppo"]="$PPO_OUTPUT"

    EVAL_PORT=$((SERVER_PORT + 10))

    for LABEL in base dpo rwr ppo; do
        MODEL_DIR="${MODELS[$LABEL]}"
        # Skip missing RLHF checkpoints
        if [[ "$LABEL" != "base" ]] && [ ! -d "$MODEL_DIR" ]; then
            echo "[eval] $LABEL checkpoint not found at $MODEL_DIR, skipping."
            continue
        fi

        # ---- Check if this model is already fully evaluated ----
        ALL_DONE=true
        for ENV in "${EVAL_ENVS[@]}"; do
            TASK=$(echo "$ENV" | cut -d'/' -f2)
            RESULTS_FILE="$EVAL_OUTPUT/$LABEL/$TASK/eval_results.json"
            VIDEO_DIR="$EVAL_OUTPUT/$LABEL/$TASK/videos"
            if [ ! -f "$RESULTS_FILE" ] || [ ! -d "$VIDEO_DIR" ]; then
                ALL_DONE=false
                break
            fi
            # Check we have all 20 videos
            N_VIDS=$(find "$VIDEO_DIR" -name "*.mp4" 2>/dev/null | wc -l)
            if [ "$N_VIDS" -lt 20 ]; then
                ALL_DONE=false
                break
            fi
        done
        if [ "$ALL_DONE" = true ]; then
            echo "[eval] $LABEL already fully evaluated (all tasks + videos). Skipping."
            continue
        fi

        # ---- Ensure no lingering server on the eval port ----
        echo "[eval] --- $LABEL ---"
        echo "[eval] Cleaning up any previous server on port $EVAL_PORT..."
        pkill -f run_gr00t_server 2>/dev/null || true
        # Wait for port to be fully released
        for i in $(seq 1 15); do
            if ! ss -tlnp 2>/dev/null | grep -q ":$EVAL_PORT "; then
                echo "[eval] Port $EVAL_PORT is free."
                break
            fi
            if [ "$i" -eq 15 ]; then
                echo "[eval] WARNING: Port $EVAL_PORT still in use after 75s! Force killing..."
                fuser -k $EVAL_PORT/tcp 2>/dev/null || true
                sleep 5
            else
                echo "[eval] Port $EVAL_PORT still in use, waiting... ($i/15)"
                sleep 5
            fi
        done

        # ---- Start server for this model ----
        echo "[eval] Starting server for $LABEL (model: $MODEL_DIR)..."
        python gr00t/eval/run_gr00t_server.py \
            --model-path "$MODEL_DIR" \
            --embodiment-tag GR1 \
            --use-sim-policy-wrapper \
            --port $EVAL_PORT &
        EVAL_SERVER_PID=$!
        echo "[eval] Server PID: $EVAL_SERVER_PID — waiting 60s for warmup..."
        sleep 60

        # Verify server is actually running
        if ! kill -0 $EVAL_SERVER_PID 2>/dev/null; then
            echo "[eval] ERROR: Server for $LABEL failed to start! Skipping."
            continue
        fi
        echo "[eval] Server is running (PID $EVAL_SERVER_PID)."

        # ---- Run eval on all tasks ----
        for ENV in "${EVAL_ENVS[@]}"; do
            TASK=$(echo "$ENV" | cut -d'/' -f2)

            # Skip tasks already completed with videos
            RESULTS_FILE="$EVAL_OUTPUT/$LABEL/$TASK/eval_results.json"
            VIDEO_DIR="$EVAL_OUTPUT/$LABEL/$TASK/videos"
            if [ -f "$RESULTS_FILE" ] && [ -d "$VIDEO_DIR" ]; then
                N_VIDS=$(find "$VIDEO_DIR" -name "*.mp4" 2>/dev/null | wc -l)
                if [ "$N_VIDS" -ge 20 ]; then
                    echo "[eval] $LABEL / $TASK already done (20 videos). Skipping."
                    continue
                fi
            fi

            # Verify server is still alive before each task
            if ! kill -0 $EVAL_SERVER_PID 2>/dev/null; then
                echo "[eval] ERROR: Server died during $LABEL eval! Breaking."
                break
            fi

            echo "[eval] $LABEL / $TASK"
            python scripts/eval_policy.py \
                --env_name "$ENV" \
                --host localhost \
                --port $EVAL_PORT \
                --n_episodes 20 \
                --output_dir "$EVAL_OUTPUT/$LABEL/$TASK" \
                --record_video
            if [ $? -ne 0 ]; then
                echo "[eval] WARNING: eval failed for $LABEL/$TASK — continuing with next task"
            fi
        done

        # ---- Stop server and wait for port release ----
        echo "[eval] Stopping $LABEL server (PID $EVAL_SERVER_PID)..."
        kill $EVAL_SERVER_PID 2>/dev/null || true
        wait $EVAL_SERVER_PID 2>/dev/null || true
        echo "[eval] Waiting for port $EVAL_PORT to be released..."
        sleep 15
    done

    set -e

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
    all)       step_install; step_download; step_sft; step_rollouts; step_dpo; step_rwr; step_eval ;;
    sft)       step_sft     ;;
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
