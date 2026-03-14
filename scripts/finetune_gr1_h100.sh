#!/bin/bash
# =============================================================================
# GR00T N1.6 SFT Fine-tuning on H100 — All 4 GR1 Tabletop Tasks
#
# Fine-tunes nvidia/GR00T-N1.6-3B on 50 demos per task (200 total) across
# all 4 evaluation tasks. The resulting checkpoint is used as the base for
# DPO and RWR (not the zero-shot base model).
#
# H100 (80GB) settings:
#   - No gradient checkpointing needed
#   - Full batch size 64 directly (no accumulation)
#   - ~2-4 hours for 30K steps on single H100
#
# Usage (from Isaac-GR00T root):
#   bash scripts/finetune_gr1_h100.sh 2>&1 | tee outputs/finetune_gr1_h100.log
# =============================================================================
set -e

export NUM_GPUS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# All 4 task demo directories (already downloaded)
TASK_DIRS=(
    "examples/robocasa-gr1-tabletop-tasks/gr1_finetune_data/gr1_arms_waist.CuttingboardToBasket"
    "examples/robocasa-gr1-tabletop-tasks/gr1_finetune_data/gr1_unified.PnPBottleToCabinetClose"
    "examples/robocasa-gr1-tabletop-tasks/gr1_finetune_data/gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA"
    "examples/robocasa-gr1-tabletop-tasks/gr1_finetune_data/gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA"
)

OUTPUT_DIR="outputs/gr1_tabletop_sft_h100"

echo "========================================"
echo "GR00T N1.6 SFT Fine-tuning (H100)"
echo "Tasks   : ${#TASK_DIRS[@]} tasks (50 demos each)"
echo "Output  : $OUTPUT_DIR"
echo "GPUs    : $NUM_GPUS"
echo "Batch   : 64 (no accumulation needed on H100)"
echo "Steps   : 30000"
echo "========================================"

# Verify all task directories exist
for TASK_DIR in "${TASK_DIRS[@]}"; do
    if [ ! -d "$TASK_DIR/meta" ]; then
        echo "ERROR: Missing demo data at $TASK_DIR"
        echo "Run: bash scripts/download_demo_tasks.sh first"
        exit 1
    fi
done
echo "[ok] All 4 task directories verified."

# Build dataset path string (comma-separated for multi-task training)
DATASET_PATHS=$(IFS=,; echo "${TASK_DIRS[*]}")

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path $DATASET_PATHS \
    --embodiment_tag gr1 \
    --num_gpus $NUM_GPUS \
    --output_dir $OUTPUT_DIR \
    --save_steps 5000 \
    --save_total_limit 3 \
    --max_steps 30000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 3e-5 \
    --global_batch_size 64 \
    --dataloader_num_workers 4 \
    --state_dropout_prob 0.8 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08

echo "========================================"
echo "SFT Fine-tuning complete!"
echo "Checkpoint: $OUTPUT_DIR/checkpoint-30000"
echo ""
echo "Next steps:"
echo "  DPO: python gr00t_rlhf/algos/dpo.py --model_path $OUTPUT_DIR/checkpoint-30000 ..."
echo "  RWR: python gr00t_rlhf/algos/rwr.py --model_path $OUTPUT_DIR/checkpoint-30000 ..."
echo "========================================"
