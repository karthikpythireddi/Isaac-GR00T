#!/bin/bash
# Fine-tune GR00T N1.6 on GR1 tabletop manipulation demos (CuttingboardToBasket)
# Single GPU: RTX 5000 Ada 16GB
#
# Memory fix: global_batch_size=64 OOMs on 16GB.
# Use per-device batch=4 + gradient_accumulation=16 (effective batch=64).
# gradient_checkpointing trades compute for ~30% VRAM savings.
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments avoids fragmentation crashes.
# Estimated time: 14-20 hours for 10K steps (gradient checkpointing ~30% slower).
#
# Usage (from Isaac-GR00T root):
#   bash scripts/finetune_gr1_tabletop.sh 2>&1 | tee outputs/finetune_gr1_tabletop.log

set -e

export NUM_GPUS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATASET_DIR=examples/robocasa-gr1-tabletop-tasks/gr1_finetune_data/gr1_arms_waist.CuttingboardToBasket
OUTPUT_DIR=outputs/gr1_tabletop_sft

echo "========================================"
echo "GR00T N1.6 SFT Fine-tuning"
echo "Dataset : $DATASET_DIR"
echo "Output  : $OUTPUT_DIR"
echo "GPUs    : $NUM_GPUS"
echo "Effective batch: 4 x 16 accum = 64"
echo "========================================"

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path $DATASET_DIR \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir $OUTPUT_DIR \
    --save_steps 500 \
    --save_total_limit 5 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --dataloader_num_workers 2 \
    --state_dropout_prob 0.8 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08

echo "========================================"
echo "Fine-tuning complete!"
echo "Checkpoint: $OUTPUT_DIR/checkpoint-10000"
echo "========================================"
