#!/bin/bash

# SuperCLEVR Counting Evaluation Script for Qwen2.5-VL

# Configuration - modify these paths as needed
MODEL_PATH="outputs/rlava/qwen2_5_vl_3b_clevr_online/checkpoints/global_step_690/converted_hf_v2"
PROMPT_PATH="rllava/eval/jsons/counting/superclevr_test200_counting_problems.jsonl"
OUTPUT_PATH="examples/eval_results/counting_results_superclevr.json"
BATCH_SIZE=4
DEVICE="cuda:0"

# Run evaluation
python -m rllava.eval.counting.eval_counting_superclevr \
    --model_path "$MODEL_PATH" \
    --prompt_path "$PROMPT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --batch_size $BATCH_SIZE \
    --device "$DEVICE"
