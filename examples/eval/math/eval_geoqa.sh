#!/bin/bash

# GeoQA Evaluation Script for Qwen2.5-VL

# Configuration - modify these paths as needed
MODEL_PATH="outputs/rlava/qwen2_5_vl_7b_geoqa3k_grpo_online/checkpoints/global_step_510/converted_hf_v2"
PROMPT_PATH="rllava/eval/jsons/math/geoqa_test_prompts.jsonl"
OUTPUT_PATH="examples/eval_results/geoqa_results.json"
BATCH_SIZE=4
DEVICE="cuda:0"

# Run evaluation
python -m rllava.eval.math.eval_geoqa \
    --model_path "$MODEL_PATH" \
    --prompt_path "$PROMPT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --batch_size $BATCH_SIZE \
    --device "$DEVICE"
