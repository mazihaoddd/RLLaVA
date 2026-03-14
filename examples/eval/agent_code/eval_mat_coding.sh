#!/bin/bash

# MAT-Coding Evaluation Script for Qwen2.5-VL

# Configuration - modify these paths as needed
MODEL_PATH="outputs/rllava/qwen2.5_vl_3b_agent_code_online/checkpoints/global_step_1350/merged"
DATA_PATH="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Benchmark/MAT-Coding.json"
IMAGE_DIR="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Benchmark/MAT-Coding-image"
OUTPUT_PATH="examples/eval_results/mat_coding_results.json"
DEVICE="cuda:0"

# Run evaluation
python -m rllava.eval.agent_code.eval_mat_coding \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --image_dir "$IMAGE_DIR" \
    --output_path "$OUTPUT_PATH" \
    --device "$DEVICE"
