#!/bin/bash

# MAT-Search Evaluation Script for Qwen2.5-VL

# Configuration - modify these paths as needed
MODEL_PATH="laolao77/Visual-ARFT-Search"
DATA_PATH="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Benchmark/MAT-Search.json"
IMAGE_DIR="../huggingface_cache/hub/datasets--laolao77--MAT/snapshots/888ea8775ff0c70b87e016fa3999d1e0c05ddf55/MAT-Benchmark/MAT-Search-image"
OUTPUT_PATH="examples/eval_results/mat_search_results.json"
DEVICE="cuda:0"

# Run evaluation
python -m rllava.eval.agent_search.eval_mat_search \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --image_dir "$IMAGE_DIR" \
    --output_path "$OUTPUT_PATH" \
    --device "$DEVICE"
