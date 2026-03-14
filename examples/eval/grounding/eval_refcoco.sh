#!/bin/bash

# RefCOCO Grounding Evaluation Script

# Configuration - modify these paths as needed
MODEL_PATH="outputs/rlava/qwen2_5_vl_3b_grounding/checkpoints/global_step_xxx/converted_hf_v2"
ANNO_DIR="path/to/refcoco/annotations"
IMAGE_DIR="path/to/coco/images"
OUTPUT_DIR="logs"
BATCH_SIZE=16
DEVICE_RANK=0

# Run evaluation
python -m rllava.eval.grounding.eval_refcoco \
    --model_path "$MODEL_PATH" \
    --anno_dir "$ANNO_DIR" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --device_rank $DEVICE_RANK
