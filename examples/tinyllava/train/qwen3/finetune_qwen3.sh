#!/bin/bash

# Source the common argument parsing library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_ARGS_PATH="$SCRIPT_DIR/../common.sh"

if [ ! -f "$COMMON_ARGS_PATH" ]; then
    echo "Error: Common args library not found at $COMMON_ARGS_PATH"
    exit 1
fi

source "$COMMON_ARGS_PATH"

# Parse arguments using the common library
parse_training_args "$0" "$@"

# Display the configuration being used
show_config

PRETRAINED_MODEL_PATH="/mnt/ve_share/zhaolei/outputs/tinyllava/tiny-llava-${VERSION}-pretrain"
RUN_NAME="tiny-llava-${VERSION}-finetune"
OUTPUT_DIR="/mnt/ve_share/zhaolei/outputs/tinyllava/${RUN_NAME}"

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29501 rllava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm full \
    --tune_type_vision_tower frozen\
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --group_by_modality_length True \
    --pretrained_model_path $PRETRAINED_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name $RUN_NAME
