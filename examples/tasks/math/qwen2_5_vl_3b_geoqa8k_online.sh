#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
# 使用统一的本地数据集路径（HF cache格式）
LOCAL_DATASET_PATH="hiyouga/geometry3k"
# LOCAL_DATASET_PATH="hiyouga/geometry3k"
TRAIN_SET="${LOCAL_DATASET_PATH}@train"
VAL_SET="${LOCAL_DATASET_PATH}@test"


OUTPUT_DIR="outputs"
export TENSORBOARD_DIR=$OUTPUT_DIR

NAME=qwen2_5_vl_3b_geoqa3k_grpo_online

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29512 -m rllava.train.pipeline.rlvr \
    config=examples/config.yaml \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.format_prompt=./examples/format_prompt/math.jinja \
    data.train_batch_size=4 \
    actor.ppo_mini_batch_size=4 \
    actor.ppo_micro_batch_size_per_gpu=1 \
    actor.log_prob_micro_batch_size_per_gpu=1 \
    actor.model.model_path=${MODEL_PATH} \
    reward.reward_function=./examples/reward_function/math.py:compute_score \
    trainer.experiment_name=${NAME} \
    trainer.outputs_dir=${OUTPUT_DIR} \
    trainer.find_last_checkpoint=false \
    trainer.val_before_train=false \
    trainer.total_epochs=5 \
    trainer.val_freq=30 \
    trainer.save_freq=30