#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
MODEL_PATH="/mnt/ve_share/zhaolei/.cache/huggingface/hub/Qwen/Qwen2.5-VL-3B-Instruct"
# 使用统一的本地数据集路径（HF cache格式）
LOCAL_DATASET_PATH="/mnt/ve_share/zhaolei/.cache/huggingface/datasets/hiyouga_geometry3k"
# LOCAL_DATASET_PATH="hiyouga/geometry3k"
TRAIN_SET="${LOCAL_DATASET_PATH}@train"
VAL_SET="${LOCAL_DATASET_PATH}@test"

OUTPUT_DIR="/mnt/ve_share/zhaolei/outputs"
export TENSORBOARD_DIR=$OUTPUT_DIR

NAME=qwen2_5_vl_3b_geoqa3k_opo

# 为确保 exact on-policy：让 ppo_mini_batch_size 等于 train_batch_size，且 ppo_epochs=1（默认即为 1）
# 同时关闭 KL 与 entropy 正则，并使用分组采样 n>=2

torchrun --nproc_per_node=2 -m rllava.train.pipeline.rlvr \
    config=examples/config.yaml \
    algorithm.adv_estimator=opo \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.train_batch_size=512 \
    data.format_prompt=./examples/format_prompt/math.jinja \
    actor.model.model_path=${MODEL_PATH} \
    actor.ppo_mini_batch_size=512 \
    actor.use_kl_loss=false \
    reward.reward_function=./examples/reward_function/math.py:compute_score \
    trainer.experiment_name=${NAME} \
    trainer.outputs_dir=${OUTPUT_DIR}


