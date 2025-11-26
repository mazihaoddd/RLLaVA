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

NAME=qwen2_5_vl_3b_geoqa3k_clipcov

torchrun --nproc_per_node=2 -m rllava.train.pipeline.rlvr \
    config=examples/config.yaml \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.format_prompt=./examples/format_prompt/math.jinja \
    actor.model.model_path=${MODEL_PATH} \
    actor.clip_ratio_low=1.0 \
    actor.clip_ratio_high=1.0 \
    actor.clip_ratio_c=10.0 \
    actor.policy_loss.loss_mode=clip_cov \
    actor.policy_loss.clip_cov_ratio=0.0002 \
    actor.policy_loss.clip_cov_lb=1.0 \
    actor.policy_loss.clip_cov_ub=5.0 \
    actor.use_kl_loss=false \
    reward.reward_function=./examples/reward_function/math.py:compute_score \
    trainer.experiment_name=${NAME} \
    trainer.outputs_dir=${OUTPUT_DIR}


