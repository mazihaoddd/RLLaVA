#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# Model
MODEL_PATH=Tongyi-MAI/MAI-UI-2B

# Environment config
ENV_CONFIG=./rllava/ppo/env/agentbay_computer.yaml  # or agentbay_computer.yaml

# Output
OUTPUT_DIR="outputs/gui_agent"
export TENSORBOARD_DIR=$OUTPUT_DIR

NAME=math_2b_osworld_grpo

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 -m rllava.train.pipeline.rlvr \
    config=examples/config_gui_agent.yaml \
    algorithm.adv_estimator=grpo \
    actor.model.model_path=${MODEL_PATH} \
    actor.ppo_mini_batch_size=2 \
    actor.ppo_micro_batch_size=1 \
    actor.optim.lr=5e-7 \
    rollout.n=3 \
    rollout.vllm.gpu_memory_utilization=0.4 \
    rollout.max_turns=10 \
    rollout.discount=0.95 \
    rollout.env_config_path=${ENV_CONFIG} \
    trainer.experiment_name=${NAME} \
    trainer.outputs_dir=${OUTPUT_DIR} \
    trainer.save_freq=10 \
    trainer.val_freq=5 \
    trainer.total_epochs=2
