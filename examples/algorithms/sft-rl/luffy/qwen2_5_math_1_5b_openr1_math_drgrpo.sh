#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

MODEL_PATH=Qwen/Qwen2.5-Math-1.5B

LOCAL_DATASET_PATH="/data/mazihao/zhaolei/code/rl/sft-rl/Unify-Post-Training/data"
TRAIN_SET="${LOCAL_DATASET_PATH}/openr1.parquet"
VAL_SET="${LOCAL_DATASET_PATH}/AIME24/test.parquet"

OUTPUT_DIR="outputs"
export TENSORBOARD_DIR=$OUTPUT_DIR

NAME=qwen2_5_math_1_5b_luffy_drgrpo_0131

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port 29506 -m rllava.train.pipeline.rlvr \
    config=examples/config.yaml \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    algorithm.use_kl_loss=false \
    data.train_files=${TRAIN_SET} \
    data.val_files=${VAL_SET} \
    data.val_batch_size=512 \
    data.format_prompt=null \
    data.prompt_key=prompt \
    data.target_key=target \
    data.answer_key=reward_model \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.max_target_length=2048 \
    actor.model.model_path=${MODEL_PATH} \
    actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor.ppo_mini_batch_size=2 \
    actor.ppo_micro_batch_size=1 \
    actor.loss_remove_clip=True \
    actor.entropy_coeff=0.001 \
    actor.policy_loss.loss_mode=luffy \
    actor.off_policy_reshape=p_div_p_0.1 \
    actor.fsdp.offload_params=False \
    reward.reward_function=./examples/algorithms/sft-rl/entropy_math_reward.py:compute_score \
    rollout.seed=none \
    rollout.vllm.dtype=float16 \
    rollout.n=8 \
    rollout.prefix_strategy=random \
    rollout.min_prefix_ratio=1.0 \
    rollout.max_prefix_ratio=1.0 \
    rollout.n_prefix=1 \
    rollout.prefix_share_across_samples=false \
    rollout.val_override_config.temperature=0.6 \
    rollout.val_override_config.top_p=0.95 \
    rollout.val_override_config.n=8 \
    trainer.experiment_name=${NAME} \
    trainer.outputs_dir=${OUTPUT_DIR} \
    trainer.val_before_train=False \
    trainer.save_freq=5

# prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
# prompt: "<|im_start|>system\n
# Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n
# <|im_start|>user\n
# [{'content': 'Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.', 'role': 'user'}] You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.<|im_end|>\n
# <|im_start|>assistant\n"

# "[{'content': 'Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.', 'role': 'user'}] You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."


# '<|im_start|>system\n
# Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n
# <|im_start|>user\n
# Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.<|im_end|>\n
# <|im_start|>assistant\n'


