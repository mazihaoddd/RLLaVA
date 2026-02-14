# RLLaVA Training Examples

We provide comprehensive examples for training vision-language models using reinforcement learning with the RLLaVA framework.

## Quick Start

Here's a basic training script template:

```bash
set -x
export PYTHONUNBUFFERED=1
MODEL_PATH=your_model_path
TRAIN_SET=your_org/dataset@train
OUTPUT_DIR=your_output_directory
export TENSORBOARD_DIR=$OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m rllava.train.pipeline.rlvr \
  config=examples/config.yaml \
  data.train_files=${TRAIN_SET} \
  data.format_prompt=./examples/format_prompt/your_task.jinja \
  data.prompt_key=prompt_key_of_your_dataset \
  data.answer_key=answer_key_of_your_dataset \
  data.image_key=image_key_of_your_dataset \
  actor.model.model_path=${MODEL_PATH} \
  reward.reward_function=./examples/reward_function/your_task.py:compute_score \
  algorithm.adv_estimator=grpo \
  trainer.outputs_dir=${OUTPUT_DIR} \
  trainer.val_freq=-1 \
  trainer.val_before_train=false
```

### Key Parameters

- **`CUDA_VISIBLE_DEVICES`**: Specifies which GPUs to use for training
- **`--nproc_per_node`**: Total number of GPUs/processes per node
- **`config`**: Base configuration file (e.g., `examples/config.yaml`)
- **`data.train_files`**: Training dataset path in format `org/dataset@split`
- **`data.prompt_key`**: Dataset field name containing the question/prompt text (default: `problem`)
- **`data.answer_key`**: Dataset field name containing the ground-truth answer (default: `answer`)
- **`data.image_key`**: Dataset field name containing image data or image paths (default: `images`)
- **`data.format_prompt`**: Prompt template file path using Jinja2 syntax to format input data into model-compatible prompts (`examples/format_prompt/your_task.jinja`)
- **`actor.model.model_path`**: Path to the base model (local or HuggingFace)
- **`reward.reward_function`**: Reward function in format `examples/reward_function/your_task.py:function_name`, where the function calculates rewards for model outputs
- **`algorithm.adv_estimator`**: The advantage estimating method which different RL algorithm to use 
- **`trainer.val_freq`**: Validation frequency in epochs (-1 to disable)
- **`trainer.val_before_train`**: Whether to validate before training starts

### Configuration Tips

- Set `trainer.val_freq=-1` and `trainer.val_before_train=false` if your dataset lacks a validation set or the validation set uses different prompt keys from the training set.
- Override any `config.yaml` parameter by adding it as a command-line argument.
- Change the algorithm by setting `algorithm.adv_estimator` and other parameters according to the bash scripts in `examples/algorithms/` for your preferred method.
- Always execute commands from the `RLLaVA` directory root.
## Downstream Tasks

- [Grounding](#grounding) - Visual referring and localization
- [OVD](#ovd) - Open-vocabulary object detection
- [Counting](#counting) - Object counting in images
- [Math](#math) - Geometry problem solving
- [Agent-Search](#agent-search) - Multi-step web search agent
- [Agent-Code](#agent-code) - Code generation from visual interfaces

### Grounding

**Task**: Visual referring expression comprehension - given an image and a descriptive sentence, predict the bounding box coordinates of the referred region.

```bash
cd RLLaVA
bash examples/tasks/grounding/qwen2_vl_2b_grounding_vllm_online.sh
```

### OVD

**Task**: Open-Vocabulary Detection - detect all objects of a specified category in an image and output bounding boxes (coordinates 0-1000, integers) with confidence scores (0-1, two decimal places).

```bash
cd RLLaVA
bash examples/tasks/OVD/qwen2_5_vl_3b_ovd_online.sh
```

### Counting

**Task**: Visual object counting - count specific objects in complex scenes with multiple instances and visual distractors.


```bash
cd RLLaVA
bash examples/tasks/counting/qwen2_5_vl_3b_clevr_online.sh
```

### Math

**Task**: Geometry problem solving - analyze geometric diagrams and answer questions requiring mathematical reasoning and spatial understanding.

```bash
cd RLLaVA
bash examples/tasks/math/qwen2_5_vl_3b_geoqa8k_online.sh
```

### Agent-Search

**Task**: Multi-step web search agent - plan and execute sequential search actions to answer complex queries requiring information synthesis across multiple sources.

```bash
cd RLLaVA
bash examples/tasks/agent_search/qwen2_5_vl_3b_agent_search_online.sh
```

### Agent-Code

**Task**: Visual-to-code generation - generate executable code from visual interface designs, wireframes, or UI screenshots.

```bash
cd RLLaVA
bash examples/tasks/agent_code/qwen2_5_vl_3b_agent_code_online.sh
```

Other tasks have not been implemented completely yet. Please refer to the provided examples for guidance on creating new tasks.

## Contributing

To add a new task:
1. Create prompt template in `format_prompt/`
2. Implement reward function in `reward_function/`
3. Add task-specific config (optional)
4. Create training script in `tasks/your_task/`
