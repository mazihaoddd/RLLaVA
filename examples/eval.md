# RLLaVA Evaluation Guide

### Dependecies Preparation

When evaluating your model, make sure to prepare the dependencies as follows:
```bash
pip install math_verify
pip install qwen-vl-utils
```

### Model Preparation

Before evaluation, convert your trained checkpoint to HuggingFace format:

```bash
python legacy_model_merger.py merge \
    --local_dir outputs/your_task/checkpoints/global_step_xxx/actor \
    --hf_model_path path/to/original/hf_model \
    --target_dir outputs/your_task/checkpoints/global_step_xxx/merged
```

Then set `MODEL_PATH` to the converted model path.

## Quick Start

```bash
#!/bin/bash

# Configuration - modify these paths as needed
MODEL_PATH="your_model_checkpoint_path"
PROMPT_PATH="rllava/eval/jsons/your_task/your_data.jsonl"
OUTPUT_PATH="examples/eval_results/your_results.json"
BATCH_SIZE=4
DEVICE="cuda:0"

# Run evaluation
python -m rllava.eval.your_task.eval_script \
    --model_path "$MODEL_PATH" \
    --prompt_path "$PROMPT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --batch_size $BATCH_SIZE \
    --device "$DEVICE"
```

### Key Parameters

Edit the corresponding `.sh` file to configure:

- **`MODEL_PATH`**: Path to your trained model checkpoint
- **`DEVICE`**: GPU device (e.g., `cuda:0`)
- **`PROMPT_PATH`**: Path to evaluation data file
- **`OUTPUT_PATH`**: Path to save evaluation results
- **`BATCH_SIZE`**: Batch size for inference (reduce if OOM)

### Data Preparation

| Task | Data Source | Location |
|------|-------------|----------|
| Math | `geoqa_test_prompts.jsonl` | `rllava/eval/jsons/math/` |
| Counting | `superclevr_test200_counting_problems.jsonl` | `rllava/eval/jsons/counting/` |
| Grounding | [Kangheng/refcoco](https://huggingface.co/datasets/Kangheng/refcoco) | Download separately |
| Agent-Code | [laolao77/MAT](https://huggingface.co/datasets/laolao77/MAT) | Download separately |
| Agent-Search | [laolao77/MAT](https://huggingface.co/datasets/laolao77/MAT) | Download separately |


## Evaluation Tasks

### Math
Geometry problem solving evaluation on GeoQA dataset.
```bash
bash examples/eval/math/eval_geoqa.sh
```

### Counting
Object counting evaluation on SuperCLEVR dataset.
```bash
bash examples/eval/counting/eval_counting_superclevr.sh
```

### Grounding (RefCOCO)
Visual grounding evaluation on RefCOCO/RefCOCO+/RefCOCOg. LISA_test is OOD.
```bash
bash examples/eval/grounding/eval_refcoco.sh
```

### Agent-Code
Code generation evaluation on MAT-Coding benchmark.
```bash
bash examples/eval/agent_code/eval_mat_coding.sh
```

### Agent-Search
Web search agent evaluation on MAT-Search benchmark.
```bash
bash examples/eval/agent_search/eval_mat_search.sh
```
