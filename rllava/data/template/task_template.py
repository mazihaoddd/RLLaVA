from dataclasses import dataclass
from .base import Template
from . import register_template
# from transformers.models.qwen2_5_vl.
    
# SYSTEM PROMPTS
AGENT_SEARCH_SYS = """# Role  
You are a step-by-step multimodal reasoning assistant.  
Given an image, a question, and optional partial reasoning chain, your task is to solve the problem **one substep at a time**.  

# Guiding Principles  
At each turn, you must **either**:  
1. Issue **one specific, text-only search** enclosed in <search> </search> tags,  
2. Or provide the **final answer** enclosed in <answer> </answer> tags.  

All outputs **must begin with a thought** enclosed in <think> </think> tags, explaining your current reasoning and what to do next.  

- Do not reference “the image” in your searches.  
- Do not repeat past queries.  
- Only output **one action per step**: either <search> or <answer>, never both.  
- When ready to conclude, summarize reasoning and give a final answer.

# Output Format (strict):  
Always start with <think>. Do not output the previous reasoning chain. Then, depending on the case, output one of the following:

## 1. If reasoning continues:  
<think> Your current reasoning and next plan </think>  
<search> One precise, retrievable textual query </search>

## 2. If ready to conclude:  
<think> Summarize all reasoning and derive the answer </think>  
<answer> Final answer, as briefly as possible </answer>

# Current reasoning chain:
"""

AGENT_CODE_SYS = """# Role
You are a step-by-step image processing assistant.
Your task is to solve an image-based task by applying OpenCV operations one step at a time, optionally using a reasoning chain.

# Output Format
At each step, output **only one** of the following, preceded by a <think> tag:
1. <problem> Describe the image issue from {'rotation90', 'rotation180', 'dark', 'overexposure', 'blur', 'noise', 'crop', 'none'} </problem>
2. <code> OpenCV code to process and save the image </code>
3. <answer> Final answer based on the processed image </answer>

# Image Processing Rules
- Always read from `'path_to_input_image.jpg'` and write to `'path_to_output_image.jpg'`.

# Output Format (strict):
Always begin with <think>. Then, depending on current reasoning chain, output one of the following:

## 1. If this is the first step and only the query is given, output in the following format:
<think> Initial analysis of the image issue. </think>
<problem> {'problem1', ...} </problem>

## 2. If <problem> is given, continue with image operations:
<think> Explain what to fix next. </think>
<code>
```python
One Python code block using OpenCV to perform the operation, and save the processed images.
```
</code>

## 3. If ready to conclude:
<think> Summarize the processing steps and provide the result or outcome </think> 
<answer> Final answer, as briefly as possible</answer>

# Current reasoning chain:
"""

LLAVA_SYS = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

QWEN2_SYS = (
    "You are a helpful assistant."
)

R1V_SYS = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

system_prompt_registry = {
    "default": QWEN2_SYS,
    "llava": LLAVA_SYS,
    "qwen": QWEN2_SYS,
    "r1v": R1V_SYS,
    "agent_code": AGENT_CODE_SYS,
    "agent_search": AGENT_SEARCH_SYS,
}

question_template_registry = {
    "default": "{question}",
    "r1v": "{question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.",
    "grounding": "Output the bounding box of the {question} in the image.",
    "counting": "Output all the bounding boxes of the {question}",
    "detection": "Please output bbox coordinates and names of person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.",
}
 
answer_template_registry = {
    "default": "{answer}",
    "r1v": "<answer> {answer} </answer>",
    "grounding": "{answer}",
    "counting": "{answer}",
    "detection": "{answer}",
}

@register_template('task')
@dataclass
class TaskTemplate(Template):

    def __init__(self, system: str, question: str, answer: str):
        self.system = system_prompt_registry[system]
        self.question = question_template_registry[question]
        self.answer = answer_template_registry[answer]
        if system.startswith("agent"):
            self.task_type = system
        else:
            self.task_type = "default"

    def make_inputs(self, example, image, problem_key, answer_key):
        if self.task_type == "agent_search":
            solution = example['solution']

            def make_conversation_image(example):
                prompt = example['problem'] 
                return [
                    {
                        "role": "user", 
                         "content": [
                            {"type": "image"},
                            {"type": "text", "text": self.system + '\n' + prompt},
                        ]
                    }
                ]
            formatted_conversation = make_conversation_image(example)
            return {"image": image, "prompt": formatted_conversation, 'solution': solution}
        elif self.task_type == "agent_code":
            solution = example['solution']
            gt = example['gt']

            def make_conversation_image(example):
                prompt = example['problem']       
                if example['type'] == "pre_problem": 
                    return [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": self.system + '\n' + prompt},
                            ],
                        }
                    ]
                elif example['type'] == "pre_code":
                    context = example['context']
                    return [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": self.system + '\n' + prompt + '\n' + context},
                            ],
                        },
                    ] 
                elif example['type'] == "pre_answer":
                    context = example['context']
                    return [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": self.system + '\n' + prompt + '\n' + context},
                            ],
                        },
                    ]
            formatted_conversation = make_conversation_image(example)
            return {"image": image, "prompt": formatted_conversation, 'solution': solution, 'gt': gt}
        else:
            def make_conversation(example):
                return [{"role": "system", "content": self.system},
                        {"role": "user", "content": self.question.format(question=example[problem_key])}]

            def make_conversation_image(example):
                return [{"role": "system", "content": self.system},
                        {"role": "user", "content": [
                            {"type": "image"},
                            {"type": "text", "text": self.question.format(question=example[problem_key])},
                        ]}]
            prompt = make_conversation(example) if 'image' not in example else make_conversation_image(example)

            return {
                'image': image,
                'problem': example[problem_key],
                'solution': example[answer_key],
                'prompt': prompt,
            }











