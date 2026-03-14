import argparse
import re
import json
import torch
import string
import numpy as np
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on MAT-Coding dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the MAT-Coding.json file")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the MAT-Coding-image directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output results")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    return parser.parse_args()


def is_chinese(text):
    """判断文本是否含中文字符"""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)

def normalize(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        chinese_punc = "！？｡＂＃＄％＆＇（）＊＋，－．／：；＜＝＞＠［＼］＾＿｀｛｜｝～""''、。：《》【】"
        exclude = set(string.punctuation + chinese_punc)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    s = lower(s)
    s = remove_punc(s)

    if is_chinese(s):
        s = s.replace(" ", "")
    else:
        s = remove_articles(s)
        s = white_space_fix(s)

    return s

def compute_f1(prediction, ground_truth):
    if prediction is None:
        return 0.0

    norm_pred = normalize(prediction)
    norm_gt = normalize(ground_truth)

    if is_chinese(norm_pred) or is_chinese(norm_gt):
        pred_tokens = list(norm_pred)
        gt_tokens = list(norm_gt)
    else:
        pred_tokens = norm_pred.split()
        gt_tokens = norm_gt.split()

    common = set(pred_tokens) & set(gt_tokens)
    num_same = len(common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    if prediction is None:
        return 0.0
    return int(normalize(prediction) == normalize(ground_truth))

def extract_coordinates(text):
    pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    matches = re.findall(pattern, text)
    coordinates = [list(map(int, match)) for match in matches]
    return coordinates

def extract_and_run_code(input_str: str):
    match = re.search(r"<code>\s*```python(.*?)```.*?</code>", input_str, re.DOTALL)

    if match:
        code_str = match.group(1)
        local_vars = {}
        try:
            exec(code_str, globals(), local_vars)
            return True, local_vars
        except Exception as e:
            print(f"Code Error {e}")
            return False, e
    else:
        print("Code is not extracted successfully")
        return False, "Code is not extracted successfully"
    
def extract_problems(text):
    match = re.search(r"<problem>\s*\{(.*?)\}\s*</problem>", text, re.DOTALL)
    if not match:
        return [[]]

    content = match.group(1)
    problems = re.findall(r"'([^']+)'", content)
    return sorted(problems)


SYSTEM_PROMPT = """# Role
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


def main():
    args = parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=args.device,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.data_path, "r") as f:
        data = json.load(f)
    print(len(data))

    f1_all = []
    em_all = []
    results = []
    import time
    time1 = time.time()
    print(time1)
    
    for item in tqdm(data[:]):
        print("########################################")
        if item['type'][0] == 'crop':
            input_image_path = item['ori_image_path']
        else:
            input_image_path = item['processed_image_path']
        input_image_path = args.image_dir + '/' + input_image_path
        query = item['question']
        data_type = item['type']
        item_id = item['id']

        output_image_path = 'cache.jpg'
        input_text = SYSTEM_PROMPT + f'<query> {query} <query>'
        print(f'<query> {query} </query>')

        messages = [
            { "role": "user", 
                "content": [{"type": "image","image": input_image_path}, {"type": "text", "text": input_text}]}
        ]

        for _ in range(5):
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result = output_text[0]
            print(result)
            
            if '<answer>' in result:
                try:
                    print(result)
                    match = re.search(r"<answer>\s*(.*?)\s*</answer>", result)   
                    if match:
                        answer_content = match.group(1).strip()
                    else: 
                        answer_content = ""
                    print("Ground Truth: " + str(item['answers']))
                    print("Pred Answer: " + answer_content)
                    pred = answer_content
                    gts = item['answers']
                    if isinstance(gts, str):
                        gts = [gts]
                    f1 = max([compute_f1(pred, gt) for gt in gts])
                    em = max([exact_match_score(pred, gt) for gt in gts])
                    if em == 1:
                        f1 = 1
                    print("F1: " + str(f1))
                    print("EM: " + str(em))
                    f1_all.append(f1)
                    em_all.append(em)
                    results.append({'id': item_id, 'gt': gts, 'pred_answer': pred})
                    break
                except Exception as e:
                    f1 = 0
                    em = 0
                    print("F1: " + str(f1))
                    print("EM: " + str(em))
                    f1_all.append(f1)
                    em_all.append(em)
                    results.append({'id': item_id, 'gt': gts, 'pred_answer': None})
                    break
                    
            if '<problem>' in result:
                problems_list = extract_problems(result)
                if problems_list[0] == 'crop':
                    input_text = input_text + '\n' + result + '\n' + f'<tips> We now need to crop the image. Please provide the Python code. Use [x_min, y_min, x_max, y_max] to represent the bounding box coordinates. </tips>' + '\n'
                elif problems_list[0] == 'none':
                    input_text = input_text + '\n' + result + '\n' + f'<tips> The image has no issues, so no code is needed in the next step. You can directly provide the answer. </tips>'
                else:
                    input_text = input_text + '\n' + result + '\n' + f'<tips> Now that we have identified the issue in the image: {problems_list}, please proceed to address it by outputting the python code. </tips>' + '\n'
                messages = [
                    {"role": "user", "content": [{"type": "image","image": input_image_path}, {"type": "text", "text": input_text}]}
                ]
                
            if '<code>' in result:
                problems_list = extract_problems(result)
                if problems_list[0] != 'crop':
                    result_replace_path = result.replace('path_to_input_image.jpg', input_image_path).replace('path_to_output_image.jpg', output_image_path)
                    flage, code_results = extract_and_run_code(result_replace_path)
                else:
                    result_replace_path = result.replace('path_to_input_image.jpg', input_image_path).replace('path_to_output_image.jpg', output_image_path)
                    img_pil = Image.open(input_image_path).convert("RGB")
                    img = np.array(img_pil)
                    width, height = img_pil.size
                    bbox = extract_coordinates(item['question'])[0]
                    x_min = int(bbox[0] / 1000 * width)
                    y_min = int(bbox[1] / 1000 * height)
                    x_max = int(bbox[2] / 1000 * width)
                    y_max = int(bbox[3] / 1000 * height)
                    pattern = r'\[\s*((?:[^\[\]:]|(?:\[[^\[\]]*\]))+?)\s*:\s*((?:[^\[\]:]|(?:\[[^\[\]]*\]))+?)\s*,\s*((?:[^\[\]:]|(?:\[[^\[\]]*\]))+?)\s*:\s*((?:[^\[\]:]|(?:\[[^\[\]]*\]))+?)\s*\]'
                    replacement = f'[{y_min}:{y_max}, {x_min}:{x_max}]'
                    result_replace_path_code = re.sub(pattern, replacement, result_replace_path)
                    flage, code_results = extract_and_run_code(result_replace_path_code)
                    
                if flage == True:
                    input_text = input_text + '\n' + result
                    input_image_path = output_image_path
                    messages = [
                        {"role": "user", "content": [{"type": "image","image": input_image_path}, {"type": "text", "text": input_text}]}
                    ]
                else:
                    input_text = input_text + '\n' + result
                    messages = [
                        {"role": "user", "content": [{"type": "image","image": input_image_path}, {"type": "text", "text": input_text}]}
                    ]

    print('Simple F1:', sum(f1_all[:70])/70)
    print('Simple EM:', sum(em_all[:70])/70)
    print('Hard F1:', sum(f1_all[70:])/130)
    print('Hard EM:', sum(em_all[70:])/130)
    print('All F1:', sum(f1_all)/200)
    print('All EM:', sum(em_all)/200)
    time2 = time.time()
    print(time2)
    print(time2-time1)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
