import argparse
import os
import sys
import re
import json
import torch
import string
import logging
import multiprocessing as mp
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Import web_search function - adjust path as needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from rllava.eval.agent_search.tools.web_search import web_search_BOCHA_API

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on MAT-Search dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the MAT-Search.json file")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the MAT-Search-image directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output results")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    return parser.parse_args()


def normalize(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
    if prediction is None:
        return 0.0
    prediction_tokens = normalize(prediction).split()
    ground_truth_tokens = normalize(ground_truth).split()

    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = len(common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    if prediction is None:
        return 0.0
    return int(normalize(prediction) == normalize(ground_truth))

def evaluate(predictions):
    total = len(predictions)
    f1_total = []
    em_total = []

    for item in predictions:
        pred = item['pred_answer']
        gts = item['gt']

        if isinstance(gts, str):
            gts = [gts]

        f1 = max([compute_f1(pred, gt) for gt in gts])
        em = max([exact_match_score(pred, gt) for gt in gts])
        if em == 1:
            f1 = 1

        f1_total.append(f1)
        em_total.append(em)

    return {
        "avg_f1": sum(f1_total) / total if total > 0 else 0,
        "avg_em": sum(em_total) / total if total > 0 else 0,
        "simple_f1": sum(f1_total[:75]) / 75 if total > 0 else 0,
        "simple_em": sum(em_total[:75]) / 75 if total > 0 else 0,
        "hard_f1": sum(f1_total[75:]) / 75 if total > 0 else 0,
        "hard_em": sum(em_total[75:]) / 75 if total > 0 else 0,
    }


SYSTEM_PROMPT = """# Role  
You are a step-by-step multimodal reasoning assistant.  
Given an image, a question, and optional partial reasoning chain, your task is to solve the problem **one substep at a time**.  

# Guiding Principles  
At each turn, you must **either**:  
1. Issue **one specific, text-only search** enclosed in <search> </search> tags,  
2. Or provide the **final answer** enclosed in <answer> </answer> tags.  

All outputs **must begin with a thought** enclosed in <think> </think> tags, explaining your current reasoning and what to do next.  

- Do not reference "the image" in your searches.  
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


def run(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=args.device,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = model.eval()

    with open(args.data_path, 'r') as file:
        wikimultihopqa = json.load(file)
    print(len(wikimultihopqa))

    combine_results = []
    for i in tqdm(range(len(wikimultihopqa))):
        pred_answer = None
        query = wikimultihopqa[i]['question']
        answer = wikimultihopqa[i]['answer']
        image_path = wikimultihopqa[i]['image_path']
        image_path = args.image_dir + '/' + image_path
        item_id = wikimultihopqa[i]['id']
        input_text = SYSTEM_PROMPT + '\n' + f'<query> {query} </query>'

        iterative_num = 0
        while iterative_num < 5:
            iterative_num += 1
            messages = [
                { 
                "role": "user", 
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": input_text}
                    ]
                }
            ]
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

            generated_ids = model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result = output_text[0]

            if '<answer>' in result:
                match = re.search(r"<answer>\s*(.*?)\s*</answer>", result)
                if match:
                    pred_answer = match.group(1).strip()
                break
            elif '<search>' in result:
                match = re.search(r"<search>\s*(.*?)\s*</search>", result)
                if match:
                    search_content = match.group(1).strip()
                search_results = web_search_BOCHA_API(search_content, 4)
                format_search_results = '<information> '
                for index, item in enumerate(search_results):
                    format_search_results += f'{index+1}.' + ' ' + "Content:" + item['body']
                format_search_results += ' </information> '
            
                input_text = input_text + '\n' + result + '\n' + format_search_results

        if pred_answer is not None:
            combine_results.append(
                {'id': item_id, 'pred_answer': pred_answer, 'gt': answer, 'query': query}
            )
        else:
            combine_results.append(
                {'id': item_id, 'pred_answer': None, 'gt': answer, 'query': query}
            )
    return combine_results


def main():
    args = parse_args()
    mp.set_start_method('spawn', force=True)
    logger.info('started generation')
    result_lists = run(args)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result_lists, f, ensure_ascii=False, indent=4)
    
    count_none = sum(1 for item in result_lists if item['pred_answer'] is None)
    print(f"None count: {count_none}")
    
    results = evaluate(result_lists)
    print(*[f"{results[k]*100:.2f}" for k in ['simple_f1', 'simple_em', 'hard_f1', 'hard_em', 'avg_f1', 'avg_em']])
    logger.info("Done")
    logger.info('finished running')


if __name__ == "__main__":
    main()
