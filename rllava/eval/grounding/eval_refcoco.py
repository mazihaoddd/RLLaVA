import argparse
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import json
import ray
from tqdm import tqdm
import re
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description="RefCOCO Grounding Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--anno_dir", type=str, required=True, help="Path to RefCOCO annotations directory")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to COCO images directory")
    parser.add_argument("--output_dir", type=str, default="logs", help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--sample_num", type=int, default=-1, help="Number of samples (for debugging)")
    parser.add_argument("--device_rank", type=int, default=0, help="Device rank for GPU")
    return parser.parse_args()


def parse_float_sequence_within(input_str):
    """Extract the first sequence of four floating-point numbers from the string"""
    patterns = [
        r"\[\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\]",
        r"\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\)",
        r"\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)\s*,\s*\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, input_str)
        if match:
            return [float(match.group(i)) for i in range(1, 5)]
    return [0.0, 0.0, 0.0, 0.0]

def extract_bbox_answer(content):
    """Extract bounding box from model output"""
    is_qwen2vl = 1
    bbox = parse_float_sequence_within(content)
    return (bbox if is_qwen2vl else [int(x*1000) for x in bbox]), is_qwen2vl

def compute_iou(box1, box2):
    """Compute IoU"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    
    return intersection / (area1 + area2 - intersection + 1e-10)

def compute_accuracy(box1, box2, threshold=0.5):
    """Compute accuracy"""
    return compute_iou(box1, box2) >= threshold

def compute_center_accuracy(box1, box2):
    """Compute center accuracy"""
    cx = (box2[0] + box2[2]) / 2
    cy = (box2[1] + box2[3]) / 2
    return (box1[0] <= cx <= box1[2]) and (box1[1] <= cy <= box1[3])


class RefCOCOEvaluator:
    """Evaluator class encapsulating the evaluation logic"""
    def __init__(self, args):
        self.args = args
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=f"cuda:{args.device_rank}", 
        )
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        
        self.scorers = {
            "IoU": compute_iou,
            "ACC@0.1": lambda x,y: compute_accuracy(x,y,0.1),
            "ACC@0.3": lambda x,y: compute_accuracy(x,y,0.3),
            "ACC@0.5": lambda x,y: compute_accuracy(x,y,0.5),
            "ACC@0.75": lambda x,y: compute_accuracy(x,y,0.75),
            "ACC@0.95": lambda x,y: compute_accuracy(x,y,0.95),
            "Center_ACC": compute_center_accuracy,
        }
    
    def evaluate_batch(self, batch_data, batch_messages):
        """Evaluate a single batch of data"""
        text = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(f"cuda:{self.args.device_rank}")
        
        outputs = self.model.generate(
            **inputs,
            use_cache=True, 
            max_new_tokens=256, 
            do_sample=True
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        outputs_decoded = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        batch_results = []
        scores = {k: 0.0 for k in self.scorers}
        
        for example, output in zip(batch_data, outputs_decoded):
            pred_box, is_normalized = extract_bbox_answer(output)
            gt_box = example['normalized_solution'] if is_normalized else example['solution']
            
            result = {
                'question': example['problem'],
                'ground_truth': gt_box,
                'model_output': output,
                'extracted_answer': pred_box,
                'scores': {}
            }
            
            for name, scorer in self.scorers.items():
                score = scorer(gt_box, pred_box)
                result['scores'][name] = score
                scores[name] += score
            
            batch_results.append(result)
        
        return batch_results, scores, len(batch_data)


def main():
    args = parse_args()
    # List of datasets to process.
    ALL_DATASETS = [
        'refcoco_val', 'refcoco_testA', 'refcoco_testB',
        'refcocop_val', 'refcocop_testA', 'refcocop_testB',
        'refcocog_val', 'refcocog_test', 'lisa_test_rl3'
    ]
    
    evaluator = RefCOCOEvaluator(args)
    
    for ds in ALL_DATASETS:
        print(f"Processing {ds}...")
        ds_path = os.path.join(args.anno_dir, f"{ds}.json")
        if not os.path.exists(ds_path):
            print(f"Skipping {ds}: file not found")
            continue
            
        data = json.load(open(ds_path, "r"))
        if args.sample_num > 0:
            data = data[:args.sample_num]
        
        messages = []
        for x in data:
            img_path = os.path.join(args.image_dir, x['image'])
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": f"{x['problem']} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."}
                ]
            }])
        
        batch_size = args.batch_size
        final_results = []
        total_samples = 0
        scores = {k: 0.0 for k in ["IoU", "ACC@0.1", "ACC@0.3", "ACC@0.5", "ACC@0.75", "ACC@0.95", "Center_ACC"]}
        
        for i in tqdm(range(0, len(data), batch_size)):
            batch_end = min(i + batch_size, len(data))
            batch_data = data[i:batch_end]
            batch_messages = messages[i:batch_end]
            
            batch_results, batch_scores, batch_len = evaluator.evaluate_batch(batch_data, batch_messages)
            final_results.extend(batch_results)
            for k in scores:
                scores[k] += batch_scores[k]
            total_samples += batch_len
        
        avg_scores = {k: round(v/total_samples*100, 2) for k,v in scores.items()}
        result = {
            'dataset': ds,
            'average_scores': avg_scores,
            'details': final_results
        }
        
        output_path = os.path.join(args.output_dir, os.path.basename(args.model_path), 'grounding', ds)
        os.makedirs(output_path, exist_ok=True)
        
        with open(os.path.join(output_path, 'rec_results.json'), 'w') as f:
            json.dump({
                'model': args.model_path,
                'config': vars(args),
                **result
            }, f, indent=2)
        
        print(f"\nResults for {ds}:")
        for k,v in result['average_scores'].items():
            print(f"{k}: {v}%")
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
