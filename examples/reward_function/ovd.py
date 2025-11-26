# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any
import json

def extract_bbox(response):
    start_tag = "<answer>"
    end_tag = "</answer>"
    input_str = response
    # Check if the start tag is in the string
    if start_tag in input_str:
        # Extract the content between the start tag and end tag
        start_idx = input_str.find(start_tag) + len(start_tag)
        end_idx = input_str.find(end_tag)
        
        # If end_tag is not found (i.e., the string is truncated), assume it should be at the end
        if end_idx == -1:
            end_idx = len(input_str)
    
        content_str = input_str[start_idx:end_idx]
    
        # Check if it ends with a closing bracket, if not, fix it
        if not content_str.endswith("]"):
            # If the string is truncated, remove the incomplete part
            content_str = content_str.rsplit("},", 1)[0] + "}]"
    
        # Replace single quotes with double quotes for valid JSON
        content_str_corrected = content_str.replace("'", '"')
    
        # Convert the corrected string to a list of dictionaries (JSON format)
        try:
            bbox_list = json.loads(content_str_corrected)
        except json.JSONDecodeError as e:
            bbox_list = None
    else:
        bbox_list = None
    return bbox_list

def normalize_bbox(bbox):
    """统一bbox格式为 [x1, y1, x2, y2]"""
    if len(bbox) == 2 and isinstance(bbox[0], (list, tuple)):
        # 格式: [(x1, y1), (x2, y2)]
        return [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
    elif len(bbox) == 4:
        # 格式: [x1, y1, x2, y2]
        return bbox
    else:
        print(f"Unknown bbox format: {bbox}")
        return [0, 0, 0, 0]  # 返回默认值

def calculate_iou(bbox1, bbox2):
    bbox1 = normalize_bbox(bbox1)
    bbox2 = normalize_bbox(bbox2)
    x1, y1, x2, y2 = bbox1    
    x1_2, y1_2, x2_2, y2_2 = bbox2
    if isinstance(x1, str) or isinstance(y1, str) or isinstance(x2, str) or isinstance(y2, str):
        return 0.0
    if isinstance(x1_2, str) or isinstance(y1_2, str) or isinstance(x2_2, str) or isinstance(y2_2, str):
        return 0.0


    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area
    return iou

def get_position_tuple(d):
    # 兼容 Position/position/bbox/xyxy/xywh 等
    if not isinstance(d, dict):
        return [0, 0, 0, 0]
    for k in ('Position', 'position', 'bbox', 'Bbox'):
        if k in d and d[k] is not None:
            try:
                return d[k]
            except Exception:
                pass
    return [0, 0, 0, 0]   



def get_confidence(d, default=0.0):
    # 兼容多种字段名与大小写，并做类型转换
    for k in ("Confidence", "confidence", "score", "Score", "prob", "Prob", "probability", "Probability"):
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)

def sort_and_calculate_iou(list1, list2, iou_threshold=0.5):
    #list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    list2_sorted = sorted(list2, key=lambda x: get_confidence(x, 0.0), reverse=True)
    iou_results = []
    
    matched_list1_indices = set()

    for bbox2 in list2_sorted:
        max_iou = 0
        matched_bbox1 = -1
        best_iou = 0
        for i, bbox1 in enumerate(list1):
            if i not in matched_list1_indices:
                #iou = calculate_iou(bbox1['Position'], bbox2['Position'])
                iou = calculate_iou(get_position_tuple(bbox1), get_position_tuple(bbox2))
                if iou > best_iou:
                    best_iou = iou
                    matched_bbox1 = i

        if best_iou > iou_threshold:
            iou_results.append((best_iou, get_confidence(bbox2, 0.0)))
            matched_list1_indices.add(matched_bbox1)
        else:
            iou_results.append((0, get_confidence(bbox2, 0.0)))
    
    ### [(0.7192676547515258, 1.0), (0, 0.7)]
    return iou_results

def remove_duplicates(bbox_list):
    seen = set()
    unique_bboxes = []
    
    for bbox in bbox_list:
        # Convert the position tuple to a tuple for set hashing
        position_tuple = tuple(get_position_tuple(bbox))
        
        if position_tuple not in seen:
            seen.add(position_tuple)
            unique_bboxes.append(bbox)
    
    return unique_bboxes

def compute_reward_iou(iou_results, len_gt):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    if len_gt>=len(iou_results):
        iou_reward = iou_reward/len_gt
    else:
        iou_reward = iou_reward/len(iou_results)
    return iou_reward

def compute_reward_confidence(iou_results):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    iou_reward = iou_reward/len(iou_results)
    confidence_reward = confidence_reward/len(iou_results)
    return confidence_reward

def format_reward(response: str) -> float:
    """Reward function that checks if the completion has a specific format."""
    #pattern = r"<answer>.*?</answer>"
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.match(pattern, response)
    #print(format_match)
    return 1.0 if format_match else 0.0


def accuracy_reward_iou(response: str, ground_truth: str) -> float:
    student_answer = re.search(r'<answer>(.*?)</answer>', response)
    if hasattr(student_answer, 'group'):
        student_answer = student_answer.group(1)
    elif not isinstance(student_answer, str):
        return 0.0
    if "Position" not in student_answer or "Confidence" not in student_answer:
        return 0.0        
    student_answer = '<answer>'+student_answer+'</answer>'
    # fix format error
    student_answer = student_answer.replace("[[",'[')  
    student_answer = student_answer.replace("]]",']')  
    student_answer = student_answer.replace("\n",'')  
    # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
    ground_truth_bbox = extract_bbox(ground_truth)
    student_answer_bbox = extract_bbox(student_answer)
    # pdb.set_trace()
    if student_answer_bbox==None or len(student_answer_bbox)==0 or type(student_answer_bbox[0])!=dict:
        reward = 0.0
    else:
        student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
        iou_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
        ### new iou reward
        reward = compute_reward_iou(iou_results, len(ground_truth_bbox))
        if reward>1:
            reward = 1.0
    return reward

def accuracy_reward_confidence(response: str, ground_truth: str) -> float:
    student_answer = re.search(r'<answer>(.*?)</answer>', response)
    if hasattr(student_answer, 'group'):
        student_answer = student_answer.group(1)
    elif not isinstance(student_answer, str):
        return 0.0
    if "Position" not in student_answer or "Confidence" not in student_answer:
        return 0.0    
    student_answer = '<answer>'+student_answer+'</answer>'
    # fix format error
    student_answer = student_answer.replace("[[",'[')  
    student_answer = student_answer.replace("]]",']')  
    student_answer = student_answer.replace("\n",'')  
    # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
    ground_truth_bbox = extract_bbox(ground_truth)
    student_answer_bbox = extract_bbox(student_answer)
    # pdb.set_trace()
    if student_answer_bbox==None or len(student_answer_bbox)==0 or type(student_answer_bbox[0])!=dict:
        reward = 0.0
    else:
        student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
        iou_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
        reward = compute_reward_confidence(iou_results)
        if reward>1:
            reward = 1.0
        if reward<0:
            reward = 0.0
    return reward


def compute_score(reward_input: dict[str, Any], weight: float = 0.33) -> dict[str, float]:
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential` for grounding reward function.")
    #print(reward_input["response"])
    format_score = format_reward(reward_input["response"])
    accuracy_iou_score = accuracy_reward_iou(reward_input["response"], reward_input["ground_truth"])
    accuracy_confidence_score = accuracy_reward_confidence(reward_input["response"], reward_input["ground_truth"])

    return {
        "overall": weight * accuracy_iou_score + weight * accuracy_confidence_score + (1 - 2 * weight) * format_score,
        "format": format_score,
        "iou": accuracy_iou_score,
        "confidence":accuracy_confidence_score,
    }