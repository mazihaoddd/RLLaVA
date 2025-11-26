import re
import os
import warnings

from datetime import datetime
from . import register_rm


def log(content, sol, other_info, reward, tag=None):
    log_dir = os.getenv("LOG_PATH", None)
    os.makedirs(log_dir, exist_ok=True)
    if log_dir is None:
        warnings.warn("LOG_DIR is not set, log will not be saved")
        return
    log_path = os.path.join(log_dir, f"{tag}.log")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    with open(log_path, "a") as f:
        try:
            f.write(f"------------- {current_time} {tag} reward: {reward} -------------\n")
            f.write(f"Content: {content}\n")
            f.write(f"Solution: {sol}\n")
            if other_info is not None:
                for k, v in other_info.items():
                    f.write(f"{k}: {v}\n")
        except:
            f.write("writing error")

def parse_float_sequence_within(input_str):
    """
    Extract the first sequence of four floating-point numbers within square brackets from a string.

    Args:
    input_str (str): A string that may contain a sequence of four floats within square brackets.

    Returns:
    list: A list of four floats if the pattern is found, or a list of four zeros if the pattern is not found.
    """
    # Define the regex pattern to find the first instance of four floats within square brackets
    # TODO: add more patterns to support various formats
    # pattern1 [num, num, num, num]
    pattern = r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]"

    # Use re.search to find the first match of the pattern in the input string
    match = re.search(pattern, input_str)

    # If a match is found, convert the captured groups into a list of floats
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    # pattern2 (num, num, num, num)
    pattern = r"\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\)"
    match = re.search(pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    # pattern3 (num, num), (num, num)
    pattern = r"\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\),\s*\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)"
    match = re.search(pattern, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    # If the input does not contain the pattern, return the null float sequence
    return [0, 0, 0, 0]

def extract_bbox_answer(content):
    is_qwen2vl = False
    if "<|box_start|>" in content:
        is_qwen2vl = True
    bbox = parse_float_sequence_within(content)
    if not is_qwen2vl:
        bbox = [int(x * 1000) for x in bbox]
    return bbox, is_qwen2vl

def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou

@register_rm("grounding_format")
def grounding_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<|object_ref_start|>.*?<|object_ref_end|><|box_start|>.*?<|box_end|><|im_end|>"
    completion_contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

@register_rm("grounding")
def grounding_reward(completions, solution, **kwargs):
    rewards = []
    contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    for completion, sol in zip(contents, solution):
        bbox, is_qwen2vl = extract_bbox_answer(completion)
        iou = compute_iou(bbox, eval(sol))
        rewards.append(iou**2)
        log(completion + f"\nBounding box: {bbox}", sol, None, iou**2, "grounding_reward")
    return rewards