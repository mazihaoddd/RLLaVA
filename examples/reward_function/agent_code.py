import re
from typing import Any
import json
import string

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

def extract_problems(text):
    match = re.search(r"<problem>\s*\{(.*?)\}\s*</problem>", text, re.DOTALL)
    if not match:
        return []

    content = match.group(1)
    # 提取所有用英文单引号包裹的单词
    problems = re.findall(r"'([^']+)'", content)
    return sorted(problems)


def accuracy_reward(response: str, ground_truth: str) -> float:
    try:
        # Extract answer from solution if it has think/answer tags
        if '<answer>' in ground_truth:
            if '<code>' in response or '<problem>' in response:
                reward = 0.0
            else:
                sol_match = re.search(r'<answer>(.*?)</answer>', ground_truth)
                ground_truth = sol_match.group(1).strip() if sol_match else ground_truth.strip()
                
                # Extract answer from response if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', response)
                student_answer = content_match.group(1).strip() if content_match else response.strip()

                # Compare the extracted answers
                reward = compute_f1(student_answer, ground_truth)
            
        elif '<code>' in ground_truth:
            if '<answer>' in response or '<problem>' in response:
                reward = 0.0
            else:
                sol_match = re.search(r'<code>(.*?)</code>', ground_truth)
                ground_truth = sol_match.group(1).strip() if sol_match else ground_truth.strip()
                
                # Extract answer from response if it has think/answer tags
                content_match = re.search(r'<code>(.*?)</code>', response)
                student_answer = content_match.group(1).strip() if content_match else response.strip()

                # if ground_truth == student_answer:
                #     reward = 1.0
                # else:
                #     reward = 0.9
                reward = 0.9
        
        elif '<problem>' in ground_truth:
            if '<answer>' in response or '<code>' in response:
                reward = 0.0
            else:
                ground_truth = extract_problems(ground_truth)
                student_answer = extract_problems(response)

                # Half correct
                if len(ground_truth) == 2 and 'rotation90' in ground_truth:
                    if len(student_answer) == 1 and 'rotation90' in student_answer:
                        reward = 0.5
                elif len(ground_truth) == 2 and 'rotation180' in ground_truth:
                    if len(student_answer) == 1 and 'rotation180' in student_answer:
                        reward = 0.5
                # All correct
                if reward == 0 : 
                    is_equal = ground_truth == student_answer
                    if is_equal:
                        reward = 1.0
                    else:
                        reward = 0.0
        return reward    
    except Exception:
        pass
    
    return 0.0


def format_reward(response: str) -> float:
    """Reward function that checks if the completion has a specific format."""
    
    pattern_answer = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern_code = r"<think>.*?</think>\s*<code>\s*```python(.*?)```.*?</code>"
    pattern_problem = r"^<think>.*?</think>\s*<problem>\s*\{\s*'[^']+'\s*(?:,\s*'[^']+'\s*)*\}\s*</problem>$"
        
    """
    pattern_answer = r"<answer>.*?</answer>"
    pattern_code = r"<code>\s*```python(.*?)```.*?</code>"
    pattern_problem = r"<problem>\s*\{\s*'[^']+'\s*(?:,\s*'[^']+'\s*)*\}\s*</problem>$"
    """

    if response.count("<answer>")>=2 or response.count("<code>")>=2 or response.count("<think>")>=2 or response.count("<problem>")>=2:
        return 0.0
    elif '<answer>' in response:
        match_answer = re.match(pattern_answer, response, re.DOTALL)
        if match_answer:
            return 1.0
        else:
            return 0.0
    elif '<code>' in response:
        match_code = re.match(pattern_code, response, re.DOTALL)
        if match_code:
            return 1.0
        else:
            return 0.0
    elif '<problem>' in response:
        match_problem = re.match(pattern_problem, response, re.DOTALL)
        if match_problem:
            return 1.0
        else:
            return 0.0
    else:
        return 0.0

def compute_score(reward_input: dict[str, Any], format_weight: float = 0.5) -> dict[str, float]:
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential` for r1v reward function.")
    #print(reward_input["response"])
    format_score = format_reward(reward_input["response"])
    accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }