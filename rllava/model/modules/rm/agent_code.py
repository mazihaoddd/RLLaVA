import re
import os
import string

from datetime import datetime
from . import register_rm


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

@register_rm("agent_code_accuracy")
def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # # Try symbolic verification first
        # try:
        #     answer = parse(content)
        #     if float(verify(answer, parse(sol))) > 0:
        #         reward = 1.0
        # except Exception:
        #     pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                if '<answer>' in sol:
                    if '<code>' in content or '<problem>' in content:
                        reward = 0.0
                    else:
                        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                        
                        # Extract answer from content if it has think/answer tags
                        content_match = re.search(r'<answer>(.*?)</answer>', content)
                        student_answer = content_match.group(1).strip() if content_match else content.strip()

                        # Compare the extracted answers
                        reward = compute_f1(student_answer, ground_truth)
                    
                elif '<code>' in sol:
                    if '<answer>' in content or '<problem>' in content:
                        reward = 0.0
                    else:
                        sol_match = re.search(r'<code>(.*?)</code>', sol)
                        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                        
                        # Extract answer from content if it has think/answer tags
                        content_match = re.search(r'<code>(.*?)</code>', content)
                        student_answer = content_match.group(1).strip() if content_match else content.strip()

                        # if ground_truth == student_answer:
                        #     reward = 1.0
                        # else:
                        #     reward = 0.9
                        reward = 0.9
                
                elif '<problem>' in sol:
                    if '<answer>' in content or '<code>' in content:
                        reward = 0.0
                    else:
                        ground_truth = extract_problems(sol)
                        student_answer = extract_problems(content)

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
                    
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

@register_rm("agent_code_format")
def format_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion matches exactly one valid format."""
    pattern_answer = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern_code = r"<think>.*?</think>\s*<code>\s*```python(.*?)```.*?</code>"
    pattern_problem = r"^<think>.*?</think>\s*<problem>\s*\{\s*'[^']+'\s*(?:,\s*'[^']+'\s*)*\}\s*</problem>$"

    completion_contents = [completion[0]["content"] for completion in completions]
    solution_contents = [sol for sol in solution]

    rewards = []
    for content, solution in zip(completion_contents, solution_contents):
        if content.count("<answer>")>=2 or content.count("<code>")>=2 or content.count("<think>")>=2 or content.count("<problem>")>=2:
            rewards.append(0.0)
        elif '<answer>' in solution:
            match_answer = re.fullmatch(pattern_answer, content, re.DOTALL)
            if match_answer:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        elif '<code>' in solution:
            match_code = re.fullmatch(pattern_code, content, re.DOTALL)
            if match_code:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        elif '<problem>' in solution:
            match_problem = re.fullmatch(pattern_problem, content, re.DOTALL)
            if match_problem:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards
