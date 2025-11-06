import re
import os
import string

from math_verify import parse, verify
from datetime import datetime
from . import register_rm
from sentence_transformers import SentenceTransformer, util



_sentence_transformers_model = None


def get_sentence_transformer_model():
    global _sentence_transformers_model
    if _sentence_transformers_model is None:
        _sentence_transformers_model = SentenceTransformer('all-MiniLM-L6-v2')  # light weight
    return _sentence_transformers_model

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

def compute_similarity(prediction, ground_truth, sentence_tranformers_model=None):
    if sentence_tranformers_model is None:
        sentence_tranformers_model = get_sentence_transformer_model()
    emb1 = sentence_tranformers_model.encode(prediction, convert_to_tensor=True)
    emb2 = sentence_tranformers_model.encode(ground_truth, convert_to_tensor=True)
    cosine_score = util.cos_sim(emb1, emb2)
    return float(cosine_score)

def extract_problems(text):
    match = re.search(r"<problem>\s*\{(.*?)\}\s*</problem>", text, re.DOTALL)
    if not match:
        return []

    content = match.group(1)
    # 提取所有用英文单引号包裹的单词
    problems = re.findall(r"'([^']+)'", content)
    return sorted(problems)

@register_rm("agent_search_accuracy")
def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                if '<answer>' in sol:
                    if '<search>' in content:
                        reward = 0.0
                    else:
                        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                        
                        # Extract answer from content if it has think/answer tags
                        content_match = re.search(r'<answer>(.*?)</answer>', content)
                        student_answer = content_match.group(1).strip() if content_match else content.strip()
                        
                        # Compare the extracted answers
                        reward = compute_f1(student_answer, ground_truth)
                    
                elif '<search>' in sol:
                    if '<answer>' in content:
                        reward = 0.0
                    else:
                        sol_match = re.search(r'<search>(.*?)</search>', sol)
                        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                        
                        # Extract answer from content if it has think/answer tags
                        content_match = re.search(r'<search>(.*?)</search>', content)
                        student_answer = content_match.group(1).strip() if content_match else content.strip()
                        
                        # Compare the extracted answers
                        reward = compute_similarity(student_answer, ground_truth)

                        if reward<0.5:
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

@register_rm("agent_search_format")
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion matches exactly one valid format."""
    pattern_answer = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern_search = r"<think>.*?</think>\s*<search>.*?</search>"

    completion_contents = [completion[0]["content"] for completion in completions]

    rewards = []
    for content in completion_contents:
        if content.count("<answer>")>=2 or content.count("<search>")>=2 or content.count("<think>")>=2:
            rewards.append(0.0)
            # print('1111111111111111', content.count("<answer>"), content.count("<search>"), content.count("<think>"))
        elif '<answer>' in content and '</answer>' in content:
            match_answer = re.fullmatch(pattern_answer, content, re.DOTALL)
            match_search = re.fullmatch(pattern_search, content, re.DOTALL)
            if match_answer and not match_search:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
                # print('2222222222222222', match_answer, match_search, content)
        elif '<search>' in content and '</search>' in content:
            match_answer = re.fullmatch(pattern_answer, content, re.DOTALL)
            match_search = re.fullmatch(pattern_search, content, re.DOTALL)
            if match_search and not match_answer:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
                # print('3333333333333333', match_answer, match_search)
        else:
            rewards.append(0.0)
            # print('4444444444444444', content)
    return rewards
