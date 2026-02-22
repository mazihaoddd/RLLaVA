"""
Reward function for We-Math dataset.

Supports two scenarios:
- Training set: ground_truth is a numerical answer -> uses mathruler for fuzzy matching
- Test set: ground_truth is a JSON-encoded dict with option letter + metadata -> letter matching + four-dimensional metrics

Usage in config:
    reward.reward_function=./examples/reward_function/wemath.py:compute_score
    reward.reward_function=./examples/reward_function/wemath.py:compute_score_without_format
    reward.reward_type=sequential
"""

import re
import json
from typing import Any
from collections import defaultdict

from mathruler.grader import grade_answer


def _parse_ground_truth(gt_raw: str):
    """Parse ground_truth which may be a plain string or JSON-encoded dict.

    Returns:
        (answer, ID, key) where ID and key are None for plain string ground_truth.
    """
    try:
        gt_info = json.loads(gt_raw)
        if isinstance(gt_info, dict) and "answer" in gt_info:
            return gt_info["answer"], gt_info.get("ID"), gt_info.get("key")
    except (json.JSONDecodeError, TypeError):
        pass
    return gt_raw, None, None


def _is_option_letter(s: str) -> bool:
    """Check if string is a single option letter (A-H)."""
    return len(s) == 1 and s in "ABCDEFGH"


def format_reward(response: str) -> float:
    """Check if response follows <think>...</think><answer>...</answer> format."""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def _extract_option_letter(text: str) -> str:
    """Extract option letter from text using multiple strategies (most specific first).

    Strategy 1: <answer>B</answer> tag
    Strategy 2: "Answer" keyword (official We-Math extraction)
    Strategy 3: \\boxed{B}
    Strategy 4: Last standalone option letter in the response
    """
    if not text:
        return ""

    # Strategy 1: <answer> tag
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        if inner and inner[0].upper() in "ABCDEFGH":
            return inner[0].upper()

    # Strategy 2: Official We-Math "Answer" keyword extraction
    parts = text.split("Answer")
    if len(parts) > 1:
        after = re.sub(r'[>><<:.]', '', parts[-1]).strip()
        if after and after[0].upper() in "ABCDEFGH":
            return after[0].upper()

    # Strategy 3: \boxed{B}
    m = re.search(r"\\boxed\{([A-Ha-h])\}", text)
    if m:
        return m.group(1).upper()

    # Strategy 4: Last standalone option letter (e.g., "The answer is B")
    m = re.findall(r'\b([A-E])\b', text)
    if m:
        return m[-1].upper()

    return ""


def _split_numbered_parts(text: str) -> list[str]:
    """Split text by (1), (2), ... numbered markers.

    Returns list of individual parts if sequential numbering is found,
    otherwise returns [text] unchanged.
    """
    markers = list(re.finditer(r'(?:^|\s)\((\d+)\)', text))
    if len(markers) < 2:
        return [text]

    nums = [int(m.group(1)) for m in markers]
    if nums != list(range(1, len(nums) + 1)):
        return [text]

    parts = []
    for i, m in enumerate(markers):
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        part = text[start:end].strip()
        if part:
            parts.append(part)

    return parts if len(parts) >= 2 else [text]


_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'it', 'its', 'this', 'that',
    'these', 'those', 'not', 'and', 'but', 'if', 'then', 'so', 'because',
    'there', 'here', 'all', 'each', 'every', 'both', 'few', 'more',
    'some', 'any', 'no', 'than', 'too', 'very', 'just', 'about',
})


def _token_grade(pred_token: str, gt_token: str) -> bool:
    """Match two individual answer tokens.

    Tries grade_answer first (handles LaTeX, numerical equivalence),
    then falls back to case-insensitive string comparison (handles words like "Cone").
    """
    if grade_answer(pred_token, gt_token):
        return True
    return pred_token.strip().lower() == gt_token.strip().lower()


def _match_multi_format_answer(prediction: str, ground_truth: str) -> float:
    """Match prediction against a potentially multi-part / multi-format ground truth.

    Handles:
    1. Numbered sub-answers:  (1) ... (2) ...    -> partial credit per sub-answer
    2. "or" alternatives:     50° or 130°        -> any match = 1.0
    3. Comma-separated:       8, 6, 37.68        -> partial credit per token
    4. Space-separated:       Cone 5 3 47.1      -> partial credit per token
    5. Single answer fallback                     -> grade_answer
    """
    gt = ground_truth.strip()
    pred = prediction.strip()

    # --- 0. Direct match first (fast path) ---
    if grade_answer(pred, gt):
        return 1.0

    # --- 1. Numbered sub-answers: (1) ... (2) ... ---
    gt_parts = _split_numbered_parts(gt)
    if len(gt_parts) >= 2:
        pred_parts = _split_numbered_parts(pred)
        if len(pred_parts) == len(gt_parts):
            matches = sum(1 for p, g in zip(pred_parts, gt_parts) if grade_answer(p, g))
            return matches / len(gt_parts)
        else:
            # Model didn't use (1)(2) format; try matching each GT part in full prediction
            matches = sum(1 for g in gt_parts if grade_answer(pred, g))
            return matches / len(gt_parts)

    # --- 2. "or" / "OR" alternatives ---
    if re.search(r'\bor\b', gt, re.IGNORECASE):
        alternatives = re.split(r'\s+or\s+', gt, flags=re.IGNORECASE)
        alternatives = [a.strip() for a in alternatives if a.strip()]
        if len(alternatives) >= 2:
            return 1.0 if any(grade_answer(pred, alt) for alt in alternatives) else 0.0

    # --- 3. Comma-separated multi-value: "8, 6, 37.68" / "17, seventeen" ---
    # Guard: skip if GT contains LaTeX commands that use commas internally
    if ',' in gt and not re.search(r'\\frac|\\binom|\\sqrt', gt):
        gt_parts = [p.strip() for p in gt.split(',') if p.strip()]
        if len(gt_parts) >= 2:
            # Tokenize prediction: try comma-split first, then space-split as fallback
            if ',' in pred:
                pred_parts = [p.strip() for p in pred.split(',') if p.strip()]
            else:
                pred_parts = pred.split()
            if not pred_parts:
                pred_parts = [pred]
            matches = sum(1 for g in gt_parts
                          if any(_token_grade(p, g) for p in pred_parts))
            return matches / len(gt_parts)

    # --- 4. Space-separated multi-token: "Cone 5 3 47.1" / "1 2 3" ---
    # Guard: skip natural-language sentences (contain stop words) and very long text
    space_parts = gt.split()
    if (len(space_parts) >= 2
            and len(space_parts) <= 8
            and not any(t.lower() in _STOP_WORDS for t in space_parts)):
        # Normalize prediction delimiters (commas -> spaces) and tokenize
        pred_tokens = pred.replace(',', ' ').split()
        if not pred_tokens:
            pred_tokens = [pred]
        matches = sum(1 for g in space_parts
                      if any(_token_grade(pt, g) for pt in pred_tokens))
        return matches / len(space_parts)

    # --- 5. Fallback: already tried direct match above ---
    return 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Compute accuracy reward.

    - If ground_truth is a single letter (A-H): extract option letter with multi-strategy fallback.
    - Otherwise: extract content from <answer> tag, use multi-format matching for
      numbered, "or", comma-separated, and space-separated answer patterns.
    """
    try:
        gt = ground_truth.strip()
        if _is_option_letter(gt):
            extracted = _extract_option_letter(response)
            return 1.0 if extracted == gt else 0.0
        else:
            content_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            given_answer = content_match.group(1).strip() if content_match else response.strip()
            gt_match = re.search(r"<answer>(.*?)</answer>", gt, re.DOTALL)
            gt = gt_match.group(1).strip() if gt_match else gt
            return _match_multi_format_answer(given_answer, gt)
    except Exception:
        pass

    return 0.0


# ---------------------------------------------------------------------------
# We-Math four-dimensional metrics (adapted from official evaluation code)
# ---------------------------------------------------------------------------

def _compute_four_dim_metrics(results: list[dict]) -> dict[str, float]:
    """Compute We-Math four-dimensional metrics from a list of per-sample results.

    Each result dict should have: ID, key, correct (bool).
    Returns metric dict with keys like wemath_score_strict, wemath_IK, etc.
    """
    if not results:
        return {}

    import pandas as pd

    df = pd.DataFrame(results)

    # Only process samples that have key field (test set)
    if "key" not in df.columns or df["key"].isna().all():
        return {}

    has_2steps = df["key"].str.contains("2steps").any()
    has_3steps = df["key"].str.contains("3steps").any()

    if not has_2steps and not has_3steps:
        return {}

    def _process_steps(df_steps, steps):
        """Group by step key, merge sub-problems, compute per-group metrics."""
        steps_data = {}
        for i in range(1, steps + 1):
            steps_data[f"{steps}steps_{i}"] = df_steps[df_steps["key"] == f"{steps}steps_{i}"].copy()
        steps_data[f"{steps}steps_multi"] = df_steps[df_steps["key"] == f"{steps}steps_multi"].copy()

        for k, data in steps_data.items():
            suffix = k.split("_")[-1]
            data.columns = [col + f"_{suffix}" for col in data.columns]

        merged = steps_data[f"{steps}steps_1"]
        for i in range(2, steps + 1):
            merged = pd.merge(merged, steps_data[f"{steps}steps_{i}"],
                              left_on="ID_1", right_on=f"ID_{i}", how="left")
        merged = pd.merge(merged, steps_data[f"{steps}steps_multi"],
                          left_on="ID_1", right_on="ID_multi", how="left")
        return merged

    merged_2steps = pd.DataFrame()
    merged_3steps = pd.DataFrame()

    if has_2steps:
        df_2 = df[df["key"].str.contains("2steps")]
        if len(df_2) > 0:
            merged_2steps = _process_steps(df_2, 2)

    if has_3steps:
        df_3 = df[df["key"].str.contains("3steps")]
        if len(df_3) > 0:
            merged_3steps = _process_steps(df_3, 3)

    # Calculate four dimensions
    IG, IK = 0, 0
    RM_loose, RM_strict = 0, 0
    CM_loose, CM_strict = 0, 0

    # Use == True / == False comparisons to match official behavior:
    # NaN values (from incomplete groups after merge) are excluded from ALL
    # categories, since NaN == True -> False, NaN == False -> False.
    if len(merged_2steps) > 0:
        c1 = merged_2steps.get("correct_1", pd.Series(dtype=object))
        c2 = merged_2steps.get("correct_2", pd.Series(dtype=object))
        cm = merged_2steps.get("correct_multi", pd.Series(dtype=object))

        if len(c1) > 0 and len(c2) > 0 and len(cm) > 0:
            RM_loose += len(merged_2steps[(c1 == False) & (c2 == False) & (cm == True)])
            RM_strict += len(merged_2steps[((c1 == False) | (c2 == False)) & (cm == True)])
            IG += len(merged_2steps[(c1 == True) & (c2 == True) & (cm == False)])
            IK += len(merged_2steps[((c1 == False) | (c2 == False)) & (cm == False)])
            CM_loose += len(merged_2steps[((c1 == True) | (c2 == True)) & (cm == True)])
            CM_strict += len(merged_2steps[(c1 == True) & (c2 == True) & (cm == True)])

    if len(merged_3steps) > 0:
        c1 = merged_3steps.get("correct_1", pd.Series(dtype=object))
        c2 = merged_3steps.get("correct_2", pd.Series(dtype=object))
        c3 = merged_3steps.get("correct_3", pd.Series(dtype=object))
        cm = merged_3steps.get("correct_multi", pd.Series(dtype=object))

        if len(c1) > 0 and len(c2) > 0 and len(c3) > 0 and len(cm) > 0:
            RM_loose += len(merged_3steps[(c1 == False) & (c2 == False) & (c3 == False) & (cm == True)])
            RM_strict += len(merged_3steps[((c1 == False) | (c2 == False) | (c3 == False)) & (cm == True)])
            IG += len(merged_3steps[(c1 == True) & (c2 == True) & (c3 == True) & (cm == False)])
            IK += len(merged_3steps[((c1 == False) | (c2 == False) | (c3 == False)) & (cm == False)])
            CM_loose += len(merged_3steps[((c1 == True) | (c2 == True) | (c3 == True)) & (cm == True)])
            CM_strict += len(merged_3steps[(c1 == True) & (c2 == True) & (c3 == True) & (cm == True)])

    total_count = len(merged_2steps) + len(merged_3steps)
    if total_count == 0:
        return {}

    score_loose = (total_count - 0.5 * IG - RM_loose - IK) / total_count
    score_strict = (total_count - 0.5 * IG - RM_strict - IK) / total_count

    metrics = {
        "wemath_score_strict": score_strict,
        "wemath_score_loose": score_loose,
        "wemath_IK": IK / total_count,
        "wemath_IG": IG / total_count,
        "wemath_CM_strict": CM_strict / total_count,
        "wemath_CM_loose": CM_loose / total_count,
        # Official RM rate = RM / (CM + RM), not RM / total_count
        "wemath_RM_strict": RM_strict / (CM_strict + RM_strict) if (CM_strict + RM_strict) > 0 else 0.0,
        "wemath_RM_loose": RM_loose / (CM_loose + RM_loose) if (CM_loose + RM_loose) > 0 else 0.0,
    }

    # Step-wise accuracy (matching official: concatenate individual step jokers)
    one_step_results = []
    if len(merged_2steps) > 0:
        for i in range(1, 3):
            col = f"correct_{i}"
            if col in merged_2steps.columns:
                one_step_results.extend(merged_2steps[col].dropna().tolist())
    if len(merged_3steps) > 0:
        for i in range(1, 4):
            col = f"correct_{i}"
            if col in merged_3steps.columns:
                one_step_results.extend(merged_3steps[col].dropna().tolist())

    if one_step_results:
        metrics["wemath_onestep_acc"] = sum(one_step_results) / len(one_step_results)

    if len(merged_2steps) > 0 and "correct_multi" in merged_2steps.columns:
        vals = merged_2steps["correct_multi"].dropna()
        if len(vals) > 0:
            metrics["wemath_twostep_acc"] = vals.mean()

    if len(merged_3steps) > 0 and "correct_multi" in merged_3steps.columns:
        vals = merged_3steps["correct_multi"].dropna()
        if len(vals) > 0:
            metrics["wemath_threestep_acc"] = vals.mean()

    return metrics


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def compute_score(reward_input: dict[str, Any], format_weight: float = 0.5) -> dict[str, float]:
    """Compute reward with format checking."""
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential` for wemath reward function.")

    answer, _, _ = _parse_ground_truth(reward_input["ground_truth"])
    format_score = format_reward(reward_input["response"])
    accuracy_score = accuracy_reward(reward_input["response"], answer)
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }


def compute_score_without_format(reward_input: dict[str, Any]) -> dict[str, float]:
    """Compute reward without format checking."""
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential` for wemath reward function.")

    answer, _, _ = _parse_ground_truth(reward_input["ground_truth"])
    accuracy_score = accuracy_reward(reward_input["response"], answer)
    return {
        "overall": accuracy_score,
    }


def compute_score_batch_with_wemath_metrics(
    reward_inputs: list[dict[str, Any]],
) -> list[dict[str, float]]:
    """Batch reward function that also computes We-Math four-dimensional metrics.

    Use with: reward.reward_type=batch

    When val_batch_size >= test set size, this function receives ALL test samples
    in one call and can compute the full four-dimensional evaluation.
    """
    scores = []
    val_results = []

    for ri in reward_inputs:
        answer, item_id, key = _parse_ground_truth(ri["ground_truth"])
        acc = accuracy_reward(ri["response"], answer)
        score = {"overall": acc, "accuracy": acc}

        if item_id is not None:
            val_results.append({
                "ID": item_id,
                "key": key,
                "correct": acc > 0,
            })

        scores.append(score)

    # Compute four-dimensional metrics if we have test set samples
    if val_results:
        four_dim = _compute_four_dim_metrics(val_results)
        # Broadcast metrics to all samples so they appear in reduce_metrics
        for metric_key, metric_val in four_dim.items():
            for s in scores:
                s[metric_key] = metric_val

    return scores
