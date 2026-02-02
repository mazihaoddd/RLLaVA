import os
import sys
from typing import Any


def _load_entropy_math():
    reward_dir = os.path.dirname(__file__)
    if reward_dir not in sys.path:
        sys.path.insert(0, reward_dir)
    from entropy_math import compute_score as entropy_compute_score

    return entropy_compute_score


def compute_score(reward_inputs: list[dict[str, Any]], fast: bool = False) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for entropy math reward function.")

    entropy_compute_score = _load_entropy_math()

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
        if isinstance(ground_truth, dict):
            ground_truth = (
                ground_truth.get("ground_truth")
                or ground_truth.get("answer")
                or ground_truth.get("target")
            )
        is_correct = entropy_compute_score(response, ground_truth, fast=fast)
        overall = 1.0 if is_correct else 0.0

        scores.append(
            {
                "overall": overall,
                "format": 0.0,
                "accuracy": overall,
            }
        )

    return scores
