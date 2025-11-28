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
from typing import Any, Optional


# Metadata
REWARD_NAME = "oil_gauge"
REWARD_TYPE = "batch"


def extract_oil_gauge_numbers(text: str) -> tuple[Optional[float], Optional[float]]:
    """
    Extract total grid count and current grid count from text.

    Expected format:
    总油格数：16
    当前油格数：4

    Returns:
        tuple of (total_grids, current_grids) or (None, None) if extraction fails
    """
    total_pattern = r"总油格数[：:]\s*(\d+(?:\.\d+)?)"
    current_pattern = r"当前油格数[：:]\s*(\d+(?:\.\d+)?)"

    total_match = re.search(total_pattern, text)
    current_match = re.search(current_pattern, text)

    total_grids = float(total_match.group(1)) if total_match else None
    current_grids = float(current_match.group(1)) if current_match else None

    return total_grids, current_grids


def format_reward(response: str) -> float:
    """
    Check if the response contains both total grid count and current grid count.

    Returns 1.0 if both are present, 0.5 if only one is present, 0.0 if neither.
    """
    total_grids, current_grids = extract_oil_gauge_numbers(response)

    if total_grids is not None and current_grids is not None:
        return 1.0
    elif total_grids is not None or current_grids is not None:
        return 0.5
    else:
        return 0.0


def accuracy_reward(response: str, ground_truth: str, tolerance: float = 0.5) -> float:
    """
    Calculate accuracy reward for oil gauge recognition.

    Args:
        response: Model's response text
        ground_truth: Ground truth text
        tolerance: Tolerance for current grid count error (default 0.5)

    Returns:
        Float between 0.0 and 1.0 representing accuracy
    """
    pred_total, pred_current = extract_oil_gauge_numbers(response)
    gt_total, gt_current = extract_oil_gauge_numbers(ground_truth)

    # If extraction failed, return 0
    if pred_total is None or pred_current is None:
        return 0.0
    if gt_total is None or gt_current is None:
        return 0.0

    # Total grid count must match exactly (integer comparison)
    total_correct = (int(pred_total) == int(gt_total))

    # Current grid count allows tolerance (for pointer positions)
    current_error = abs(pred_current - gt_current)
    current_correct = current_error <= tolerance

    # Calculate partial credit
    if total_correct and current_correct:
        return 1.0
    elif total_correct:
        # Give partial credit if total is correct but current has error
        # Scale from 0.5 to 0.0 based on error magnitude
        if current_error <= gt_total / 2:  # Error within half range
            return 0.5 * (1.0 - current_error / (gt_total / 2))
        else:
            return 0.0
    elif current_correct:
        # Give small credit if only current is within tolerance
        return 0.3
    else:
        return 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    tolerance: float = 0.5
) -> list[dict[str, float]]:
    """
    Compute reward scores for oil gauge recognition task.

    Args:
        reward_inputs: List of reward input dictionaries
        format_weight: Weight for format score (default 0.1)
        tolerance: Tolerance for current grid count error (default 0.5)

    Returns:
        List of reward score dictionaries
    """
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]

        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, ground_truth, tolerance=tolerance)

        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
