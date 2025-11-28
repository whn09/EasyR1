#!/usr/bin/env python3
"""
Test the oil gauge reward function
"""
import sys
sys.path.insert(0, '/home/ubuntu/EasyR1')

from examples.reward_function.oil_gauge import compute_score


def test_reward_function():
    """Test the reward function with sample inputs"""

    test_cases = [
        {
            "name": "Perfect match",
            "response": "总油格数：16\n当前油格数：4",
            "ground_truth": "总油格数：16\n当前油格数：4",
            "expected_accuracy": 1.0,
        },
        {
            "name": "Within tolerance",
            "response": "总油格数：16\n当前油格数：4.3",
            "ground_truth": "总油格数：16\n当前油格数：4",
            "expected_accuracy": 1.0,  # Within 0.5 tolerance
        },
        {
            "name": "Total correct, current slightly off",
            "response": "总油格数：16\n当前油格数：5",
            "ground_truth": "总油格数：16\n当前油格数：4",
            "expected_accuracy_min": 0.4,  # Partial credit
        },
        {
            "name": "Wrong total",
            "response": "总油格数：8\n当前油格数：4",
            "ground_truth": "总油格数：16\n当前油格数：4",
            "expected_accuracy_max": 0.3,  # Small credit for current match
        },
        {
            "name": "Missing format",
            "response": "The gauge shows 4 out of 16",
            "ground_truth": "总油格数：16\n当前油格数：4",
            "expected_accuracy": 0.0,
            "expected_format": 0.0,
        },
    ]

    print("Testing oil gauge reward function...\n")
    all_passed = True

    for i, test_case in enumerate(test_cases, 1):
        reward_inputs = [
            {
                "response": test_case["response"],
                "ground_truth": test_case["ground_truth"],
                "response_length": len(test_case["response"]),
            }
        ]

        scores = compute_score(reward_inputs)
        score = scores[0]

        print(f"Test {i}: {test_case['name']}")
        print(f"  Response: {test_case['response']}")
        print(f"  Ground truth: {test_case['ground_truth']}")
        print(f"  Scores: overall={score['overall']:.3f}, "
              f"format={score['format']:.3f}, "
              f"accuracy={score['accuracy']:.3f}")

        # Check expectations
        passed = True
        if "expected_accuracy" in test_case:
            if abs(score["accuracy"] - test_case["expected_accuracy"]) > 0.01:
                print(f"  ❌ FAILED: Expected accuracy {test_case['expected_accuracy']}, "
                      f"got {score['accuracy']:.3f}")
                passed = False

        if "expected_accuracy_min" in test_case:
            if score["accuracy"] < test_case["expected_accuracy_min"]:
                print(f"  ❌ FAILED: Expected accuracy >= {test_case['expected_accuracy_min']}, "
                      f"got {score['accuracy']:.3f}")
                passed = False

        if "expected_accuracy_max" in test_case:
            if score["accuracy"] > test_case["expected_accuracy_max"]:
                print(f"  ❌ FAILED: Expected accuracy <= {test_case['expected_accuracy_max']}, "
                      f"got {score['accuracy']:.3f}")
                passed = False

        if "expected_format" in test_case:
            if abs(score["format"] - test_case["expected_format"]) > 0.01:
                print(f"  ❌ FAILED: Expected format {test_case['expected_format']}, "
                      f"got {score['format']:.3f}")
                passed = False

        if passed:
            print("  ✓ PASSED")
        else:
            all_passed = False

        print()

    if all_passed:
        print("All tests passed! ✓")
        return 0
    else:
        print("Some tests failed! ❌")
        return 1


if __name__ == '__main__':
    sys.exit(test_reward_function())
