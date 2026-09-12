"""Pairwise evaluation of two LangSmith experiments."""

from __future__ import annotations

import os
import argparse
from typing import Any, Sequence

from dotenv import load_dotenv
from langsmith import Client
from langsmith.schemas import Example, Run

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

def pairwise_evaluator(
    runs: Sequence[Run],
    example: Example,
) -> dict[str, Any]:
    """Compare outputs from the two experiments.

    Runs are provided in experiment order unless randomization is enabled.
    """

    if len(runs) != 2:
        raise ValueError(f"Expected exactly 2 runs, received {len(runs)}")

    output_a = runs[0].outputs or {}
    output_b = runs[1].outputs or {}

    # Customize this comparison for your task.
    # This example compares an `output` field against the reference answer.
    expected = (example.outputs or {}).get("answer")
    answer_a = output_a.get("output")
    answer_b = output_b.get("output")

    score_a = int(answer_a == expected)
    score_b = int(answer_b == expected)

    if score_a == score_b:
        comment = "Both experiments received the same score."
    elif score_a > score_b:
        comment = "Experiment A was preferred."
    else:
        comment = "Experiment B was preferred."

    return {
        "key": "pairwise_correctness",
        "scores": [score_a, score_b],
        "comment": comment,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-a",
        required=True,
        help="LangSmith experiment/project name or ID for experiment A.",
    )
    parser.add_argument(
        "--experiment-b",
        required=True,
        help="LangSmith experiment/project name or ID for experiment B.",
    )
    parser.add_argument(
        "--description",
        default="Pairwise comparison of two LangSmith experiments",
    )
    parser.add_argument(
        "--randomize-order",
        action="store_true",
        help="Randomize experiment order to reduce position bias.",
    )
    args = parser.parse_args()

    if not os.environ.get("LANGSMITH_API_KEY"):
        raise ValueError(
            "Please set the LANGSMITH_API_KEY environment variable to your LangSmith API key."
        )

    client = Client()

    results = client.evaluate(
        [args.experiment_a, args.experiment_b],
        evaluators=[pairwise_evaluator],
        description=args.description,
        randomize_order=args.randomize_order,
    )

    print(results)


if __name__ == "__main__":
    main()