"""
ARC AGI Solver using the Simplified v2 Framework

Only 3 methods needed in ProblemConfig!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frameworks"))

import json
from typing import Dict, Any, List
from multiagent_framework_v2 import ProblemConfig, solve_problem_collaborative


def format_grid(grid: List[List[int]], indent: str = "  ") -> str:
    """Format a grid as a readable string."""
    if not grid:
        return f"{indent}(empty)"
    lines = []
    for row in grid:
        lines.append(indent + " ".join(str(cell) for cell in row))
    return "\n".join(lines)


class ARCProblemConfig(ProblemConfig[dict, List[List[int]]]):
    """
    ARC AGI problem configuration.

    Problem data format:
    {
        "train": [{"input": [[...]], "output": [[...]]}, ...],
        "test_input": [[...]]
    }
    """

    def __init__(self, train_pairs: List[dict], test_input: List[List[int]]):
        problem_data = {
            "train": train_pairs,
            "test_input": test_input,
        }
        super().__init__(problem_data=problem_data, problem_type="arc_agi_visual_reasoning")

    def format_problem_for_prompt(self) -> str:
        lines = [
            "ARC AGI Problem",
            "=" * 40,
            "",
            "Find the transformation rule from the training examples,",
            "then apply it to the test input.",
            "",
            "TRAINING EXAMPLES:",
        ]

        for i, pair in enumerate(self.problem_data["train"], 1):
            lines.append(f"\n--- Example {i} ---")
            lines.append("Input:")
            lines.append(format_grid(pair["input"]))
            lines.append("Output:")
            lines.append(format_grid(pair["output"]))

        lines.append("\n" + "=" * 40)
        lines.append("TEST INPUT (apply the transformation):")
        lines.append(format_grid(self.problem_data["test_input"]))

        return "\n".join(lines)

    def get_answer_format(self) -> str:
        return "A 2D grid (list of lists of integers), e.g., [[0,1],[1,0]]"

    def parse_final_answer(self, final: Dict[str, Any]) -> List[List[int]]:
        # Try different keys the model might use
        answer = final.get("answer") or final.get("grid") or final.get("output")
        if isinstance(answer, list):
            return answer
        # Try to parse if it's a string
        if isinstance(answer, str):
            try:
                return json.loads(answer)
            except:
                pass
        return []

    def validate_answer(self, answer: List[List[int]]) -> bool:
        """Check that answer is a valid non-empty grid."""
        if not isinstance(answer, list) or len(answer) == 0:
            return False
        if not all(isinstance(row, list) for row in answer):
            return False
        if len(answer[0]) == 0:
            return False
        # Check all rows have same length
        row_len = len(answer[0])
        return all(len(row) == row_len for row in answer)


# Convenience function
def solve_arc_problem(
    train_pairs: List[dict],
    test_input: List[List[int]],
    problem_id: str = "arc_001",
    **kwargs,
):
    """Solve an ARC AGI problem using the multiagent framework."""
    config = ARCProblemConfig(train_pairs, test_input)
    return solve_problem_collaborative(
        problem_config=config,
        problem_id=problem_id,
        **kwargs,
    )


if __name__ == "__main__":
    # Example: Invert the grid (0 becomes 1, 1 becomes 0)
    train_pairs = [
        {
            "input": [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
            "output": [[1, 1, 0], [1, 0, 1], [0, 1, 1]],
        },
        {
            "input": [[1, 1], [0, 0]],
            "output": [[0, 0], [1, 1]],
        },
        {
            "input": [[0, 1, 0, 1], [1, 0, 1, 0]],
            "output": [[1, 0, 1, 0], [0, 1, 0, 1]],
        },
    ]
    test_input = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    result = solve_arc_problem(
        train_pairs=train_pairs,
        test_input=test_input,
        problem_id="arc_invert",
        model="gpt-5-nano",
        out_dir="output/arc",
        # num_agents auto-determined (2-8), max_rounds auto (verifier decides)
        verbose=True,
    )

    print(f"\nFinal Answer:")
    print(format_grid(result['answer'], indent=""))
    print(f"\nExpected: [[0,1,1], [1,0,1], [1,1,0]]")
    print(f"Valid grid: {result['is_valid']}")
    print(f"Rounds: {result['num_rounds']}")
